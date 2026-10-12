"""Bounded synthetic single- versus double-buffer host-to-CUDA stream lab.

Never reads existing model weights; only creates and removes its own private
scratch fixture. Explicit CUDA completion events govern buffer ownership.
No claim of inference acceleration, cold-cache NVMe throughput, or measured
GPU-compute overlap. License-independent, original implementation.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import platform
import shutil
import statistics
import tempfile
import time

from .hardware import physical_memory
from .reuse import _duration
from .transfer import MIB

SCHEMA = "PHISTREAM_CHUNK_PIPELINE_V1"
PROVENANCE = "MEASURED_LOCAL_SYNTHETIC_CHUNK_STREAM_NO_MODEL"
METHODS = ("single", "double")


def _check_parameters(chunk_mib: int, chunks: int, repeats: int, warmups: int):
    if type(chunk_mib) is not int or chunk_mib not in (1, 2, 4, 8, 16):
        raise ValueError("chunk-mib must be one of 1, 2, 4, 8, 16")
    if type(chunks) is not int or not 4 <= chunks <= 32:
        raise ValueError("chunks must be an integer from 4 to 32")
    if chunk_mib * chunks > 256:
        raise ValueError("Total synthetic fixture must not exceed 256 MiB")
    if type(repeats) is not int or repeats not in (4, 6, 8):
        raise ValueError("repeats must be 4, 6, or 8")
    if type(warmups) is not int or not 1 <= warmups <= 3:
        raise ValueError("warmups must be an integer from 1 to 3")


def _fixture(path: Path, chunk_bytes: int, chunks: int):
    """Unique content per chunk, deterministic; write only exclusively owned file."""
    whole = hashlib.sha256()
    hashes = []
    started = time.perf_counter()
    with path.open("xb") as f:
        for index in range(chunks):
            block = bytes((x + index * 17) % 256 for x in range(256))
            # The supported chunks are MiB multiples, always divisible by 256.
            data = block * (chunk_bytes // 256)
            f.write(data)
            whole.update(data)
            hashes.append(hashlib.sha256(data).hexdigest())
        f.flush()
        os.fsync(f.fileno())
    return whole.hexdigest(), hashes, time.perf_counter() - started


def _read_at(path: Path, offset: int, view: memoryview):
    if view.readonly or not view.contiguous or view.nbytes < 1:
        raise ValueError("Owned read buffer must be a contiguous writable memoryview")
    start = time.perf_counter()
    with path.open("rb", buffering=0) as f:
        f.seek(offset)
        actual = f.readinto(view)
    elapsed = time.perf_counter() - start
    if actual != view.nbytes:
        raise RuntimeError("Short read in owned scratch chunk")
    return elapsed


def _run_sequence(path: Path, chunk_bytes: int, chunks: int, mode: str,
                  buffers: list[memoryview], submit, wait, query=None):
    """Coordinate buffer ownership: wait for CUDA completion BEFORE next read.

    submit(slot) returns an event/ticket whose wait must establish completion.
    CPU read is executed on a worker for double mode, potentially concurrent
    with the previous CUDA copy. We only schedule a read on a released slot.
    """
    if mode not in METHODS:
        raise ValueError("Unsupported method")
    if len(buffers) != 2:
        raise ValueError("The experiment preallocates two equal host buffers")
    if any(b.nbytes != chunk_bytes or b.readonly for b in buffers):
        raise ValueError("Incorrect host buffers")
    events = [None, None]
    tickets = []
    read_seconds = []
    wait_seconds = 0.0
    possible_overlap = 0
    wall_start = time.perf_counter()

    def release(slot):
        nonlocal wait_seconds
        event = events[slot]
        if event is not None:
            start = time.perf_counter()
            wait(event)
            wait_seconds += time.perf_counter() - start
            events[slot] = None

    try:
        if mode == "single":
            for index in range(chunks):
                release(0)
                read_seconds.append(_read_at(path, index * chunk_bytes, buffers[0]))
                event = submit(0)
                events[0] = event
                tickets.append(event)
                release(0)
        else:
            # Read chunk 0 first. Thereafter the worker reads the next
            # released buffer while current CUDA H2D may still be pending.
            with ThreadPoolExecutor(max_workers=1) as worker:
                next_read = worker.submit(_read_at, path, 0, buffers[0])
                for index in range(chunks):
                    slot = index % 2
                    read_seconds.append(next_read.result())
                    event = submit(slot)
                    events[slot] = event
                    tickets.append(event)
                    if index + 1 < chunks:
                        next_slot = (index + 1) % 2
                        release(next_slot)
                        if query is not None and not query(event):
                            possible_overlap += 1
                        next_read = worker.submit(
                            _read_at, path, (index + 1) * chunk_bytes,
                            buffers[next_slot],
                        )
    finally:
        # Even on a short read, failed thread or CUDA exception, wait for
        # outstanding GPU reads of pinned buffers before they are destroyed
        # or returned to the allocator. Propagate the primary failure.
        for slot in (0, 1):
            release(slot)
    elapsed = time.perf_counter() - wall_start
    return {
        "elapsed_seconds": elapsed,
        "read_samples_seconds": read_seconds,
        "wait_seconds_total": wait_seconds,
        "read_scheduled_while_previous_copy_pending": possible_overlap,
        "tickets": tickets,
    }


class _CudaBackend:
    def __init__(self, chunk_bytes: int):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no fallback")
        free, total = torch.cuda.mem_get_info()
        if free < 4 * chunk_bytes + 256 * MIB:
            raise RuntimeError("Insufficient free CUDA memory for bounded stream lab")
        self.torch = torch
        self.host = [torch.empty(chunk_bytes, dtype=torch.uint8, pin_memory=True)
                     for _ in range(2)]
        self.device = [torch.empty(chunk_bytes, dtype=torch.uint8, device="cuda")
                       for _ in range(2)]
        self.views = [memoryview(t.numpy()).cast("B") for t in self.host]
        if any(v.readonly or v.nbytes != chunk_bytes for v in self.views):
            raise RuntimeError("Pinned host memory cannot export writable buffer")
        self.stream = torch.cuda.Stream()
        self.free_before = int(free)
        self.total = int(total)
        self.name = torch.cuda.get_device_name()
        self.version = str(torch.__version__)
        self.build = str(torch.version.cuda) if torch.version.cuda else None

    def submit(self, slot):
        torch = self.torch
        before = torch.cuda.Event(enable_timing=True)
        after = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self.stream):
            before.record(self.stream)
            self.device[slot].copy_(self.host[slot], non_blocking=True)
            after.record(self.stream)
        return {"complete": after, "start": before, "slot": slot}

    @staticmethod
    def wait(ticket):
        ticket["complete"].synchronize()

    @staticmethod
    def pending(ticket):
        return ticket["complete"].query()

    @staticmethod
    def transfer_ms(ticket):
        return float(ticket["start"].elapsed_time(ticket["complete"]))

    def replay_audit(self, path: Path, digests: list[str], chunk_bytes: int):
        """Outside timing: full CPU SHA and GPU-returned SHA for every chunk."""
        for index, expected in enumerate(digests):
            slot = index % 2
            _read_at(path, index * chunk_bytes, self.views[slot])
            if hashlib.sha256(self.views[slot]).hexdigest() != expected:
                raise RuntimeError("Full host chunk SHA256 mismatch")
            ticket = self.submit(slot)
            self.wait(ticket)
            returned = self.device[slot].cpu().numpy().tobytes()
            if hashlib.sha256(returned).hexdigest() != expected:
                raise RuntimeError("Full GPU-returned chunk SHA256 mismatch")
        return {"full_sha256_host_and_gpu_audit": True,
                "audit_chunks": len(digests), "audit_is_outside_timing": True}


class _CpuBackend:
    """CI-only fake transfer semantics; *not* simulated GPU benchmark evidence."""
    def __init__(self, chunk_bytes):
        self.views = [memoryview(bytearray(chunk_bytes)) for _ in range(2)]
        self.device = [bytearray(chunk_bytes) for _ in range(2)]

    def submit(self, slot):
        self.device[slot][:] = self.views[slot]
        return {"complete": True, "slot": slot}

    @staticmethod
    def wait(ticket):
        if ticket["complete"] is not True:
            raise RuntimeError("CPU fake transfer not complete")

    @staticmethod
    def pending(ticket):
        return True

    @staticmethod
    def transfer_ms(ticket):
        return 0.0

    def replay_audit(self, path, digests, chunk_bytes):
        for index, expected in enumerate(digests):
            slot = index % 2
            _read_at(path, index * chunk_bytes, self.views[slot])
            if hashlib.sha256(self.views[slot]).hexdigest() != expected:
                raise RuntimeError("Full host chunk integrity failure")
            ticket = self.submit(slot)
            self.wait(ticket)
            if hashlib.sha256(self.device[slot]).hexdigest() != expected:
                raise RuntimeError("CPU transfer integrity failure")
        return {"full_sha256_host_and_gpu_audit": False,
                "audit_chunks": len(digests), "audit_is_outside_timing": True,
                "cpu_only_simulation": True}


def _alternating_orders(repeats):
    return [list(METHODS) if i % 2 == 0 else list(reversed(METHODS))
            for i in range(repeats)]


def run_chunk_lab(workspace: str | Path, *, gpu="none", chunk_mib=4,
                  chunks=16, repeats=6, warmups=1):
    if gpu not in ("none", "cuda"):
        raise ValueError("gpu must be 'none' or 'cuda'")
    _check_parameters(chunk_mib, chunks, repeats, warmups)
    root = Path(workspace)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Existing nonsymlink scratch workspace required")
    chunk_bytes = chunk_mib * MIB
    total_bytes = chunk_bytes * chunks
    if shutil.disk_usage(root).free < total_bytes * 3 + 128 * MIB:
        raise RuntimeError("Insufficient scratch disk free space")
    ram_avail = physical_memory()["available_bytes"]
    if ram_avail < chunk_bytes * 6 + 256 * MIB:
        raise RuntimeError("Insufficient available host RAM")
    backend = _CudaBackend(chunk_bytes) if gpu == "cuda" else _CpuBackend(chunk_bytes)
    orders = _alternating_orders(repeats)
    started = datetime.now(timezone.utc).isoformat()
    samples = {m: [] for m in METHODS}
    wait_times = {m: [] for m in METHODS}
    read_times = {m: [] for m in METHODS}
    transfer_times = {m: [] for m in METHODS}
    overlap_hint = {m: 0 for m in METHODS}
    with tempfile.TemporaryDirectory(prefix="phistream-chunk-", dir=root) as tmp:
        scratch = Path(tmp) / "owned-checkpoint-like-bytes.bin"
        digest, chunk_hashes, write_seconds = _fixture(scratch, chunk_bytes, chunks)
        def trial(method):
            output = _run_sequence(
                scratch, chunk_bytes, chunks, method,
                backend.views, backend.submit, backend.wait, backend.pending,
            )
            # Events are synchronized before return; CUDA event times exclude
            # CPU reads and thread scheduling, unlike full wall-clock time.
            transfers = [backend.transfer_ms(t) / 1000 for t in output["tickets"]]
            return output, transfers
        for _ in range(warmups):
            for method in METHODS:
                trial(method)
        for pair in orders:
            for method in pair:
                observation, cuda_seconds = trial(method)
                samples[method].append(observation["elapsed_seconds"])
                wait_times[method].append(observation["wait_seconds_total"])
                read_times[method].append(sum(observation["read_samples_seconds"]))
                transfer_times[method].append(sum(cuda_seconds))
                overlap_hint[method] += observation[
                    "read_scheduled_while_previous_copy_pending"]
        audit = backend.replay_audit(scratch, chunk_hashes, chunk_bytes)
        with scratch.open("rb") as f:
            if hashlib.sha256(f.read()).hexdigest() != digest:
                raise RuntimeError("Original synthetic fixture was modified")
    by_method = {
        name: {
            "end_to_end": _duration(samples[name]),
            "sum_file_reads": _duration(read_times[name]),
            "cuda_copy_event_sum": _duration(transfer_times[name]),
            "synchronization_wait_total": _duration(wait_times[name]),
            "reads_scheduled_with_pending_copy": overlap_hint[name],
        } for name in METHODS
    }
    double_wins = sum(b < a for a, b in zip(samples["single"], samples["double"]))
    median_ratio = statistics.median(
        a / b for a, b in zip(samples["single"], samples["double"])
    )
    return {
        "schema": SCHEMA, "provenance": PROVENANCE,
        "status": "PASS_CUDA_CHUNK_LAB" if gpu == "cuda" else "PASS_CPU_CHUNK_SCHEDULER",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "workspace": str(root),
        "gpu_requested": gpu, "chunk_mib": chunk_mib, "chunks": chunks,
        "fixture": {
            "total_bytes": total_bytes, "sha256": digest,
            "per_chunk_sha256_count": len(chunk_hashes),
            "write_fsync_seconds": round(write_seconds, 9),
            "scratch_deleted": True, "synthetic": True,
        },
        "design": {
            "methods": list(METHODS), "orders": orders,
            "paired_rounds": repeats, "warmups_per_method": warmups,
            "warmups_excluded": True, "cuda_pinned_host_buffers": 2 if gpu == "cuda" else 0,
            "same_two_buffers_preallocated_for_both_methods": True,
            "cuda_transfer_non_blocking": gpu == "cuda",
            "wait_for_completion_before_host_buffer_reuse": True,
            "os_cache_controlled": False, "gpu_compute_overlap_tested": False,
        },
        "methods": by_method,
        "paired_outcome": {
            "double_wins": double_wins,
            "single_wins_or_ties": repeats - double_wins,
            "median_single_over_double_time": round(median_ratio, 5),
            "time_ratio_favors": ("DOUBLE" if median_ratio > 1 else
                                  "SINGLE" if median_ratio < 1 else "TIE"),
        },
        "validation": audit,
        "hardware": ({
            "gpu_name": backend.name, "free_vram_before_bytes": backend.free_before,
            "gpu_total_bytes": backend.total, "torch_version": backend.version,
            "cuda_build": backend.build,
        } if gpu == "cuda" else None),
        "model_execution": False, "automatic_layer_placement": False,
        "limitations": [
            "Temporary synthetic bytes only, NOT Llama inference, layer prefetch or tensor format conversion.",
            "OS cache is uncontrolled, and this experiment does not measure cold-cache SSD throughput.",
            "Both methods preallocate two pinned CPU and GPU buffers to keep allocation budgets equal.",
            "Double buffering uses a worker to read upcoming bytes while a CUDA copy might be pending.",
            "A pending-event observation is NOT proof of measured device/file-read time overlap.",
            "CUDA events time the copies only; the CPU wall clock includes file I/O, thread scheduling and waits.",
            "Full SHA256 host/GPU integrity is checked in a separate replay OUTSIDE benchmark timing.",
            "Asynchronous trial destination is not independently hashed before slot reuse; correctness relies on event waits and audited identical transfer code.",
            "No compute kernels or GPU compute/data overlap, no driver-controlled DMA, no production model files.",
            "Local receipts are user supplied, unauthenticated; review paths before sharing.",
        ],
    }
