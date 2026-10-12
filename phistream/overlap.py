"""Overlap Intelligence: bounded synthetic single/ephemeral/persistent worker lab.

All traces use CPU monotonic clock. CUDA events measure GPU duration in a
different clock domain: never conflate host enqueue and event completion with
a calibrated GPU execution interval. No model code or existing file inputs.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import platform
import shutil
import statistics
import tempfile
import time

from .chunklab import (
    _CudaBackend, _CpuBackend, _check_parameters, _fixture, _read_at
)
from .hardware import physical_memory
from .reuse import _duration
from .transfer import MIB

SCHEMA = "PHISTREAM_OVERLAP_LAB_V1"
PROVENANCE = "MEASURED_LOCAL_SYNTHETIC_HOST_TIMELINE_NO_MODEL"
METHODS = ("single", "double_new_worker", "double_reuse_worker")
THRESHOLD = 0.05
MIN_CONSISTENCY = 0.75


def _order(rounds: int) -> list[list[str]]:
    """For six rounds, all six method permutations exactly once."""
    from itertools import permutations
    if type(rounds) is not int or rounds not in (6, 12):
        raise ValueError("rounds must be 6 or 12")
    base = [list(p) for p in permutations(METHODS)]
    return base * (rounds // 6)


def recommend(single: list[float], reused: list[float], *, gpu: bool) -> dict:
    if not gpu:
        return {"decision": "NO_GPU_EVIDENCE",
                "default": "SINGLE",
                "reason": "CPU scheduler tests cannot justify a CUDA policy."}
    if not single or len(single) != len(reused) or any(
        not math.isfinite(x) or x <= 0 for x in single + reused
    ):
        raise ValueError("Need paired, finite positive elapsed times")
    if len(single) < 6:
        raise ValueError("At least six paired trials required")
    gains = [(s - d) / s for s, d in zip(single, reused)]
    median_gain = statistics.median(gains)
    double_wins = sum(g > 0 for g in gains)
    single_wins = sum(g < 0 for g in gains)
    min_wins = math.ceil(MIN_CONSISTENCY * len(gains))
    if median_gain >= THRESHOLD and double_wins >= min_wins:
        verdict = "DOUBLE_CANDIDATE"
        reason = "Persistent double won by >=5% median paired time and >=75% of trials."
    elif median_gain <= -THRESHOLD and single_wins >= min_wins:
        verdict = "SINGLE_CANDIDATE"
        reason = "Single won by >=5% median paired time and >=75% of trials."
    else:
        verdict = "INCONCLUSIVE"
        reason = "Speed difference or paired consistency is below the predeclared threshold."
    return {
        "decision": verdict,
        "default": "SINGLE",
        "median_paired_relative_gain_double": round(median_gain, 6),
        "double_wins": double_wins, "single_wins": single_wins,
        "minimum_wins": min_wins,
        "minimum_relative_gain": THRESHOLD,
        "reason": reason,
        "policy_applied": False,
    }


def _trial(path: Path, chunk_bytes: int, chunks: int, method: str,
           backend, permanent: ThreadPoolExecutor | None = None) -> dict:
    if method not in METHODS:
        raise ValueError("Unrecognized overlap method")
    if method == "double_reuse_worker" and permanent is None:
        raise ValueError("Persistent worker required")

    active = [None, None]
    timings = [None] * chunks
    tickets = []
    wait_total_ns = 0
    pending_at_read_start = 0
    pending_at_queue = 0
    start_ns = time.perf_counter_ns()
    local_pool = None

    def ms(stamp):
        return round((stamp - start_ns) / 1e6, 6)

    def drain(slot):
        nonlocal wait_total_ns
        ticket = active[slot]
        if ticket is not None:
            before = time.perf_counter_ns()
            backend.wait(ticket)
            finished = time.perf_counter_ns()
            wait_total_ns += finished - before
            timings[ticket["index"]]["completion_observed_ms"] = ms(finished)
            timings[ticket["index"]]["wait_ms"] = round((finished-before) / 1e6, 6)
            active[slot] = None

    def read_chunk(index: int, slot: int, prior_ticket=None):
        nonlocal pending_at_read_start
        began = time.perf_counter_ns()
        pending = bool(prior_ticket is not None and not backend.pending(prior_ticket))
        if pending:
            pending_at_read_start += 1
        duration = _read_at(path, index * chunk_bytes, backend.views[slot])
        finished = time.perf_counter_ns()
        return {
            "chunk": index, "slot": slot,
            "read_start_ms": ms(began), "read_end_ms": ms(finished),
            "read_ms": round(duration * 1e3, 6),
            "copy_pending_at_read_start": pending,
        }

    def submit(index, slot, row):
        before = time.perf_counter_ns()
        ticket = backend.submit(slot)
        after = time.perf_counter_ns()
        row["enqueue_start_ms"] = ms(before)
        row["enqueue_return_ms"] = ms(after)
        row["queue_to_enqueue_ms"] = round((after - before) / 1e6, 6)
        timings[index] = row
        ticket["index"] = index
        active[slot] = ticket
        tickets.append(ticket)

    try:
        if method == "single":
            for index in range(chunks):
                drain(0)
                row = read_chunk(index, 0)
                submit(index, 0, row)
                drain(0)
        else:
            pool = permanent
            if pool is None:
                # Worker creation happens inside measured trial only in this
                # mode, while permanent worker startup is reported separately.
                pool = ThreadPoolExecutor(max_workers=1)
                local_pool = pool

            def queue_read(index, slot, previous=None):
                nonlocal pending_at_queue
                queued = time.perf_counter_ns()
                if previous is not None and not backend.pending(previous):
                    pending_at_queue += 1
                return pool.submit(read_chunk, index, slot, previous), queued

            future, queued = queue_read(0, 0)
            for index in range(chunks):
                slot = index % 2
                row = future.result()
                row["queued_ms"] = ms(queued)
                row["worker_queue_delay_ms"] = max(0, round(
                    row["read_start_ms"] - ms(queued), 6
                ))
                submit(index, slot, row)
                if index + 1 < chunks:
                    next_slot = (index + 1) % 2
                    drain(next_slot)
                    future, queued = queue_read(index + 1, next_slot, tickets[-1])
    finally:
        # Shut down a temporary worker before releasing pinned host buffers.
        # Any failed read/submit must still wait for all active GPU events.
        try:
            if local_pool is not None:
                local_pool.shutdown(wait=True)
        finally:
            for slot in (0, 1):
                drain(slot)

    end_ns = time.perf_counter_ns()
    return {
        "elapsed_seconds": (end_ns - start_ns) / 1e9,
        "read_sum_seconds": sum(row["read_ms"] for row in timings) / 1e3,
        "worker_queue_sum_seconds": sum(row.get("worker_queue_delay_ms", 0) for row in timings) / 1e3,
        "enqueue_sum_seconds": sum(row["queue_to_enqueue_ms"] for row in timings) / 1e3,
        "wait_sum_seconds": wait_total_ns / 1e9,
        "gpu_event_copy_sum_seconds": sum(backend.transfer_ms(t) for t in tickets) / 1e3,
        "pending_at_read_start": pending_at_read_start,
        "pending_at_queue": pending_at_queue,
        "host_timeline": timings,
    }


def run_overlap_lab(workspace: str | Path, *, gpu="none", chunk_mib=8,
                    chunks=8, rounds=6, warmups=1) -> dict:
    if gpu not in ("none", "cuda"):
        raise ValueError("gpu must be none or cuda")
    _check_parameters(chunk_mib, chunks, 6, warmups)
    order = _order(rounds)
    root = Path(workspace)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Existing nonsymlink scratch workspace required")
    count = chunk_mib * MIB
    total_bytes = count * chunks
    if shutil.disk_usage(root).free < 3 * total_bytes + 128 * MIB:
        raise RuntimeError("Not enough scratch disk space")
    if physical_memory()["available_bytes"] < 6 * count + 256 * MIB:
        raise RuntimeError("Insufficient available system RAM")
    backend = _CudaBackend(count) if gpu == "cuda" else _CpuBackend(count)
    results = {m: [] for m in METHODS}
    first_timeline = {}
    started = datetime.now(timezone.utc).isoformat()

    with tempfile.TemporaryDirectory(prefix="phistream-overlap-", dir=root) as tmp:
        fixture = Path(tmp) / "owned-synthetic-chunks.bin"
        checksum, chunk_hashes, write_s = _fixture(fixture, count, chunks)
        with ThreadPoolExecutor(max_workers=1) as worker:
            # A separately accounted persistent worker exists across all
            # trials. Its warm start is intentionally outside trial clocks.
            startup_ns = time.perf_counter_ns()
            worker.submit(lambda: None).result()
            permanent_setup_seconds = (time.perf_counter_ns() - startup_ns) / 1e9
            def run(method):
                return _trial(fixture, count, chunks, method, backend, worker)
            for _ in range(warmups):
                for method in METHODS:
                    run(method)
            for sequence in order:
                for method in sequence:
                    reading = run(method)
                    results[method].append(reading)
                    if method not in first_timeline:
                        first_timeline[method] = reading["host_timeline"]
        audit = backend.replay_audit(fixture, chunk_hashes, count)
        with fixture.open("rb") as file:
            if hashlib.sha256(file.read()).hexdigest() != checksum:
                raise RuntimeError("Scratch fixture SHA256 differs")
    stats = {
        method: {
            "end_to_end": _duration([r["elapsed_seconds"] for r in rows]),
            "read_sum": _duration([r["read_sum_seconds"] for r in rows]),
            "worker_queue_sum": _duration([r["worker_queue_sum_seconds"] for r in rows]),
            "host_enqueue_sum": _duration([r["enqueue_sum_seconds"] for r in rows]),
            "cuda_wait_sum": _duration([r["wait_sum_seconds"] for r in rows]),
            "gpu_event_copy_sum": _duration([r["gpu_event_copy_sum_seconds"] for r in rows]),
            "pending_at_worker_read_start_total": sum(r["pending_at_read_start"] for r in rows),
            "pending_at_queue_total": sum(r["pending_at_queue"] for r in rows),
            "first_measured_trial_host_timeline": first_timeline[method],
        } for method, rows in results.items()
    }
    recommendation = recommend(
        [r["elapsed_seconds"] for r in results["single"]],
        [r["elapsed_seconds"] for r in results["double_reuse_worker"]],
        gpu=gpu == "cuda",
    )
    return {
        "schema": SCHEMA, "provenance": PROVENANCE,
        "status": "PASS_CUDA_OVERLAP_LAB" if gpu == "cuda" else "PASS_CPU_OVERLAP_SCHEDULER",
        "started_utc": started, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "workspace": str(root),
        "gpu_requested": gpu, "chunk_mib": chunk_mib, "chunks": chunks,
        "fixture": {"synthetic": True, "total_bytes": total_bytes,
                    "sha256": checksum, "scratch_deleted": True,
                    "write_fsync_seconds": round(write_s, 9)},
        "design": {
            "methods": list(METHODS), "balanced_orders": order,
            "paired_rounds": rounds, "warmups_per_method": warmups,
            "same_preallocated_host_device_buffer_budget": True,
            "persistent_worker_startup_seconds_excluded": round(permanent_setup_seconds, 9),
            "scratch_cache_controlled": False,
            "gpu_clock_aligned_with_cpu": False,
            "gpu_compute_overlap_tested": False,
            "host_buffers_reused_only_after_cuda_completion_event": True,
        },
        "methods": stats, "recommendation": recommendation,
        "validation": audit,
        "hardware": ({
            "gpu_name": backend.name, "cuda_build": backend.build,
            "torch_version": backend.version,
            "free_vram_before_bytes": backend.free_before,
        } if gpu == "cuda" else None),
        "model_execution": False, "automatic_layer_placement": False,
        "limitations": [
            "Only synthetic owned temporary bytes; no pretrained checkpoint, model inference or actual layer prefetch.",
            "CPU host timestamps are comparable with one another; GPU events have an uncalibrated separate clock domain.",
            "Read and CUDA event durations cannot establish physical I/O/GPU overlap without cross-clock calibration or a profiler.",
            "Recorded CUDA completion is a host observation after wait, not exact device completion timestamp.",
            "Background worker setup is included in new-worker trials but excluded from persistent-worker trials; disclosed separately.",
            "Host filesystem cache is uncontrolled. Warm reads are not raw NVMe drive measurements.",
            "No GPU compute kernel or computation/data-transfer overlap is tested.",
            "Full chunk-by-chunk GPU integrity is checked in a separate replay outside timing.",
            "An evidence-gated result is an advisory candidate only; no scheduler policy is applied.",
            "Generated receipt contains user-selected paths; review before sharing.",
        ],
    }
