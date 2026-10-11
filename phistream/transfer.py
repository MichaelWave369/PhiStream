"""Bounded, opt-in transfer laboratory for SSD/host/CUDA tier experiments.

Original implementation. Never reads or modifies model files. Only uses a
random, owned temporary directory within an explicitly selected workspace.
No 'cold cache', sustained SSD bandwidth, GPU acceleration, or LLM claims.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import tempfile
import time

SCHEMA = "PHISTREAM_TRANSFER_LAB_V1"
PROVENANCE = "MEASURED_LOCAL_SYNTHETIC_BYTES_NO_MODEL"
MIB = 1024 * 1024
MAX_MIB = 256
MIN_MIB = 1
MIN_REPEAT = 3
MAX_REPEAT = 15


def _stats(samples: list[float], size_bytes: int) -> dict:
    if not samples or any(not math.isfinite(s) or s <= 0 for s in samples):
        raise RuntimeError("Invalid elapsed measurements; cannot report a bandwidth")
    middle = statistics.median(samples)
    return {
        "samples_seconds": [round(t, 9) for t in samples],
        "median_seconds": round(middle, 9),
        "min_seconds": round(min(samples), 9),
        "max_seconds": round(max(samples), 9),
        "median_mib_per_second": round(size_bytes / MIB / middle, 3),
        "repetitions": len(samples),
        "byte_count_per_sample": size_bytes,
    }


def _write_fixture(path: Path, total_bytes: int) -> tuple[str, float]:
    """Create a deterministic bounded byte fixture, exclusive write only."""
    block = bytes(range(256)) * 16384   # 4 MiB; no random external content
    digest = hashlib.sha256()
    start = time.perf_counter()
    with path.open("xb") as fh:
        remaining = total_bytes
        while remaining:
            part = block[: min(remaining, len(block))]
            fh.write(part)
            digest.update(part)
            remaining -= len(part)
        fh.flush()
        os.fsync(fh.fileno())
    return digest.hexdigest(), time.perf_counter() - start


def _read_exact(path: Path, total_bytes: int, expected_digest: str) -> tuple[bytes, float]:
    start = time.perf_counter()
    with path.open("rb") as fh:
        data = fh.read(total_bytes + 1)
    elapsed = time.perf_counter() - start
    if len(data) != total_bytes:
        raise RuntimeError("Scratch file changed size; aborting transfer lab")
    if hashlib.sha256(data).hexdigest() != expected_digest:
        raise RuntimeError("Scratch content hash mismatch; aborting transfer lab")
    return data, elapsed


def _host_staging(data: bytes, repeats: int) -> dict:
    samples = []
    copied = None
    for _ in range(repeats):
        started = time.perf_counter()
        copied = bytearray(data)
        elapsed = time.perf_counter() - started
        if copied != data:
            raise RuntimeError("Host RAM copy differs from read fixture")
        samples.append(elapsed)
    return _stats(samples, len(data))


def _gpu_transfers(data: bytes, repeats: int) -> dict:
    """CUDA pageable vs pinned transfer wall-time, synchronization included.

    All CUDA buffers are bounded by input fixture, allocated once and reused.
    Staging pinned memory is measured separately from the H2D copy, so users
    do not mistake pinned H2D alone for total end-to-end throughput.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA explicitly requested but unavailable; no CPU fallback")
    free_before, total = torch.cuda.mem_get_info()
    count = len(data)
    # Conservative allocation safety margin. Actual runtime uses more memory;
    # CUDA OOM may still occur and must be treated as failed, not retried larger.
    if free_before < count * 4 + 256 * MIB:
        raise RuntimeError("Insufficient CUDA free memory for guarded transfer experiment")
    source = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    if source.is_pinned():
        raise RuntimeError("Pageable source unexpectedly pinned")
    try:
        pinned = torch.empty(count, dtype=torch.uint8, pin_memory=True)
        dest = torch.empty(count, dtype=torch.uint8, device="cuda")
    except RuntimeError as exc:
        raise RuntimeError("CUDA/pinned host allocation failed; no fallback") from exc
    stage_times = []
    pageable_times = []
    pinned_times = []
    for run in range(repeats):
        start = time.perf_counter()
        pinned.copy_(source)
        stage_times.append(time.perf_counter() - start)
        # Swap benchmark order every pass to reduce one-direction order bias.
        for mode in (("pageable", "pinned") if run % 2 == 0 else ("pinned", "pageable")):
            torch.cuda.synchronize()
            start = time.perf_counter()
            dest.copy_(source if mode == "pageable" else pinned, non_blocking=False)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if mode == "pageable":
                pageable_times.append(elapsed)
            else:
                pinned_times.append(elapsed)
    # A small deterministic check of both source and device output; this is a
    # correctness guard, not a throughput measurement.
    sample_points = sorted({0, count // 2, count - 1})
    actual = dest[sample_points].cpu().tolist()
    expected = [data[i] for i in sample_points]
    if actual != expected:
        raise RuntimeError("CUDA transfer content check failed")
    return {
        "gpu_name": torch.cuda.get_device_name(),
        "torch_version": str(torch.__version__),
        "cuda_build": str(torch.version.cuda) if torch.version.cuda else None,
        "device_free_bytes_before": int(free_before),
        "device_total_bytes": int(total),
        "copy_direction": "HOST_TO_DEVICE",
        "pageable_h2d": _stats(pageable_times, count),
        "pinned_h2d": _stats(pinned_times, count),
        "pageable_to_pinned_host_stage": _stats(stage_times, count),
        "pinned_total_median_seconds_approx": round(
            statistics.median(stage_times) + statistics.median(pinned_times), 9),
        "correctness_sampled_bytes_checked": len(sample_points),
        "measurement_type": "WALL_CLOCK_WITH_EXPLICIT_CUDA_SYNCHRONIZATION",
        "note": "Pinned-only H2D rate excludes CPU pin staging; CUDA buffers are preallocated.",
    }


def run_transfer_lab(
    workspace: str | Path, *, gpu: str = "none", size_mib: int = 16,
    repeats: int = 5,
) -> dict:
    if gpu not in ("none", "cuda"):
        raise ValueError("gpu must be none or cuda; no implicit fallback")
    if type(size_mib) is not int or not MIN_MIB <= size_mib <= MAX_MIB:
        raise ValueError("size_mib must be an integer from 1 to 256")
    if type(repeats) is not int or not MIN_REPEAT <= repeats <= MAX_REPEAT:
        raise ValueError("repeats must be an integer from 3 to 15")
    folder = Path(workspace)
    if not folder.is_dir() or folder.is_symlink():
        raise ValueError("Workspace must be an existing nonsymlink directory")
    bytes_count = size_mib * MIB
    drive = shutil.disk_usage(folder)
    # Limit scratch space consumption, keeping several fixture sizes free.
    if drive.free < bytes_count * 4 + 128 * MIB:
        raise RuntimeError("Insufficient free disk space for bounded scratch fixture")
    # Detect absent CUDA *before* creating a scratch fixture.
    if gpu == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no CPU fallback")

    started = datetime.now(timezone.utc).isoformat()
    try:
        # All filesystem writes are confined to a temporary subdirectory
        # created exclusively by this process and removed on exit.
        with tempfile.TemporaryDirectory(prefix="phistream-transfer-", dir=folder) as tmp:
            fixture = Path(tmp) / "owned-fixture.dat"
            digest, write_elapsed = _write_fixture(fixture, bytes_count)
            data, first_elapsed = _read_exact(fixture, bytes_count, digest)
            samples = []
            for _ in range(repeats):
                check, elapsed = _read_exact(fixture, bytes_count, digest)
                if check != data:
                    raise RuntimeError("Inconsistent repeated read contents")
                samples.append(elapsed)
            host = _host_staging(data, repeats)
            cuda = _gpu_transfers(data, repeats) if gpu == "cuda" else None
    except OSError as exc:
        raise RuntimeError("Scratch workspace IO failed; no benchmark receipt") from exc

    return {
        "schema": SCHEMA,
        "provenance": PROVENANCE,
        "status": "PASS_CUDA_TRANSFERS" if cuda else "PASS_CPU_BASELINE",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "workspace": str(folder),
        "fixture": {
            "type": "SYNTHETIC_DETERMINISTIC_BYTES",
            "size_bytes": bytes_count,
            "sha256": digest,
            "temporary_owned_file_deleted": True,
        },
        "repetitions": repeats,
        "gpu_requested": gpu,
        "storage": {
            "write_create_fsync_seconds": round(write_elapsed, 9),
            "first_read_after_write_seconds": round(first_elapsed, 9),
            "first_read_is_cold_cache": False,
            "warm_reads": _stats(samples, bytes_count),
            "cache_control": "UNCONTROLLED_OS_FILESYSTEM_CACHE",
        },
        "host": {"bytearray_copy": host},
        "cuda": cuda,
        "model_execution": False,
        "automatic_layer_placement": False,
        "limitations": [
            "Synthetic byte movement only: no Llama layer computation or model checkpoint.",
            "The first disk read follows a write and fsync; it is NOT a cold-cache measurement.",
            "Warm reads are likely served from filesystem cache. Do not claim raw NVMe throughput.",
            "File-open overhead and host-memory copies may dominate small sizes.",
            "CPU host copy and H2D transfers are separate; pinned stage cost matters end-to-end.",
            "No transfer-compute overlap or PCIe link bandwidth is independently measured.",
            "A single sequential series is not proof of long-term sustained performance.",
            "No actual adaptive scheduling, GPU model fit or performance victory claim.",
            "The workspace path may be identifying. Review receipts before sharing.",
        ],
    }
