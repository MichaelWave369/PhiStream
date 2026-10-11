"""Opt-in paired end-to-end staging lab: file read -> host buffer -> CUDA.

Measures real elapsed pipeline stages and complete wall time for an existing
controlled scratch fixture. The scratch file is NOT a model/checkpoint.
File reads may be satisfied from OS cache; output never claims raw SSD speed.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import tempfile
import time

from .transfer import MIB, _stats, _write_fixture

SCHEMA = "PHISTREAM_PIPELINE_LAB_V1"
PROVENANCE = "MEASURED_LOCAL_PAIRED_SYNTHETIC_PIPELINE_NO_MODEL"
ORDER_SEED = 369


def balanced_orders(repeats: int, seed: int = ORDER_SEED) -> list[list[str]]:
    """Reproducible balanced order, one observation per strategy per round."""
    if type(repeats) is not int or not 4 <= repeats <= 20:
        raise ValueError("repeats must be an integer from 4 to 20")
    random_order = random.Random(seed)
    variants = []
    for _ in range(repeats // 2):
        first = ["pageable", "pinned"]
        if random_order.randrange(2):
            first.reverse()
        variants.extend([first, first[::-1]])
    if repeats % 2:
        variants.append(["pageable", "pinned"] if random_order.randrange(2)
                        else ["pinned", "pageable"])
    return variants


def _read_fixture(path: Path, count: int) -> tuple[bytes, float]:
    """Time open/read, hash result OUTSIDE timed interval, reject corruption."""
    start = time.perf_counter()
    with path.open("rb") as fh:
        data = fh.read(count + 1)
    elapsed = time.perf_counter() - start
    if len(data) != count:
        raise RuntimeError("Pipeline scratch file changed size")
    # Hash outside timed read and outside complete pipeline wall time.
    return data, elapsed


def _cpu_trial(path: Path, count: int, digest: str, mode: str, buffer: bytearray):
    start = time.perf_counter()
    data, read_seconds = _read_fixture(path, count)
    prep_start = time.perf_counter()
    if mode == "pageable":
        copy = bytearray(data)
    else:
        buffer[:] = data
        copy = buffer
    prep_seconds = time.perf_counter() - prep_start
    total = time.perf_counter() - start
    if hashlib.sha256(data).hexdigest() != digest:
        raise RuntimeError("Scratch file failed SHA256 integrity check")
    if copy != data:
        raise RuntimeError("CPU staging differs from fixture")
    return {"total": total, "read": read_seconds, "prepare": prep_seconds,
            "h2d": None, "pinned_stage": None,
            "pre_h2d_sync": None, "checked": True}


def _cuda_environment(count: int):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing CPU fallback")
    free, total = torch.cuda.mem_get_info()
    if free < count * 4 + 256 * MIB:
        raise RuntimeError("Insufficient free CUDA VRAM for guarded pipeline lab")
    try:
        destination = torch.empty(count, dtype=torch.uint8, device="cuda")
        pinned = torch.empty(count, dtype=torch.uint8, pin_memory=True)
    except RuntimeError as exc:
        raise RuntimeError("Required CUDA/pinned buffer allocation failed") from exc
    return {
        "torch": torch, "dest": destination, "pinned": pinned,
        "name": torch.cuda.get_device_name(), "free": int(free), "total": int(total),
        "torch_version": str(torch.__version__),
        "cuda_build": str(torch.version.cuda) if torch.version.cuda else None,
    }


def _cuda_trial(path: Path, count: int, digest: str, mode: str, state: dict):
    torch = state["torch"]
    # A full synchronization prevents operations from previous trials
    # contaminating this paired wall-clock observation.
    torch.cuda.synchronize()
    started = time.perf_counter()
    data, read_seconds = _read_fixture(path, count)
    prepare_start = time.perf_counter()
    # Retain the same 64 MiB-scale pageable allocation until AFTER the
    # full wall-clock interval for BOTH modes. In the previous implementation
    # reassigning `src` to the pinned tensor dropped the last reference to
    # its host buffer *inside* the pinned timing interval. The pageable path
    # released its buffer only after timing, biasing the comparison.
    host_src = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    prep_seconds = time.perf_counter() - prepare_start
    device_src = host_src
    staging_seconds = None
    if mode == "pinned":
        t0 = time.perf_counter()
        state["pinned"].copy_(host_src)
        staging_seconds = time.perf_counter() - t0
        device_src = state["pinned"]
    sync_start = time.perf_counter()
    torch.cuda.synchronize()
    pre_h2d_sync_seconds = time.perf_counter() - sync_start
    copy_start = time.perf_counter()
    state["dest"].copy_(device_src, non_blocking=False)
    torch.cuda.synchronize()
    h2d_seconds = time.perf_counter() - copy_start
    total = time.perf_counter() - started

    # Deliberately release temporary host tensor ownership after the clock
    # stops for both methods. This cost is tracked as a separate diagnostic,
    # rather than silently charged only to pinned staging.
    release_start = time.perf_counter()
    del device_src
    del host_src
    source_release_seconds = time.perf_counter() - release_start

    if hashlib.sha256(data).hexdigest() != digest:
        raise RuntimeError("Scratch file failed SHA256 integrity check")

    # This is a sampled device check, deliberately outside the timing.
    indices = [0, count // 2, count - 1]
    actual = state["dest"][indices].cpu().tolist()
    if actual != [data[x] for x in indices]:
        raise RuntimeError("Pipeline device byte sample mismatch")
    return {"total": total, "read": read_seconds,
            "prepare": prep_seconds, "h2d": h2d_seconds,
            "pinned_stage": staging_seconds,
            "pre_h2d_sync": pre_h2d_sync_seconds,
            "source_release_after_timing": source_release_seconds,
            "checked": True}


def _duration_stats(samples: list[float]) -> dict:
    """Describe a time interval without inventing a meaningful byte throughput."""
    if not samples or any(not math.isfinite(x) or x < 0 for x in samples):
        raise RuntimeError("Time attribution samples must be finite and nonnegative")
    return {
        "samples_seconds": [round(x, 9) for x in samples],
        "median_seconds": round(statistics.median(samples), 9),
        "min_seconds": round(min(samples), 9),
        "max_seconds": round(max(samples), 9),
        "repetitions": len(samples),
    }


def _aggregate(trials: list[dict], size: int, gpu: bool) -> dict:
    result = {
        "end_to_end": _stats([r["total"] for r in trials], size),
        "file_read": _stats([r["read"] for r in trials], size),
        "host_buffer_prepare": _stats([r["prepare"] for r in trials], size),
    }
    if gpu:
        result["h2d"] = _stats([r["h2d"] for r in trials], size)
        result["pre_h2d_sync"] = _duration_stats([r["pre_h2d_sync"] for r in trials])
        result["source_release_after_timing"] = _duration_stats(
            [r["source_release_after_timing"] for r in trials]
        )
        if all(t["pinned_stage"] is not None for t in trials):
            result["pinned_host_stage"] = _stats([r["pinned_stage"] for r in trials], size)
    # Each trial's reported wall time includes interval gaps between explicit
    # stage timers, bookkeeping, tensor wrappers and Python dispatch.
    # Attribute the entire residual *per trial*, never by subtracting medians.
    residuals = [
        max(0.0, r["total"] - r["read"] - r["prepare"] -
            (r["h2d"] or 0) - (r["pinned_stage"] or 0) -
            (r["pre_h2d_sync"] or 0))
        for r in trials
    ]
    result["unattributed_wall_time"] = _duration_stats(residuals)
    return result


def run_pipeline_lab(workspace: str | Path, *, gpu: str = "none",
                     size_mib: int = 16, repeats: int = 8,
                     warmups: int = 2) -> dict:
    if gpu not in ("none", "cuda"):
        raise ValueError("gpu must be none or cuda")
    if type(size_mib) is not int or not 1 <= size_mib <= 128:
        raise ValueError("size_mib must be an integer from 1 to 128")
    order = balanced_orders(repeats)
    if type(warmups) is not int or not 1 <= warmups <= 5:
        raise ValueError("warmups must be an integer from 1 to 5")
    root = Path(workspace)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Specify an existing nonsymlink workspace")
    count = size_mib * MIB
    if shutil.disk_usage(root).free < count * 4 + 128 * MIB:
        raise RuntimeError("Not enough free scratch disk space for pipeline lab")
    # Validate CUDA before creating or writing any files.
    state = _cuda_environment(count) if gpu == "cuda" else None
    cpu_buffer = bytearray(count) if gpu == "none" else None
    device = gpu == "cuda"
    begin = datetime.now(timezone.utc).isoformat()
    results = {"pageable": [], "pinned": []}
    try:
        with tempfile.TemporaryDirectory(prefix="phistream-pipeline-", dir=root) as tmp:
            fixture = Path(tmp) / "scratch.dat"
            digest, write_seconds = _write_fixture(fixture, count)
            def trial(method: str) -> dict:
                if device:
                    return _cuda_trial(fixture, count, digest, method, state)
                return _cpu_trial(fixture, count, digest, method, cpu_buffer)
            # Identical warmup for both strategies. Excluded from all statistics.
            for _ in range(warmups):
                for method in ("pageable", "pinned"):
                    trial(method)
            for pair in order:
                for method in pair:
                    results[method].append(trial(method))
    except OSError as exc:
        raise RuntimeError("Pipeline scratch I/O failed; no receipt generated") from exc
    aggregate = {k: _aggregate(v, count, device) for k, v in results.items()}
    # A paired comparison aligns adjacent observations in the same round.
    ratios = [
        results["pageable"][i]["total"] / results["pinned"][i]["total"]
        for i in range(repeats)
    ]
    wins = sum(results["pinned"][i]["total"] < results["pageable"][i]["total"]
               for i in range(repeats))
    paired_median_ratio = statistics.median(ratios)
    return {
        "schema": SCHEMA,
        "status": "PASS_CUDA_PIPELINE" if device else "PASS_CPU_PIPELINE",
        "provenance": PROVENANCE,
        "started_utc": begin,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "workspace": str(root),
        "gpu_requested": gpu,
        "hardware": ({
            "name": state["name"], "free_bytes_before": state["free"],
            "total_bytes": state["total"], "torch_version": state["torch_version"],
            "cuda_build": state["cuda_build"],
        } if state else None),
        "fixture": {"bytes": count, "sha256": digest, "write_create_fsync_seconds": round(write_seconds, 9),
                    "synthetic": True, "owned_scratch_deleted": True},
        "trial_design": {
            "paired_repetitions": repeats, "warmups_per_mode": warmups,
            "warmup_excluded_from_statistics": True, "order_seed": ORDER_SEED,
            "orders": order, "os_file_cache_controlled": False,
            "checksum_checked_outside_timing": True,
            "host_tensor_lifetime": "RETAIN_UNTIL_END_OF_TIMED_INTERVAL_BOTH_METHODS",
            "temporary_source_release_measured_after_timing": device,
            "cpu_fallback": False,
        },
        "methods": {
            "pageable": {"kind": "FILE_READ_TO_FRESH_PAGEABLE_BUFFER" if device
                          else "FILE_READ_TO_NEW_BYTEARRAY",
                          "measurements": aggregate["pageable"]},
            "pinned": {"kind": "FILE_READ_TO_REUSABLE_PINNED_BUFFER" if device
                        else "FILE_READ_TO_REUSABLE_BYTEARRAY",
                        "measurements": aggregate["pinned"]},
        },
        "paired_outcome": {
            "pinned_wins": wins,
            "pageable_wins_or_ties": repeats - wins,
            "median_pageable_total_over_pinned_total": round(paired_median_ratio, 5),
            "median_ratio_favors": (
                "PINNED" if paired_median_ratio > 1 else
                "PAGEABLE" if paired_median_ratio < 1 else "TIE"
            ),
            "interpretation": (
                "A ratio ABOVE 1.0 favors pinned; BELOW 1.0 favors pageable. "
                "This is a paired synthetic experiment, not model inference "
                "or a statistical confidence interval."
            ),
        },
        "model_execution": False,
        "automatic_layer_placement": False,
        "limitations": [
            "No model tensors, inference, layer cache, or prefetch/compute overlap are measured.",
            "OS filesystem caching is uncontrolled; file reads are not cold-cache NVMe bandwidth.",
            "Warmups and deterministic balanced order reduce but do not eliminate noise.",
            "GPU wall-clock spans file reads, fresh pageable-buffer creation, optional pinned staging and synchronized H2D.",
            "Pre-H2D synchronization is timed, and source-tensor release is measured after full-pipeline timing in both CUDA modes.",
            "Residual wall time includes any other Python bookkeeping and uninstrumented gaps.",
            "Reference bytes are hashed outside measured file-read interval; buffers are allocated before the trial.",
            "End-to-end here is one synchronous synthetic stage pipeline, not model token latency.",
            "CPU-only mode compares fresh and reused bytearray staging, not pinned RAM.",
            "Results are local, user-provided and unauthenticated; review paths before publishing.",
        ],
    }
