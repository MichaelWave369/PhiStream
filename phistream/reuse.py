"""Synthetic CUDA buffer lifetime experiment, opt-in and bounded.

Compares pageable H2D, newly allocated pinned H2D, and preallocated pinned
H2D at three user-selected data sizes. No model weights, imports, uploads or
real model runtime; caching of scratch-file reads is uncontrolled.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import itertools
import math
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import tempfile
import time

from .pipeline import _read_fixture
from .transfer import MIB, _write_fixture

SCHEMA = "PHISTREAM_REUSE_LAB_V1"
PROVENANCE = "MEASURED_LOCAL_SYNTHETIC_HOST_BUFFER_REUSE_NO_MODEL"
CUDA_METHODS = ("pageable", "pinned_new", "pinned_reuse")
CPU_METHODS = ("fresh_host", "reused_host")
ORDER_SEED = 369


def parse_sizes(value: str) -> tuple[int, ...]:
    if not isinstance(value, str):
        raise ValueError("sizes-mib must be comma-separated integers")
    pieces = value.split(",")
    if not 1 <= len(pieces) <= 3:
        raise ValueError("Select 1–3 data sizes")
    if any(not v or not v.isascii() or not v.isdecimal() or
           str(int(v)) != v for v in pieces):
        raise ValueError("Sizes must be canonical positive decimal integers")
    sizes = tuple(int(v) for v in pieces)
    if len(set(sizes)) != len(sizes) or any(not 1 <= v <= 128 for v in sizes):
        raise ValueError("Sizes must be unique integers from 1 to 128 MiB")
    if sum(sizes) > 192:
        raise ValueError("Combined selected sizes must not exceed 192 MiB")
    return sizes


def _orders(methods: tuple[str, ...], rounds: int, seed: int):
    """Six Latin-like balanced CUDA permutations; alternating CPU pairs.

    Every group of six CUDA rounds contains all 3! orderings once.
    """
    if type(rounds) is not int or rounds not in (6, 12):
        raise ValueError("Rounds must be 6 or 12 for order balance")
    rng = random.Random(seed)
    block = [list(p) for p in itertools.permutations(methods)]
    if len(methods) == 2:
        return [list(methods) if i % 2 else list(reversed(methods))
                for i in range(rounds)]
    all_orders = []
    for _ in range(rounds // 6):
        chunk = [p.copy() for p in block]
        rng.shuffle(chunk)
        all_orders.extend(chunk)
    return all_orders


def _duration(samples: list[float]) -> dict:
    if not samples or any(not math.isfinite(v) or v < 0 for v in samples):
        raise RuntimeError("Invalid timing samples")
    return {
        "samples_seconds": [round(x, 9) for x in samples],
        "median_seconds": round(statistics.median(samples), 9),
        "min_seconds": round(min(samples), 9),
        "max_seconds": round(max(samples), 9),
        "repetitions": len(samples),
    }


def _cuda_state(count: int):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU fallback")
    free, total = torch.cuda.mem_get_info()
    if free < count * 5 + 256 * MIB:
        raise RuntimeError("Insufficient CUDA free memory for guarded test")
    try:
        pinned = torch.empty(count, dtype=torch.uint8, pin_memory=True)
        dest = torch.empty(count, dtype=torch.uint8, device="cuda")
    except RuntimeError as exc:
        raise RuntimeError("CUDA preallocation failed; no CPU fallback") from exc
    return {
        "torch": torch, "pinned": pinned, "dest": dest,
        "name": torch.cuda.get_device_name(),
        "free_before": int(free), "total": int(total),
        "version": str(torch.__version__),
        "cuda_build": str(torch.version.cuda) if torch.version.cuda else None,
    }


def _cuda_trial(path: Path, count: int, digest: str, method: str, state: dict):
    torch = state["torch"]
    torch.cuda.synchronize()
    started = time.perf_counter()
    data, read_time = _read_fixture(path, count)
    prep_start = time.perf_counter()
    original = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    prep_time = time.perf_counter() - prep_start
    src = original
    alloc_time = 0.0
    stage_time = 0.0
    if method != "pageable":
        if method == "pinned_new":
            start = time.perf_counter()
            src = torch.empty(count, dtype=torch.uint8, pin_memory=True)
            alloc_time = time.perf_counter() - start
        else:
            src = state["pinned"]
        start = time.perf_counter()
        src.copy_(original)
        stage_time = time.perf_counter() - start
    start = time.perf_counter()
    torch.cuda.synchronize()
    sync_time = time.perf_counter() - start
    start = time.perf_counter()
    state["dest"].copy_(src, non_blocking=False)
    torch.cuda.synchronize()
    h2d_time = time.perf_counter() - start
    elapsed = time.perf_counter() - started
    # Symmetric release of temporary pageable ownership AFTER timing. The
    # new pinned allocation, when present, is also released after timing;
    # the reuse buffer is retained. Allocation time IS included for new.
    release_start = time.perf_counter()
    del src
    del original
    release_time = time.perf_counter() - release_start
    if hashlib.sha256(data).hexdigest() != digest:
        raise RuntimeError("Scratch SHA256 changed; aborting experiment")
    positions = [0, count // 2, count - 1]
    if state["dest"][positions].cpu().tolist() != [data[p] for p in positions]:
        raise RuntimeError("CUDA sample bytes differ from input")
    return {
        "total": elapsed, "read": read_time, "prepare": prep_time,
        "pinned_allocation": alloc_time, "pinned_staging": stage_time,
        "pre_h2d_sync": sync_time, "h2d": h2d_time,
        "release_after_timing": release_time,
    }


def _cpu_trial(path: Path, count: int, digest: str, method: str, host: bytearray):
    began = time.perf_counter()
    data, read_time = _read_fixture(path, count)
    t0 = time.perf_counter()
    if method == "fresh_host":
        buf = bytearray(data)
    else:
        host[:] = data
        buf = host
    copy_time = time.perf_counter() - t0
    elapsed = time.perf_counter() - began
    if hashlib.sha256(data).hexdigest() != digest or buf != data:
        raise RuntimeError("Host scratch integrity check failed")
    return {
        "total": elapsed, "read": read_time,
        "prepare": copy_time, "pinned_allocation": 0.0,
        "pinned_staging": 0.0, "pre_h2d_sync": 0.0,
        "h2d": 0.0, "release_after_timing": 0.0,
    }


def _summary(rows: list[dict]) -> dict:
    keys = ("total", "read", "prepare", "pinned_allocation",
            "pinned_staging", "pre_h2d_sync", "h2d", "release_after_timing")
    result = {key: _duration([row[key] for row in rows]) for key in keys}
    result["unattributed_time"] = _duration([
        max(0.0, row["total"] - row["read"] - row["prepare"] -
            row["pinned_allocation"] - row["pinned_staging"] -
            row["pre_h2d_sync"] - row["h2d"])
        for row in rows
    ])
    return result


def run_reuse_lab(workspace: str | Path, *, gpu: str = "none",
                  sizes_mib: str = "4,16,64", rounds: int = 6,
                  warmups: int = 2) -> dict:
    if gpu not in ("none", "cuda"):
        raise ValueError("gpu must be none or cuda")
    sizes = parse_sizes(sizes_mib)
    methods = CUDA_METHODS if gpu == "cuda" else CPU_METHODS
    _orders(methods, rounds, ORDER_SEED)
    if type(warmups) is not int or not 1 <= warmups <= 4:
        raise ValueError("warmups must be from 1 to 4")
    root = Path(workspace)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Workspace must be an existing nonsymlink directory")
    max_count = max(sizes) * MIB
    if shutil.disk_usage(root).free < max_count * 4 + 128 * MIB:
        raise RuntimeError("Insufficient free scratch disk space")
    # Fail before any scratch write when CUDA isn't present.
    if gpu == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no CPU fallback")
    began = datetime.now(timezone.utc).isoformat()
    size_reports = []
    for size_mib in sizes:
        count = size_mib * MIB
        state = _cuda_state(count) if gpu == "cuda" else None
        host = bytearray(count) if gpu == "none" else None
        orders = _orders(methods, rounds, ORDER_SEED + size_mib)
        rows = {method: [] for method in methods}
        try:
            with tempfile.TemporaryDirectory(prefix="phistream-reuse-", dir=root) as temp:
                path = Path(temp) / "owned-scratch.dat"
                digest, write_sec = _write_fixture(path, count)
                def trial(method: str):
                    return (_cuda_trial(path, count, digest, method, state)
                            if state else _cpu_trial(path, count, digest, method, host))
                for _ in range(warmups):
                    for method in methods:
                        trial(method)
                for sequence in orders:
                    for method in sequence:
                        rows[method].append(trial(method))
        except OSError as exc:
            raise RuntimeError("Owned scratch I/O failed; no valid receipt") from exc
        summary = {name: _summary(measurements) for name, measurements in rows.items()}
        wins = {name: 0 for name in methods}
        for i in range(rounds):
            lowest = min(methods, key=lambda name: rows[name][i]["total"])
            wins[lowest] += 1
        size_reports.append({
            "size_mib": size_mib, "fixture_sha256": digest,
            "scratch_deleted": True,
            "write_fsync_seconds": round(write_sec, 9),
            "orders": orders,
            "methods": summary,
            "round_winner_counts": wins,
            "gpu": ({
                "name": state["name"], "free_before_bytes": state["free_before"],
                "total_bytes": state["total"], "torch_version": state["version"],
                "cuda_build": state["cuda_build"],
            } if state else None),
        })
        del state
    return {
        "schema": SCHEMA,
        "status": "PASS_CUDA_REUSE_LAB" if gpu == "cuda" else "PASS_CPU_REUSE_BASELINE",
        "provenance": PROVENANCE,
        "started_utc": began,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(root),
        "platform": platform.platform(),
        "gpu_requested": gpu,
        "design": {
            "methods": list(methods), "sizes_mib": list(sizes),
            "rounds_per_size": rounds, "warmups_per_method": warmups,
            "warmups_excluded": True, "order_seed_base": ORDER_SEED,
            "cuda_orders_are_six_permutation_blocks": gpu == "cuda",
            "file_cache_controlled": False,
            "source_buffer_release_outside_timing": gpu == "cuda",
            "pinned_new_allocation_included_in_timing": gpu == "cuda",
            "pinned_reuse_buffer_preallocated": gpu == "cuda",
        },
        "size_reports": size_reports,
        "model_execution": False,
        "automatic_layer_placement": False,
        "limitations": [
            "Synthetic bytes only, NOT model inference or model layer prefetch.",
            "The file cache is uncontrolled: reads are warm and not raw NVMe throughput.",
            "The pinned-new path includes new pinned allocation but not its final deallocation.",
            "Pageable temp buffers release outside timed interval for all CUDA methods.",
            "CUDA destination and pinned-reuse buffers are preallocated per size.",
            "No CPU/GPU compute overlap, interleaved streaming, or adaptive placement.",
            "Six balanced position orders reduce systematic position bias but not hardware noise.",
            "CPU-only modes compare fresh vs reused bytearrays; NO pinned CUDA evidence.",
            "Receipts are generated locally, not independently authenticated or uploaded.",
        ],
    }
