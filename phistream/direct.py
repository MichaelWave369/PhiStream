"""Opt-in synthetic direct-read laboratory: avoid intermediate Python copies.

Three CUDA paths: legacy file -> bytes -> bytearray -> reused pinned -> GPU,
direct file.readinto(preallocated pageable tensor) -> GPU, and direct
file.readinto(preallocated pinned tensor) -> GPU.

No model weights, model execution, GPU prefetch, direct NVMe DMA or cold-cache
claims. Local temporary scratch only; bounded allocations and SHA256 checks.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform
import shutil
import tempfile
import time

from .reuse import _duration, _orders, parse_sizes
from .transfer import MIB, _write_fixture
from .hardware import physical_memory

SCHEMA = "PHISTREAM_DIRECT_READ_LAB_V1"
PROVENANCE = "MEASURED_LOCAL_SYNTHETIC_READINTO_NO_MODEL"
CUDA_METHODS = ("legacy_copy", "direct_pageable", "direct_pinned")
CPU_METHODS = ("legacy_host_copy", "direct_host_readinto")
ORDER_SEED = 369


def _readinto_exact(path: Path, view: memoryview, count: int) -> float:
    """Read directly into a writable existing buffer, with no bytes result."""
    if view.readonly or view.nbytes != count or not view.contiguous:
        raise ValueError("Writable contiguous destination of exact size required")
    start = time.perf_counter()
    with path.open("rb", buffering=0) as stream:
        received = stream.readinto(view)
        # Reject a changed/oversized fixture; the extra single byte is not
        # transferred into the destination.
        extra = stream.read(1)
    elapsed = time.perf_counter() - start
    if received != count or extra:
        raise RuntimeError("Owned scratch changed size or was truncated")
    return elapsed


def _read_legacy(path: Path, count: int) -> tuple[bytes, float]:
    start = time.perf_counter()
    with path.open("rb", buffering=0) as stream:
        data = stream.read(count + 1)
    elapsed = time.perf_counter() - start
    if len(data) != count:
        raise RuntimeError("Owned scratch changed size or was truncated")
    return data, elapsed


def _state_cuda(count: int) -> dict:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
    free, total = torch.cuda.mem_get_info()
    if free < count * 5 + 256 * MIB:
        raise RuntimeError("Insufficient CUDA free memory for bounded direct read lab")
    try:
        cpu = torch.empty(count, dtype=torch.uint8, device="cpu")
        pinned = torch.empty(count, dtype=torch.uint8, pin_memory=True)
        gpu = torch.empty(count, dtype=torch.uint8, device="cuda")
    except RuntimeError as exc:
        raise RuntimeError("CUDA/pinned staging allocation failed; no fallback") from exc
    # numpy() is a zero-copy writable view of an ordinary CPU tensor. Create
    # memoryview ONCE per size; readinto writes directly to tensor storage.
    try:
        cpu_view = memoryview(cpu.numpy()).cast("B")
        pinned_view = memoryview(pinned.numpy()).cast("B")
    except (ValueError, TypeError, RuntimeError) as exc:
        raise RuntimeError("Writable CPU tensor buffer export is unavailable") from exc
    if cpu_view.readonly or pinned_view.readonly:
        raise RuntimeError("Pinned/pageable buffer export must be writable")
    return {
        "torch": torch, "cpu": cpu, "pinned": pinned, "gpu": gpu,
        "cpu_view": cpu_view, "pinned_view": pinned_view,
        "name": torch.cuda.get_device_name(), "free_before": int(free),
        "total": int(total), "torch_version": str(torch.__version__),
        "cuda_build": str(torch.version.cuda) if torch.version.cuda else None,
    }


def _trial_cuda(path: Path, count: int, digest: str, mode: str, state: dict) -> dict:
    torch = state["torch"]
    torch.cuda.synchronize()
    start_total = time.perf_counter()
    legacy_data = None
    legacy_tensor = None
    prep_elapsed = 0.0
    stage_elapsed = 0.0

    if mode == "legacy_copy":
        legacy_data, read_elapsed = _read_legacy(path, count)
        start = time.perf_counter()
        legacy_tensor = torch.frombuffer(bytearray(legacy_data), dtype=torch.uint8)
        prep_elapsed = time.perf_counter() - start
        start = time.perf_counter()
        state["pinned"].copy_(legacy_tensor)
        stage_elapsed = time.perf_counter() - start
        source = state["pinned"]
        digest_view = state["pinned_view"]
    elif mode in ("direct_pageable", "direct_pinned"):
        key = "cpu_view" if mode == "direct_pageable" else "pinned_view"
        read_elapsed = _readinto_exact(path, state[key], count)
        source = state["cpu"] if mode == "direct_pageable" else state["pinned"]
        digest_view = state[key]
    else:
        raise ValueError("Unsupported direct-read CUDA method")

    sync_start = time.perf_counter()
    torch.cuda.synchronize()
    sync_elapsed = time.perf_counter() - sync_start
    h2d_start = time.perf_counter()
    state["gpu"].copy_(source, non_blocking=False)
    torch.cuda.synchronize()
    h2d_elapsed = time.perf_counter() - h2d_start
    total_elapsed = time.perf_counter() - start_total
    # Preserve all pageable temporary references until AFTER timing in every
    # mode, to avoid the earlier 64 MiB CPython destructor timing bias.
    if hashlib.sha256(digest_view).hexdigest() != digest:
        raise RuntimeError("Direct CUDA buffer SHA256 mismatch")
    positions = [0, count // 2, count - 1]
    if state["gpu"][positions].cpu().tolist() != [digest_view[i] for i in positions]:
        raise RuntimeError("CUDA returned sampled bytes differ from source")
    if legacy_data is not None and hashlib.sha256(legacy_data).hexdigest() != digest:
        raise RuntimeError("Legacy file bytes SHA256 mismatch")
    release_start = time.perf_counter()
    del source
    del legacy_tensor
    del legacy_data
    release_elapsed = time.perf_counter() - release_start
    return {
        "total": total_elapsed, "file_read": read_elapsed,
        "temporary_host_copy": prep_elapsed, "host_to_pinned": stage_elapsed,
        "pre_h2d_sync": sync_elapsed, "h2d": h2d_elapsed,
        "source_release_outside_timing": release_elapsed,
        "checksum_verified": True, "cuda_sample_verified": True,
    }


def _trial_cpu(path: Path, count: int, digest: str, mode: str, buffer: bytearray) -> dict:
    started = time.perf_counter()
    prep_elapsed = 0.0
    if mode == "legacy_host_copy":
        data, read_elapsed = _read_legacy(path, count)
        copy_start = time.perf_counter()
        source = bytearray(data)
        prep_elapsed = time.perf_counter() - copy_start
    elif mode == "direct_host_readinto":
        read_elapsed = _readinto_exact(path, memoryview(buffer), count)
        source = buffer
        data = None
    else:
        raise ValueError("Unsupported direct-read CPU method")
    total_elapsed = time.perf_counter() - started
    if hashlib.sha256(source).hexdigest() != digest:
        raise RuntimeError("Direct CPU scratch SHA256 mismatch")
    return {
        "total": total_elapsed, "file_read": read_elapsed,
        "temporary_host_copy": prep_elapsed, "host_to_pinned": 0.0,
        "pre_h2d_sync": 0.0, "h2d": 0.0,
        "source_release_outside_timing": 0.0,
        "checksum_verified": True, "cuda_sample_verified": False,
    }


def _summarize(samples: list[dict]) -> dict:
    fields = ("total", "file_read", "temporary_host_copy", "host_to_pinned",
              "pre_h2d_sync", "h2d", "source_release_outside_timing")
    stats = {field: _duration([row[field] for row in samples]) for field in fields}
    stats["unattributed_wall_time"] = _duration([
        max(0.0, row["total"] - sum(row[field] for field in fields[1:-1]))
        for row in samples
    ])
    return stats


def run_direct_lab(workspace: str | Path, *, gpu: str = "none",
                   sizes_mib: str = "4,16,64", rounds: int = 6,
                   warmups: int = 2) -> dict:
    if gpu not in ("none", "cuda"):
        raise ValueError("gpu must be none or cuda")
    sizes = parse_sizes(sizes_mib)
    methods = CUDA_METHODS if gpu == "cuda" else CPU_METHODS
    _orders(methods, rounds, ORDER_SEED)
    if type(warmups) is not int or not 1 <= warmups <= 4:
        raise ValueError("warmups must be an integer 1–4")
    root = Path(workspace)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Workspace must be an existing nonsymlink directory")
    max_bytes = max(sizes) * MIB
    if shutil.disk_usage(root).free < max_bytes * 4 + 128 * MIB:
        raise RuntimeError("Insufficient scratch disk space")
    available_ram = physical_memory()["available_bytes"]
    if available_ram < max_bytes * 5 + 256 * MIB:
        raise RuntimeError("Insufficient available host RAM for bounded direct read lab")
    if gpu == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no CPU fallback")

    started_utc = datetime.now(timezone.utc).isoformat()
    reports = []
    for size_mib in sizes:
        count = size_mib * MIB
        state = _state_cuda(count) if gpu == "cuda" else None
        cpu_buf = bytearray(count) if state is None else None
        rows = {name: [] for name in methods}
        ordering = _orders(methods, rounds, ORDER_SEED + size_mib)
        with tempfile.TemporaryDirectory(prefix="phistream-direct-", dir=root) as temp:
            fixture = Path(temp) / "scratch.bin"
            digest, write_seconds = _write_fixture(fixture, count)
            def trial(name: str):
                return (_trial_cuda(fixture, count, digest, name, state) if state
                        else _trial_cpu(fixture, count, digest, name, cpu_buf))
            for _ in range(warmups):
                for name in methods:
                    trial(name)
            for group in ordering:
                for name in group:
                    rows[name].append(trial(name))
        winners = {name: 0 for name in methods}
        for i in range(rounds):
            winner = min(methods, key=lambda key: rows[key][i]["total"])
            winners[winner] += 1
        reports.append({
            "size_mib": size_mib, "fixture_sha256": digest,
            "temporary_scratch_deleted": True, "write_fsync_seconds": round(write_seconds, 9),
            "orders": ordering, "methods": {k: _summarize(v) for k, v in rows.items()},
            "round_winner_counts": winners,
            "gpu": ({
                "name": state["name"], "free_before_bytes": state["free_before"],
                "total_bytes": state["total"],
                "torch_version": state["torch_version"], "cuda_build": state["cuda_build"],
            } if state else None),
        })
        del state
    return {
        "schema": SCHEMA, "provenance": PROVENANCE,
        "status": "PASS_CUDA_DIRECT_READ" if gpu == "cuda" else "PASS_CPU_DIRECT_READ",
        "started_utc": started_utc, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "workspace": str(root),
        "gpu_requested": gpu,
        "design": {
            "methods": list(methods), "sizes_mib": list(sizes),
            "rounds_per_size": rounds, "warmups_per_method": warmups,
            "warmups_excluded": True, "balanced_method_order": True,
            "filesystem_cache_controlled": False,
            "legacy_source_lifetime_matches_timed_interval": True,
            "direct_buffers_preallocated": True,
            "readinto_writes_into_tensor_host_storage": gpu == "cuda",
            "checksums_checked_outside_timing": True,
        },
        "size_reports": reports, "model_execution": False,
        "automatic_layer_placement": False,
        "limitations": [
            "Synthetic bytes only; no pretrained checkpoint, model layer execution or inference.",
            "readinto avoids intermediate Python bytes construction for direct methods, not OS cache.",
            "OS filesystem cache uncontrolled; this is NOT measured cold-cache SSD speed.",
            "Python/PyTorch/NumPy host storage export may affect this workload's behavior.",
            "The legacy path includes Python bytes and bytearray copy; direct paths preallocate buffers.",
            "Scratch write/fsync, warmups, buffer allocation and source release are excluded from timed trials.",
            "Sequential blocking H2D; no asynchronous prefetch, compute overlap or NVMe-to-GPU DMA.",
            "No conclusion about performance of models or dynamic layer schedulers.",
            "Receipts are local, user-provided and unauthenticated; review paths before sharing.",
        ],
    }
