"""Read-only, opt-in hardware observation for the PhiStream research lab.

No web calls, no disk-speed benchmarks, no model weights and no execution
routing. Platform RAM and optional CUDA memory are sampled once per call;
the measurements are ephemeral and can change immediately.
"""
from __future__ import annotations

import ctypes
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import shutil
from typing import Any

SCHEMA = "PHISTREAM_HARDWARE_SNAPSHOT_V1"
MAX_DRIVES = 8


def _windows_physical_memory() -> dict[str, int]:
    from ctypes import wintypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    data = MemoryStatus()
    data.dwLength = ctypes.sizeof(data)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(data)):
        raise OSError("GlobalMemoryStatusEx could not read system memory")
    return {"total_bytes": int(data.ullTotalPhys), "available_bytes": int(data.ullAvailPhys)}


def parse_linux_meminfo(contents: str) -> dict[str, int]:
    fields: dict[str, int] = {}
    for line in contents.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parts = value.split()
        if len(parts) >= 2 and parts[1] == "kB" and parts[0].isdigit():
            fields[key] = int(parts[0]) * 1024
    if "MemTotal" not in fields or "MemAvailable" not in fields:
        raise RuntimeError("Linux MemAvailable/MemTotal are missing; no RAM estimate")
    total, available = fields["MemTotal"], fields["MemAvailable"]
    if not 0 < total or not 0 <= available <= total:
        raise RuntimeError("Linux memory counters are invalid")
    return {"total_bytes": total, "available_bytes": available}


def physical_memory() -> dict[str, Any]:
    system = platform.system()
    if system == "Windows":
        result = _windows_physical_memory()
        origin = "WIN32_GLOBAL_MEMORY_STATUS_EX"
    elif system == "Linux":
        with open("/proc/meminfo", "r", encoding="ascii") as source:
            result = parse_linux_meminfo(source.read())
        origin = "LINUX_PROC_MEMINFO_MEMAVAILABLE"
    else:
        raise RuntimeError(
            f"No supported physical-memory collector for {system}; refusing to guess"
        )
    if result["total_bytes"] <= 0 or not (
        0 <= result["available_bytes"] <= result["total_bytes"]
    ):
        raise RuntimeError("Memory collector reported inconsistent values")
    return {**result, "source": origin}


def cuda_memory() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no fabricated GPU report")
    device_index = torch.cuda.current_device()
    prop = torch.cuda.get_device_properties(device_index)
    free, total = torch.cuda.mem_get_info(device_index)
    if not 0 <= free <= total:
        raise RuntimeError("CUDA free-memory counters are inconsistent")
    return {
        "device": "cuda",
        "name": str(prop.name),
        "device_index": int(device_index),
        "compute_capability": f"{prop.major}.{prop.minor}",
        "total_bytes": int(total),
        "free_bytes": int(free),
        "source": "TORCH_CUDA_MEM_GET_INFO",
        "torch_version": str(torch.__version__),
        "cuda_build": str(torch.version.cuda) if torch.version.cuda else None,
    }


def hardware_snapshot(*, gpu: str = "none", drives: tuple[str, ...] = ()) -> dict:
    if gpu not in ("none", "cuda"):
        raise ValueError("gpu must be 'none' or 'cuda'")
    if len(drives) > MAX_DRIVES or any(not isinstance(p, str) or not p.strip() for p in drives):
        raise ValueError("Specify at most eight nonempty drive roots")
    if len(set(drives)) != len(drives):
        raise ValueError("Duplicate drive roots are not permitted")
    memory = physical_memory()
    disks = []
    for entry in drives:
        # Disk *capacity* query only. No data writes or disk throughput probes.
        path = Path(entry)
        if not path.exists() or not path.is_dir():
            raise ValueError(f"Drive directory is missing or not accessible: {entry}")
        capacity = shutil.disk_usage(path)
        disks.append({
            "requested_path": entry,
            "free_bytes": int(capacity.free),
            "total_bytes": int(capacity.total),
            "source": "PYTHON_SHUTIL_DISK_USAGE",
            "throughput_measured": False,
        })
    observed_gpu = cuda_memory() if gpu == "cuda" else None
    return {
        "schema": SCHEMA,
        "status": "OBSERVED_READ_ONLY",
        "provenance": "LOCAL_HARDWARE_COUNTERS_NOT_A_BENCHMARK",
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "ram": memory,
        "gpu": observed_gpu,
        "gpu_requested": gpu,
        "drives": disks,
        "limitations": [
            "All capacities are snapshots. Concurrent apps may consume RAM/VRAM immediately.",
            "RAM available is OS-reported, not memory dedicated to PhiStream.",
            "CUDA free memory may change during initialization or after capture.",
            "Disk capacity queries measure neither read/write throughput nor latency.",
            "No inference is launched and no layer residency changes are performed.",
            "Local drive paths are included in JSON: review receipts before sharing.",
        ],
    }
