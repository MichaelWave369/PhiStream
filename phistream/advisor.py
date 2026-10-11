"""Human-review-only, evidence-scoped tier-residency recommendations.

This emits an advisory policy proposal. It NEVER mutates model execution,
allocates persistent GPU weights, launches tests or approves a backend route.
"""
from __future__ import annotations

import math
from typing import Any

from .hardware import SCHEMA as HARDWARE_SCHEMA
from .memory import MIB, memory_preflight
from .llama import LlamaSettings
from .safeweights import LocalCheckpoint

SCHEMA = "PHISTREAM_RESIDENCY_ADVICE_V1"


def _reserve(value: float, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite and nonnegative")
    result = float(value)
    if not math.isfinite(result) or result < 0 or result > 1048576:
        raise ValueError(f"{label} must be between 0 and 1048576 MiB")
    return int(result * MIB)


def recommend_residency(
    checkpoint: LocalCheckpoint,
    snapshot: dict[str, Any],
    *,
    context_tokens: int,
    gpu_reserve_mib: float = 1024.0,
    ram_reserve_mib: float = 4096.0,
    max_tensor_mib: int = 512,
    head_rows: int = 128,
) -> dict:
    if not isinstance(snapshot, dict) or snapshot.get("schema") != HARDWARE_SCHEMA:
        raise ValueError("A PhiStream V1 read-only hardware snapshot is required")
    if snapshot.get("status") != "OBSERVED_READ_ONLY":
        raise ValueError("Hardware snapshot is not a local observation")
    ram = snapshot.get("ram")
    if not isinstance(ram, dict) or any(
        type(ram.get(k)) is not int for k in ("available_bytes", "total_bytes")
    ):
        raise ValueError("Missing valid RAM observation")
    if not 0 <= ram["available_bytes"] <= ram["total_bytes"]:
        raise ValueError("Inconsistent RAM availability")
    gpu = snapshot.get("gpu")
    if not isinstance(gpu, dict) or gpu.get("device") != "cuda":
        raise ValueError("A measured CUDA free-memory snapshot is required")
    if any(type(gpu.get(k)) is not int for k in ("free_bytes", "total_bytes")):
        raise ValueError("Invalid CUDA memory counters")
    if not 0 <= gpu["free_bytes"] <= gpu["total_bytes"] or gpu["total_bytes"] <= 0:
        raise ValueError("Inconsistent CUDA free memory")
    gpu_reserve = _reserve(gpu_reserve_mib, "gpu_reserve_mib")
    ram_reserve = _reserve(ram_reserve_mib, "ram_reserve_mib")
    if gpu["free_bytes"] <= 0:
        raise ValueError("No available GPU free memory to plan against")

    plan = memory_preflight(
        checkpoint, context_tokens=context_tokens,
        capacity_mib=gpu["free_bytes"] / MIB,
        reserve_mib=gpu_reserve / MIB,
        max_tensor_mib=max_tensor_mib,
        head_rows=head_rows,
        observed_free_bytes=gpu["free_bytes"],
    )
    settings = LlamaSettings.from_checkpoint(checkpoint)
    fp32_layers = plan["layer_estimates"]
    # Advisory headroom is conservative: still reserves the entire estimated
    # active-layer peak rather than subtracting any shared working buffers.
    gpu_after_active = max(
        0,
        plan["available_policy_bytes"] - plan["estimated_working_set_bytes"],
    )
    staged = []
    remaining = gpu_after_active
    for layer in fp32_layers:
        required = layer["fp32_execution_weight_bytes"]
        if required <= remaining:
            staged.append(layer["layer"])
            remaining -= required
    max_raw_layer = max(
        (layer["checkpoint_weight_bytes"] for layer in fp32_layers), default=0
    )
    ram_headroom = max(0, ram["available_bytes"] - ram_reserve)
    reasons = list(plan["blocked_reasons"])
    if max_raw_layer > ram_headroom:
        reasons.append("HOST_LAYER_STAGING_EXCEEDS_RAM_POLICY")
    if not isinstance(snapshot.get("observed_utc"), str):
        raise ValueError("Snapshot timestamp is required")

    return {
        "schema": SCHEMA,
        "status": "BLOCKED_BY_POLICY" if reasons else "REVIEW_REQUIRED",
        "provenance": "LIVE_MEMORY_COUNTERS_PLUS_METADATA_ESTIMATE_NOT_BENCHMARKED",
        "snapshot_observed_utc": snapshot["observed_utc"],
        "execution_approved": False,
        "auto_placement_enabled": False,
        "no_fit_guarantee": True,
        "model_type": "llama",
        "model_layer_count": settings.layers,
        "context_tokens": plan["context_tokens"],
        "observations": {
            "gpu_free_bytes_at_snapshot": gpu["free_bytes"],
            "ram_available_bytes_at_snapshot": ram["available_bytes"],
            "drives": snapshot.get("drives", []),
        },
        "policy": {
            "gpu_reserve_bytes": gpu_reserve,
            "ram_reserve_bytes": ram_reserve,
            "max_single_tensor_read_bytes": plan["max_tensor_read_bytes"],
        },
        "checkpoint": {
            "weight_bytes_on_disk": plan["checkpoint_weight_bytes"],
            "largest_single_layer_fp32_bytes": plan["largest_single_layer_fp32_bytes"],
            "largest_layer_on_disk_bytes": max_raw_layer,
        },
        "proposed_gpu": {
            "estimated_current_executor_working_set_bytes": plan["estimated_working_set_bytes"],
            "estimated_kv_bytes": plan["kv_cache_fp32_bytes"],
            "additional_layer_advice_budget_bytes": gpu_after_active,
            "candidate_pinned_layer_indices": staged if not reasons else [],
            "candidate_pinned_layer_bytes": (
                gpu_after_active - remaining if not reasons else 0
            ),
            "interpretation": (
                "What could be explored after engineering a new cache implementation, "
                "not a layer allocation performed by the present Llama executor."
            ),
        },
        "proposed_host": {
            "ram_staging_budget_bytes": ram_headroom,
            "one_layer_staging_appears_within_budget": max_raw_layer <= ram_headroom,
            "storage_policy": "USER_SELECTED_DRIVES_ONLY_SPEED_NOT_MEASURED",
        },
        "blocked_reasons": reasons,
        "preflight": plan,
        "limitations": [
            "Read-only proposal. The Llama engine does not implement this pinned-layer placement policy.",
            "Free GPU/RAM counters change; recommendations are stale immediately after sampling.",
            "No drive read-speed, PCIe bandwidth, transfer latency or CPU throughput measured.",
            "No throughput or VRAM fit guarantees. RAM staging estimates exclude host Python temporaries.",
            "No pretrained model reference parity, accelerator approval, or Vessie route change.",
        ],
    }
