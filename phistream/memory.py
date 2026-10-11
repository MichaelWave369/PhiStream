"""Evidence-labeled memory preflight for PhiStream's narrow Llama executor.

Pure metadata accounting, not a benchmark, guaranteed peak, CUDA allocation
or claim that a particular checkpoint can run on a given GPU. Fail closed
when a user's *selected policy budget* is exceeded or a single tensor
would violate the executor's bounded read size.
"""
from __future__ import annotations

import math

from .llama import LAYER_NAMES, LlamaSettings
from .safeweights import LocalCheckpoint

MIB = 1024 * 1024
SCHEMA = "PHISTREAM_MEMORY_PREFLIGHT_V1"


def _positive(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be a finite positive number")
    v = float(value)
    if not math.isfinite(v) or v <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return v


def _nonnegative(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    v = float(value)
    if not math.isfinite(v) or v < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return v


def _integer(value, name: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer from 1 to {maximum}")
    return value


def memory_preflight(
    checkpoint: LocalCheckpoint,
    *,
    context_tokens: int,
    capacity_mib: float,
    reserve_mib: float = 512.0,
    max_tensor_mib: int = 512,
    head_rows: int = 128,
    observed_free_bytes: int | None = None,
) -> dict:
    """Compute an execution-footprint estimate from checkpoint metadata alone.

    capacity_mib is a *user-declared memory policy*, not detected device VRAM.
    observed_free_bytes may cap that policy from an actual CUDA memory query,
    but never proves a model fits. Context estimates assume FP32 KV cache,
    one active FP32 layer, and a streamed vocabulary projection.
    """
    settings = LlamaSettings.from_checkpoint(checkpoint)
    context = _integer(context_tokens, "context_tokens", settings.max_context)
    capacity = _positive(capacity_mib, "capacity_mib")
    reserve = _nonnegative(reserve_mib, "reserve_mib")
    tensor_limit = _integer(max_tensor_mib, "max_tensor_mib", 4096)
    rows = _integer(head_rows, "head_rows", 16384)
    if observed_free_bytes is not None and (
        type(observed_free_bytes) is not int or observed_free_bytes < 0
    ):
        raise ValueError("observed_free_bytes must be a nonnegative integer")
    target_bytes = int(capacity * MIB)
    observed_bound = observed_free_bytes if observed_free_bytes is not None else target_bytes
    effective_bytes = min(target_bytes, observed_bound)
    allowed_bytes = max(0, effective_bytes - int(reserve * MIB))
    tensor_cap_bytes = tensor_limit * MIB

    layers = []
    oversize = []
    largest_layer = 0
    # The decoder casts *every* layer tensor to FP32 and creates a nine-tensor
    # dictionary per block. On-disk BF16/FP16 sizes are not GPU execution sizes.
    for i in range(settings.layers):
        prefix = f"model.layers.{i}."
        raw_bytes = 0
        fp32_bytes = 0
        for suffix in LAYER_NAMES:
            entry = checkpoint.entries[prefix + suffix]
            raw_bytes += entry.nbytes
            fp32_bytes += math.prod(entry.shape) * 4
            if entry.nbytes > tensor_cap_bytes:
                oversize.append(prefix + suffix)
        layers.append({
            "layer": i, "checkpoint_weight_bytes": raw_bytes,
            "fp32_execution_weight_bytes": fp32_bytes,
        })
        largest_layer = max(largest_layer, fp32_bytes)

    d = settings.dim
    head_dim = d // settings.heads
    # KV tensors are kept on device and use FP32 in the current engine.
    kv_bytes = settings.layers * context * settings.kv_heads * head_dim * 2 * 4
    # torch.stack copies all seen KV for a layer, then repeat_interleave expands
    # from KV heads to attention heads. Softmax scores and probabilities remain.
    stacked_kv = context * settings.kv_heads * head_dim * 2 * 4
    repeated_kv = context * settings.heads * head_dim * 2 * 4
    attention_scores = context * settings.heads * 2 * 4
    attention_workspace = stacked_kv + repeated_kv + attention_scores
    norm_bytes = d * 4
    row_bytes = min(rows, settings.vocab) * d * 4
    logits_bytes = settings.vocab * 4
    embed_bytes = d * 4
    # These are approximate concurrent footprints. Runtime activations,
    # allocator reservations, CUDA libraries and Python process RAM are NOT
    # included and may make actual usage much higher.
    active_layer_estimate = largest_layer + kv_bytes + attention_workspace + norm_bytes
    output_estimate = kv_bytes + norm_bytes + row_bytes + logits_bytes + embed_bytes
    estimated_bytes = max(active_layer_estimate, output_estimate)

    blocked_reasons = []
    if oversize:
        blocked_reasons.append("SINGLE_TENSOR_READ_CAP_EXCEEDED")
    if allowed_bytes < estimated_bytes:
        blocked_reasons.append("ESTIMATED_WORKING_SET_EXCEEDS_POLICY_BUDGET")
    result = {
        "schema": SCHEMA,
        "status": "BLOCKED_BY_POLICY" if blocked_reasons else "REVIEW_REQUIRED",
        "provenance": "DERIVED_FROM_CHECKPOINT_METADATA_NOT_MEASURED",
        "no_fit_guarantee": True,
        "checkpoint_model_type": "llama",
        "execution_dtype": "float32",
        "context_tokens": context,
        "capacity_source": (
            "USER_POLICY_CAPPED_BY_CUDA_FREE" if observed_free_bytes is not None
            else "USER_SPECIFIED_SCENARIO"
        ),
        "capacity_policy_bytes": target_bytes,
        "cuda_observed_free_bytes": observed_free_bytes,
        "effective_capacity_bytes": effective_bytes,
        "reserved_bytes": int(reserve * MIB),
        "available_policy_bytes": allowed_bytes,
        "checkpoint_weight_bytes": sum(t.nbytes for t in checkpoint.entries.values()),
        "largest_single_layer_fp32_bytes": largest_layer,
        "kv_cache_fp32_bytes": kv_bytes,
        "attention_workspace_estimate_bytes": attention_workspace,
        "output_head_chunk_fp32_bytes": row_bytes,
        "logits_fp32_bytes": logits_bytes,
        "estimated_working_set_bytes": estimated_bytes,
        "max_tensor_read_bytes": tensor_cap_bytes,
        "oversize_tensor_count": len(oversize),
        "oversize_tensor_examples": oversize[:5],
        "blocked_reasons": blocked_reasons,
        "layer_estimates": layers,
        "limitations": [
            "Estimate only; REVIEW_REQUIRED never means GPU memory fit is proven.",
            "Actual CUDA allocation includes allocator fragmentation, context, kernels, temporaries and other processes.",
            "Checkpoint on-disk FP16/BF16 weights are staged in FP32.",
            "Temporary CPU RAM, host memory mappings, disk reads and transfer latency are not budgeted.",
            "Context capacity and per-tensor cap are evaluated for the current single-sequence implementation.",
            "This is a policy gate, not a system-level hard memory allocator or crash-prevention guarantee.",
        ],
    }
    return result


def refuse_blocked(plan: dict) -> None:
    if plan["status"] == "BLOCKED_BY_POLICY":
        raise MemoryError(
            "PhiStream memory preflight blocked execution: " +
            ", ".join(plan["blocked_reasons"])
        )
