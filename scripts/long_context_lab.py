#!/usr/bin/env python3
"""PR30: metadata-only long-context protocol. Deliberately has NO run command.

This creates a held, NON-EXECUTABLE plan for 128/512/2048-token input
experiments. It NEVER runs PyTorch operators, CUDA, reference-model inference,
model downloads, checkpoint modification, or scheduler changes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import math
from pathlib import Path

from phistream.llama import LlamaSettings
from phistream.memory import memory_preflight
from phistream.prefill import MAX_BATCH_PREFILL_TOKENS
from phistream.safeweights import LocalCheckpoint
from scripts.real_weight_matrix import bound_output, write_new

SCHEMA = "PHISTREAM_LONG_CONTEXT_PROTOCOL_V1"
TARGETS = (128, 512, 2048)
GENERATED_STEPS = 32
MIB = 1024 * 1024


def _positive_number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(label + " must be a positive finite number")
    return float(value)


def _nonnegative_number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(label + " must be finite and nonnegative")
    return float(value)


def _positive_int(value, label, limit):
    if type(value) is not int or not 1 <= value <= limit:
        raise ValueError(label + " must be an integer in 1.." + str(limit))
    return value


def make_plan(checkpoint, *, capacity_mib=8192.0, reserve_mib=1024.0,
              max_tensor_mib=512, head_rows=128):
    """Metadata-only pure planning once an explicit read-only checkpoint is opened."""
    capacity_mib = _positive_number(capacity_mib, "capacity_mib")
    reserve_mib = _nonnegative_number(reserve_mib, "reserve_mib")
    max_tensor_mib = _positive_int(max_tensor_mib, "max_tensor_mib", 4096)
    head_rows = _positive_int(head_rows, "head_rows", 16384)
    cfg = LlamaSettings.from_checkpoint(checkpoint)
    cases = []

    for tokens in TARGETS:
        total_context = tokens + GENERATED_STEPS - 1
        holds = ["NO_LONG_CONTEXT_BATCHED_IMPLEMENTATION",
                 "NO_FROZEN_LONG_CONTEXT_REFERENCE_TRACE",
                 "NO_LONG_CONTEXT_EXECUTION_APPROVAL",
                 "NO_INDEPENDENT_LONG_CONTEXT_PARITY_EVIDENCE"]
        memory = None
        if total_context > cfg.max_context:
            holds.append("FULL_CONTEXT_EXCEEDS_CHECKPOINT_CONTEXT_LIMIT")
        else:
            memory = memory_preflight(
                checkpoint,
                context_tokens=total_context,
                capacity_mib=capacity_mib, reserve_mib=reserve_mib,
                max_tensor_mib=max_tensor_mib, head_rows=head_rows)
            if memory["status"] == "BLOCKED_BY_POLICY":
                holds.extend(memory["blocked_reasons"])

        # A single theoretical FP32 [heads, prompt, prompt] score matrix.
        # The current 24-token BATCHED kernel is not extrapolated or invoked.
        theoretical_score_bytes = cfg.heads * tokens * tokens * 4
        cases.append({
            "id": "prompt_" + str(tokens),
            "prompt_tokens": tokens,
            "teacher_forced_continuation_steps": GENERATED_STEPS,
            "full_context_tokens_required": total_context,
            "status": "HELD_PROTOCOL_ONLY_NO_EXECUTION",
            "execution_approved": False,
            "existing_batched_kernel_supports_prompt": tokens <= MAX_BATCH_PREFILL_TOKENS,
            "full_context_fits_checkpoint_limit": total_context <= cfg.max_context,
            "hold_reasons": holds,
            "memory_preflight": memory,
            "theoretical_one_fp32_attention_score_matrix_bytes": theoretical_score_bytes,
            "theoretical_score_note": (
                "One dense prompt-by-prompt score matrix only. NOT a PyTorch "
                "peak, GPU-fit estimate, verified future architecture, or a "
                "claimed execution profile."
            ),
            "reference_trace_ready": False,
            "candidate_implementation_ready": False,
            "benchmark_measured": False,
        })

    return {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ALL_LONG_CONTEXT_TARGETS_HELD_PROTOCOL_ONLY",
        "model_execution": False,
        "cuda_queried": False,
        "reference_generated": False,
        "checkpoint_mutated": False,
        "scheduler_policy_changed": False,
        "production_promoted": False,
        "speedup_proven": False,
        "source_reference_authenticated": False,
        "provenance": "LOCAL_CHECKPOINT_METADATA_ONLY_NO_WEIGHT_HASH",
        "checkpoint_binding_scope": "METADATA_LAYOUT_ONLY_NOT_SHA_BOUND_TO_WEIGHT_BYTES",
        "model": {
            "architecture": "llama_canonical_limited",
            "layers": cfg.layers,
            "attention_heads": cfg.heads,
            "kv_heads": cfg.kv_heads,
            "hidden_size": cfg.dim,
            "intermediate_size": cfg.intermediate,
            "max_context_tokens": cfg.max_context,
            "model_weight_tensor_count": len(checkpoint.entries),
        },
        "policy": {
            "user_declared_capacity_mib": capacity_mib,
            "user_declared_reserve_mib": reserve_mib,
            "max_tensor_read_mib": max_tensor_mib,
            "head_rows": head_rows,
            "observed_cuda_free_bytes": None,
            "no_hardware_fit_guarantee": True,
        },
        "existing_batched_prefill_max_prompt_tokens": MAX_BATCH_PREFILL_TOKENS,
        "continuation_steps": GENERATED_STEPS,
        "targets": cases,
        "separate_future_gates": [
            "DESIGN_AND_TEST_CAUSAL_MEMORY_BOUNDED_CHUNKED_PREFILL",
            "FREEZE_HELD_OUT_LITERAL_TOKEN_ID_PROMPTS_AND_SOURCE_PROVENANCE",
            "GENERATE_NEW_INDEPENDENT_USER_DECLARED_REFERENCES_WITH_SEPARATE_APPROVAL",
            "CHECK_ALL_32_GREEDY_IDS_AND_SPARSE_LOGITS_FOR_EACH_EXECUTION",
            "REVIEW_ACTUAL_GPU_FREE_MEMORY_THERMALS_PEAKS_AND_BUFFER_LIMITS",
            "APPROVE_EACH_NEW_CUDA_RUN_IN_A_SEPARATE_FUTURE_COMMAND",
            "KEEP_SINGLE_DEFAULT_AND_PRESERVE_IMMUTABLE_RECEIPTS",
        ],
        "limitations": [
            "Only metadata and tensor-header shape inspection, NOT checkpoint SHA256 identity or reference authentication.",
            "REVIEW_REQUIRED memory estimate is NOT a safe-fit or peak-allocation guarantee.",
            "One score matrix grows quadratically with prompt length; actual future chunked workspace is undefined.",
            "Current batched candidate rejects inputs longer than 24 tokens, and the PR25 A/B harness rejects non-frozen cases.",
            "2,048 input tokens plus 31 continuation positions require at least 2,079 context positions.",
            "No 128/512/2,048-token inference, speedup, token parity, memory safety or production readiness is implied.",
        ],
    }


def plan_local(checkpoint_path, output, **settings):
    dest = bound_output(output)
    checkpoint = LocalCheckpoint(checkpoint_path)
    report = make_plan(checkpoint, **settings)
    # Config file binds part of the declared metadata; it does not bind model weights.
    config_path = checkpoint.base / "config.json"
    if config_path.is_symlink() or not config_path.is_file():
        raise ValueError("Expected local non-symlinked config.json")
    if config_path.stat().st_size > 8 * MIB:
        raise ValueError("Checkpoint config exceeds metadata-only planning limit")
    report["config_sha256_only"] = sha256(config_path.read_bytes()).hexdigest()
    write_new(dest, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan",), help="Only metadata-only planning exists; no execution command")
    parser.add_argument("--checkpoint", required=True, help="Explicit local safetensors checkpoint directory")
    parser.add_argument("--capacity-mib", type=float, default=8192.0,
                        help="User-declared SCENARIO memory policy, not detected CUDA VRAM")
    parser.add_argument("--reserve-mib", type=float, default=1024.0)
    parser.add_argument("--max-tensor-mib", type=int, default=512)
    parser.add_argument("--head-rows", type=int, default=128)
    parser.add_argument("--output", required=True, help="Exclusive new JSON receipt")
    args = parser.parse_args(argv)
    result = plan_local(
        args.checkpoint, args.output,
        capacity_mib=args.capacity_mib, reserve_mib=args.reserve_mib,
        max_tensor_mib=args.max_tensor_mib, head_rows=args.head_rows)
    print({"status": result["status"],
           "targets": [(v["prompt_tokens"], v["status"]) for v in result["targets"]],
           "model_execution": result["model_execution"], "output": args.output})


if __name__ == "__main__":
    main()
