"""Offline ΦPocketRotor E0 study. Default PLAN never executes a model.

Opt-in RUN compares identical canonical token IDs and weights across ordinary
streaming, conventional LRU, and deterministic recursive layer pinning.
No speculative prefetch, no MoE support, and no production activation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import torch

from phistream.llama import LlamaSettings, StreamingLlama
from phistream.memory import memory_preflight, refuse_blocked
from phistream.pocketrotor import PocketRotorCache, PocketRotorLlama
from phistream.reference import checkpoint_fingerprint
from phistream.safeweights import LocalCheckpoint

SCHEMA = "PHISTREAM_POCKETROTOR_E0_RESEARCH_V1"
MIB = 1024 * 1024


def plan_experiment(
    ckpt: LocalCheckpoint, ids: list[int], *,
    cache_bytes: int, device: str, capacity_mib: float,
    reserve_mib: float, max_tensor_mib: int, head_rows: int,
) -> dict:
    cfg = LlamaSettings.from_checkpoint(ckpt)
    if (not isinstance(ids, list) or not 2 <= len(ids) <= 32 or
            any(type(x) is not int or not 0 <= x < cfg.vocab for x in ids) or
            len(ids) > cfg.max_context):
        raise ValueError("Require 2..32 valid token IDs within context/vocabulary")
    if type(cache_bytes) is not int or not 1 <= cache_bytes <= 64 * 1024**3:
        raise ValueError("Cache size must be 1 byte..64 GiB")
    if device not in ("cpu", "cuda"):
        raise ValueError("Explicit cpu or cuda required")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing silent CPU fallback")
    observed = int(torch.cuda.mem_get_info()[0]) if device == "cuda" else None
    policy = memory_preflight(
        ckpt, context_tokens=len(ids), capacity_mib=capacity_mib,
        reserve_mib=reserve_mib, max_tensor_mib=max_tensor_mib,
        head_rows=head_rows, observed_free_bytes=observed,
    )
    nested = PocketRotorCache(ckpt, budget_bytes=cache_bytes, policy="nested-pin")
    # Conservative additive charge: active layer + optional cache. An executing
    # layer may overlap cache; do not subtract overlap from the estimate.
    fits = (policy["status"] != "BLOCKED_BY_POLICY" and
            policy["estimated_working_set_bytes"] + cache_bytes <= policy["available_policy_bytes"])
    return {
        "schema": SCHEMA, "status": "REVIEW_REQUIRED" if fits else "HELD_BY_POLICY",
        "model_execution": False, "production_promoted": False,
        "input_count": len(ids), "device": device,
        "cache_budget_bytes": cache_bytes,
        "pocket_count": len(nested.inventory),
        "nested_pinned_layer_ids": nested.pinned_layers,
        "nested_planned_fp32_bytes": nested.pinned_planned_bytes,
        "memory_preflight": policy,
        "limitations": [
            "Memory estimates exclude CUDA allocator/context and other temporary allocations.",
            "Dense-model layers are all executed; a pocket cache cannot skip weights.",
            "No asynchronous prefetch or Mixture-of-Experts routing is implemented.",
            "Plan contains no inference measurement or guaranteed memory fit.",
        ],
    }


def run_experiment(
    ckpt: LocalCheckpoint, ids: list[int], *,
    cache_bytes: int, device: str = "cpu", capacity_mib: float = 8192,
    reserve_mib: float = 1024, max_tensor_mib: int = 512,
    head_rows: int = 128, approve_local_execution: bool = False,
) -> dict:
    plan = plan_experiment(
        ckpt, ids, cache_bytes=cache_bytes, device=device,
        capacity_mib=capacity_mib, reserve_mib=reserve_mib,
        max_tensor_mib=max_tensor_mib, head_rows=head_rows,
    )
    if not approve_local_execution or type(approve_local_execution) is not bool:
        raise PermissionError("Execution requires --run AND --approve-local-execution")
    if plan["status"] != "REVIEW_REQUIRED":
        raise MemoryError("PocketRotor policy held: active decoder workspace + cache exceeds budget")
    refuse_blocked(plan["memory_preflight"])
    before = checkpoint_fingerprint(ckpt)
    reference = []
    records = []
    for policy in ("stream", "flat-lru", "nested-pin"):
        if policy == "stream":
            engine = StreamingLlama(ckpt, device=device,
                                    max_tensor_mib=max_tensor_mib, head_rows=head_rows)
        else:
            engine = PocketRotorLlama(ckpt, cache_bytes=cache_bytes, policy=policy,
                                      device=device, max_tensor_mib=max_tensor_mib,
                                      head_rows=head_rows)
        greedy = []
        worst_delta = 0.0
        max_peak = 0 if device == "cuda" else None
        started = time.perf_counter()
        for i, token in enumerate(ids):
            logits = engine.step(token)
            greedy.append(int(logits.argmax().item()))
            if policy == "stream":
                reference.append(logits.detach().cpu().clone())
            else:
                delta = torch.abs(logits.detach().cpu() - reference[i])
                worst_delta = max(worst_delta, float(delta.max().item()))
                if not torch.allclose(logits.detach().cpu(), reference[i], rtol=1e-5, atol=1e-5):
                    raise RuntimeError(f"PocketRotor output parity failed for {policy} at token {i}")
            if device == "cuda":
                max_peak = max(max_peak, int(engine.last_peak_cuda_allocated))
        elapsed = time.perf_counter() - started
        records.append({
            "policy": policy,
            "token_ids_processed": len(ids), "greedy_ids": greedy,
            "matches_streamed_baseline": True,
            "max_absolute_logit_delta": worst_delta,
            "decoder_wall_seconds": elapsed,
            "load_events": engine.weight_load_count,
            "weight_bytes_staged_total": engine.weight_bytes_staged,
            "weight_load_wall_seconds": engine.weight_load_seconds,
            "max_cuda_allocated_bytes_observed_per_step": max_peak,
            "cache": engine.pocket_cache.stats() if policy != "stream" else None,
        })
        del engine
    if checkpoint_fingerprint(ckpt) != before:
        raise RuntimeError("Checkpoint changed across experiment")
    return {
        "schema": SCHEMA, "status": "LOCAL_EXPERIMENT_PARITY_PASSED",
        "provenance": "OPERATOR_LOCAL_UNAUTHENTICATED",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_execution": True, "production_promoted": False,
        "checkpoint_fingerprint": before,
        "independent_model_reference": False,
        "device": device, "input_token_ids": ids,
        "plan": plan, "runs": records,
        "interpretation": (
            "One fixed-order local three-arm study, NOT a randomized benchmark or "
            "proof of faster inference. Streaming is a same-code baseline, not "
            "an independent transformer reference. No MoE, GPU prefetch or "
            "real checkpoint provenance authentication."
        ),
    }


def _write_or_print(receipt: dict, output: str | None):
    body = json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    if output:
        target = Path(output)
        if not target.parent.is_dir() or target.is_symlink():
            raise ValueError("Output parent must exist and must not be a symlink target")
        with target.open("x", encoding="utf-8") as fh:
            fh.write(body)
    else:
        print(body, end="")


def main() -> None:
    parser = argparse.ArgumentParser(description="PocketRotor E0 offline, default-read-only research")
    parser.add_argument("checkpoint", help="Explicit local canonical Llama safetensors directory")
    parser.add_argument("--ids", default="1,2,3,4", help="Comma-separated 2..32 token IDs")
    parser.add_argument("--cache-mib", type=float, default=256.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--capacity-mib", type=float, default=8192.0)
    parser.add_argument("--reserve-mib", type=float, default=1024.0)
    parser.add_argument("--max-tensor-mib", type=int, default=512)
    parser.add_argument("--head-rows", type=int, default=128)
    parser.add_argument("--run", action="store_true", help="Opt in to actual model inference")
    parser.add_argument("--approve-local-execution", action="store_true")
    parser.add_argument("--output", help="Optional exclusive NEW local JSON receipt")
    args = parser.parse_args()
    if not math.isfinite(args.cache_mib) or args.cache_mib <= 0:
        parser.error("--cache-mib must be finite and positive")
    if args.approve_local_execution and not args.run:
        parser.error("--approve-local-execution requires --run")
    try:
        ids = [int(part) for part in args.ids.split(",")]
    except ValueError:
        parser.error("--ids must be comma-separated integers")
    cache_bytes = int(args.cache_mib * MIB)
    checkpoint = LocalCheckpoint(args.checkpoint)
    params = dict(
        cache_bytes=cache_bytes, device=args.device,
        capacity_mib=args.capacity_mib, reserve_mib=args.reserve_mib,
        max_tensor_mib=args.max_tensor_mib, head_rows=args.head_rows,
    )
    if args.run:
        receipt = run_experiment(
            checkpoint, ids, approve_local_execution=args.approve_local_execution, **params
        )
    else:
        receipt = plan_experiment(checkpoint, ids, **params)
    _write_or_print(receipt, args.output)


if __name__ == "__main__":
    main()
