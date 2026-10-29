"""ΦPocketRotor E1: SHA-bound, balanced real-weight residency research.

Metadata-only PLAN is the default. RUN is separately approved and must consume
an unchanged plan file. This is not a production inference router.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import statistics
import time

import torch

from phistream.llama import StreamingLlama
from phistream.memory import memory_preflight
from phistream.pocketrotor import PocketRotorCache, PocketRotorLlama
from phistream.reference import (
    _hash_file, checkpoint_fingerprint, read_reference_trace, validate_reference_trace,
)
from phistream.safeweights import LocalCheckpoint, _no_duplicate_fields

SCHEMA = "PHISTREAM_POCKETROTOR_E1_BALANCED_V1"
METHODS = ("stream", "flat-lru", "nested-pin")
MIB = 1024 * 1024
MAX_PLAN_BYTES = 1024 * 1024


def schedule(rounds: int, seed: int) -> list[list[str]]:
    """Each method occupies every trial position equally often."""
    if type(rounds) is not int or rounds not in (3, 6):
        raise ValueError("Require exactly 3 or 6 balanced rounds")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Seed must be an unsigned 32-bit integer")
    offsets = list(range(3)) * (rounds // 3)
    random.Random(seed).shuffle(offsets)
    return [[METHODS[(i + shift) % 3] for i in range(3)] for shift in offsets]


def _read_plan(path: str | Path) -> dict:
    target = Path(path)
    if target.is_symlink() or not target.is_file() or target.stat().st_size > MAX_PLAN_BYTES:
        raise ValueError("Plan must be a bounded local regular JSON file")
    with target.open(encoding="utf-8") as fh:
        data = json.load(fh, object_pairs_hook=_no_duplicate_fields,
                         parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if not isinstance(data, dict):
        raise ValueError("Plan must be an object")
    return data


def _number(value, label, *, min_value=0, max_value=65536) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{label} must be a number")
    value = float(value)
    if not math.isfinite(value) or not min_value <= value <= max_value:
        raise ValueError(f"{label} is outside finite bounds")
    return value


def proposed_plan(
    checkpoint: LocalCheckpoint, trace_path: str | Path, *,
    device: str = "cpu", cache_mib: float = 256,
    capacity_mib: float = 8192, reserve_mib: float = 1024,
    max_tensor_mib: int = 512, head_rows: int = 128,
    rounds: int = 3, seed: int = 369,
) -> tuple[dict, dict]:
    """No inference, full checkpoint reads, or CUDA device queries."""
    if device not in ("cpu", "cuda"):
        raise ValueError("Explicit cpu/cuda selection is required")
    cache_mib = _number(cache_mib, "cache_mib", min_value=1 / MIB)
    cache_bytes = int(cache_mib * MIB)
    if cache_bytes < 1 or cache_bytes > 64 * 1024**3:
        raise ValueError("Cache must be 1 byte to 64 GiB")
    capacity_mib = _number(capacity_mib, "capacity_mib", min_value=1 / MIB)
    reserve_mib = _number(reserve_mib, "reserve_mib")
    orders = schedule(rounds, seed)
    trace = validate_reference_trace(read_reference_trace(trace_path), checkpoint)
    if len(trace["expected_ids"]) != 32:
        raise ValueError("E1 requires exactly 32 reference continuation steps")
    ctx = len(trace["prompt"]) + 31
    policy = memory_preflight(
        checkpoint, context_tokens=ctx, capacity_mib=capacity_mib,
        reserve_mib=reserve_mib, max_tensor_mib=max_tensor_mib,
        head_rows=head_rows,
    )
    nested = PocketRotorCache(checkpoint, budget_bytes=cache_bytes, policy="nested-pin")
    fits = (policy["status"] != "BLOCKED_BY_POLICY" and
            policy["estimated_working_set_bytes"] + cache_bytes <= policy["available_policy_bytes"])
    params = {
        "device": device, "cache_mib": cache_mib,
        "capacity_mib": capacity_mib, "reserve_mib": reserve_mib,
        "max_tensor_mib": max_tensor_mib, "head_rows": head_rows,
        "rounds": rounds, "seed": seed,
    }
    result = {
        "schema": SCHEMA,
        "status": "READY_FOR_REVIEW" if fits else "HELD_BY_MEMORY_POLICY",
        "model_execution": False, "production_promoted": False,
        "reference_authentication": "USER_ASSERTED_ONLY",
        "trace_sha256": _hash_file(Path(trace_path)),
        "checkpoint_fingerprint_declared": trace["fingerprint"],
        "parameters": params,
        "reference_coverage": {
            "prompt_tokens": len(trace["prompt"]),
            "continuation_steps": len(trace["expected_ids"]),
            "sampled_logit_probes": sum(len(p) for p in trace["probes"]),
            "complete_context_tokens": ctx,
        },
        "schedule": orders,
        "nested_pinned_layer_ids": nested.pinned_layers,
        "nested_pinned_fp32_bytes": nested.pinned_planned_bytes,
        "estimated_active_plus_cache_bytes": policy["estimated_working_set_bytes"] + cache_bytes,
        "memory_policy": policy,
        "execution_modes": list(METHODS),
        "phases": ["cold_logical_cache", "warm_logical_cache"],
        "limitations": [
            "Warm = same decoder after KV reset, persistent weight cache; OS file cache is uncontrolled.",
            "Cold = fresh decoder object, NOT guaranteed cold NVMe, process RAM or GPU allocator.",
            "Planning validates metadata but does not hash the full checkpoint or invoke inference.",
            "Execution requires explicit approval, matching frozen plan and checkpoint SHA256.",
            "Runtime CUDA allocation and other process memory remain outside the policy estimate.",
            "All methods execute every layer; no MoE routing or asynchronous prefetch.",
        ],
    }
    return result, trace


def _snapshot(engine) -> dict:
    cache = getattr(engine, "pocket_cache", None)
    return {
        "staged_bytes": engine.weight_bytes_staged,
        "load_count": engine.weight_load_count,
        "load_seconds": engine.weight_load_seconds,
        "cache": cache.stats() if cache is not None else None,
    }


def _phase(engine, reference: dict, phase: str, *, started: float) -> dict:
    before = _snapshot(engine)
    expected = reference["expected_ids"]
    prompt = reference["prompt"]
    greedy_ok = probes_ok = 0
    worst_error = 0.0
    cuda_peak = None
    first_logits_seconds = None
    first_logits = None
    for token in prompt:
        first_logits = engine.step(token)
    if engine.device.type == "cuda":
        torch.cuda.synchronize(engine.device)
    first_logits_seconds = time.perf_counter() - started
    logits = first_logits
    for step, expected_token in enumerate(expected):
        if step:
            logits = engine.step(expected[step - 1])
        if engine.device.type == "cuda":
            torch.cuda.synchronize(engine.device)
        actual = int(torch.argmax(logits).item())
        greedy_ok += int(actual == expected_token)
        for token, target in reference["probes"][step].items():
            actual_logit = float(logits[token].item())
            delta = abs(actual_logit - target)
            if math.isfinite(actual_logit) and math.isfinite(delta):
                worst_error = max(worst_error, delta)
                probes_ok += int(delta <= reference["atol"] + reference["rtol"] * abs(target))
        if engine.last_peak_cuda_allocated is not None:
            cuda_peak = max(cuda_peak or 0, int(engine.last_peak_cuda_allocated))
    if engine.device.type == "cuda":
        torch.cuda.synchronize(engine.device)
    elapsed = time.perf_counter() - started
    after = _snapshot(engine)
    total_probes = sum(map(len, reference["probes"]))
    cache_delta = None
    if before["cache"] is not None:
        fields = ("hits", "misses", "admissions", "evictions", "bypasses")
        cache_delta = {field: after["cache"][field] - before["cache"][field] for field in fields}
        cache_delta.update({
            "peak_resident_bytes": after["cache"]["peak_bytes_resident"],
            "end_resident_bytes": after["cache"]["bytes_resident"],
        })
    return {
        "phase": phase,
        "status": "PARITY_MATCH_USER_REFERENCE_UNAUTHENTICATED"
                  if greedy_ok == len(expected) and probes_ok == total_probes
                  else "FAIL_PARITY",
        "greedy_matched": greedy_ok, "greedy_total": len(expected),
        "probes_matched": probes_ok, "probes_total": total_probes,
        "max_abs_sampled_logit_error": worst_error,
        "wall_seconds": elapsed,
        "time_to_first_logits_seconds": first_logits_seconds,
        "staged_bytes": after["staged_bytes"] - before["staged_bytes"],
        "load_events": after["load_count"] - before["load_count"],
        "weight_loading_seconds": after["load_seconds"] - before["load_seconds"],
        "peak_cuda_allocated_per_step_bytes": cuda_peak,
        "cache": cache_delta,
        "full_logits_reference_tested": False,
        "source_reference_authenticated": False,
    }


def measure_method(
    checkpoint: LocalCheckpoint, trace: dict, policy: str, *,
    device: str, cache_bytes: int,
    max_tensor_mib: int, head_rows: int,
) -> dict:
    """One fresh decoder, followed by same-engine KV reset and repeated sequence."""
    if policy not in METHODS:
        raise ValueError("Unknown policy")
    started = time.perf_counter()
    if policy == "stream":
        engine = StreamingLlama(checkpoint, device=device,
                                max_tensor_mib=max_tensor_mib, head_rows=head_rows)
    else:
        engine = PocketRotorLlama(checkpoint, cache_bytes=cache_bytes, policy=policy,
                                  device=device, max_tensor_mib=max_tensor_mib,
                                  head_rows=head_rows)
    if device == "cuda":
        torch.cuda.synchronize(engine.device)
    cold = _phase(engine, trace, "cold_logical_cache", started=started)
    if cold["status"] != "PARITY_MATCH_USER_REFERENCE_UNAUTHENTICATED":
        return {"policy": policy, "cold": cold, "warm": None, "status": "FAIL_PARITY"}
    engine.reset()
    started = time.perf_counter()
    warm = _phase(engine, trace, "warm_logical_cache", started=started)
    return {
        "policy": policy, "cold": cold, "warm": warm,
        "status": "PARITY_MATCH_USER_REFERENCE_UNAUTHENTICATED" if
                  warm["status"] == "PARITY_MATCH_USER_REFERENCE_UNAUTHENTICATED"
                  else "FAIL_PARITY",
    }


def summarize(round_results: list[dict]) -> dict:
    if len(round_results) not in (3, 6) or any(
            not item.get("complete") or set(item["methods"]) != set(METHODS)
            for item in round_results):
        raise ValueError("Require 3 or 6 complete three-arm rounds")
    summaries = {}
    for candidate in ("flat-lru", "nested-pin"):
        comparisons = {}
        for phase in ("cold", "warm"):
            gains = []
            wins = 0
            byte_deltas = []
            for item in round_results:
                base = item["methods"]["stream"][phase]
                trial = item["methods"][candidate][phase]
                if (base["status"] != "PARITY_MATCH_USER_REFERENCE_UNAUTHENTICATED" or
                    trial["status"] != "PARITY_MATCH_USER_REFERENCE_UNAUTHENTICATED"):
                    raise ValueError("Unmatched reference cannot become a performance win")
                b, c = base["wall_seconds"], trial["wall_seconds"]
                if not math.isfinite(b) or not math.isfinite(c) or b <= 0 or c <= 0:
                    raise ValueError("Invalid timing")
                gains.append((b - c) / b)
                wins += int(c < b)
                byte_deltas.append(base["staged_bytes"] - trial["staged_bytes"])
            threshold = (statistics.median(gains) >= 0.05 and
                         wins / len(gains) >= 2 / 3)
            comparisons[phase] = {
                "paired_relative_gains": [round(g, 6) for g in gains],
                "median_relative_gain": round(statistics.median(gains), 6),
                "faster_rounds": wins, "total_rounds": len(gains),
                "median_staged_bytes_avoided": statistics.median(byte_deltas),
                "meets_further_review_thresholds": threshold,
            }
        summaries[candidate] = comparisons
    nested = summaries["nested-pin"]
    return {
        "candidate_results": summaries,
        "nested_candidate_for_further_review": (
            nested["cold"]["meets_further_review_thresholds"] and
            nested["warm"]["meets_further_review_thresholds"]
        ),
        "interpretation": "DESCRIPTIVE_ONLY_NO_CERTIFIED_SPEEDUP",
        "independent_source_authenticated": False,
        "production_promoted": False,
    }


def _write(receipt: dict, target: Path, *, create: bool) -> None:
    data = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    # "x" protects the original output. Subsequent writes apply only after
    # this run itself created the file exclusively.
    with target.open("x" if create else "w", encoding="utf-8") as stream:
        stream.write(data)


def run_study(checkpoint_path: str | Path, trace_path: str | Path,
              plan_path: str | Path, output_path: str | Path, *,
              approve_local_execution: bool = False) -> dict:
    if approve_local_execution is not True:
        raise PermissionError("E1 execution requires explicit --approve-local-execution")
    frozen = _read_plan(plan_path)
    params = frozen.get("parameters")
    if not isinstance(params, dict):
        raise ValueError("Missing frozen plan parameters")
    ckpt = LocalCheckpoint(checkpoint_path)
    current, trace = proposed_plan(ckpt, trace_path, **params)
    if frozen != current:
        raise ValueError("Plan drift: checkpoint metadata, trace, schedule or policy changed")
    if current["status"] != "READY_FOR_REVIEW":
        raise MemoryError("The frozen memory policy holds this experiment")
    device = params["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU fallback")
    fp = checkpoint_fingerprint(ckpt)
    if fp["files"] != trace["fingerprint"]:
        raise ValueError("Full checkpoint SHA256 mismatch; refusing any inference")
    observed_free = int(torch.cuda.mem_get_info()[0]) if device == "cuda" else None
    guard = memory_preflight(
        ckpt, context_tokens=len(trace["prompt"]) + 31,
        capacity_mib=params["capacity_mib"], reserve_mib=params["reserve_mib"],
        max_tensor_mib=params["max_tensor_mib"], head_rows=params["head_rows"],
        observed_free_bytes=observed_free,
    )
    cache_bytes = int(params["cache_mib"] * MIB)
    if (guard["status"] == "BLOCKED_BY_POLICY" or
            guard["estimated_working_set_bytes"] + cache_bytes > guard["available_policy_bytes"]):
        raise MemoryError("Observed CUDA free/policy budget holds E1")
    target = Path(output_path)
    if target.is_symlink() or not target.parent.is_dir():
        raise ValueError("Receipt parent must exist; symlinks are refused")
    receipt = {
        "schema": SCHEMA, "status": "INCOMPLETE_UNVERIFIED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_execution": False, "production_promoted": False,
        "reference_source_authenticated": False,
        "checkpoint_fingerprint": fp,
        "trace_sha256": current["trace_sha256"],
        "plan_sha256": _hash_file(Path(plan_path)),
        "device": device, "memory_policy": guard,
        "parameters": params, "schedule": current["schedule"],
        "round_results": [], "limitations": current["limitations"],
    }
    _write(receipt, target, create=True)
    try:
        for round_number, ordered in enumerate(current["schedule"], 1):
            by_policy = {}
            for policy in ordered:
                # Re-observe free GPU memory as a warning, not a reliable hard
                # allocator cap. The original hard budget remains conservative.
                result = measure_method(
                    ckpt, trace, policy, device=device, cache_bytes=cache_bytes,
                    max_tensor_mib=params["max_tensor_mib"], head_rows=params["head_rows"],
                )
                receipt["model_execution"] = True
                by_policy[policy] = result
                receipt["round_results"].append({
                    "round": round_number, "order": ordered, "methods": dict(by_policy),
                    "complete": False,
                }) if policy == ordered[0] else None
                receipt["round_results"][-1]["methods"] = dict(by_policy)
                _write(receipt, target, create=False)
                if result["status"] != "PARITY_MATCH_USER_REFERENCE_UNAUTHENTICATED":
                    raise ValueError(f"Reference parity failed in round {round_number}: {policy}")
            receipt["round_results"][-1]["complete"] = True
            _write(receipt, target, create=False)
        if _hash_file(Path(trace_path)) != receipt["trace_sha256"]:
            raise ValueError("Reference JSON changed during experiment")
        if checkpoint_fingerprint(ckpt) != fp:
            raise ValueError("Checkpoint changed during experiment")
        if _hash_file(Path(plan_path)) != receipt["plan_sha256"]:
            raise ValueError("Frozen plan changed during experiment")
        receipt["summary"] = summarize(receipt["round_results"])
        receipt["status"] = "LOCAL_PAIRED_REVIEW_ONLY"
        _write(receipt, target, create=False)
    except Exception as exc:
        receipt["status"] = "FAIL_CLOSED_INCOMPLETE"
        receipt["failure_type"] = type(exc).__name__
        receipt["failure_reason"] = str(exc)[:300]
        _write(receipt, target, create=False)
        raise
    return receipt


def main() -> None:
    ap = argparse.ArgumentParser(description="PocketRotor E1 default-off paired reference lab")
    sub = ap.add_subparsers(dest="action", required=True)
    for name in ("plan", "run"):
        p = sub.add_parser(name)
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--trace", required=True)
        p.add_argument("--output", required=True)
        if name == "plan":
            p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
            p.add_argument("--cache-mib", type=float, default=256)
            p.add_argument("--capacity-mib", type=float, default=8192)
            p.add_argument("--reserve-mib", type=float, default=1024)
            p.add_argument("--max-tensor-mib", type=int, default=512)
            p.add_argument("--head-rows", type=int, default=128)
            p.add_argument("--rounds", type=int, choices=(3, 6), default=3)
            p.add_argument("--seed", type=int, default=369)
        else:
            p.add_argument("--plan", required=True)
            p.add_argument("--approve-local-execution", action="store_true")
    args = ap.parse_args()
    if args.action == "plan":
        ckpt = LocalCheckpoint(args.checkpoint)
        plan, _ = proposed_plan(
            ckpt, args.trace, device=args.device,
            cache_mib=args.cache_mib, capacity_mib=args.capacity_mib,
            reserve_mib=args.reserve_mib, max_tensor_mib=args.max_tensor_mib,
            head_rows=args.head_rows, rounds=args.rounds, seed=args.seed,
        )
        target = Path(args.output)
        if target.is_symlink() or not target.parent.is_dir():
            raise ValueError("Plan output parent must exist")
        _write(plan, target, create=True)
        print(plan["status"])
    else:
        receipt = run_study(
            args.checkpoint, args.trace, args.plan, args.output,
            approve_local_execution=args.approve_local_execution,
        )
        print(receipt["status"])


if __name__ == "__main__":
    main()
