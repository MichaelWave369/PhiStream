#!/usr/bin/env python3
"""PR25: default-read-only paired A/B research gate for local PhiStream models.

No pretrained model is run without the explicit per-case CUDA run command.
No scheduler changes, automatic downloads, model rewrites, or performance
promotion. Existing SHA-bound reference evidence remains the correctness gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import random
import statistics
import time

from scripts.batched_prefill_matrix import (
    CASES_WITH_BATCH, STEPS, verified_baseline,
)
from scripts.export_hf_reference import file_digest
from scripts.real_weight_matrix import _read_bounded, _require, bound_output, write_new
from phistream.reference import read_reference_trace, validate_reference_trace
from phistream.safeweights import LocalCheckpoint

SCHEMA = "PHISTREAM_PAIRED_REAL_PREFILL_AB_V1"
CANDIDATE_SCHEMA = "PHISTREAM_BATCHED_PREFILL_REAL_GATE_V1"
MIN_PAIRS, MAX_PAIRS = 4, 20
MIN_MEDIAN_RELATIVE_GAIN = 0.05
MIN_PAIRED_WIN_FRACTION = 0.75


def schedule(pairs: int, seed: int) -> list[tuple[str, str]]:
    """Predeclare a randomized, exactly balanced AB/BA crossover order."""
    if type(pairs) is not int or pairs < MIN_PAIRS or pairs > MAX_PAIRS or pairs % 2:
        raise ValueError("Use an even number of pairs from 4 through 20")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Randomization seed must be uint32")
    orders = [("SINGLE", "BATCHED")] * (pairs // 2)
    orders += [("BATCHED", "SINGLE")] * (pairs // 2)
    random.Random(seed).shuffle(orders)
    return orders


def existing_candidate(folder: Path, case: str, baseline: dict) -> dict:
    path = folder / "batched-prefill" / ("candidate_" + case + ".json")
    receipt = _read_bounded(path)
    origin = baseline["origins"][case]
    _require(receipt.get("schema") == CANDIDATE_SCHEMA
             and receipt.get("status") == "MATCH_USER_REFERENCE_UNAUTHENTICATED"
             and receipt.get("case") == case
             and receipt.get("model_execution") is True
             and receipt.get("candidate_selected") is True
             and receipt.get("speedup_measured") is False
             and receipt.get("checkpoint_fingerprint") == baseline["checkpoint_fingerprint"]
             and receipt.get("manifest_sha256") == baseline["manifest_sha256"]
             and receipt.get("matrix_review_sha256") == baseline["review_sha256"]
             and receipt.get("reference_sha256") == origin["reference_sha256"]
             and receipt.get("baseline_parity_receipt_sha256") == origin["baseline_parity_sha256"],
             "Historical candidate/manifest/checkpoint reference contract invalid")
    result = receipt.get("result") or {}
    _require(result.get("status") == "MATCH_USER_REFERENCE_UNAUTHENTICATED"
             and result.get("greedy_ids_total") == STEPS
             and result.get("greedy_ids_matched") == STEPS
             and type(result.get("sampled_logit_probes_total")) is int
             and result["sampled_logit_probes_total"] >= STEPS * 8
             and result.get("sampled_logit_probes_matched") == result["sampled_logit_probes_total"]
             and result.get("production_ready") is False
             and result.get("speedup_measured") is False,
             "Previous candidate trial is not complete, parity-matched evidence")
    return {"sha256": file_digest(path), "path": path,
            "historical_matched_probes": result["sampled_logit_probes_total"]}


def preflight(checkpoint, preflight_receipt, matrix_dir, case: str, pairs: int,
              seed: int, warmups: int) -> tuple[dict, Path]:
    if case not in CASES_WITH_BATCH:
        raise ValueError("A/B requires mixed_7, repeat_12 or spread_24; short_1 is control only")
    orders = schedule(pairs, seed)
    if type(warmups) is not int or not 0 <= warmups <= 2:
        raise ValueError("Warmups per method must be 0, 1, or 2")
    folder = Path(matrix_dir)
    baseline = verified_baseline(checkpoint, preflight_receipt, folder)
    previous = existing_candidate(folder, case, baseline)
    trace_path = baseline["origins"][case]["reference_path"]
    ckpt = LocalCheckpoint(checkpoint)
    reference_sha = file_digest(trace_path)
    _require(reference_sha == baseline["origins"][case]["reference_sha256"],
             "Historical reference SHA differs from current trace")
    validated = validate_reference_trace(read_reference_trace(trace_path), ckpt)
    _require(validated["fingerprint"] == baseline["checkpoint_fingerprint"]["files"]
             and len(validated["expected_ids"]) == STEPS
             and 2 <= len(validated["prompt"]) <= 24
             and sum(len(p) for p in validated["probes"]) == previous["historical_matched_probes"],
             "Original 32-step reference coverage/identity differs from historical trial")
    from scripts.real_weight_matrix import frozen_cases
    expected_prompt = next(row["prompt_token_ids"] for row in frozen_cases()
                           if row["id"] == case)
    _require(validated["prompt"] == expected_prompt,
             "Reference prompt differs from frozen test case")
    report = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SHA_BOUND_AB_PLAN_NOT_EXECUTED",
        "case": case,
        "device_requested": "cuda",
        "model_execution": False,
        "scheduler_policy_changed": False,
        "production_promoted": False,
        "speedup_proven": False,
        "source_reference_authenticated": False,
        "checkpoint_fingerprint": baseline["checkpoint_fingerprint"],
        "manifest_sha256": baseline["manifest_sha256"],
        "matrix_review_sha256": baseline["review_sha256"],
        "reference_sha256": reference_sha,
        "previous_candidate_receipt_sha256": previous["sha256"],
        "baseline_parity_receipt_sha256": baseline["origins"][case]["baseline_parity_sha256"],
        "design": {
            "pairs": pairs,
            "warmups_per_method": warmups,
            "seed": seed,
            "orders": [list(x) for x in orders],
            "modes": ["SINGLE", "BATCHED"],
            "min_median_relative_paired_gain": MIN_MEDIAN_RELATIVE_GAIN,
            "min_pair_win_fraction": MIN_PAIRED_WIN_FRACTION,
            "full_32_step_exact_token_and_sparse_logit_parity_required_every_trial": True,
            "candidate_approved_execution_only": True,
            "trial_timer_includes_engine_creation_prompt_and_31_decode_steps": True,
            "full_checkpoint_hash_and_policy_preflight_outside_timer": True,
            "batch_candidate_rechecks_its_own_preflight_inside_timer": True,
            "gpu_clocks_cache_thermals_and_other_processes_uncontrolled": True,
            "confidence_interval_computed": False,
            "results_only_apply_to_one_gpu_one_checkpoint_one_prompt": True,
            "no_further_review_without_all_correctness_gates": True,
        },
    }
    return report, trace_path


def one_method(ckpt: LocalCheckpoint, validated: dict, method: str,
               *, approve_local_execution: bool, device="cuda",
               capacity_mib=8192.0, reserve_mib=1024.0,
               max_tensor_mib=512, head_rows=128) -> dict:
    """Equivalent reference checks for each mode, fresh engine for each trial."""
    if approve_local_execution is not True:
        raise PermissionError("Trial requires separate explicit local model execution approval")
    if method not in ("SINGLE", "BATCHED"):
        raise ValueError("Unknown method")
    if device not in ("cpu", "cuda"):
        raise ValueError("Only explicit cpu/cuda execution supported")
    import torch
    from phistream.llama import StreamingLlama, LlamaSettings
    from phistream.memory import memory_preflight, refuse_blocked
    from phistream.prefill import approved_batched_prefill, estimated_extra_workspace_bytes
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA unavailable; never fallback to CPU")
    expected_ids = validated["expected_ids"]
    prompt = validated["prompt"]
    if len(expected_ids) != STEPS or not 2 <= len(prompt) <= 24:
        raise ValueError("Only frozen 32-step reference cases with 2..24 token prompts")
    observed = int(torch.cuda.mem_get_info()[0]) if device == "cuda" else None
    plan = memory_preflight(
        ckpt, context_tokens=len(prompt) + STEPS - 1,
        capacity_mib=capacity_mib, reserve_mib=reserve_mib,
        max_tensor_mib=max_tensor_mib, head_rows=head_rows,
        observed_free_bytes=observed,
    )
    refuse_blocked(plan)
    if method == "BATCHED":
        extra = estimated_extra_workspace_bytes(LlamaSettings.from_checkpoint(ckpt), len(prompt))
        if plan["estimated_working_set_bytes"] + extra > plan["available_policy_bytes"]:
            raise MemoryError("Full continuation plus batched workspace exceeds estimate")
    start = time.perf_counter()
    if method == "SINGLE":
        engine = StreamingLlama(ckpt, device=device, max_tensor_mib=max_tensor_mib,
                                head_rows=head_rows)
        for token in prompt:
            logits = engine.step(token)
    else:
        engine, logits, _ = approved_batched_prefill(
            ckpt, prompt, approve_local_execution=True, device=device,
            capacity_mib=capacity_mib, reserve_mib=reserve_mib,
            max_tensor_mib=max_tensor_mib, head_rows=head_rows,
        )
    if engine.device.type == "cuda":
        torch.cuda.synchronize(engine.device)
    first_logits_seconds = time.perf_counter() - start
    max_peak = engine.last_peak_cuda_allocated
    greedy_matched, logit_matched, max_error = 0, 0, 0.0
    step_seconds = []
    probe_count = sum(len(p) for p in validated["probes"])
    rows = []
    for index, expected_id in enumerate(expected_ids):
        if index:
            tick = time.perf_counter()
            logits = engine.step(expected_ids[index - 1])
            if engine.device.type == "cuda":
                torch.cuda.synchronize(engine.device)
            step_seconds.append(time.perf_counter() - tick)
        if engine.last_peak_cuda_allocated is not None:
            max_peak = max(max_peak or 0, int(engine.last_peak_cuda_allocated))
        actual_id = int(torch.argmax(logits).item())
        id_ok = actual_id == expected_id
        greedy_matched += int(id_ok)
        probes_ok = True
        for probe_id, reference_logit in validated["probes"][index].items():
            actual_logit = float(logits[probe_id].item())
            delta = abs(actual_logit - reference_logit)
            finite = math.isfinite(actual_logit) and math.isfinite(delta)
            if finite:
                max_error = max(max_error, delta)
            if finite and delta <= validated["atol"] + validated["rtol"] * abs(reference_logit):
                logit_matched += 1
            else:
                probes_ok = False
        rows.append({"step": index, "token_match": id_ok, "probe_match": probes_ok})
    whole_seconds = time.perf_counter() - start
    matched = greedy_matched == STEPS and logit_matched == probe_count
    return {
        "method": method,
        "status": "PARITY_MATCH_USER_REFERENCE_UNAUTHENTICATED" if matched else "FAIL_PARITY",
        "greedy_matched": greedy_matched,
        "greedy_total": STEPS,
        "logit_matched": logit_matched,
        "logit_total": probe_count,
        "max_abs_sampled_logit_error": max_error,
        "comparisons": rows,
        "decoder_seconds": round(whole_seconds, 6),
        "time_to_first_logits_seconds": round(first_logits_seconds, 6),
        "continuation_step_seconds": [round(x, 6) for x in step_seconds],
        "weight_staging_bytes": engine.weight_bytes_staged,
        "weight_load_operations": engine.weight_load_count,
        "weight_loading_seconds_including_transfers": round(engine.weight_load_seconds, 6),
        "peak_pytorch_cuda_allocated_bytes": max_peak,
        "memory_policy": plan,
        "method_specific_policy_recheck_in_timed_region": method == "BATCHED",
    }


def summarize_pairs(pairs: list[dict]) -> dict:
    """Descriptive paired comparison; no statistical significance or speedup claim."""
    if not pairs:
        raise ValueError("At least one complete pair required")
    gains, ttft_gains, bytes_saved, wins = [], [], [], 0
    for pair in pairs:
        a = pair["SINGLE"]
        b = pair["BATCHED"]
        if a["status"] != "PARITY_MATCH_USER_REFERENCE_UNAUTHENTICATED" or b["status"] != "PARITY_MATCH_USER_REFERENCE_UNAUTHENTICATED":
            raise ValueError("Invalid paired result parity")
        single = a["decoder_seconds"]
        batched = b["decoder_seconds"]
        if single <= 0 or batched <= 0:
            raise ValueError("Nonpositive timing")
        gains.append((single - batched) / single)
        at, bt = a["time_to_first_logits_seconds"], b["time_to_first_logits_seconds"]
        if at <= 0 or bt <= 0:
            raise ValueError("Nonpositive TTFT timing")
        ttft_gains.append((at - bt) / at)
        bytes_saved.append(a["weight_staging_bytes"] - b["weight_staging_bytes"])
        wins += batched < single
    median_gain = statistics.median(gains)
    win_fraction = wins / len(pairs)
    review_candidate = (
        len(pairs) >= MIN_PAIRS
        and median_gain >= MIN_MEDIAN_RELATIVE_GAIN
        and win_fraction >= MIN_PAIRED_WIN_FRACTION
    )
    return {
        "pairs": len(pairs),
        "candidate_faster_pair_count": wins,
        "candidate_faster_fraction": win_fraction,
        "paired_relative_gain_values": [round(x, 6) for x in gains],
        "median_paired_relative_gain": round(median_gain, 6),
        "median_relative_ttft_gain": round(statistics.median(ttft_gains), 6),
        "median_cumulative_weight_bytes_reduction": statistics.median(bytes_saved),
        "thresholds_met_for_human_review": review_candidate,
        "claim": "CANDIDATE_FOR_REPLICATED_REVIEW_NOT_PROVEN_SPEEDUP" if review_candidate
                 else "INCONCLUSIVE_OR_NO_PERFORMANCE_BENEFIT",
        "statistical_significance_not_established": True,
        "production_scheduler_unchanged": True,
    }


def process(checkpoint, preflight_receipt, matrix_dir, case, output, *,
            mode="plan", pairs=4, seed=369, warmups=1,
            approve_local_execution=False, device="cuda",
            capacity_mib=8192.0, reserve_mib=1024.0,
            max_tensor_mib=512, head_rows=128) -> dict:
    target = bound_output(output)
    if mode not in ("plan", "run"):
        raise ValueError("Invalid A/B mode")
    if mode == "plan" and approve_local_execution:
        raise ValueError("Planning must never authorize execution")
    if mode == "run" and (approve_local_execution is not True or device != "cuda"):
        raise PermissionError("Real A/B execution requires explicit --approve-local-execution on CUDA")
    record, trace_path = preflight(checkpoint, preflight_receipt, matrix_dir, case, pairs, seed, warmups)
    if mode == "plan":
        write_new(target, record)
        return record
    record["status"] = "INCOMPLETE_OR_UNVERIFIED"
    record["model_execution"] = False
    record["warmup_observations"] = []
    record["paired_observations"] = []
    record["candidate_selected"] = True
    record["run_device"] = device
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable, no CPU fallback")
        ckpt = LocalCheckpoint(checkpoint)
        validated = validate_reference_trace(read_reference_trace(trace_path), ckpt)
        _require(file_digest(trace_path) == record["reference_sha256"],
                 "Reference changed since plan")
        for warmup in range(warmups):
            for method in ("SINGLE", "BATCHED"):
                trial = one_method(
                    ckpt, validated, method,
                    approve_local_execution=True, device=device,
                    capacity_mib=capacity_mib, reserve_mib=reserve_mib,
                    max_tensor_mib=max_tensor_mib, head_rows=head_rows,
                )
                record["model_execution"] = True
                record["warmup_observations"].append({"warmup": warmup + 1, **trial})
                _require(trial["status"] == "PARITY_MATCH_USER_REFERENCE_UNAUTHENTICATED",
                         "Warmup parity failed; stop")
        for pair_index, methods in enumerate(record["design"]["orders"], start=1):
            outcomes = {}
            for method in methods:
                trial = one_method(
                    ckpt, validated, method,
                    approve_local_execution=True, device=device,
                    capacity_mib=capacity_mib, reserve_mib=reserve_mib,
                    max_tensor_mib=max_tensor_mib, head_rows=head_rows,
                )
                record["model_execution"] = True
                outcomes[method] = trial
                if trial["status"] != "PARITY_MATCH_USER_REFERENCE_UNAUTHENTICATED":
                    record["paired_observations"].append({"pair": pair_index, "order": methods,
                                                          "methods": outcomes, "complete": False})
                    raise ValueError("Parity failure in pair " + str(pair_index) + " " + method)
            record["paired_observations"].append({
                "pair": pair_index, "order": methods, "methods": outcomes, "complete": True
            })
        summary = summarize_pairs([item["methods"] for item in record["paired_observations"]])
        record["summary"] = summary
        if file_digest(trace_path) != record["reference_sha256"]:
            raise ValueError("Reference changed during benchmark")
        from scripts.batched_prefill_matrix import pinned_checkpoint
        _, final_fp, _ = pinned_checkpoint(checkpoint, preflight_receipt)
        if final_fp != record["checkpoint_fingerprint"]:
            raise ValueError("Checkpoint changed during benchmark")
        prev = Path(matrix_dir) / "batched-prefill" / ("candidate_" + case + ".json")
        if file_digest(prev) != record["previous_candidate_receipt_sha256"]:
            raise ValueError("Historical candidate evidence changed during benchmark")
        record["status"] = ("MATCHED_CANDIDATE_FOR_FURTHER_REVIEW" if summary["thresholds_met_for_human_review"]
                            else "MATCHED_BUT_PERFORMANCE_NOT_QUALIFIED")
    except Exception as exc:
        record["status"] = "FAIL_CLOSED_INCOMPLETE_OR_UNTRUSTED"
        record["error_type"] = type(exc).__name__
        record["error_detail"] = str(exc)[:240]
    record["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_new(target, record)
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("plan", "run"), nargs="?", default="plan")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--preflight-receipt", required=True)
    p.add_argument("--matrix-dir", required=True)
    p.add_argument("--case", choices=CASES_WITH_BATCH, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--pairs", type=int, default=4)
    p.add_argument("--seed", type=int, default=369)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--device", choices=("cuda",), default="cuda")
    p.add_argument("--capacity-mib", type=float, default=8192)
    p.add_argument("--reserve-mib", type=float, default=1024)
    p.add_argument("--max-tensor-mib", type=int, default=512)
    p.add_argument("--head-rows", type=int, default=128)
    p.add_argument("--approve-local-execution", action="store_true")
    args = p.parse_args(argv)
    result = process(
        args.checkpoint, args.preflight_receipt, args.matrix_dir, args.case,
        args.output, mode=args.mode, pairs=args.pairs, seed=args.seed,
        warmups=args.warmups, approve_local_execution=args.approve_local_execution,
        device=args.device, capacity_mib=args.capacity_mib,
        reserve_mib=args.reserve_mib, max_tensor_mib=args.max_tensor_mib,
        head_rows=args.head_rows,
    )
    print({"status": result["status"], "case": result["case"],
           "model_execution": result["model_execution"], "output": args.output,
           "production_promoted": False})
    if result["status"] not in (
        "SHA_BOUND_AB_PLAN_NOT_EXECUTED",
        "MATCHED_CANDIDATE_FOR_FURTHER_REVIEW",
        "MATCHED_BUT_PERFORMANCE_NOT_QUALIFIED",
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
