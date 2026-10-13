#!/usr/bin/env python3
"""PR26: read-only SHA-bound census of THREE prior PR25 paired A/B cohorts.

No model execution, no CUDA calls, no network or checkpoint writes.
This is local consistency/receipt arithmetic, not external authentication,
statistical inference, a speedup certification, or scheduler promotion.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path

from scripts.export_hf_reference import file_digest
from scripts.paired_real_prefill_ab import (
    SCHEMA as PAIR_SCHEMA, preflight, summarize_pairs,
)
from scripts.real_weight_matrix import _read_bounded, _require, bound_output, write_new
from phistream.reference import read_reference_trace

SCHEMA = "PHISTREAM_THREE_CASE_PAIRED_AB_CENSUS_V1"
CASES = ("mixed_7", "repeat_12", "spread_24")
STEPS = 32
PAIRS, SEED, WARMUPS_PER_METHOD = 4, 369, 1
MATCH = "PARITY_MATCH_USER_REFERENCE_UNAUTHENTICATED"


def _positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be finite and positive")
    return float(value)


def _nonnegative(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(name + " must be finite and nonnegative")
    return float(value)


def validate_trial(trial, method, expected_probes, full_context):
    """Independently inspect each 32-step parity ledger and metric contract."""
    _require(isinstance(trial, dict)
             and trial.get("method") == method and trial.get("status") == MATCH,
             "Trial mode or parity status invalid")
    _require(type(trial.get("greedy_total")) is int
             and trial["greedy_total"] == STEPS
             and type(trial.get("greedy_matched")) is int
             and trial["greedy_matched"] == STEPS
             and type(trial.get("logit_total")) is int
             and trial["logit_total"] == expected_probes
             and type(trial.get("logit_matched")) is int
             and trial["logit_matched"] == expected_probes,
             "Trial does not have all 32 greedy and all sparse probe matches")
    rows = trial.get("comparisons")
    _require(isinstance(rows, list) and len(rows) == STEPS
             and all(type(x) is dict and x.get("step") == i
                     and x.get("token_match") is True
                     and x.get("probe_match") is True
                     for i, x in enumerate(rows)),
             "Trial step-by-step parity flags incomplete or forged inconsistently")
    _require(trial.get("method_specific_policy_recheck_in_timed_region")
             is (method == "BATCHED"), "Timed policy-recheck declaration wrong")
    policy = trial.get("memory_policy")
    _require(isinstance(policy, dict) and policy.get("status") == "REVIEW_REQUIRED"
             and policy.get("context_tokens") == full_context
             and policy.get("capacity_source") == "USER_POLICY_CAPPED_BY_CUDA_FREE",
             "Trial lacks accepted full-context observed-CUDA memory policy")
    _positive(trial.get("decoder_seconds"), "decoder_seconds")
    first = _positive(trial.get("time_to_first_logits_seconds"), "ttft")
    _require(first <= trial["decoder_seconds"] + 1e-5,
             "Reported first-logits duration exceeds trial total")
    _nonnegative(trial.get("max_abs_sampled_logit_error"), "max probe error")
    _nonnegative(trial.get("weight_loading_seconds_including_transfers"), "load seconds")
    _require(trial["weight_loading_seconds_including_transfers"]
             <= trial["decoder_seconds"] + 1e-5,
             "Weight-load timers exceed decoder total")
    for key in ("weight_staging_bytes", "weight_load_operations",
                "peak_pytorch_cuda_allocated_bytes"):
        _require(type(trial.get(key)) is int and trial[key] > 0,
                 "Invalid or missing positive CUDA/staging counter: " + key)
    step_times = trial.get("continuation_step_seconds")
    _require(isinstance(step_times, list) and len(step_times) == STEPS - 1,
             "Missing 31 step timing observations")
    for value in step_times:
        _positive(value, "continuation step seconds")
    return {"greedy_matches": STEPS, "probe_matches": expected_probes}


def validate_case(plan, run, expected, *, case, expected_probes, prompt_length):
    """Pure receipt-only review for use with synthetic CPU CI fixtures."""
    _require(isinstance(plan, dict) and isinstance(run, dict)
             and isinstance(expected, dict), "Expected three JSON records")
    _require(plan.get("schema") == PAIR_SCHEMA
             and plan.get("status") == "SHA_BOUND_AB_PLAN_NOT_EXECUTED"
             and plan.get("case") == case
             and plan.get("model_execution") is False
             and plan.get("speedup_proven") is False
             and plan.get("scheduler_policy_changed") is False
             and plan.get("production_promoted") is False
             and plan.get("source_reference_authenticated") is False,
             "Plan did not preserve trust/non-execution boundaries")
    # Only the plan's creation timestamp may differ from a fresh re-preflight.
    def without_timestamp(record):
        return {key: value for key, value in record.items()
                if key != "generated_utc"}
    _require(without_timestamp(plan) == without_timestamp(expected),
             "Archived plan differs from current SHA-bound preflight")
    _require(run.get("schema") == PAIR_SCHEMA
             and run.get("case") == case
             and run.get("run_device") == "cuda"
             and run.get("candidate_selected") is True
             and run.get("model_execution") is True
             and run.get("speedup_proven") is False
             and run.get("scheduler_policy_changed") is False
             and run.get("production_promoted") is False
             and run.get("source_reference_authenticated") is False,
             "Run does not satisfy explicit local-CUDA/no-promotion contract")
    for key, value in expected.items():
        if key in ("generated_utc", "status", "model_execution"):
            continue
        _require(run.get(key) == value,
                 "Run differs from predeclared frozen plan binding: " + key)
    _require(run.get("status") in (
                 "MATCHED_CANDIDATE_FOR_FURTHER_REVIEW",
                 "MATCHED_BUT_PERFORMANCE_NOT_QUALIFIED"),
             "Run is not fully successful and complete")
    design = expected["design"]
    _require(design["pairs"] == PAIRS
             and design["warmups_per_method"] == WARMUPS_PER_METHOD
             and design["seed"] == SEED,
             "Wrong frozen cross-case design")
    warmups = run.get("warmup_observations")
    paired = run.get("paired_observations")
    _require(isinstance(warmups, list)
             and len(warmups) == 2 * WARMUPS_PER_METHOD
             and isinstance(paired, list) and len(paired) == PAIRS,
             "Missing expected warmups or measured pairs")
    full_context = prompt_length + STEPS - 1
    checked_trials = 0
    total_ids = 0
    total_probes = 0
    for w, row in enumerate(warmups):
        method = ("SINGLE", "BATCHED")[w % 2]
        _require(isinstance(row, dict) and row.get("warmup") == (w // 2) + 1,
                 "Warmup order/number differs from declared design")
        counts = validate_trial(row, method, expected_probes, full_context)
        checked_trials += 1
        total_ids += counts["greedy_matches"]
        total_probes += counts["probe_matches"]
    results = []
    for i, entry in enumerate(paired):
        _require(isinstance(entry, dict) and entry.get("pair") == i + 1
                 and entry.get("complete") is True
                 and entry.get("order") == design["orders"][i],
                 "Measured pair not complete or not in randomized planned order")
        methods = entry.get("methods")
        _require(isinstance(methods, dict)
                 and set(methods) == {"SINGLE", "BATCHED"},
                 "Measured pair lacks either original method")
        for method in design["orders"][i]:
            counts = validate_trial(methods[method], method, expected_probes, full_context)
            checked_trials += 1
            total_ids += counts["greedy_matches"]
            total_probes += counts["probe_matches"]
        results.append(methods)
    recomputed = summarize_pairs(results)
    _require(run.get("summary") == recomputed,
             "Recorded summary differs from recomputed individual paired trials")
    expected_status = ("MATCHED_CANDIDATE_FOR_FURTHER_REVIEW"
                       if recomputed["thresholds_met_for_human_review"]
                       else "MATCHED_BUT_PERFORMANCE_NOT_QUALIFIED")
    _require(run["status"] == expected_status,
             "Run status contradicts individual-pair performance thresholds")
    return {
        "case": case,
        "pairs": len(paired),
        "measured_executions": 2 * len(paired),
        "warmup_executions": len(warmups),
        "all_executions_with_complete_parity": checked_trials,
        "greedy_id_matches": total_ids,
        "sampled_logit_matches": total_probes,
        "median_paired_relative_gain": recomputed["median_paired_relative_gain"],
        "median_relative_ttft_gain": recomputed["median_relative_ttft_gain"],
        "candidate_faster_pair_count": recomputed["candidate_faster_pair_count"],
        "median_cumulative_weight_bytes_reduction": recomputed[
            "median_cumulative_weight_bytes_reduction"],
        "review_thresholds_met": recomputed["thresholds_met_for_human_review"],
        "qualification": recomputed["claim"],
        "caution": "ONE_LOCAL_GPU_ONE_CHECKPOINT_FROZEN_PROMPT_NOT_SPEEDUP_PROOF",
    }


def audit(checkpoint, preflight_receipt, matrix_dir, output):
    """Re-hash all referenced evidence; write one exclusive new review JSON."""
    target = bound_output(output)
    root = Path(matrix_dir)
    folder = root / "batched-prefill"
    _require(folder.is_dir() and not folder.is_symlink(),
             "Original batched-prefill receipt directory required")
    cases = []
    hashes = {}
    for case in CASES:
        expected, trace_path = preflight(
            checkpoint, preflight_receipt, matrix_dir, case,
            PAIRS, SEED, WARMUPS_PER_METHOD)
        plan_path = folder / ("paired-ab-plan_" + case + ".json")
        run_path = folder / ("paired-ab-run_" + case + ".json")
        plan_sha = file_digest(plan_path)
        run_sha = file_digest(run_path)
        plan = _read_bounded(plan_path)
        run = _read_bounded(run_path)
        original_reference = read_reference_trace(trace_path)
        case_review = validate_case(
            plan, run, expected, case=case,
            expected_probes=sum(len(row) for row in original_reference["logit_probes"]),
            prompt_length=len(original_reference["prompt_token_ids"]),
        )
        _require(plan_sha == file_digest(plan_path)
                 and run_sha == file_digest(run_path)
                 and expected["reference_sha256"] == file_digest(trace_path),
                 "Receipt or reference mutated during three-case audit")
        case_review["plan_sha256"] = plan_sha
        case_review["run_sha256"] = run_sha
        cases.append(case_review)
        hashes[case] = {"plan_sha256": plan_sha, "run_sha256": run_sha}
    result = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": ("THREE_CASES_REVIEW_CANDIDATE_NOT_SPEEDUP_CERTIFIED"
                   if all(c["review_thresholds_met"] for c in cases)
                   else "THREE_CASES_VALID_BUT_MIXED_PERFORMANCE"),
        "provenance": "READ_ONLY_USER_LOCAL_SHA_BOUND_RECEIPT_DERIVATION",
        "source_reference_authenticated": False,
        "model_execution": False,
        "scheduler_policy_changed": False,
        "production_promoted": False,
        "statistical_significance_established": False,
        "general_speedup_proven": False,
        "checkpoint_fingerprint": expected["checkpoint_fingerprint"],
        "original_manifest_sha256": expected["manifest_sha256"],
        "original_matrix_review_sha256": expected["matrix_review_sha256"],
        "cases": cases,
        "case_count": len(cases),
        "total_pairs": sum(c["pairs"] for c in cases),
        "total_measured_executions": sum(c["measured_executions"] for c in cases),
        "total_warmup_executions": sum(c["warmup_executions"] for c in cases),
        "total_checked_model_executions_in_original_receipts": sum(
            c["all_executions_with_complete_parity"] for c in cases),
        "total_matching_greedy_ids_in_original_receipts": sum(
            c["greedy_id_matches"] for c in cases),
        "total_matching_sparse_logits_in_original_receipts": sum(
            c["sampled_logit_matches"] for c in cases),
        "total_candidate_faster_pairs": sum(
            c["candidate_faster_pair_count"] for c in cases),
        "original_local_receipt_hashes": hashes,
        "next_gate": "SEPARATELY_AUTHORIZED_INDEPENDENT_REPLICATION_NEW_SEED_AND_DAY",
        "scope": "3_FROZEN_WORKLOADS_1_MODEL_1_GPU_NO_GENERALIZATION",
    }
    write_new(target, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--preflight-receipt", required=True)
    parser.add_argument("--matrix-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = audit(args.checkpoint, args.preflight_receipt, args.matrix_dir,
                   args.output)
    print({"status": report["status"], "cases": report["case_count"],
           "pairs": report["total_pairs"],
           "model_execution": report["model_execution"],
           "output": args.output})


if __name__ == "__main__":
    main()
