#!/usr/bin/env python3
"""PR28: offline SHA-bound receipt audit of the eight-pair spread_24 replication.

No inference, CUDA, network, scheduler mutation, source/publisher attestation,
cross-day clock verification or statistical-speedup certification.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from phistream.reference import read_reference_trace
from scripts.export_hf_reference import file_digest
from scripts.paired_real_prefill_ab import (
    SCHEMA as PAIR_SCHEMA, preflight, summarize_pairs,
)
from scripts.real_weight_matrix import _read_bounded, _require, bound_output, write_new
from scripts.three_case_ab_audit import (
    CASES, SCHEMA as CENSUS_SCHEMA, validate_case, validate_trial,
)

SCHEMA = "PHISTREAM_REPLICATION_REVIEW_V1"
ORIGINAL_SEED = 369
REPLICATION_SEED = 70421
CASE = "spread_24"
REPLICATION_PAIRS = 8
WARMUPS = 1


def validate_replication(plan, run, expected, *, probes: int, prompt_length: int):
    """Pure metadata/trial validation, safe for synthetic tests and offline review."""
    _require(isinstance(plan, dict) and isinstance(run, dict)
             and isinstance(expected, dict), "Three receipt records required")
    def without_timestamp(record):
        return {k: v for k, v in record.items() if k != "generated_utc"}
    _require(plan.get("schema") == PAIR_SCHEMA
             and plan.get("case") == CASE
             and plan.get("status") == "SHA_BOUND_AB_PLAN_NOT_EXECUTED"
             and plan.get("model_execution") is False
             and plan.get("production_promoted") is False
             and plan.get("speedup_proven") is False
             and plan.get("source_reference_authenticated") is False
             and plan.get("scheduler_policy_changed") is False
             and without_timestamp(plan) == without_timestamp(expected),
             "Saved replication plan not identical to frozen current SHA-bound preflight")
    _require(run.get("schema") == PAIR_SCHEMA
             and run.get("case") == CASE
             and run.get("run_device") == "cuda"
             and run.get("model_execution") is True
             and run.get("candidate_selected") is True
             and run.get("production_promoted") is False
             and run.get("scheduler_policy_changed") is False
             and run.get("speedup_proven") is False
             and run.get("source_reference_authenticated") is False,
             "Replication run missing required no-promotion CUDA contract")
    for k, v in expected.items():
        if k not in ("generated_utc", "status", "model_execution"):
            _require(run.get(k) == v,
                     "Replication run differs from frozen plan reference: " + k)
    _require(run.get("status") in (
                 "MATCHED_CANDIDATE_FOR_FURTHER_REVIEW",
                 "MATCHED_BUT_PERFORMANCE_NOT_QUALIFIED"),
             "Replication not completed with matched references")
    design = expected["design"]
    _require(design["pairs"] == REPLICATION_PAIRS
             and design["seed"] == REPLICATION_SEED
             and design["warmups_per_method"] == WARMUPS,
             "Unexpected replication design")
    warmups = run.get("warmup_observations")
    pairs = run.get("paired_observations")
    _require(isinstance(warmups, list) and len(warmups) == 2 * WARMUPS
             and isinstance(pairs, list) and len(pairs) == REPLICATION_PAIRS,
             "Missing warmups or replication pairs")
    full_context = prompt_length + 31
    for i, row in enumerate(warmups):
        method = ("SINGLE", "BATCHED")[i % 2]
        _require(isinstance(row, dict)
                 and row.get("warmup") == i // 2 + 1,
                 "Wrong warmup index")
        validate_trial(row, method, probes, full_context)
    paired_trials = []
    for i, row in enumerate(pairs):
        _require(isinstance(row, dict)
                 and row.get("pair") == i + 1
                 and row.get("complete") is True
                 and row.get("order") == design["orders"][i],
                 "Replicated pair does not match frozen randomized schedule")
        methods = row.get("methods")
        _require(isinstance(methods, dict)
                 and set(methods) == {"SINGLE", "BATCHED"},
                 "Incomplete method pair")
        for method in design["orders"][i]:
            validate_trial(methods[method], method, probes, full_context)
        paired_trials.append(methods)
    computed = summarize_pairs(paired_trials)
    _require(run.get("summary") == computed,
             "Replication summary differs from per-trial recomputation")
    qualified = computed["thresholds_met_for_human_review"]
    _require(run["status"] ==
             ("MATCHED_CANDIDATE_FOR_FURTHER_REVIEW"
              if qualified else "MATCHED_BUT_PERFORMANCE_NOT_QUALIFIED"),
             "Status differs from original predeclared qualification threshold")
    return {
        "case": CASE,
        "seed": REPLICATION_SEED,
        "pairs": len(pairs),
        "model_executions_recorded": len(warmups) + 2 * len(pairs),
        "measured_executions": 2 * len(pairs),
        "warmups": len(warmups),
        "greedy_id_matches": (len(warmups) + 2 * len(pairs)) * 32,
        "sampled_logit_matches": (len(warmups) + 2 * len(pairs)) * probes,
        "median_paired_relative_gain": computed["median_paired_relative_gain"],
        "median_relative_ttft_gain": computed["median_relative_ttft_gain"],
        "median_cumulative_weight_bytes_reduction": computed[
            "median_cumulative_weight_bytes_reduction"],
        "candidate_faster_pair_count": computed["candidate_faster_pair_count"],
        "review_thresholds_met": qualified,
        "qualification": computed["claim"],
    }


def validate_original_census(census, original_cases, original_hashes):
    """Require the existing PR26 census to agree with revalidated source receipts."""
    _require(isinstance(census, dict)
             and census.get("schema") == CENSUS_SCHEMA
             and census.get("status") == "THREE_CASES_REVIEW_CANDIDATE_NOT_SPEEDUP_CERTIFIED"
             and census.get("source_reference_authenticated") is False
             and census.get("model_execution") is False
             and census.get("production_promoted") is False
             and census.get("general_speedup_proven") is False
             and census.get("statistical_significance_established") is False
             and census.get("case_count") == 3
             and census.get("cases") == original_cases
             and census.get("original_local_receipt_hashes") == original_hashes,
             "PR26 census does not match all three revalidated frozen cases")
    sums = {
        "total_pairs": sum(c["pairs"] for c in original_cases),
        "total_measured_executions": sum(c["measured_executions"] for c in original_cases),
        "total_warmup_executions": sum(c["warmup_executions"] for c in original_cases),
        "total_checked_model_executions_in_original_receipts": sum(
            c["all_executions_with_complete_parity"] for c in original_cases),
        "total_matching_greedy_ids_in_original_receipts": sum(
            c["greedy_id_matches"] for c in original_cases),
        "total_matching_sparse_logits_in_original_receipts": sum(
            c["sampled_logit_matches"] for c in original_cases),
        "total_candidate_faster_pairs": sum(
            c["candidate_faster_pair_count"] for c in original_cases),
    }
    for key, value in sums.items():
        _require(type(census.get(key)) is int and census[key] == value,
                 "Original PR26 aggregate mismatch: " + key)
    _require(all(c["review_thresholds_met"] for c in original_cases),
             "Original predeclared review thresholds no longer all satisfied")
    return sums


def audit(checkpoint, preflight_receipt, matrix_dir, output):
    """Rehash and audit 8 original files plus fresh plan/run; write new-only receipt."""
    output_path = bound_output(output)
    folder = Path(matrix_dir) / "batched-prefill"
    _require(folder.is_dir() and not folder.is_symlink(),
             "Existing original local receipt directory required")
    original_cases = []
    original_hashes = {}
    reference = None
    for case in CASES:
        expected, trace_path = preflight(
            checkpoint, preflight_receipt, matrix_dir, case, 4, ORIGINAL_SEED, 1)
        original_plan = folder / ("paired-ab-plan_" + case + ".json")
        original_run = folder / ("paired-ab-run_" + case + ".json")
        plan_sha, run_sha = file_digest(original_plan), file_digest(original_run)
        trace = read_reference_trace(trace_path)
        row = validate_case(
            _read_bounded(original_plan), _read_bounded(original_run),
            expected, case=case,
            expected_probes=sum(len(v) for v in trace["logit_probes"]),
            prompt_length=len(trace["prompt_token_ids"]),
        )
        _require(plan_sha == file_digest(original_plan)
                 and run_sha == file_digest(original_run)
                 and expected["reference_sha256"] == file_digest(trace_path),
                 "Original receipt/reference changed during review")
        row["plan_sha256"] = plan_sha
        row["run_sha256"] = run_sha
        original_cases.append(row)
        original_hashes[case] = {"plan_sha256": plan_sha, "run_sha256": run_sha}
        if case == CASE:
            reference = trace

    census_path = folder / "three-case-paired-ab-census.json"
    census_sha = file_digest(census_path)
    census = _read_bounded(census_path)
    original = validate_original_census(census, original_cases, original_hashes)
    _require(census_sha == file_digest(census_path),
             "Original census changed during read")
    expected_new, trace_path = preflight(
        checkpoint, preflight_receipt, matrix_dir,
        CASE, REPLICATION_PAIRS, REPLICATION_SEED, WARMUPS)
    replication_plan_path = folder / "replication-seed70421-plan_spread_24.json"
    replication_run_path = folder / "replication-seed70421-run_spread_24.json"
    new_plan_sha = file_digest(replication_plan_path)
    new_run_sha = file_digest(replication_run_path)
    _require(reference is not None
             and expected_new["reference_sha256"] == file_digest(trace_path)
             and expected_new["checkpoint_fingerprint"] == census["checkpoint_fingerprint"]
             and expected_new["manifest_sha256"] == census["original_manifest_sha256"]
             and expected_new["matrix_review_sha256"] == census["original_matrix_review_sha256"],
             "Replication checkpoint/reference differs from original census")
    new = validate_replication(
        _read_bounded(replication_plan_path), _read_bounded(replication_run_path),
        expected_new, probes=sum(len(v) for v in reference["logit_probes"]),
        prompt_length=len(reference["prompt_token_ids"]),
    )
    _require(new_plan_sha == file_digest(replication_plan_path)
             and new_run_sha == file_digest(replication_run_path)
             and expected_new["reference_sha256"] == file_digest(trace_path),
             "Replication file or reference changed during review")
    old_spread = next(row for row in original_cases if row["case"] == CASE)
    combined_wins = original["total_candidate_faster_pairs"] + new[
        "candidate_faster_pair_count"]
    total_pairs = original["total_pairs"] + new["pairs"]
    combined = {
        "paired_comparisons": total_pairs,
        "batched_faster_pairs": combined_wins,
        "recorded_executions_including_warmups": original[
            "total_checked_model_executions_in_original_receipts"] + new[
            "model_executions_recorded"],
        "greedy_id_matches": original[
            "total_matching_greedy_ids_in_original_receipts"] + new[
            "greedy_id_matches"],
        "sparse_logit_matches": original[
            "total_matching_sparse_logits_in_original_receipts"] + new[
            "sampled_logit_matches"],
        "cohort_count": 4,
        "replication_minus_original_spread_median_gain_percentage_points":
            round(100 * (new["median_paired_relative_gain"] -
                         old_spread["median_paired_relative_gain"]), 4),
    }
    status = ("REPLICATION_REVIEW_CANDIDATE_NOT_SPEEDUP_CERTIFIED"
              if new["review_thresholds_met"]
              else "REPLICATION_MATCHED_BUT_PERFORMANCE_NOT_QUALIFIED")
    report = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "model_execution": False,
        "source_reference_authenticated": False,
        "statistical_significance_established": False,
        "cross_day_calendar_verified_independently": False,
        "production_promoted": False,
        "production_scheduler_unchanged": True,
        "general_speedup_proven": False,
        "original_seed": ORIGINAL_SEED,
        "original_census_sha256": census_sha,
        "original_spread_24": old_spread,
        "replication": new,
        "replication_plan_sha256": new_plan_sha,
        "replication_run_sha256": new_run_sha,
        "checkpoint_fingerprint": expected_new["checkpoint_fingerprint"],
        "reference_sha256": expected_new["reference_sha256"],
        "combined": combined,
        "provenance": "OPERATOR_LOCAL_SHA_BOUND_RECEIPTS_NOT_PUBLISHER_ATTESTATION",
        "scope": "SMOLLM2_135M_ONE_RTX_5070_ONE_FROZEN_PROMPT_FOR_REPLICATION",
        "next_gate": "HELD_OUT_PROMPTS_VRAM_SAFETY_OTHER_HARDWARE_SEPARATE_AUTHORIZATION",
    }
    write_new(output_path, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--preflight-receipt", required=True)
    parser.add_argument("--matrix-dir", required=True)
    parser.add_argument("--output", required=True)
    a = parser.parse_args(argv)
    report = audit(a.checkpoint, a.preflight_receipt, a.matrix_dir, a.output)
    print({"status": report["status"],
           "replication_pairs": report["replication"]["pairs"],
           "combined_pairs": report["combined"]["paired_comparisons"],
           "model_execution": report["model_execution"],
           "output": a.output})


if __name__ == "__main__":
    main()
