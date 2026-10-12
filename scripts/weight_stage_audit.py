#!/usr/bin/env python3
"""PR22: audit existing SHA-bound real-model weight staging receipts.

Reads only local already-produced matrix evidence plus explicit checkpoint.
No inference, no Transformers/PyTorch/CUDA import, no benchmark repetitions,
no scheduler/prefetch/production changes and no model writes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics

from scripts.export_hf_reference import file_digest
from scripts.real_weight_matrix import (
    CASES, SCHEMA as MATRIX_SCHEMA, _case_pair, _read_bounded,
    _require, bound_output, frozen_cases, pinned_checkpoint, write_new,
)

SCHEMA = "PHISTREAM_WEIGHT_STAGE_AUDIT_V1"
MATRIX_REVIEW_SCHEMA = "PHISTREAM_REAL_WEIGHT_MATRIX_REVIEW_V1"
PERF_SCOPE = "DECODER_ONLY_EXCLUDES_CHECKPOINT_SHA_AND_TRACE_VALIDATION"


def measured_number(data, key, *, integral=False, positive=False):
    value = data.get(key)
    if integral:
        if type(value) is not int:
            raise ValueError(f"Missing/noninteger measured field: {key}")
    elif type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError(f"Missing/nonfinite measured field: {key}")
    if value < 0 or (positive and value <= 0):
        raise ValueError(f"Negative/nonpositive measured field: {key}")
    return value


def audit(checkpoint, preflight, manifest_path, receipts_dir, matrix_summary,
          output):
    """Re-verify all actual case pairs, then derive counters, not speed claims."""
    target = bound_output(output)
    folder = Path(receipts_dir)
    _require(folder.is_dir() and not folder.is_symlink(),
             "Existing non-symlink evidence folder required")
    manifest_file = Path(manifest_path)
    manifest_sha = file_digest(manifest_file)
    manifest = _read_bounded(manifest_file)
    _require(manifest.get("schema") == MATRIX_SCHEMA
             and manifest.get("status") == "FROZEN_PLAN_NOT_EXECUTED"
             and manifest.get("cases") == frozen_cases()
             and manifest.get("model_execution") is False
             and manifest.get("double_buffer_enabled") is False,
             "Frozen matrix manifest contract invalid")
    _, fingerprint, source = pinned_checkpoint(checkpoint, preflight)
    _require(manifest.get("checkpoint_fingerprint") == fingerprint
             and manifest.get("user_asserted_source") == source,
             "Manifest checkpoint/source differs from current explicitly selected files")
    summary_file = Path(matrix_summary)
    summary_sha = file_digest(summary_file)
    summary = _read_bounded(summary_file)
    _require(summary.get("schema") == MATRIX_REVIEW_SCHEMA
             and summary.get("manifest_sha256") == manifest_sha
             and summary.get("checkpoint_fingerprint") == fingerprint
             and summary.get("status") in (
                 "MATCHED_LIMITED_FOUR_CASES",
                 "MATCHED_LIMITED_LOW_OUTPUT_DIVERSITY")
             and summary.get("required_cases") == len(CASES)
             and summary.get("passing_cases") == len(CASES)
             and summary.get("missing_or_invalid") == []
             and summary.get("model_execution") is False
             and summary.get("real_model_double_buffer_promoted") is False
             and summary.get("production_readiness") == "NOT_ESTABLISHED"
             and summary.get("model_speedup") == "NOT_MEASURED",
             "Prior four-case summary must be complete, matching, not production-qualified")

    case_rows = []
    expected_audit_rows = []
    all_ids = set()
    distinct_prefixes = set()
    payload_bytes = None
    counter_pairs = []
    for item in manifest["cases"]:
        cid = item["id"]
        trace_path = folder / ("reference_" + cid + ".json")
        parity_path = folder / ("parity_" + cid + ".json")
        _require(trace_path.is_file() and parity_path.is_file(),
                 f"Required trace and parity receipt missing: {cid}")
        case_row, ids = _case_pair(
            item, trace_path, parity_path, fingerprint, source)
        _require(case_row["status"] == "CASE_MATCH_UNAUTHENTICATED_REFERENCE",
                 f"Case did not pass previously consented parity: {cid}")
        expected_audit_rows.append(case_row)
        all_ids.update(ids)
        distinct_prefixes.add(tuple(ids[:5]))
        original = _read_bounded(parity_path)
        _require(file_digest(trace_path) == case_row["trace_sha256"],
                 "Reference trace changed during staging audit")
        execution = original.get("execution") or {}
        perf = execution.get("performance") or {}
        _require(perf.get("measurement_scope") == PERF_SCOPE
                 and perf.get("no_reference_speedup_claim") is True,
                 "Unrecognized decoder timing measurement scope")
        model = original.get("checkpoint") or {}
        w = measured_number(model, "weight_bytes", integral=True, positive=True)
        if payload_bytes is None:
            payload_bytes = w
        _require(w == payload_bytes, "Different checkpoint payload bytes across cases")
        count = measured_number(perf, "weight_load_count", integral=True, positive=True)
        staged = measured_number(perf, "weight_bytes_staged_total",
                                 integral=True, positive=True)
        timing = measured_number(perf, "total_with_engine_setup_seconds",
                                 positive=True)
        staging_sec = measured_number(perf, "weight_load_seconds_including_transfers")
        _require(staging_sec <= timing + 1e-9,
                 "Staging timer exceeds measured decoder timer")
        steps = len(item["prompt_token_ids"]) + len(ids) - 1
        restaged_per_steps = staged - steps * w
        _require(restaged_per_steps >= 0,
                 "Stage bytes lower than claimed full-weight stage per step")
        counter_pairs.append((steps,count,restaged_per_steps))
        case_rows.append({
            "id": cid,
            "prompt_tokens": len(item["prompt_token_ids"]),
            "generated_tokens": len(ids),
            "single_token_decoder_steps": steps,
            "checkpoint_weight_payload_bytes": w,
            "recorded_staging_bytes": staged,
            "recorded_weight_load_operations": count,
            "recorded_staging_seconds": round(float(staging_sec),6),
            "recorded_decoder_seconds": round(float(timing),6),
            "recorded_staging_fraction": round(staging_sec/timing,6),
            "recorded_nonstaging_seconds": round(timing-staging_sec,6),
            "staged_bytes_minus_steps_times_payload": restaged_per_steps,
            "incremental_prompt_restaging_bytes_relative_to_one_token": (
                (len(item["prompt_token_ids"])-1)*w),
            "reference_sha256": case_row["trace_sha256"],
            "parity_receipt_sha256": file_digest(parity_path),
        })
    _require(summary.get("case_results") == expected_audit_rows,
             "Prior summary case results differ from revalidated receipt pairs")
    _require(summary.get("unique_reference_greedy_ids_across_cases") == len(all_ids)
             and summary.get("distinct_first_five_greedy_sequences") ==
             len(distinct_prefixes),
             "Prior summary output-diversity measurements differ")
    expected_status = ("MATCHED_LIMITED_FOUR_CASES"
                       if len(all_ids) >= 4 and len(distinct_prefixes) >= 2
                       else "MATCHED_LIMITED_LOW_OUTPUT_DIVERSITY")
    _require(summary["status"] == expected_status,
             "Prior summary overall status inconsistent")

    offsets = {row[2] for row in counter_pairs}
    bytes_exact = len(offsets) == 1
    # Infer operation count from the 1- and 7-token cases, not an assumed
    # number of layers or output head rows.
    (s0,c0,_),(s1,c1,_) = counter_pairs[:2]
    delta_count, delta_steps = c1-c0, s1-s0
    loads_per_step = (delta_count // delta_steps
                      if delta_count > 0 and delta_count % delta_steps == 0
                      else None)
    initialization_loads = (c0 - loads_per_step*s0
                            if loads_per_step is not None else None)
    operation_linear = (
        loads_per_step is not None and initialization_loads is not None
        and initialization_loads >= 0
        and all(c == loads_per_step*s + initialization_loads
                for s,c,_ in counter_pairs)
    )
    if not operation_linear:
        loads_per_step = None
        initialization_loads = None
    status = ("EXACT_STAGE_COUNTER_LINEARITY_OBSERVED"
              if bytes_exact and operation_linear else
              "COUNTER_PATTERN_NOT_FULLY_LINEAR")
    stage_fractions = [r["recorded_staging_fraction"] for r in case_rows]
    audit_report = {
        "schema": SCHEMA,
        "status": status,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": "DERIVED_FROM_USER_SUPPLIED_LOCAL_SHA_BOUND_PARITY_RECEIPTS",
        "matrix_status": summary["status"],
        "source_reference_authenticated": False,
        "model_execution": False,
        "scheduler_policy_changed": False,
        "real_model_optimization_speedup_measured": False,
        "production_ready": False,
        "input_bindings": {
            "manifest_sha256": manifest_sha,
            "matrix_review_sha256": summary_sha,
            "checkpoint_fingerprint": fingerprint,
        },
        "cases": case_rows,
        "four_cases": len(case_rows),
        "weight_payload_bytes": payload_bytes,
        "exact_bytes_step_pattern": bytes_exact,
        "per_run_fixed_staged_bytes_outside_single_token_steps": (
            next(iter(offsets)) if bytes_exact else None),
        "exact_weight_load_operation_count_pattern": operation_linear,
        "weight_load_operations_per_decoder_step": loads_per_step,
        "fixed_initial_weight_load_operations": initialization_loads,
        "median_staging_time_fraction_observed": statistics.median(stage_fractions),
        "max_staging_time_fraction_observed": max(stage_fractions),
        "performance_scope": {
            "timings_descriptive": True,
            "each_case_only_one_run": True,
            "prompt_length_confounded_with_staging_count": True,
            "staging_timer_includes_file_read_cpu_conversion_gpu_transfer_sync_and_python_overhead": True,
            "no_overlapped_or_batched_prefill_executed": True,
            "incremental_prompt_bytes_are_a_counter_fact_not_achieved_savings": True,
            "cannot_infer_transfer_bandwidth_or_gpu_speedup": True,
        },
        "next_falsifiable_experiment": (
            "Compare a separate opt-in, causal-mask-correct batched-prefill "
            "candidate against these identical four trace hashes with full "
            "32-step greedy and sparse-logit parity, AND then do repeated "
            "paired timing trials without modifying SINGLE baseline."),
    }
    _require(file_digest(summary_file) == summary_sha
             and file_digest(manifest_file) == manifest_sha,
             "Matrix summary or manifest changed during audit")
    write_new(target,audit_report)
    return audit_report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Offline weight-stage audit over existing four-case SHA-bound receipts")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--preflight-receipt", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--receipts-dir", required=True)
    parser.add_argument("--matrix-summary", required=True)
    parser.add_argument("--output", required=True,help="New JSON, never overwrite")
    args = parser.parse_args(argv)
    report = audit(args.checkpoint,args.preflight_receipt,args.manifest,
                   args.receipts_dir,args.matrix_summary,args.output)
    print(json.dumps(report,indent=2,allow_nan=False))
    if report["status"] != "EXACT_STAGE_COUNTER_LINEARITY_OBSERVED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
