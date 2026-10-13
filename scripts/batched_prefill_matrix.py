#!/usr/bin/env python3
"""PR24: SHA-bound, explicitly approved batched-prefill parity gate.

Default 'plan' reads old evidence and hashes the checkpoint; it cannot run
inference. 'run' is operator-only, no model downloads, no scheduler changes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import statistics
import time

from scripts.export_hf_reference import file_digest
from scripts.real_weight_matrix import (
    _case_pair, _read_bounded, _require, bound_output, frozen_cases,
    pinned_checkpoint, write_new, SCHEMA as MATRIX_SCHEMA,
)

SCHEMA = "PHISTREAM_BATCHED_PREFILL_REAL_GATE_V1"
MATRIX_SUMMARY_SCHEMA = "PHISTREAM_REAL_WEIGHT_MATRIX_REVIEW_V1"
CASES_WITH_BATCH = ("mixed_7", "repeat_12", "spread_24")
STEPS = 32


def verified_baseline(checkpoint, preflight, matrix_dir):
    """Recheck ALL four prior SHA-bound baseline case pairs, not one alone."""
    folder = Path(matrix_dir)
    _require(folder.is_dir() and not folder.is_symlink(), "Original matrix directory missing or symlinked")
    manifest_path = folder / "manifest.json"
    review_path = folder / "matrix-review.json"
    manifest = _read_bounded(manifest_path)
    review = _read_bounded(review_path)
    _require(manifest.get("schema") == MATRIX_SCHEMA
             and manifest.get("status") == "FROZEN_PLAN_NOT_EXECUTED"
             and manifest.get("cases") == frozen_cases()
             and manifest.get("model_execution") is False
             and manifest.get("double_buffer_enabled") is False,
             "Original frozen matrix manifest is invalid")
    _, fingerprint, source = pinned_checkpoint(checkpoint, preflight)
    _require(manifest.get("checkpoint_fingerprint") == fingerprint
             and manifest.get("user_asserted_source") == source,
             "Original manifest/checkpoint/source not SHA-bound")
    _require(review.get("schema") == MATRIX_SUMMARY_SCHEMA
             and review.get("manifest_sha256") == file_digest(manifest_path)
             and review.get("checkpoint_fingerprint") == fingerprint
             and review.get("required_cases") == 4
             and review.get("passing_cases") == 4
             and review.get("missing_or_invalid") == []
             and review.get("model_execution") is False
             and review.get("real_model_double_buffer_promoted") is False
             and review.get("production_readiness") == "NOT_ESTABLISHED"
             and review.get("model_speedup") == "NOT_MEASURED",
             "Baseline review metadata incomplete or invalid")
    rows, all_ids, prefix5, origins = [], set(), set(), {}
    for item in manifest["cases"]:
        case = item["id"]
        trace_path = folder / ("reference_" + case + ".json")
        parity_path = folder / ("parity_" + case + ".json")
        _require(trace_path.is_file() and parity_path.is_file(),
                 "Original baseline reference/parity pair missing: " + case)
        result, ids = _case_pair(item, trace_path, parity_path, fingerprint, source)
        _require(result["status"] == "CASE_MATCH_UNAUTHENTICATED_REFERENCE",
                 "Original baseline parity no longer passes: " + case)
        rows.append(result)
        origins[case] = {
            "reference_sha256": file_digest(trace_path),
            "baseline_parity_sha256": file_digest(parity_path),
            "reference_path": trace_path,
        }
        all_ids.update(ids)
        prefix5.add(tuple(ids[:5]))
    _require(review.get("case_results") == rows
             and review.get("unique_reference_greedy_ids_across_cases") == len(all_ids)
             and review.get("distinct_first_five_greedy_sequences") == len(prefix5),
             "Original matrix case evidence has changed since its review")
    expected_status = ("MATCHED_LIMITED_FOUR_CASES"
                       if len(all_ids) >= 4 and len(prefix5) >= 2 else
                       "MATCHED_LIMITED_LOW_OUTPUT_DIVERSITY")
    _require(review.get("status") == expected_status,
             "Original matrix review status inconsistent")
    return {
        "checkpoint_fingerprint": fingerprint, "source_declared": source,
        "manifest_sha256": file_digest(manifest_path),
        "review_sha256": file_digest(review_path),
        "matrix_status": expected_status,
        "origins": origins,
    }


def evaluate_candidate(ckpt, validated, *, approve_local_execution=False,
                       device="cuda", capacity_mib=8192, reserve_mib=1024,
                       max_tensor_mib=512, head_rows=128):
    """Full 32-step teacher-forced parity after one batched prompt.

    The caller MUST authenticate unchanged checkpoint bytes and reference
    SHA before and after this function. For CPU tests only, tiny fixtures
    are allowed; actual CLI run mode requires CUDA and explicit approval.
    """
    if approve_local_execution is not True:
        raise PermissionError("Explicit local candidate-model execution approval required")
    if len(validated["expected_ids"]) != STEPS:
        raise ValueError("Candidate gate requires exactly 32 reference steps")
    if not 2 <= len(validated["prompt"]) <= 24:
        raise ValueError("Candidate gate requires 2..24 prompt tokens")
    if device not in ("cuda", "cpu"):
        raise ValueError("Only cpu/cuda explicitly supported by candidate")

    import torch
    from phistream.prefill import approved_batched_prefill

    started = time.perf_counter()
    engine, logits, initial = approved_batched_prefill(
        ckpt, validated["prompt"], approve_local_execution=True,
        device=device, capacity_mib=capacity_mib,
        reserve_mib=reserve_mib, max_tensor_mib=max_tensor_mib,
        head_rows=head_rows,
    )
    first_seconds = time.perf_counter() - started
    stage_first_bytes = engine.weight_bytes_staged
    stage_first_ops = engine.weight_load_count
    total_matched = 0
    probes_matched = 0
    largest_error = 0.0
    worst_step = None
    steps = []
    other_times = []
    max_peak = initial.get("peak_cuda_allocated_bytes")
    for index, token in enumerate(validated["expected_ids"]):
        if index:
            tick = time.perf_counter()
            logits = engine.step(validated["expected_ids"][index - 1])
            if engine.device.type == "cuda":
                torch.cuda.synchronize(engine.device)
            other_times.append(time.perf_counter() - tick)
        if engine.last_peak_cuda_allocated is not None:
            max_peak = max(max_peak or 0, int(engine.last_peak_cuda_allocated))
        greedy = int(torch.argmax(logits).item())
        id_match = (greedy == token)
        if id_match:
            total_matched += 1
        step_probes_match = True
        for probe_id, expected in validated["probes"][index].items():
            actual = float(logits[probe_id].item())
            delta = abs(actual - expected)
            if not math.isfinite(actual) or not math.isfinite(delta):
                step_probes_match = False
                worst_step = index
                continue
            if delta > largest_error:
                largest_error = delta
                worst_step = index
            if delta <= validated["atol"] + validated["rtol"] * abs(expected):
                probes_matched += 1
            else:
                step_probes_match = False
        steps.append({
            "step": index, "expected_next_id": token,
            "actual_next_id": greedy, "token_match": id_match,
            "probe_match": step_probes_match,
        })
    total_probes = sum(len(v) for v in validated["probes"])
    matched = total_matched == STEPS and probes_matched == total_probes
    return {
        "status": "MATCH_USER_REFERENCE_UNAUTHENTICATED" if matched else "PARITY_MISMATCH",
        "model_execution": True,
        "prompt_tokens": len(validated["prompt"]),
        "greedy_ids_matched": total_matched,
        "greedy_ids_total": STEPS,
        "sampled_logit_probes_matched": probes_matched,
        "sampled_logit_probes_total": total_probes,
        "max_abs_sampled_logit_error": largest_error,
        "worst_probe_step": worst_step,
        "comparisons": steps,
        "execution": {
            "time_to_first_logits_including_batched_prefill_seconds": round(first_seconds, 6),
            "continuation_step_seconds": [round(x, 6) for x in other_times],
            "decoder_wall_seconds_including_setup": round(time.perf_counter() - started, 6),
            "first_logits_weight_load_operations": stage_first_ops,
            "first_logits_weight_staging_bytes": stage_first_bytes,
            "full_run_weight_load_operations": engine.weight_load_count,
            "full_run_weight_staging_bytes": engine.weight_bytes_staged,
            "full_run_weight_loading_seconds_including_transfers": round(engine.weight_load_seconds, 6),
            "max_observed_cuda_allocated_bytes": max_peak,
            "measurement_scope": "ONE_SHOT_CANDIDATE_ONLY_NOT_RANDOMIZED_AB",
        },
        "candidate_evidence": initial,
        "reference_independence_authenticated": False,
        "production_ready": False,
        "speedup_measured": False,
        "scheduler_policy_changed": False,
    }


def process(checkpoint, preflight, matrix_dir, case, output, *,
            mode="plan", approve_local_execution=False, device="cuda",
            capacity_mib=8192, reserve_mib=1024,
            max_tensor_mib=512, head_rows=128):
    target = bound_output(output)
    if mode not in ("plan", "run"):
        raise ValueError("Use plan (default) or run")
    if case not in CASES_WITH_BATCH:
        raise ValueError("Only mixed_7, repeat_12 and spread_24 support batch prefill")
    if type(approve_local_execution) is not bool or (mode == "plan" and approve_local_execution):
        raise ValueError("Plan mode never authorizes execution")
    if mode == "run" and (not approve_local_execution or device != "cuda"):
        raise PermissionError("Real-model run requires --approve-local-execution and --device cuda")

    base = verified_baseline(checkpoint, preflight, matrix_dir)
    reference_path = base["origins"][case]["reference_path"]
    record = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PLAN_SHA_BOUND_NOT_EXECUTED",
        "case": case,
        "original_matrix_status": base["matrix_status"],
        "checkpoint_fingerprint": base["checkpoint_fingerprint"],
        "manifest_sha256": base["manifest_sha256"],
        "matrix_review_sha256": base["review_sha256"],
        "reference_sha256": base["origins"][case]["reference_sha256"],
        "baseline_parity_receipt_sha256": base["origins"][case]["baseline_parity_sha256"],
        "source_independence_authenticated": False,
        "reference_runtime_authenticated": False,
        "model_execution": False,
        "candidate_selected": mode == "run",
        "candidate_optimization_production_enabled": False,
        "single_baseline_modified": False,
        "speedup_measured": False,
        "randomized_timing_completed": False,
    }
    if mode == "run":
        from phistream.safeweights import LocalCheckpoint
        from phistream.reference import read_reference_trace, validate_reference_trace
        ckpt = LocalCheckpoint(checkpoint)
        before = file_digest(reference_path)
        _require(before == record["reference_sha256"],
                 "Reference trace SHA changed before candidate run")
        validated = validate_reference_trace(read_reference_trace(reference_path), ckpt)
        _require(validated["prompt"] == next(
            x["prompt_token_ids"] for x in frozen_cases() if x["id"] == case),
            "Reference prompt differs from original frozen prompt")
        _require(validated["fingerprint"] == base["checkpoint_fingerprint"]["files"],
                 "Reference fingerprint differs from selected checkpoint")
        result = evaluate_candidate(
            ckpt, validated, approve_local_execution=approve_local_execution,
            device=device, capacity_mib=capacity_mib, reserve_mib=reserve_mib,
            max_tensor_mib=max_tensor_mib, head_rows=head_rows,
        )
        _require(file_digest(reference_path) == before,
                 "Reference trace changed during candidate execution")
        _require(pinned_checkpoint(checkpoint, preflight)[1] == base["checkpoint_fingerprint"],
                 "Checkpoint bytes changed during candidate execution")
        record["result"] = result
        record["model_execution"] = True
        record["status"] = result["status"]
    write_new(target, record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "run"), nargs="?", default="plan")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--preflight-receipt", required=True)
    parser.add_argument("--matrix-dir", required=True)
    parser.add_argument("--case", choices=CASES_WITH_BATCH, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--capacity-mib", type=float, default=8192)
    parser.add_argument("--reserve-mib", type=float, default=1024)
    parser.add_argument("--max-tensor-mib", type=int, default=512)
    parser.add_argument("--head-rows", type=int, default=128)
    parser.add_argument("--approve-local-execution", action="store_true")
    args = parser.parse_args(argv)
    outcome = process(
        args.checkpoint, args.preflight_receipt, args.matrix_dir,
        args.case, args.output, mode=args.mode,
        approve_local_execution=args.approve_local_execution,
        device=args.device, capacity_mib=args.capacity_mib,
        reserve_mib=args.reserve_mib, max_tensor_mib=args.max_tensor_mib,
        head_rows=args.head_rows,
    )
    print({
        "status": outcome["status"], "case": outcome["case"],
        "model_execution": outcome["model_execution"],
        "output": args.output, "speedup_measured": False,
    })
    if outcome["status"] == "PARITY_MISMATCH":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
