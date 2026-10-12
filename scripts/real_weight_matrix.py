#!/usr/bin/env python3
"""Offline, read-only four-case pretrained Llama parity evidence matrix.

'plan' only fingerprints the explicitly selected checkpoint and creates a
fixed-case manifest; 'review' only reads local evidence. No model load,
Transformers import, package installation, synthetic promotion or scheduler.
Actual inference remains separately user-approved in the existing CLIs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

from scripts.export_hf_reference import check_checkpoint, file_digest, read_json

SCHEMA = "PHISTREAM_REAL_WEIGHT_MATRIX_V1"
TRACES = "PHISTREAM_REFERENCE_TRACE_V1"
RESULT = "PHISTREAM_REAL_MODEL_BRIDGE_V1"
STEPS = 32
# Frozen, deliberately different literal token-ID shapes. Not natural language.
# These cases are safe for the user's inspected SmolLM2 vocab (49,152 entries),
# but still validated against the explicitly selected config.
CASES = (
    ("short_1", (1,)),
    ("mixed_7", (9, 28, 7, 139, 302, 900, 4096)),
    ("repeat_12", (42, 42, 42, 42, 42, 42, 13, 13, 13, 13, 1, 1)),
    ("spread_24", (495, 871, 1604, 2361, 3141, 4004,
                   5127, 6203, 7129, 8261, 9347, 10589,
                   11827, 13219, 14641, 15883, 17257, 18631,
                   19937, 21247, 22621, 23917, 25127, 26393)),
)


def bound_output(target):
    if target is None:
        raise ValueError("--output required for a receipt")
    target = Path(target)
    if target.suffix.lower() != ".json" or target.exists() or target.is_symlink():
        raise ValueError("Output must be a new JSON file; never overwrite")
    return target


def write_new(target, data):
    dest = bound_output(target)
    payload = json.dumps(data, allow_nan=False, indent=2) + "\n"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x", encoding="utf-8") as output:
        output.write(payload)


def pinned_checkpoint(path, preflight):
    # This rehashes complete existing local model files and is read-only.
    receipt = read_json(preflight)
    cfg, hashes, source = check_checkpoint(path, receipt)
    if type(cfg.get("vocab_size")) is not int:
        raise ValueError("Invalid checkpoint vocabulary")
    for _, prompt in CASES:
        if len(set(prompt)) > cfg["vocab_size"] or any(
                id_ >= cfg["vocab_size"] or id_ < 0 for id_ in prompt):
            raise ValueError("Matrix prompt does not fit checkpoint vocabulary")
        if len(prompt) + STEPS - 1 > cfg.get("max_position_embeddings", 0):
            raise ValueError("Matrix prompt exceeds checkpoint max context")
    return cfg, {"algorithm": "sha256", "files": hashes}, source


def frozen_cases():
    return [{"id": name, "prompt_token_ids": list(tokens),
             "expected_generated_steps": STEPS}
            for name, tokens in CASES]


def plan(checkpoint, preflight, output):
    target = bound_output(output)
    cfg, fp, source = pinned_checkpoint(checkpoint, preflight)
    manifest = {
        "schema": SCHEMA,
        "status": "FROZEN_PLAN_NOT_EXECUTED",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_fingerprint": fp,
        "user_asserted_source": source,
        "vocab_size": cfg["vocab_size"],
        "cases": frozen_cases(),
        "provenance": "USER_CHECKPOINT_LICENSE_AND_SOURCE_UNAUTHENTICATED",
        "local_execution_authorized": False,
        "model_execution": False,
        "double_buffer_enabled": False,
        "limits": [
            "A plan is not evidence of any execution.",
            "Four literal-token prompt cases only, each requiring 32 steps.",
            "Every reference trace and PhiStream receipt requires separate explicit operator approval.",
            "Parity compares sparse logits under reported tolerances, not all logits.",
            "Reference origin and checkpoint publisher are not authenticated.",
            "No real-model speedup or production scheduling promotion.",
        ],
    }
    write_new(target, manifest)
    return {"status": manifest["status"], "manifest_path": str(target),
            "cases": [item["id"] for item in manifest["cases"]],
            "model_execution": False}


def _read_bounded(path):
    # Never follow a symlink to receipts or accept huge/untrusted JSON.
    return read_json(path, max_bytes=8 * 1024 * 1024)


def _require(flag, explanation):
    if not flag:
        raise ValueError(explanation)


def _case_pair(item, trace_path, comparison_path, expected_fp, source):
    trace_sha_before = file_digest(trace_path)
    trace = _read_bounded(trace_path)
    _require(file_digest(trace_path) == trace_sha_before,
             "trace altered during matrix review")
    result = _read_bounded(comparison_path)
    case = item["id"]
    _require(trace.get("schema") == TRACES,
             "wrong external reference trace schema")
    origin = trace.get("reference") or {}
    _require(origin.get("runtime") == "Hugging Face Transformers AutoModelForCausalLM"
             and origin.get("algorithm") == "FULL_SEQUENCE_GREEDY_NO_CACHE"
             and origin.get("asserted_independent") is True,
             "unexpected external reference declaration")
    _require(origin.get("execution_device") == "cuda",
             "matrix only accepts CUDA reference traces")
    _require(trace.get("prompt_token_ids") == item["prompt_token_ids"],
             "trace has different token prompt from frozen case")
    expected_ids = trace.get("expected_generated_token_ids")
    probes = trace.get("logit_probes")
    _require(isinstance(expected_ids, list) and len(expected_ids) == STEPS
             and all(type(x) is int and x >= 0 for x in expected_ids),
             "trace lacks exactly 32 valid greedy token IDs")
    _require(isinstance(probes, list) and len(probes) == STEPS
             and all(isinstance(x, dict) and len(x) >= 8 for x in probes),
             "trace lacks 32 adequately sampled logit maps")
    _require(trace.get("checkpoint_fingerprint") == expected_fp,
             "reference trace checkpoint fingerprint mismatch")
    exp = trace.get("experiment") or {}
    _require(exp.get("status") == "REFERENCE_OBSERVED_NOT_AUTHENTICATED"
             and exp.get("checkpoint_bytes_rehashed_after_execution") is True
             and exp.get("phistream_decoder_not_imported") is True,
             "reference provenance contract not satisfied")
    _require(exp.get("source_revision_license_user_asserted") == source,
             "reference source/revision/license differs from frozen plan")

    _require(result.get("schema") == RESULT, "wrong PhiStream receipt schema")
    _require(result.get("device_requested") == "cuda"
             and result.get("model_execution") is True
             and result.get("real_model_run_authorized") is True
             and result.get("double_buffer_enabled") is False
             and result.get("scheduler_policy_applied") is False,
             "PhiStream consent/device/scheduler contract failed")
    _require(result.get("checkpoint_fingerprint") == expected_fp,
             "PhiStream checkpoint fingerprint differs")
    declaration = result.get("user_asserted_source") or {}
    _require(all(declaration.get(k) == source[k] for k in
                 ("source", "revision", "license_id"))
             and declaration.get("declared") is True,
             "PhiStream checkpoint declarations disagree")
    receipt_ref = result.get("reference_trace") or {}
    _require(receipt_ref.get("trace_sha256") == trace_sha_before,
             "PARITY RECEIPT REFERENCE SHA256 DOES NOT MATCH TRACE FILE")
    _require(receipt_ref.get("steps") == STEPS,
             "PhiStream receipt did not claim exactly 32 steps")
    run = result.get("execution") or {}
    _require(run.get("reference") == origin
             and run.get("checkpoint_fingerprint") == expected_fp
             and run.get("device") == "cuda"
             and run.get("steps") == STEPS,
             "PhiStream execution does not bind to expected reference")
    comparisons = run.get("comparisons")
    _require(isinstance(comparisons, list) and len(comparisons) == STEPS,
             "Missing step-by-step PhiStream results")
    for i, row in enumerate(comparisons):
        _require(isinstance(row, dict) and row.get("step") == i
                 and row.get("expected_next_id") == expected_ids[i],
                 f"PhiStream compared different token at step {i}")
    expected_probes = sum(len(x) for x in probes)
    _require(run.get("total_logit_probes") == expected_probes,
             "PhiStream did not compare all reference probes")
    _require(result.get("status") in (
        "PARITY_MATCH_USER_REFERENCE_UNAUTHENTICATED", "PARITY_MISMATCH"
    ), "Unexpected result status")
    match = (
        result["status"] == "PARITY_MATCH_USER_REFERENCE_UNAUTHENTICATED"
        and run.get("status") == "MATCH_USER_REFERENCE_UNVERIFIED"
        and run.get("exact_greedy_tokens_matched") == STEPS
        and run.get("logit_probes_matched") == expected_probes
        and all(r.get("token_match") is True and r.get("probe_match") is True
                for r in comparisons)
    )
    # Bound parsed trace bytes also after reading comparison receipt.
    _require(file_digest(trace_path) == trace_sha_before,
             "trace changed during matrix verification")
    perf = run.get("performance") or {}
    def nonnegative(key):
        n = perf.get(key)
        if type(n) not in (int, float) or not math.isfinite(n) or n < 0:
            return None
        return float(n)
    return {
        "id": case,
        "status": "CASE_MATCH_UNAUTHENTICATED_REFERENCE" if match else
                  "CASE_PARITY_MISMATCH",
        "prompt_length": len(item["prompt_token_ids"]),
        "tokens_matched": run.get("exact_greedy_tokens_matched"),
        "total_steps": STEPS,
        "logit_probes_matched": run.get("logit_probes_matched"),
        "logit_probes_total": expected_probes,
        "unique_reference_greedy_ids": len(set(expected_ids)),
        "max_abs_sampled_logit_error": run.get("max_abs_sampled_logit_error"),
        "decoder_seconds": nonnegative("total_with_engine_setup_seconds"),
        "weight_staging_seconds": nonnegative("weight_load_seconds_including_transfers"),
        "weight_stage_bytes": nonnegative("weight_bytes_staged_total"),
        "trace_sha256": trace_sha_before,
        "reference_file": trace_path.name,
        "parity_file": comparison_path.name,
    }, expected_ids


def review(checkpoint, preflight, manifest_path, receipts_dir, output):
    target = bound_output(output)
    manifest = _read_bounded(manifest_path)
    _require(manifest.get("schema") == SCHEMA
             and manifest.get("status") == "FROZEN_PLAN_NOT_EXECUTED"
             and manifest.get("model_execution") is False
             and manifest.get("double_buffer_enabled") is False
             and manifest.get("cases") == frozen_cases(),
             "Invalid/modified frozen matrix plan")
    _, fp, source = pinned_checkpoint(checkpoint, preflight)
    _require(manifest.get("checkpoint_fingerprint") == fp
             and manifest.get("user_asserted_source") == source,
             "Matrix plan checkpoint fingerprint/source differs from local bytes")
    evidence_dir = Path(receipts_dir)
    _require(evidence_dir.is_dir() and not evidence_dir.is_symlink(),
             "Existing nonsymlink evidence directory required")
    reports = []
    unmatched = []
    all_ids = set()
    first_five_patterns = set()
    for item in manifest["cases"]:
        case = item["id"]
        trace_path = evidence_dir / ("reference_" + case + ".json")
        comparison_path = evidence_dir / ("parity_" + case + ".json")
        if not trace_path.is_file() or not comparison_path.is_file():
            unmatched.append({"id": case, "reason": "MISSING_REFERENCE_OR_PARITY_FILE"})
            continue
        try:
            outcome, reference_ids = _case_pair(
                item, trace_path, comparison_path, fp, source
            )
            reports.append(outcome)
            all_ids.update(reference_ids)
            first_five_patterns.add(tuple(reference_ids[:5]))
        except (ValueError, OSError, KeyError, TypeError) as exc:
            unmatched.append({"id": case, "reason": "INVALID_EVIDENCE",
                              "detail": str(exc)[:160]})
    all_match = len(reports) == len(CASES) and all(
        r["status"] == "CASE_MATCH_UNAUTHENTICATED_REFERENCE" for r in reports)
    if unmatched:
        status = ("INCOMPLETE_EVIDENCE" if all(
            e["reason"] == "MISSING_REFERENCE_OR_PARITY_FILE" for e in unmatched
        ) else "FAIL_TRUST_CONTRACT")
    elif not all_match:
        status = "FAIL_PARITY"
    elif len(all_ids) < 4 or len(first_five_patterns) < 2:
        status = "MATCHED_LIMITED_LOW_OUTPUT_DIVERSITY"
    else:
        status = "MATCHED_LIMITED_FOUR_CASES"
    valid_elapsed = [r["decoder_seconds"] for r in reports
                     if r["decoder_seconds"] is not None]
    stage_shares = [r["weight_staging_seconds"] / r["decoder_seconds"]
                    for r in reports if r["decoder_seconds"] and
                    r["weight_staging_seconds"] is not None]
    final = {
        "schema": "PHISTREAM_REAL_WEIGHT_MATRIX_REVIEW_V1",
        "status": status,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": file_digest(Path(manifest_path)),
        "checkpoint_fingerprint": fp,
        "required_cases": len(CASES),
        "case_results": reports,
        "missing_or_invalid": unmatched,
        "passing_cases": sum(r["status"] == "CASE_MATCH_UNAUTHENTICATED_REFERENCE"
                             for r in reports),
        "unique_reference_greedy_ids_across_cases": len(all_ids),
        "distinct_first_five_greedy_sequences": len(first_five_patterns),
        "median_decoder_seconds_descriptive_only": (
            statistics.median(valid_elapsed) if valid_elapsed else None
        ),
        "median_weight_staging_fraction_descriptive_only": (
            statistics.median(stage_shares) if stage_shares else None
        ),
        "all_reference_origins_authenticated": False,
        "real_model_double_buffer_promoted": False,
        "model_execution": False,
        "production_readiness": "NOT_ESTABLISHED",
        "model_speedup": "NOT_MEASURED",
        "limitations": [
            "Offline user-local evidence review, no neural model execution.",
            "Each greedy/token/logit comparison is to user-declared external reference.",
            "Sparse logit probes are not full-vocabulary equality.",
            "Cases use arbitrary literal token IDs, not tokenizer-encoded language.",
            "All cases target only one model/revision, one GPU, one precision and one runtime configuration.",
            "No claim of cross-platform production readiness or real-model scheduling speedup.",
        ],
    }
    write_new(target, final)
    return final


def main(argv=None):
    cli = argparse.ArgumentParser(
        description="Strict four-case, SHA-bound real-weight evidence matrix, offline")
    sub = cli.add_subparsers(dest="command", required=True)
    for name in ("plan", "review"):
        parser = sub.add_parser(name, help="No inference: create frozen plan or review supplied evidence")
        parser.add_argument("--checkpoint", required=True)
        parser.add_argument("--preflight-receipt", required=True)
        parser.add_argument("--output", required=True, help="New local JSON; exclusive create")
        if name == "review":
            parser.add_argument("--manifest", required=True)
            parser.add_argument("--receipts-dir", required=True)
    args = cli.parse_args(argv)
    if args.command == "plan":
        data = plan(args.checkpoint, args.preflight_receipt, args.output)
    else:
        data = review(args.checkpoint, args.preflight_receipt, args.manifest,
                      args.receipts_dir, args.output)
    print(json.dumps(data, indent=2, allow_nan=False))
    if args.command == "review" and data["status"].startswith(
            ("FAIL_", "INCOMPLETE_")):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
