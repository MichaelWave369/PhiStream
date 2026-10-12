"""PhiStream PR18: offline qualification of every supplied guarded CUDA receipt.

This code consumes local, unauthenticated self-reports, never model files or
CUDA devices. Qualifications are synthetic ONLY and cannot mutate policy.
"""
from __future__ import annotations

from datetime import datetime, timezone
import glob
import hashlib
import json
import math
from pathlib import Path
import statistics

SCHEMA = "PHISTREAM_SCHEDULER_QUALIFICATION_V1"
PROVENANCE = "REVIEWED_UNAUTHENTICATED_GUARDED_SYNTHETIC_COHORT"
RUN_SCHEMA = "PHISTREAM_GUARDED_SYNTHETIC_RUN_V1"
RUN_PROVENANCE = "LOCALLY_EXECUTED_SYNTHETIC_GPU_SCRATCH_NO_MODEL"
MIN_RUNS = 3
MIN_RANDOMIZED = 3
MAX_FILES = 16
MAX_BYTES = 512 * 1024
GAIN_FLOOR = 0.05
WIN_FLOOR = 0.75


def find_guarded_receipts(pattern: str) -> list[str]:
    if (not isinstance(pattern, str) or not pattern or len(pattern) > 1024
            or "**" in pattern):
        raise ValueError("Use one local, nonrecursive guarded receipt glob")
    paths = sorted(glob.glob(pattern))
    if not 1 <= len(paths) <= MAX_FILES:
        raise ValueError("Expected 1–16 matching guarded receipts")
    return paths


def _utc(value):
    if not isinstance(value, str):
        raise ValueError("Timestamp required")
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("UTC-offset timestamp required")
    return stamp.astimezone(timezone.utc)


def _positive(v):
    return type(v) in (int, float) and math.isfinite(v) and v > 0


def _check_series(obj):
    if not isinstance(obj, dict):
        raise ValueError("Missing series")
    x = obj.get("samples_seconds")
    if not isinstance(x, list) or len(x) != 6 or any(not _positive(t) for t in x):
        raise ValueError("Exactly six finite positive raw trials required")
    if type(obj.get("repetitions")) is not int or obj["repetitions"] != 6:
        raise ValueError("Series repetition count must be 6")
    for label, expected in (
        ("median_seconds", statistics.median(x)),
        ("min_seconds", min(x)), ("max_seconds", max(x))
    ):
        if not _positive(obj.get(label)) or abs(obj[label] - expected) > 0.000002:
            raise ValueError("Raw samples disagree with reported " + label)
    return x


def _inspect(obj):
    if not isinstance(obj, dict):
        raise ValueError("Expected object")
    if (obj.get("schema") != RUN_SCHEMA or obj.get("provenance") != RUN_PROVENANCE
        or obj.get("status") != "PASS_GUARDED_CUDA_SYNTHETIC"
        or obj.get("requested_double") is not True
        or obj.get("explicit_operator_approval") is not True
        or obj.get("selected_mode") != "double_reuse_worker"
        or obj.get("fallback_reasons") != []
        or obj.get("control_is_deliberately_nonoverlapping") is not True
        or obj.get("temporary_scratch_deleted") is not True
        or obj.get("model_execution") is not False
        or obj.get("production_policy_applied") is not False
        or obj.get("automatic_layer_placement") is not False
        or obj.get("original_gate_status") != "ELIGIBLE_FOR_SYNTHETIC_OPT_IN"
        or obj.get("live_hardware_check") != "MATCHED_UNAUTHENTICATED_RECEIPT_COHORT"):
        raise ValueError("Not an approved, matched, passing synthetic double run")
    if (type(obj.get("chunk_mib")) is not int or obj["chunk_mib"] != 8
        or type(obj.get("chunks")) is not int or obj["chunks"] != 8):
        raise ValueError("Not the qualified 8 MiB × 8 synthetic shape")
    integrity = obj.get("integrity") or {}
    if (integrity.get("full_sha256_host_and_gpu_audit") is not True
        or type(integrity.get("audit_chunks")) is not int
        or integrity["audit_chunks"] != 8
        or integrity.get("audit_is_outside_timing") is not True):
        raise ValueError("Missing full out-of-band GPU replay integrity audit")
    gate = obj.get("gate_summary") or {}
    hw = gate.get("hardware_claim") or {}
    if (gate.get("schema") != "PHISTREAM_GUARDED_SCHEDULER_V1"
        or gate.get("status") != "ELIGIBLE_FOR_SYNTHETIC_OPT_IN"
        or gate.get("unverified_external_inputs") is not True
        or gate.get("production_policy_applied") is not False
        or gate.get("model_execution") is not False
        or gate.get("default") != "SINGLE"
        or gate.get("guards") != []
        or type(gate.get("validated_receipt_count")) is not int
        or gate["validated_receipt_count"] < 3
        or (gate.get("conditions") or {}).get("rounds_per_run") != 12
        or (gate.get("conditions") or {}).get("chunk_mib") != 8
        or (gate.get("conditions") or {}).get("chunks") != 8):
        raise ValueError("Invalid original conservative eligibility gate")
    key_fields = ("platform", "gpu_name", "torch_version", "cuda_build")
    if any(not isinstance(hw.get(k), str) or not hw[k] for k in key_fields):
        raise ValueError("Missing original hardware identity")
    start, end = _utc(obj.get("started_utc")), _utc(obj.get("completed_utc"))
    if not start < end:
        raise ValueError("Invalid capture interval")
    double = _check_series(obj.get("selected_end_to_end"))
    control = _check_series(obj.get("serialized_same_worker_negative_control"))
    gains = [(c - d) / c for c, d in zip(control, double)]
    wins = sum(x > 0 for x in gains)
    ratio = (statistics.median(control) - statistics.median(double)) / statistics.median(control)
    # Randomized controls added after legacy PR17. Legacy receipts remain
    # included, but cannot satisfy the randomized qualification requirement.
    randomization = obj.get("order_randomization")
    randomized = False
    if randomization is not None:
        if not isinstance(randomization, dict):
            raise ValueError("Malformed order randomization")
        flags = randomization.get("control_first_per_trial")
        if (randomization.get("mode") == "BALANCED_SHUFFLE"
            and type(randomization.get("seed")) is int
            and 0 <= randomization["seed"] < 2**64
            and type(flags) is list and len(flags) == 6
            and all(type(x) is bool for x in flags)
            and sum(flags) == 3):
            randomized = True
        elif (randomization.get("mode") == "DETERMINISTIC_ALTERNATION"
              and flags == [True, False, True, False, True, False]
              and randomization.get("seed") is None):
            randomized = False
        else:
            raise ValueError("Invalid randomized or deterministic control order")
    return {
        "started": start, "ended": end,
        "hardware": tuple(hw[k] for k in key_fields),
        "randomized": randomized,
        "samples": len(double),
        "wins": wins, "win_share": wins / 6,
        "median_gain_from_method_medians": ratio,
        "median_paired_gain": statistics.median(gains),
        "passed_5pct_and_75pct": ratio >= GAIN_FLOOR and wins >= 5,
        "gains": gains,
    }


def qualify_guarded(receipt_files: list[str | Path]) -> dict:
    if (not isinstance(receipt_files, list)
        or not 1 <= len(receipt_files) <= MAX_FILES):
        raise ValueError("Provide 1–16 local receipt filenames")
    rejected = []
    accepted = []
    seen_digest = set()
    seen_starts = set()
    for item in receipt_files:
        file = Path(item)
        try:
            if not file.is_file() or file.is_symlink():
                raise ValueError("Missing, symlink or non-file receipt")
            if file.stat().st_size > MAX_BYTES:
                raise ValueError("Receipt exceeds 512 KiB cap")
            raw = file.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if digest in seen_digest:
                raise ValueError("Duplicate receipt SHA256 content")
            seen_digest.add(digest)
            obj = json.loads(raw)
            data = _inspect(obj)
            if data["started"] in seen_starts:
                raise ValueError("Duplicate experiment start time")
            seen_starts.add(data["started"])
            accepted.append((file.name, digest, data))
        except (OSError, ValueError, TypeError, OverflowError, UnicodeError, json.JSONDecodeError) as exc:
            rejected.append({"filename": file.name, "reason": str(exc)[:200]})
    blockers = []
    if rejected: blockers.append("INVALID_OR_DUPLICATE_RECEIPT")
    if len(accepted) < MIN_RUNS: blockers.append("INSUFFICIENT_VALID_RUNS")
    identities = {data["hardware"] for _, _, data in accepted}
    if len(identities) > 1: blockers.append("MIXED_HARDWARE_OR_SOFTWARE")
    by_time = sorted([data for _, _, data in accepted], key=lambda r: r["started"])
    if any(b["started"] <= a["ended"] for a, b in zip(by_time, by_time[1:])):
        blockers.append("OVERLAPPING_CAPTURE_WINDOWS")
    all_gains = [gain for _, _, data in accepted for gain in data["gains"]]
    pass_count = sum(data["passed_5pct_and_75pct"] for _, _, data in accepted)
    randomized_count = sum(data["randomized"] for _, _, data in accepted)
    notes = []
    if accepted and pass_count != len(accepted):
        notes.append("AT_LEAST_ONE_RUN_BELOW_PREDECLARED_5PCT_OR_75PCT_GATE")
    if randomized_count < MIN_RANDOMIZED:
        notes.append("FEWER_THAN_THREE_BALANCED_RANDOMIZED_ORDER_RUNS")
    if all_gains and (statistics.median(all_gains) < GAIN_FLOOR
                      or sum(g > 0 for g in all_gains)/len(all_gains) < WIN_FLOOR):
        notes.append("POOLED_EVIDENCE_BELOW_GATE")
    if blockers:
        status = "REJECTED"
    elif notes:
        status = "INCONCLUSIVE"
    else:
        status = "QUALIFIED_SYNTHETIC"
    identity = accepted[0][2]["hardware"] if len(identities) == 1 and accepted else None
    return {
        "schema": SCHEMA, "provenance": PROVENANCE,
        "status": status, "default": "SINGLE",
        "production_policy_applied": False, "model_execution": False,
        "automatic_layer_placement": False,
        "input_receipt_count": len(receipt_files),
        "accepted_count": len(accepted),
        "rejected_receipts": rejected,
        "blockers": blockers, "qualification_notes": notes,
        "qualified_run_count": pass_count, "randomized_run_count": randomized_count,
        "hardware_claim": (dict(zip(("platform","gpu_name","torch_version","cuda_build"), identity))
                           if identity is not None else None),
        "conditions": {
            "min_valid_runs": MIN_RUNS, "min_randomized_runs": MIN_RANDOMIZED,
            "per_run_min_median_gain": GAIN_FLOOR,
            "per_run_min_win_share": WIN_FLOOR,
            "all_included_runs_must_pass": True,
            "min_pooled_median_gain": GAIN_FLOOR,
            "min_pooled_win_share": WIN_FLOOR,
        },
        "aggregate": {
            "paired_trials": len(all_gains),
            "double_wins": sum(g > 0 for g in all_gains),
            "median_paired_gain": round(statistics.median(all_gains), 6) if all_gains else None,
            "win_share": round(sum(g > 0 for g in all_gains)/len(all_gains), 6)
                         if all_gains else None,
        },
        "ledger": [{
            "filename": name, "sha256": digest,
            "started_utc": data["started"].isoformat(),
            "randomized_order": data["randomized"],
            "pair_wins": data["wins"],
            "median_paired_gain": round(data["median_paired_gain"], 6),
            "gain_from_method_medians": round(data["median_gain_from_method_medians"], 6),
            "passes_individual_gate": data["passed_5pct_and_75pct"],
        } for name, digest, data in accepted],
        "review_only": True,
        "unverified_external_inputs": True,
        "limitations": [
            "JSON is user-supplied and unauthenticated; SHA256 of the receipt detects exact duplicates, not fabrication.",
            "Legacy deterministic-order runs remain visible and never silently discarded.",
            "The benchmark reads synthetic owned scratch only and does not execute a language model.",
            "No cold-cache I/O claim; CUDA and host CPU clocks were not calibrated.",
            "QUALIFIED_SYNTHETIC is research classification only, not runtime permission.",
        ],
    }
