"""Read-only independent arithmetic/structure review of PocketRotor E1 receipts.

Standard library only. This checks a supplied evidence chain and the
consistency of the claimed measurements; it cannot attest that a GPU ran,
that the independent reference was genuinely independent, or that timings
were measured honestly.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import statistics

SCHEMA = "PHISTREAM_POCKETROTOR_E1_BALANCED_V1"
AUDIT_SCHEMA = "PHISTREAM_POCKETROTOR_E1_AUDIT_V1"
METHODS = ("stream", "flat-lru", "nested-pin")
MATCH = "PARITY_MATCH_USER_REFERENCE_UNAUTHENTICATED"
MIB = 1024 * 1024


def _require(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError(why)


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate JSON key " + key)
        result[key] = value
    return result


def _read(path: str | Path, limit: int) -> dict:
    path = Path(path)
    _require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= limit,
             "Missing, linked, or oversized JSON file")
    with path.open(encoding="utf-8") as stream:
        result = json.load(stream, object_pairs_hook=_pairs,
                           parse_constant=lambda s: (_ for _ in ()).throw(ValueError("Nonfinite JSON number")))
    _require(isinstance(result, dict), "JSON root must be an object")
    return result


def _digest(path: str | Path) -> str:
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), "Missing or linked source")
    before = path.stat()
    hash_value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * MIB), b""):
            hash_value.update(block)
    after = path.stat()
    _require((before.st_size, before.st_mtime_ns, before.st_ino)
             == (after.st_size, after.st_mtime_ns, after.st_ino),
             "File changed while hashing")
    return hash_value.hexdigest()


def _finite(value, label: str, *, positive: bool = False):
    _require(type(value) in (int, float), label + ": expected number")
    _require(math.isfinite(value) and (value > 0 if positive else value >= 0),
             label + ": nonfinite or negative")
    return value


def _count(value, label: str, *, minimum: int = 0):
    _require(type(value) is int and value >= minimum, label + ": expected integer")
    return value


def _schedule(rounds: int, seed: int):
    _require(type(rounds) is int and rounds in (3, 6), "Invalid round count")
    _require(type(seed) is int and 0 <= seed <= 2**32 - 1, "Invalid schedule seed")
    offsets = list(range(3)) * (rounds // 3)
    random.Random(seed).shuffle(offsets)
    return [[METHODS[(i + shift) % 3] for i in range(3)] for shift in offsets]


def _replay_summary(rows: list[dict]) -> dict:
    computed = {}
    for candidate in ("flat-lru", "nested-pin"):
        phases = {}
        for phase in ("cold", "warm"):
            gains, wins, bytes_saved = [], 0, []
            for row in rows:
                baseline = row["methods"]["stream"][phase]
                selected = row["methods"][candidate][phase]
                b, c = baseline["wall_seconds"], selected["wall_seconds"]
                gains.append((b - c) / b)
                wins += int(c < b)
                bytes_saved.append(baseline["staged_bytes"] - selected["staged_bytes"])
            threshold = statistics.median(gains) >= .05 and wins / len(gains) >= 2 / 3
            phases[phase] = {
                "paired_relative_gains": [round(g, 6) for g in gains],
                "median_relative_gain": round(statistics.median(gains), 6),
                "faster_rounds": wins,
                "total_rounds": len(gains),
                "median_staged_bytes_avoided": statistics.median(bytes_saved),
                "meets_further_review_thresholds": threshold,
            }
        computed[candidate] = phases
    nested = computed["nested-pin"]
    return {
        "candidate_results": computed,
        "nested_candidate_for_further_review": (
            nested["cold"]["meets_further_review_thresholds"] and
            nested["warm"]["meets_further_review_thresholds"]),
        "interpretation": "DESCRIPTIVE_ONLY_NO_CERTIFIED_SPEEDUP",
        "independent_source_authenticated": False,
        "production_promoted": False,
    }


def audit(receipt_file: str | Path, plan_file: str | Path, trace_file: str | Path,
          checkpoint_dir: str | Path) -> dict:
    """Read only files. No imported torch, CUDA call, model or network activity."""
    receipt = _read(receipt_file, 32 * MIB)
    plan = _read(plan_file, 1 * MIB)
    trace = _read(trace_file, 8 * MIB)
    _require(plan.get("schema") == SCHEMA and
             receipt.get("schema") == SCHEMA,
             "Unexpected E1 schema")
    _require(plan.get("status") == "READY_FOR_REVIEW" and
             plan.get("model_execution") is False and
             plan.get("production_promoted") is False,
             "Unapproved or changed E1 plan")
    _require(receipt.get("status") == "LOCAL_PAIRED_REVIEW_ONLY"
             and receipt.get("model_execution") is True
             and receipt.get("production_promoted") is False
             and receipt.get("reference_source_authenticated") is False,
             "E1 receipt incomplete, unexecuted, or overclaims authenticity")
    _require(plan.get("reference_authentication") == "USER_ASSERTED_ONLY",
             "Unexpected reference authority claim")
    _require(receipt.get("plan_sha256") == _digest(plan_file)
             and plan.get("trace_sha256") == receipt.get("trace_sha256") == _digest(trace_file),
             "Plan/trace SHA binding mismatch")
    params = plan.get("parameters")
    _require(isinstance(params, dict) and receipt.get("parameters") == params,
             "Frozen parameters mismatch")
    rounds = _count(params.get("rounds"), "rounds", minimum=3)
    orders = _schedule(rounds, params.get("seed"))
    _require(plan.get("schedule") == orders and receipt.get("schedule") == orders
             and plan.get("execution_modes") == list(METHODS),
             "Balanced scheduler drift")
    _require(params.get("device") in ("cpu", "cuda")
             and receipt.get("device") == params["device"],
             "Device policy changed")
    cache_mib = _finite(params.get("cache_mib"), "cache MiB", positive=True)
    cache_bytes = int(cache_mib * MIB)
    _require(1 <= cache_bytes <= 64 * 1024**3, "Cache cap invalid")
    _require(trace.get("schema") == "PHISTREAM_REFERENCE_TRACE_V1"
             and isinstance(trace.get("checkpoint_fingerprint"), dict)
             and trace["checkpoint_fingerprint"].get("algorithm") == "sha256",
             "Reference schema/fingerprint missing")
    fp = trace["checkpoint_fingerprint"].get("files")
    _require(isinstance(fp, dict) and len(fp) >= 2
             and plan.get("checkpoint_fingerprint_declared") == fp
             and receipt.get("checkpoint_fingerprint") == {"algorithm": "sha256", "files": fp},
             "Checkpoint fingerprint or source identity changed")
    root = Path(checkpoint_dir)
    _require(root.is_dir() and not root.is_symlink(), "Checkpoint must be local directory")
    _require("config.json" in fp, "Missing checkpoint config")
    for name, expected in sorted(fp.items()):
        _require(isinstance(name, str) and Path(name).name == name
                 and name not in (".", "..") and "/" not in name and "\\" not in name,
                 "Noncanonical checkpoint filename")
        _require(isinstance(expected, str) and len(expected) == 64
                 and all(char in "0123456789abcdef" for char in expected),
                 "Invalid SHA256 digest")
        _require(_digest(root / name) == expected,
                 "Checkpoint bytes disagree with recorded SHA256")
    _require(len(trace.get("expected_generated_token_ids", [])) == 32
             and isinstance(trace.get("logit_probes"), list)
             and len(trace["logit_probes"]) == 32
             and isinstance(trace.get("prompt_token_ids"), list),
             "Reference does not cover 32 continuation steps")
    probes = 0
    for entry in trace["logit_probes"]:
        _require(isinstance(entry, dict) and 1 <= len(entry) <= 128,
                 "Missing sparse probes")
        probes += len(entry)
    coverage = plan.get("reference_coverage", {})
    _require(coverage.get("continuation_steps") == 32
             and coverage.get("sampled_logit_probes") == probes
             and coverage.get("prompt_tokens") == len(trace["prompt_token_ids"])
             and coverage.get("complete_context_tokens") == len(trace["prompt_token_ids"]) + 31,
             "Reference coverage drift")
    for key in ("memory_policy",):
        guard = receipt.get(key)
        _require(isinstance(guard, dict) and guard.get("status") == "REVIEW_REQUIRED"
                 and guard.get("context_tokens") == coverage["complete_context_tokens"],
                 "Memory policy held or drifted")
        _require(_count(guard.get("estimated_working_set_bytes"), "working-set bytes")
                 + cache_bytes <= _count(guard.get("available_policy_bytes"), "policy bytes"),
                 "Cache exceeds executable memory policy")
    results = receipt.get("round_results")
    _require(isinstance(results, list) and len(results) == rounds,
             "Incomplete number of rounds")
    for index, row in enumerate(results):
        _require(isinstance(row, dict) and row.get("round") == index + 1
                 and row.get("complete") is True and row.get("order") == orders[index],
                 "Incomplete or reordered round")
        methods = row.get("methods")
        _require(isinstance(methods, dict) and set(methods) == set(METHODS),
                 "Missing or unexpected methods")
        for method in METHODS:
            trial = methods[method]
            _require(isinstance(trial, dict) and trial.get("policy") == method
                     and trial.get("status") == MATCH,
                     "Trial policy mismatch or failed reference")
            for phase, label in (("cold", "cold_logical_cache"),
                                 ("warm", "warm_logical_cache")):
                observation = trial.get(phase)
                _require(isinstance(observation, dict)
                         and observation.get("phase") == label
                         and observation.get("status") == MATCH,
                         "Phase missing or parity failed")
                _require(observation.get("greedy_total") == 32
                         and observation.get("greedy_matched") == 32
                         and observation.get("probes_total") == probes
                         and observation.get("probes_matched") == probes,
                         "Reference-token/probe coverage mismatch")
                elapsed = _finite(observation.get("wall_seconds"), "wall_seconds", positive=True)
                ttft = _finite(observation.get("time_to_first_logits_seconds"), "TTFT", positive=True)
                _require(ttft <= elapsed, "TTFT exceeds complete trial elapsed")
                _finite(observation.get("max_abs_sampled_logit_error"), "sampled error")
                _finite(observation.get("weight_loading_seconds"), "weight load time")
                _count(observation.get("staged_bytes"), "staged bytes")
                _count(observation.get("load_events"), "load operations")
                _require(observation.get("full_logits_reference_tested") is False
                         and observation.get("source_reference_authenticated") is False,
                         "Reference scope falsely promoted")
                memory = observation.get("peak_cuda_allocated_per_step_bytes")
                _require((memory is None) if params["device"] == "cpu"
                         else type(memory) is int and memory >= 0,
                         "Observed CUDA memory measurement malformed")
                cache = observation.get("cache")
                if method == "stream":
                    _require(cache is None, "Stream must not expose cache residency")
                else:
                    _require(isinstance(cache, dict), "Cache counters missing")
                    for key in ("hits", "misses", "admissions", "evictions", "bypasses"):
                        _count(cache.get(key), key)
                    for key in ("peak_resident_bytes", "end_resident_bytes"):
                        _require(_count(cache.get(key), key) <= cache_bytes,
                                 "Reported resident bytes exceeded cache budget")
                    _require(cache["end_resident_bytes"] <= cache["peak_resident_bytes"],
                             "Resident accounting exceeds peak")
    independently_recomputed = _replay_summary(results)
    _require(receipt.get("summary") == independently_recomputed,
             "Summary arithmetic or claims disagree with trials")
    return {
        "schema": AUDIT_SCHEMA,
        "status": "STRUCTURE_AND_ARITHMETIC_MATCH_UNAUTHENTICATED",
        "model_executed_by_auditor": False,
        "gpu_queried_by_auditor": False,
        "original_receipt_modified": False,
        "source_reference_authenticated": False,
        "measured_execution_independently_attested": False,
        "performance_certified": False,
        "production_promoted": False,
        "receipt_sha256": _digest(receipt_file),
        "plan_sha256": _digest(plan_file),
        "trace_sha256": _digest(trace_file),
        "checkpoint_file_count": len(fp),
        "reviewed_rounds": rounds,
        "reviewed_trial_phases": rounds * 3 * 2,
        "recomputed_summary": independently_recomputed,
        "limitations": [
            "SHA256 and arithmetic consistency cannot authenticate who executed the model.",
            "Reference runtime independence is user asserted, not cryptographically attested.",
            "No model was executed and no GPU performance measurement was repeated.",
            "Cold and warm describe logical cache residency, not guaranteed SSD state.",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline read-only PocketRotor E1 evidence audit")
    ap.add_argument("--receipt", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", help="Optional exclusive new audit report")
    args = ap.parse_args()
    report = audit(args.receipt, args.plan, args.trace, args.checkpoint)
    body = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        target = Path(args.output)
        _require(target.parent.is_dir() and not target.is_symlink(),
                 "Output folder invalid")
        with target.open("x", encoding="utf-8") as stream:
            stream.write(body)
    else:
        print(body, end="")


if __name__ == "__main__":
    main()
