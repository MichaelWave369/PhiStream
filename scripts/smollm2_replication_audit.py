"""Offline, archive-only verification of a frozen SmolLM2 real-weight replication.

Reads a user-supplied ZIP without extracting, importing torch/transformers,
downloading weights, or querying a GPU. Hash links, token/probe counts,
paired speed statistics and non-promotion flags are recalculated. The
archive does NOT authenticate reference provenance, hardware activity,
source-model identity, or a performance speedup.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import statistics
import zipfile

SCHEMA = "PHISTREAM_EXTERNAL_SMOLLM2_REPLICATION_AUDIT_V1"
MAX_ARCHIVE = 20 * 1024 * 1024
MAX_FILES = 100
MAX_UNCOMPRESSED = 20 * 1024 * 1024
EXPECTED_FILES = {
    "config.json": "8eb740e8bbe4cff95ea7b4588d17a2432deb16e8075bc5828ff7ba9be94d982a",
    "model.safetensors": "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c",
}
SOURCE = "HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
FROZEN_CASES = {
    "short_1": (1,),
    "mixed_7": (9, 28, 7, 139, 302, 900, 4096),
    "repeat_12": (42, 42, 42, 42, 42, 42, 13, 13, 13, 13, 1, 1),
    "spread_24": (495, 871, 1604, 2361, 3141, 4004, 5127, 6203,
                  7129, 8261, 9347, 10589, 11827, 13219, 14641,
                  15883, 17257, 18631, 19937, 21247, 22621, 23917,
                  25127, 26393),
}
STEP_COUNT = 32


def require(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError(why)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_archive(z: zipfile.ZipFile) -> None:
    items = z.infolist()
    require(len(items) <= MAX_FILES, "Too many ZIP entries")
    seen = set()
    total = 0
    for item in items:
        p = PurePosixPath(item.filename)
        require(item.filename not in seen, "Duplicate ZIP entry")
        seen.add(item.filename)
        require(not p.is_absolute() and ".." not in p.parts and
                not item.filename.startswith(("/", "\\")) and
                "\\" not in item.filename, "Unsafe ZIP path")
        require((item.external_attr >> 16) & 0o170000 != 0o120000,
                "ZIP symlink refused")
        total += item.file_size
        require(total <= MAX_UNCOMPRESSED, "ZIP contents exceed bound")
    require("matrix/manifest.json" in seen and
            "matrix/matrix-review.json" in seen, "Missing frozen matrix")


def read_json(z: zipfile.ZipFile, name: str) -> dict:
    info = z.getinfo(name)
    require(info.file_size <= 3 * 1024 * 1024, "Unbounded JSON receipt")
    data = json.loads(z.read(info))
    require(type(data) is dict, "Receipt must be an object: " + name)
    return data


def check_fingerprint(obj: dict) -> None:
    require(obj.get("checkpoint_fingerprint") ==
            {"algorithm": "sha256", "files": EXPECTED_FILES},
            "Checkpoint fingerprint mismatch")


def check_pair(pair: dict, nprobes: int) -> tuple[float, float]:
    methods = pair["methods"]
    require(set(methods) == {"SINGLE", "BATCHED"},
            "Paired trial missing method")
    for key in ("SINGLE", "BATCHED"):
        o = methods[key]
        require(o["status"] == "PARITY_MATCH_USER_REFERENCE_UNAUTHENTICATED" and
                o["greedy_matched"] == o["greedy_total"] == STEP_COUNT and
                o["logit_matched"] == o["logit_total"] == nprobes and
                len(o["comparisons"]) == STEP_COUNT and
                all(c.get("token_match") is True and
                    c.get("probe_match") is True for c in o["comparisons"]),
                "Per-trial parity incomplete or mismatch")
        require(type(o["decoder_seconds"]) in (int, float) and
                type(o["time_to_first_logits_seconds"]) in (int, float) and
                0 < o["decoder_seconds"] < 3600 and
                0 < o["time_to_first_logits_seconds"] < 3600,
                "Invalid timing values")
    a, b = methods["BATCHED"], methods["SINGLE"]
    return ((b["decoder_seconds"] - a["decoder_seconds"]) / b["decoder_seconds"],
            (b["time_to_first_logits_seconds"] -
             a["time_to_first_logits_seconds"]) / b["time_to_first_logits_seconds"])


def approximately(a: float, b: float) -> bool:
    return type(b) in (int, float) and abs(a - b) <= 0.000002


def audit_bytes(archive: bytes, expected_sha256: str, contended_cases=()) -> dict:
    require(type(expected_sha256) is str and
            re.fullmatch("[0-9a-f]{64}", expected_sha256) is not None,
            "Full expected SHA-256 required (not a truncated prefix)")
    require(0 < len(archive) <= MAX_ARCHIVE, "Archive exceeds size bound")
    require(sha(archive) == expected_sha256, "Archive SHA-256 mismatch")
    require(set(contended_cases) <= set(FROZEN_CASES),
            "Unknown externally declared contended case")
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        validate_archive(z)
        pre_names = [n for n in z.namelist() if
                     n.startswith("smollm2-cuda-preflight-") and n.endswith(".json")]
        require(len(pre_names) == 1, "Expected one pinned preflight")
        pre = read_json(z, pre_names[0])
        check_fingerprint(pre)
        source = pre["user_asserted_source"]
        require(source.get("source") == SOURCE and
                source.get("revision") == REVISION and
                source.get("license_id") == "Apache-2.0" and
                source.get("verified_by_phistream") is False,
                "Source declaration differs from frozen 5070 study")
        require(pre.get("status") == "PREFLIGHT_REVIEW_REQUIRED" and
                pre.get("model_execution") is False and
                pre.get("double_buffer_enabled") is False and
                pre.get("scheduler_policy_applied") is False and
                pre["checkpoint"]["format"] == "SAFETENSORS" and
                pre["checkpoint"]["model_type"] == "llama" and
                pre["memory_preflight"]["status"] == "REVIEW_REQUIRED" and
                pre["memory_preflight"]["capacity_policy_bytes"] ==
                1600 * 1024 * 1024 and
                pre["memory_preflight"]["reserved_bytes"] ==
                512 * 1024 * 1024,
                "Unsafe or inconsistent preflight metadata")

        manifest = read_json(z, "matrix/manifest.json")
        review = read_json(z, "matrix/matrix-review.json")
        check_fingerprint(manifest)
        check_fingerprint(review)
        require(review["manifest_sha256"] == sha(z.read("matrix/manifest.json")) and
                review.get("status") == "MATCHED_LIMITED_FOUR_CASES" and
                review.get("passing_cases") == 4 and
                review.get("required_cases") == 4 and
                review.get("production_readiness") == "NOT_ESTABLISHED" and
                review.get("all_reference_origins_authenticated") is False,
                "Four-case matrix review mismatch")
        cases = {c["id"]: tuple(c["prompt_token_ids"])
                 for c in manifest["cases"]}
        require(cases == FROZEN_CASES and
                all(c["expected_generated_steps"] == STEP_COUNT
                    for c in manifest["cases"]),
                "Frozen prompt cases or steps changed")
        result_rows = {c["id"]: c for c in review["case_results"]}
        require(set(result_rows) == set(FROZEN_CASES),
                "Matrix review has missing/extra cases")

        reference_totals = []
        unique_tokens = set()
        prefixes = set()
        for case in FROZEN_CASES:
            refpath = f"matrix/reference_{case}.json"
            parpath = f"matrix/parity_{case}.json"
            ref = read_json(z, refpath)
            parity = read_json(z, parpath)
            check_fingerprint(ref)
            check_fingerprint(parity)
            expected = ref["expected_generated_token_ids"]
            probes = ref["logit_probes"]
            probe_count = sum(len(row) for row in probes)
            require(tuple(ref["prompt_token_ids"]) == FROZEN_CASES[case] and
                    len(expected) == len(probes) == STEP_COUNT and
                    all(type(x) is int for x in expected),
                    "Reference does not match frozen case: " + case)
            unique_tokens.update(expected)
            prefixes.add(tuple(expected[:5]))
            row = result_rows[case]
            ex = parity["execution"]
            require(row["trace_sha256"] == sha(z.read(refpath)) ==
                    parity["reference_trace"]["trace_sha256"] and
                    row["tokens_matched"] == ex["exact_greedy_tokens_matched"] ==
                    STEP_COUNT and
                    row["logit_probes_matched"] == ex["logit_probes_matched"] ==
                    probe_count and
                    row["logit_probes_total"] == ex["total_logit_probes"] ==
                    probe_count and
                    parity["status"] ==
                    "PARITY_MATCH_USER_REFERENCE_UNAUTHENTICATED" and
                    len(ex["comparisons"]) == STEP_COUNT and
                    all(c.get("token_match") is True and
                        c.get("probe_match") is True for c in ex["comparisons"]),
                    "External reference and baseline parity mismatch: " + case)
            reference_totals.append(probe_count)
        require(review["unique_reference_greedy_ids_across_cases"] ==
                len(unique_tokens) and
                review["distinct_first_five_greedy_sequences"] ==
                len(prefixes), "Reference diversity count mismatch")

        paired = {}
        for case in ("mixed_7", "repeat_12", "spread_24"):
            base = "matrix/batched-prefill/"
            run = read_json(z, base + f"paired-ab-run_{case}.json")
            candidate = read_json(z, base + f"candidate_{case}.json")
            check_fingerprint(run)
            check_fingerprint(candidate)
            require(run["case"] == case and
                    run["status"] == "MATCHED_CANDIDATE_FOR_FURTHER_REVIEW" and
                    run["model_execution"] is True and
                    run["scheduler_policy_changed"] is False and
                    run["production_promoted"] is False and
                    run["speedup_proven"] is False and
                    run["source_reference_authenticated"] is False and
                    candidate["result"]["status"] ==
                    "MATCH_USER_REFERENCE_UNAUTHENTICATED" and
                    run["manifest_sha256"] == sha(z.read("matrix/manifest.json")) and
                    run["matrix_review_sha256"] ==
                    sha(z.read("matrix/matrix-review.json")) and
                    run["reference_sha256"] ==
                    sha(z.read(f"matrix/reference_{case}.json")) and
                    run["baseline_parity_receipt_sha256"] ==
                    sha(z.read(f"matrix/parity_{case}.json")) and
                    run["previous_candidate_receipt_sha256"] ==
                    sha(z.read(base + f"candidate_{case}.json")),
                    "Candidate/paired SHA chain or governance flags mismatch: " + case)
            d = run["design"]
            require((d["pairs"], d["seed"], d["warmups_per_method"]) ==
                    (4, 369, 1), "Paired experiment design changed")
            expected_probes = result_rows[case]["logit_probes_total"]
            for warm in run["warmup_observations"]:
                require(warm["greedy_matched"] == warm["greedy_total"] ==
                        STEP_COUNT and
                        warm["logit_matched"] == warm["logit_total"] ==
                        expected_probes and
                        all(c.get("token_match") is True and
                            c.get("probe_match") is True
                            for c in warm["comparisons"]),
                        "Warmup parity mismatch: " + case)
            require(len(run["warmup_observations"]) == 2 and
                    Counter(w["method"] for w in run["warmup_observations"]) ==
                    Counter({"SINGLE": 1, "BATCHED": 1}) and
                    len(run["paired_observations"]) == 4,
                    "Missing warmups or pairs: " + case)
            order_count = Counter()
            gains, first = [], []
            for i, pair in enumerate(run["paired_observations"]):
                require(pair["pair"] == i + 1 and
                        pair["order"] == d["orders"][i],
                        "Pair ordering changed")
                order_count[tuple(pair["order"])] += 1
                g, t = check_pair(pair, expected_probes)
                gains.append(g)
                first.append(t)
            require(order_count == Counter({
                ("BATCHED", "SINGLE"): 2, ("SINGLE", "BATCHED"): 2}),
                "Not balanced A/B ordering")
            summary = run["summary"]
            require(approximately(statistics.median(gains),
                                  summary["median_paired_relative_gain"]) and
                    approximately(statistics.median(first),
                                  summary["median_relative_ttft_gain"]) and
                    summary["candidate_faster_pair_count"] ==
                    sum(g > 0 for g in gains) and
                    summary["production_scheduler_unchanged"] is True and
                    summary["statistical_significance_not_established"] is True,
                    "Paired statistics or non-promotion flags mismatch")
            paired[case] = {
                "pairs": 4, "greedy_token_parity_per_trial": "32/32",
                "sampled_logit_probes_per_trial": expected_probes,
                "faster_pairs": sum(g > 0 for g in gains),
                "median_paired_relative_gain": round(statistics.median(gains), 6),
                "median_ttft_gain": round(statistics.median(first), 6),
                "timing_interference": ("EXTERNALLY_REPORTED_CONTENDED"
                    if case in contended_cases else "NOT_INDEPENDENTLY_VERIFIED_CLEAN"),
            }
    return {
        "schema": SCHEMA, "status": "LIMITED_EXTERNAL_RECEIPT_CONSISTENCY_PASS",
        "archive_sha256": expected_sha256,
        "checkpoint_fingerprint_claim_matches_5070": True,
        "checkpoint_files_in_archive": False,
        "model_executed_by_auditor": False,
        "reference_runtime_independence_authenticated": False,
        "gpu_execution_independently_witnessed": False,
        "matched_greedy_tokens": 128,
        "matched_sampled_logits": sum(reference_totals),
        "distinct_generated_ids": len(unique_tokens),
        "distinct_five_token_prefixes": len(prefixes),
        "paired": paired, "production_promoted": False,
        "speedup_certified": False,
        "cross_gpu_performance_comparison_certified": False,
        "limitations": [
            "Receipts are user-supplied. No GPU, model weights or network queried.",
            "Matching declared checkpoint hashes do not authenticate publisher or actual model bytes.",
            "Sparse logit samples are not full-logit parity.",
            "Timing and GPU contention are not independently verified.",
            "An externally reported late model reload contaminated spread_24 timing.",
            "Cross-machine speed claims require matched, controlled independent repetitions.",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", required=True, type=Path)
    ap.add_argument("--expected-sha256", required=True)
    ap.add_argument("--contended-case", action="append", default=[])
    ap.add_argument("--output", type=Path, help="Exclusive new JSON receipt")
    args = ap.parse_args()
    require(args.archive.is_file() and not args.archive.is_symlink(),
            "Normal existing archive required")
    require(args.archive.stat().st_size <= MAX_ARCHIVE, "Archive too large")
    result = audit_bytes(args.archive.read_bytes(), args.expected_sha256,
                         args.contended_case)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        require(args.output.suffix.lower() == ".json" and
                not args.output.exists() and not args.output.is_symlink(),
                "New .json path required; do not overwrite")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as fp:
            fp.write(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
