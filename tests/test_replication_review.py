"""PR28 offline-only synthetic regression tests; no model/CUDA/network execution."""
from copy import deepcopy
import unittest

from scripts.paired_real_prefill_ab import SCHEMA as AB_SCHEMA, schedule, summarize_pairs
from scripts.replication_review import (
    CASE, REPLICATION_PAIRS, REPLICATION_SEED,
    validate_original_census, validate_replication,
)
from scripts.three_case_ab_audit import SCHEMA as CENSUS_SCHEMA


def trial(method, seconds, probes=288):
    return {
        "method": method,
        "status": "PARITY_MATCH_USER_REFERENCE_UNAUTHENTICATED",
        "greedy_total": 32, "greedy_matched": 32,
        "logit_total": probes, "logit_matched": probes,
        "comparisons": [
            {"step": i, "token_match": True, "probe_match": True} for i in range(32)],
        "method_specific_policy_recheck_in_timed_region": method == "BATCHED",
        "memory_policy": {
            "status": "REVIEW_REQUIRED", "context_tokens": 55,
            "capacity_source": "USER_POLICY_CAPPED_BY_CUDA_FREE",
        },
        "decoder_seconds": seconds,
        "time_to_first_logits_seconds": 1.0 if method == "SINGLE" else 0.1,
        "max_abs_sampled_logit_error": 0.00000001,
        "weight_loading_seconds_including_transfers": seconds * 0.6,
        "weight_staging_bytes": 1000000 if method == "SINGLE" else 400000,
        "weight_load_operations": 500 if method == "SINGLE" else 300,
        "peak_pytorch_cuda_allocated_bytes": 12000000,
        "continuation_step_seconds": [0.01] * 31,
    }


def fixture():
    design = {
        "pairs": 8, "seed": REPLICATION_SEED, "warmups_per_method": 1,
        "orders": [list(p) for p in schedule(8, REPLICATION_SEED)],
    }
    plan = {
        "schema": AB_SCHEMA,
        "status": "SHA_BOUND_AB_PLAN_NOT_EXECUTED",
        "generated_utc": "plan timestamp",
        "case": CASE,
        "model_execution": False,
        "production_promoted": False,
        "speedup_proven": False,
        "source_reference_authenticated": False,
        "scheduler_policy_changed": False,
        "checkpoint_fingerprint": {"sha": "a" * 64},
        "manifest_sha256": "b" * 64,
        "reference_sha256": "c" * 64,
        "design": design,
    }
    expected = deepcopy(plan)
    expected["generated_utc"] = "fresh preflight timestamp"
    run = deepcopy(plan)
    run.update(
        generated_utc="run timestamp",
        status="MATCHED_CANDIDATE_FOR_FURTHER_REVIEW",
        model_execution=True, candidate_selected=True,
        run_device="cuda",
        warmup_observations=[
            {"warmup": 1, **trial("SINGLE", 10)},
            {"warmup": 1, **trial("BATCHED", 6)},
        ],
    )
    run["paired_observations"] = []
    for i, order in enumerate(design["orders"], start=1):
        methods = {"SINGLE": trial("SINGLE", 10),
                   "BATCHED": trial("BATCHED", 6)}
        run["paired_observations"].append({
            "pair": i, "order": order, "complete": True, "methods": methods})
    run["summary"] = summarize_pairs([p["methods"] for p in run["paired_observations"]])
    return plan, run, expected


class ReplicationValidationTests(unittest.TestCase):
    def test_eight_pair_valid_recomputation(self):
        p, r, e = fixture()
        result = validate_replication(p, r, e, probes=288, prompt_length=24)
        self.assertEqual(result["pairs"], REPLICATION_PAIRS)
        self.assertEqual(result["candidate_faster_pair_count"], 8)
        self.assertEqual(result["model_executions_recorded"], 18)
        self.assertEqual(result["greedy_id_matches"], 576)
        self.assertEqual(result["sampled_logit_matches"], 5184)
        self.assertTrue(result["review_thresholds_met"])

    def test_reject_modified_plan_or_run_and_missing_parity(self):
        mutators = (
            lambda p, r: p.update(manifest_sha256="x" * 64),
            lambda p, r: r.update(checkpoint_fingerprint={"sha": "x" * 64}),
            lambda p, r: r["paired_observations"].pop(),
            lambda p, r: r["paired_observations"][0].update(order=["BAD"]),
            lambda p, r: r["paired_observations"][0]["methods"]["BATCHED"].update(greedy_matched=31),
            lambda p, r: r["warmup_observations"][0]["comparisons"][9].update(probe_match=False),
            lambda p, r: r["paired_observations"][5]["methods"]["SINGLE"]["memory_policy"].update(context_tokens=1),
            lambda p, r: r["summary"].update(median_paired_relative_gain=0.99),
            lambda p, r: r.update(production_promoted=True),
            lambda p, r: r.update(speedup_proven=True),
            lambda p, r: r.update(source_reference_authenticated=True),
            lambda p, r: r.update(status="MATCHED_BUT_PERFORMANCE_NOT_QUALIFIED"),
        )
        for change in mutators:
            p, r, e = fixture()
            change(p, r)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_replication(p, r, e, probes=288, prompt_length=24)

    def test_negative_replication_valid_but_not_promoted(self):
        p, r, e = fixture()
        for item in r["paired_observations"]:
            item["methods"]["BATCHED"]["decoder_seconds"] = 11
        r["summary"] = summarize_pairs([x["methods"] for x in r["paired_observations"]])
        r["status"] = "MATCHED_BUT_PERFORMANCE_NOT_QUALIFIED"
        result = validate_replication(p, r, e, probes=288, prompt_length=24)
        self.assertFalse(result["review_thresholds_met"])
        self.assertEqual(result["candidate_faster_pair_count"], 0)

    def test_original_census_hashes_and_totals_required(self):
        rows = [
            {"case": case, "pairs": 4, "measured_executions": 8,
             "warmup_executions": 2, "all_executions_with_complete_parity": 10,
             "greedy_id_matches": 320, "sampled_logit_matches": probes * 10,
             "candidate_faster_pair_count": 4, "review_thresholds_met": True}
            for case, probes in (("mixed_7", 285), ("repeat_12", 286), ("spread_24", 288))
        ]
        hashes = {x["case"]: {"plan_sha256": "a" * 64, "run_sha256": "b" * 64}
                  for x in rows}
        census = {
            "schema": CENSUS_SCHEMA,
            "status": "THREE_CASES_REVIEW_CANDIDATE_NOT_SPEEDUP_CERTIFIED",
            "model_execution": False, "production_promoted": False,
            "general_speedup_proven": False, "source_reference_authenticated": False,
            "statistical_significance_established": False,
            "case_count": 3, "cases": rows, "original_local_receipt_hashes": hashes,
            "total_pairs": 12, "total_measured_executions": 24,
            "total_warmup_executions": 6,
            "total_checked_model_executions_in_original_receipts": 30,
            "total_matching_greedy_ids_in_original_receipts": 960,
            "total_matching_sparse_logits_in_original_receipts": 8590,
            "total_candidate_faster_pairs": 12,
        }
        self.assertEqual(validate_original_census(census, rows, hashes)["total_pairs"], 12)
        tamper = deepcopy(census)
        tamper["total_matching_sparse_logits_in_original_receipts"] = 10000
        with self.assertRaises(ValueError):
            validate_original_census(tamper, rows, hashes)
        tamper = deepcopy(census)
        tamper["original_local_receipt_hashes"]["spread_24"]["run_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_original_census(tamper, rows, hashes)
        tamper = deepcopy(census)
        tamper["general_speedup_proven"] = True
        with self.assertRaises(ValueError):
            validate_original_census(tamper, rows, hashes)


if __name__ == "__main__":
    unittest.main()
