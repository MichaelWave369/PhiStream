"""PR26 synthetic-only tests of a strict receipt census, no real model or GPU."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.export_hf_reference import file_digest
from scripts.paired_real_prefill_ab import SCHEMA as PAIR_SCHEMA, summarize_pairs, schedule
from scripts.three_case_ab_audit import (
    SCHEMA, audit, validate_case, validate_trial, CASES,
)


def method_trial(method, seconds, probes=288, context=38):
    return {
        "method": method,
        "status": "PARITY_MATCH_USER_REFERENCE_UNAUTHENTICATED",
        "greedy_matched": 32, "greedy_total": 32,
        "logit_matched": probes, "logit_total": probes,
        "max_abs_sampled_logit_error": 0.000001,
        "comparisons": [
            {"step": i, "token_match": True, "probe_match": True}
            for i in range(32)
        ],
        "decoder_seconds": seconds,
        "time_to_first_logits_seconds": 1.0 if method == "SINGLE" else 0.2,
        "continuation_step_seconds": [0.01] * 31,
        "weight_staging_bytes": 1000000 if method == "SINGLE" else 500000,
        "weight_load_operations": 600 if method == "SINGLE" else 300,
        "weight_loading_seconds_including_transfers": seconds / 2,
        "peak_pytorch_cuda_allocated_bytes": 1000000,
        "memory_policy": {
            "status": "REVIEW_REQUIRED",
            "context_tokens": context,
            "capacity_source": "USER_POLICY_CAPPED_BY_CUDA_FREE",
        },
        "method_specific_policy_recheck_in_timed_region": method == "BATCHED",
    }


def make_records(case="mixed_7", probes=288, prompt_length=7, seed=369):
    design = {
        "pairs": 4, "warmups_per_method": 1, "seed": seed,
        "orders": [list(x) for x in schedule(4, seed)],
        "modes": ["SINGLE", "BATCHED"],
    }
    plan = {
        "schema": PAIR_SCHEMA, "generated_utc": "synthetic plan",
        "status": "SHA_BOUND_AB_PLAN_NOT_EXECUTED",
        "case": case, "device_requested": "cuda",
        "model_execution": False, "scheduler_policy_changed": False,
        "production_promoted": False, "speedup_proven": False,
        "source_reference_authenticated": False,
        "checkpoint_fingerprint": {"algorithm": "sha256", "files": {"config.json": "a" * 64}},
        "manifest_sha256": "b" * 64,
        "matrix_review_sha256": "c" * 64,
        "reference_sha256": "d" * 64,
        "previous_candidate_receipt_sha256": "e" * 64,
        "baseline_parity_receipt_sha256": "f" * 64,
        "design": design,
    }
    run = deepcopy(plan)
    run["generated_utc"] = "synthetic run"
    run.update(status="MATCHED_CANDIDATE_FOR_FURTHER_REVIEW",
               model_execution=True, run_device="cuda", candidate_selected=True)
    context = prompt_length + 31
    run["warmup_observations"] = [
        {"warmup": 1, **method_trial(method, 10 if method == "SINGLE" else 8,
                                      probes, context)}
        for method in ("SINGLE", "BATCHED")
    ]
    run["paired_observations"] = []
    for i, order in enumerate(design["orders"], 1):
        pair = {
            "SINGLE": method_trial("SINGLE", 10, probes, context),
            "BATCHED": method_trial("BATCHED", 8, probes, context),
        }
        run["paired_observations"].append(
            {"pair": i, "order": order, "methods": pair, "complete": True}
        )
    run["summary"] = summarize_pairs([r["methods"] for r in run["paired_observations"]])
    expected = deepcopy(plan)
    expected["generated_utc"] = "recomputed preflight"
    return plan, run, expected


class CensusPureTests(unittest.TestCase):
    def test_valid_four_pairs_and_two_warmups(self):
        plan, run, expected = make_records()
        result = validate_case(plan, run, expected, case="mixed_7",
                               expected_probes=288, prompt_length=7)
        self.assertEqual(result["pairs"], 4)
        self.assertEqual(result["all_executions_with_complete_parity"], 10)
        self.assertEqual(result["greedy_id_matches"], 320)
        self.assertEqual(result["sampled_logit_matches"], 2880)
        self.assertEqual(result["candidate_faster_pair_count"], 4)
        self.assertTrue(result["review_thresholds_met"])

    def test_step_level_forgery_and_probe_count_detected(self):
        for mutation in (
            lambda run: run["paired_observations"][0]["methods"]["SINGLE"]["comparisons"][7].update(token_match=False),
            lambda run: run["paired_observations"][0]["methods"]["BATCHED"].update(logit_matched=287),
            lambda run: run["paired_observations"][1]["methods"]["SINGLE"].update(greedy_total=True),
            lambda run: run["warmup_observations"][0]["comparisons"][2].update(probe_match=False),
            lambda run: run["paired_observations"][0]["methods"]["SINGLE"].update(
                peak_pytorch_cuda_allocated_bytes=None),
        ):
            plan, run, expected = make_records()
            mutation(run)
            with self.assertRaises(ValueError):
                validate_case(plan, run, expected, case="mixed_7",
                              expected_probes=288, prompt_length=7)

    def test_plan_run_order_and_aggregate_mutations_fail_closed(self):
        for mutation in (
            lambda plan, run: plan.update(manifest_sha256="0" * 64),
            lambda plan, run: run.update(reference_sha256="0" * 64),
            lambda plan, run: run.update(status="MATCHED_BUT_PERFORMANCE_NOT_QUALIFIED"),
            lambda plan, run: run["paired_observations"][0].update(order=["SINGLE", "BATCHED"]),
            lambda plan, run: run["paired_observations"].pop(),
            lambda plan, run: run["summary"].update(median_paired_relative_gain=0.9),
            lambda plan, run: run.update(production_promoted=True),
            lambda plan, run: run["paired_observations"][0]["methods"]["SINGLE"].update(
                method_specific_policy_recheck_in_timed_region=True),
        ):
            plan, run, expected = make_records()
            mutation(plan, run)
            # If seeded first pair ever becomes S->B, force an actual order change.
            if run["paired_observations"][0]["order"] == expected["design"]["orders"][0] and run["paired_observations"][0]["pair"] == 1:
                if run["paired_observations"][0]["order"] != ["SINGLE", "BATCHED"]:
                    pass
            with self.assertRaises(ValueError):
                validate_case(plan, run, expected, case="mixed_7",
                              expected_probes=288, prompt_length=7)

    def test_nonqualifying_but_correct_cannot_be_mislabeled(self):
        plan, run, expected = make_records()
        for entry in run["paired_observations"]:
            entry["methods"]["BATCHED"]["decoder_seconds"] = 10.5
        run["summary"] = summarize_pairs([x["methods"] for x in run["paired_observations"]])
        run["status"] = "MATCHED_BUT_PERFORMANCE_NOT_QUALIFIED"
        result = validate_case(plan, run, expected, case="mixed_7",
                               expected_probes=288, prompt_length=7)
        self.assertFalse(result["review_thresholds_met"])
        self.assertEqual(result["candidate_faster_pair_count"], 0)


class CensusFileTests(unittest.TestCase):
    def test_three_synthetic_receipts_are_assembled_without_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "batched-prefill"
            folder.mkdir()
            fake = {}
            for case, prompt_length in zip(CASES, (7, 12, 24)):
                trace = root / ("reference_" + case + ".json")
                trace.write_text(json.dumps({
                    "prompt_token_ids": list(range(prompt_length)),
                    "logit_probes": [{str(i): 0.0 for i in range(9)} for _ in range(32)],
                }))
                plan, run, expected = make_records(case, prompt_length=prompt_length)
                digest = file_digest(trace)
                for record in (plan, run, expected):
                    record["reference_sha256"] = digest
                (folder / ("paired-ab-plan_" + case + ".json")).write_text(json.dumps(plan))
                (folder / ("paired-ab-run_" + case + ".json")).write_text(json.dumps(run))
                fake[case] = (expected, trace)
            output = root / "new-census.json"
            with patch("scripts.three_case_ab_audit.preflight",
                       side_effect=lambda a, b, c, case, *rest: fake[case]):
                report = audit(root, "irrelevant", root, output)
            self.assertEqual(report["schema"], SCHEMA)
            self.assertEqual(report["case_count"], 3)
            self.assertEqual(report["total_pairs"], 12)
            self.assertEqual(report["total_checked_model_executions_in_original_receipts"], 30)
            self.assertEqual(report["total_matching_greedy_ids_in_original_receipts"], 960)
            self.assertEqual(report["total_matching_sparse_logits_in_original_receipts"], 8640)
            self.assertFalse(report["model_execution"])
            self.assertFalse(report["production_promoted"])
            self.assertFalse(report["general_speedup_proven"])
            self.assertTrue(output.is_file())
            with patch("scripts.three_case_ab_audit.preflight",
                       side_effect=lambda a, b, c, case, *rest: fake[case]):
                with self.assertRaises(ValueError):
                    audit(root, "irrelevant", root, output)


if __name__ == "__main__":
    unittest.main()
