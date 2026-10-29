"""Day-two PhiStream preregistration: deterministic, offline negative controls.

No CUDA, model execution, network or service pausing. This suite freezes
the seed, case order and the A/B crossover schedule before measurements.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.smollm2_day2_prereg import (
    AB_CASES, DAY1_SEED, FROZEN_PROMPTS, MASTER_SEED, PAIR_COUNT,
    balanced_orders, canonical_bytes, case_seed, manifest,
)
from scripts.paired_real_prefill_ab import schedule as canonical_schedule


class Day2PreregistrationTests(unittest.TestCase):
    def test_master_seed_differs_and_case_seeds_are_frozen(self):
        self.assertEqual(MASTER_SEED, 4277796034)
        self.assertNotEqual(MASTER_SEED, DAY1_SEED)
        self.assertLessEqual(MASTER_SEED, 2 ** 32 - 1)
        self.assertEqual(
            {case: case_seed(case) for case in AB_CASES},
            {"mixed_7": 1389626234,
             "repeat_12": 757448,
             "spread_24": 3796163371}
        )
        with self.assertRaises(ValueError):
            case_seed("short_1")

    def test_eight_balanced_pairs_match_original_runner_schedule(self):
        for case in AB_CASES:
            with self.subTest(case=case):
                observed = balanced_orders(case)
                self.assertEqual(len(observed), PAIR_COUNT)
                self.assertEqual(
                    Counter(tuple(pair) for pair in observed),
                    Counter({("SINGLE", "BATCHED"): 4,
                             ("BATCHED", "SINGLE"): 4})
                )
                self.assertEqual(
                    observed,
                    [list(p) for p in canonical_schedule(PAIR_COUNT, case_seed(case))]
                )

    def test_all_frozen_prompts_and_cases_have_32_steps(self):
        from scripts.real_weight_matrix import frozen_cases
        canonical = {row["id"]: row["prompt_token_ids"] for row in frozen_cases()}
        self.assertEqual(canonical, FROZEN_PROMPTS)
        plan = manifest()
        self.assertEqual(plan["design"]["greedy_steps_per_case"], 32)
        self.assertEqual(plan["design"]["cases_order"], list(FROZEN_PROMPTS))
        self.assertEqual(plan["design"]["paired_cases"], list(AB_CASES))
        self.assertEqual(plan["design"]["warmups_per_method"], 1)

    def test_manifest_canonical_stable_and_exclusive_no_execution(self):
        one = canonical_bytes()
        two = canonical_bytes()
        self.assertEqual(one, two)
        self.assertEqual(one, (json.dumps(manifest(), sort_keys=True,
                                          indent=2, ensure_ascii=True) + "\n").encode())
        self.assertEqual(len(hashlib.sha256(one).hexdigest()), 64)
        self.assertTrue(manifest()["authorization"]["this_manifest_authorizes_execution"] is False)
        self.assertIn("NO_GPU_EXECUTION", manifest()["status"])
        self.assertTrue(manifest()["memory_policy"]["policy_is_not_device_fit_attestation"])

    def test_monitoring_unknown_is_not_clean_and_contended_is_preserved(self):
        plan = manifest()["contention_plan"]
        self.assertEqual(plan["poll_interval_seconds"], 1)
        self.assertEqual(plan["sample_source_unavailable_or_gapped"],
                         "UNKNOWN_NOT_CLEAN")
        self.assertIn("CONTENDED", plan["unexpected_process_overlap"])
        self.assertIn("OBSERVED_CLEAR_NOT_EXCLUSIVITY_PROOF",
                      plan["no_other_process_observed"])
        self.assertTrue(plan["monitor_or_trial_timestamp_binding_must_be_implemented_and_tested_before_execution"])
        self.assertTrue(plan["no_automatic_unload_or_kill"])

    def test_no_forked_scheduler_or_performance_certification(self):
        study = manifest()
        self.assertTrue(study["analysis"]["single_default_remains_unchanged"])
        self.assertTrue(study["analysis"]["day_one_archive_and_contended_spread_24_preserved"])
        self.assertTrue(study["analysis"]["no_statistical_significance_or_production_speedup_presumed"])
        self.assertFalse(study["authorization"]["this_manifest_authorizes_execution"])

    def test_doc_warns_day_one_auditor_is_not_eight_pair_compatible(self):
        body = (Path(__file__).resolve().parents[1] /
                "docs/SMOLLM2_5060_DAY2_PREREGISTRATION.md").read_text()
        self.assertIn("It **cannot** validate the new study unchanged", body)
        self.assertIn("UNKNOWN", body)
        self.assertIn("OBSERVED_CLEAR", body)
        self.assertIn("CONTENDED", body)
        self.assertIn("not a runnable GPU", body)


if __name__ == "__main__":
    unittest.main()
