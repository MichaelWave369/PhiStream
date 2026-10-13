"""PR25 bounded CPU-only A/B gate tests; no real models, CUDA, or downloads."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from phistream.reference import validate_reference_trace
from phistream.safeweights import LocalCheckpoint
from scripts.paired_real_prefill_ab import (
    MIN_PAIRS, existing_candidate, one_method, process, schedule, summarize_pairs,
)
from test_llama_execution import local_llama
from test_reference_gate import make_trace


def fake_trial(method, seconds, *, ttft=0.5, bytes_=10000):
    return {
        "status": "PARITY_MATCH_USER_REFERENCE_UNAUTHENTICATED",
        "method": method, "decoder_seconds": seconds,
        "time_to_first_logits_seconds": ttft,
        "weight_staging_bytes": bytes_,
    }


class PairedABTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config, self.weights = local_llama(self.root)
        self.ckpt = LocalCheckpoint(self.root)
        self.trace = make_trace(self.root, self.config, self.weights)
        self.validated = validate_reference_trace(self.trace, self.ckpt)
        self.kw = dict(approve_local_execution=True, device="cpu",
                       capacity_mib=16, reserve_mib=1,
                       max_tensor_mib=1, head_rows=7)

    def tearDown(self):
        self.temp.cleanup()

    def test_seeded_random_pair_schedule_is_exactly_balanced(self):
        for pairs in (4, 8, 20):
            order = schedule(pairs, 369)
            self.assertEqual(order, schedule(pairs, 369))
            self.assertEqual(order.count(("SINGLE", "BATCHED")), pairs // 2)
            self.assertEqual(order.count(("BATCHED", "SINGLE")), pairs // 2)
            self.assertTrue(all(set(x) == {"SINGLE", "BATCHED"} for x in order))
        for invalid in (0, 1, 3, 5, 21, 22, True, 4.0):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                schedule(invalid, 369)
        for seed in (-1, 2**32, False, 1.5):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                schedule(4, seed)

    def test_real_method_parity_and_continuation_on_synthetic_cpu(self):
        single = one_method(self.ckpt, self.validated, "SINGLE", **self.kw)
        batch = one_method(self.ckpt, self.validated, "BATCHED", **self.kw)
        for result in (single, batch):
            self.assertEqual(result["status"], "PARITY_MATCH_USER_REFERENCE_UNAUTHENTICATED")
            self.assertEqual(result["greedy_matched"], 32)
            self.assertEqual(result["greedy_total"], 32)
            self.assertEqual(result["logit_matched"], result["logit_total"])
            self.assertEqual(len(result["comparisons"]), 32)
            self.assertEqual(len(result["continuation_step_seconds"]), 31)
            self.assertGreater(result["decoder_seconds"], 0)
            json.dumps(result, allow_nan=False)
        self.assertLess(batch["weight_staging_bytes"], single["weight_staging_bytes"])
        self.assertLess(batch["weight_load_operations"], single["weight_load_operations"])
        self.assertTrue(batch["method_specific_policy_recheck_in_timed_region"])
        self.assertFalse(single["method_specific_policy_recheck_in_timed_region"])

    def test_each_method_rejects_changed_reference_token_or_logit(self):
        for method in ("SINGLE", "BATCHED"):
            wrong_id = dict(self.validated)
            wrong_id["expected_ids"] = list(self.validated["expected_ids"])
            wrong_id["expected_ids"][0] = (wrong_id["expected_ids"][0] + 1) % 32
            a = one_method(self.ckpt, wrong_id, method, **self.kw)
            self.assertEqual(a["status"], "FAIL_PARITY")
            self.assertLess(a["greedy_matched"], 32)
            wrong_probe = dict(self.validated)
            wrong_probe["probes"] = [dict(item) for item in self.validated["probes"]]
            first = next(iter(wrong_probe["probes"][0]))
            wrong_probe["probes"][0][first] += 2.0
            b = one_method(self.ckpt, wrong_probe, method, **self.kw)
            self.assertEqual(b["status"], "FAIL_PARITY")
            self.assertLess(b["logit_matched"], b["logit_total"])

    def test_no_unapproved_model_execution(self):
        for method in ("SINGLE", "BATCHED"):
            with self.assertRaises(PermissionError):
                one_method(self.ckpt, self.validated, method,
                           approve_local_execution=False,
                           device="cpu", capacity_mib=16, reserve_mib=1,
                           max_tensor_mib=1, head_rows=7)
        with self.assertRaises(PermissionError):
            process(self.root, "unused", self.root, "mixed_7",
                    self.root / "no.json", mode="run")
        with self.assertRaises(PermissionError):
            process(self.root, "unused", self.root, "mixed_7",
                    self.root / "no.json", mode="run",
                    approve_local_execution=True, device="cpu")
        with self.assertRaises(ValueError):
            process(self.root, "unused", self.root, "mixed_7",
                    self.root / "no.json", mode="plan",
                    approve_local_execution=True)

    def test_plan_writes_exclusive_nonexecution_receipt(self):
        dummy = {"schema": "PHISTREAM_PAIRED_REAL_PREFILL_AB_V1",
                 "status": "SHA_BOUND_AB_PLAN_NOT_EXECUTED",
                 "case": "mixed_7", "model_execution": False,
                 "design": {"orders": [["SINGLE", "BATCHED"]] * 4}}
        output = self.root / "out" / "plan.json"
        with patch("scripts.paired_real_prefill_ab.preflight",
                   return_value=(dummy, self.root / "unused.json")):
            with patch("scripts.paired_real_prefill_ab.one_method",
                       side_effect=AssertionError("plan cannot run models")):
                r = process(self.root, "unused", self.root, "mixed_7", output)
        self.assertEqual(r["status"], "SHA_BOUND_AB_PLAN_NOT_EXECUTED")
        self.assertFalse(json.loads(output.read_text())["model_execution"])
        with patch("scripts.paired_real_prefill_ab.preflight",
                   return_value=(dummy, self.root / "unused.json")):
            with self.assertRaises(ValueError):
                process(self.root, "unused", self.root, "mixed_7", output)

    def test_balanced_paired_summary_is_descriptive_only(self):
        pairs = [
            {"SINGLE": fake_trial("SINGLE", 10), "BATCHED": fake_trial("BATCHED", 8)},
            {"SINGLE": fake_trial("SINGLE", 10), "BATCHED": fake_trial("BATCHED", 9)},
            {"SINGLE": fake_trial("SINGLE", 10), "BATCHED": fake_trial("BATCHED", 9)},
            {"SINGLE": fake_trial("SINGLE", 10), "BATCHED": fake_trial("BATCHED", 11)},
        ]
        result = summarize_pairs(pairs)
        self.assertEqual(result["candidate_faster_pair_count"], 3)
        self.assertEqual(result["candidate_faster_fraction"], 0.75)
        self.assertTrue(result["thresholds_met_for_human_review"])
        self.assertTrue(result["statistical_significance_not_established"])
        self.assertTrue(result["production_scheduler_unchanged"])
        pairs[1]["BATCHED"]["decoder_seconds"] = 11
        weak = summarize_pairs(pairs)
        self.assertFalse(weak["thresholds_met_for_human_review"])
        pairs[0]["BATCHED"]["status"] = "FAIL_PARITY"
        with self.assertRaises(ValueError):
            summarize_pairs(pairs)
        with self.assertRaises(ValueError):
            summarize_pairs([])

    def test_previous_candidate_sha_and_status_must_match(self):
        directory = self.root / "batched-prefill"
        directory.mkdir()
        fake = {
            "schema": "PHISTREAM_BATCHED_PREFILL_REAL_GATE_V1",
            "status": "MATCH_USER_REFERENCE_UNAUTHENTICATED",
            "case": "mixed_7", "model_execution": True,
            "candidate_selected": True, "speedup_measured": False,
            "checkpoint_fingerprint": {"files": {"model": "x"}},
            "manifest_sha256": "a" * 64,
            "matrix_review_sha256": "b" * 64,
            "reference_sha256": "c" * 64,
            "baseline_parity_receipt_sha256": "d" * 64,
            "result": {
                "status": "MATCH_USER_REFERENCE_UNAUTHENTICATED",
                "greedy_ids_total": 32, "greedy_ids_matched": 32,
                "sampled_logit_probes_total": 285,
                "sampled_logit_probes_matched": 285,
                "production_ready": False, "speedup_measured": False,
            },
        }
        path = directory / "candidate_mixed_7.json"
        baseline = {
            "checkpoint_fingerprint": fake["checkpoint_fingerprint"],
            "manifest_sha256": fake["manifest_sha256"],
            "review_sha256": fake["matrix_review_sha256"],
            "origins": {"mixed_7": {"reference_sha256": fake["reference_sha256"],
                                   "baseline_parity_sha256": fake["baseline_parity_receipt_sha256"]}},
        }
        path.write_text(json.dumps(fake))
        trusted = existing_candidate(self.root, "mixed_7", baseline)
        self.assertEqual(trusted["historical_matched_probes"], 285)
        fake["result"]["sampled_logit_probes_matched"] = 284
        path.write_text(json.dumps(fake))
        with self.assertRaises(ValueError):
            existing_candidate(self.root, "mixed_7", baseline)


if __name__ == "__main__":
    unittest.main()
