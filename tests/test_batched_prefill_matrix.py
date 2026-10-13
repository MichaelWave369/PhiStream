"""PR24 synthetic-only tests for the SHA-bound prefill gate.

No pretrained checkpoints, CUDA, or automatic real-model execution.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from phistream.reference import validate_reference_trace
from phistream.safeweights import LocalCheckpoint
from scripts.batched_prefill_matrix import (
    CASES_WITH_BATCH, evaluate_candidate, process,
)
from test_llama_execution import local_llama
from test_reference_gate import make_trace


class CandidateRealGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg, self.weights = local_llama(self.root)
        self.ckpt = LocalCheckpoint(self.root)
        self.trace = make_trace(self.root, self.cfg, self.weights)
        self.validated = validate_reference_trace(self.trace, self.ckpt)
        self.kw = dict(
            approve_local_execution=True, device="cpu",
            capacity_mib=16, reserve_mib=1, max_tensor_mib=1, head_rows=7,
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_candidate_32_step_reference_and_token_forced_continuation(self):
        result = evaluate_candidate(self.ckpt, self.validated, **self.kw)
        self.assertEqual(result["status"], "MATCH_USER_REFERENCE_UNAUTHENTICATED")
        self.assertEqual(result["greedy_ids_matched"], 32)
        self.assertEqual(result["sampled_logit_probes_matched"],
                         result["sampled_logit_probes_total"])
        self.assertEqual(len(result["comparisons"]), 32)
        self.assertEqual(len(result["execution"]["continuation_step_seconds"]), 31)
        self.assertGreater(result["execution"]["first_logits_weight_staging_bytes"], 0)
        self.assertEqual(result["execution"]["first_logits_weight_load_operations"] <
                         result["execution"]["full_run_weight_load_operations"], True)
        self.assertFalse(result["production_ready"])
        self.assertFalse(result["speedup_measured"])
        json.dumps(result, allow_nan=False)

    def test_modified_reference_ids_or_probes_do_not_pass(self):
        bad_id = dict(self.validated)
        bad_id["expected_ids"] = list(self.validated["expected_ids"])
        bad_id["expected_ids"][0] = (
            bad_id["expected_ids"][0] + 1) % self.cfg["vocab_size"]
        self.assertEqual(evaluate_candidate(self.ckpt, bad_id, **self.kw)["status"],
                         "PARITY_MISMATCH")

        bad_probe = dict(self.validated)
        bad_probe["probes"] = [dict(x) for x in self.validated["probes"]]
        probe_id = next(iter(bad_probe["probes"][3]))
        bad_probe["probes"][3][probe_id] += 2.0
        result = evaluate_candidate(self.ckpt, bad_probe, **self.kw)
        self.assertEqual(result["status"], "PARITY_MISMATCH")
        self.assertLess(result["sampled_logit_probes_matched"],
                        result["sampled_logit_probes_total"])

    def test_missing_approval_fails_before_model_is_initialized(self):
        with patch("phistream.prefill.StreamingLlama",
                   side_effect=AssertionError("no decoder allowed")):
            with self.assertRaises(PermissionError):
                evaluate_candidate(self.ckpt, self.validated, device="cpu",
                                   capacity_mib=16, reserve_mib=1,
                                   max_tensor_mib=1, head_rows=7)

    def test_short_trace_rejected(self):
        shortened = dict(self.validated)
        shortened["expected_ids"] = self.validated["expected_ids"][:31]
        with self.assertRaisesRegex(ValueError, "32"):
            evaluate_candidate(self.ckpt, shortened, **self.kw)

    def test_default_plan_writes_offline_receipt_without_decoder(self):
        fake = {
            "checkpoint_fingerprint": {"algorithm": "sha256",
                                       "files": {"config.json": "a" * 64}},
            "source_declared": {}, "manifest_sha256": "a" * 64,
            "review_sha256": "b" * 64,
            "matrix_status": "MATCHED_LIMITED_FOUR_CASES",
            "origins": {
                "mixed_7": {
                    "reference_path": self.root / "unused.json",
                    "reference_sha256": "c" * 64,
                    "baseline_parity_sha256": "d" * 64,
                },
            },
        }
        target = self.root / "receipt.json"
        with patch("scripts.batched_prefill_matrix.verified_baseline", return_value=fake):
            with patch("scripts.batched_prefill_matrix.evaluate_candidate",
                       side_effect=AssertionError("no candidate execution")):
                receipt = process(self.root, "unused", self.root, "mixed_7", target)
        self.assertEqual(receipt["status"], "PLAN_SHA_BOUND_NOT_EXECUTED")
        self.assertFalse(receipt["model_execution"])
        self.assertFalse(receipt["candidate_selected"])
        self.assertEqual(json.loads(target.read_text())["schema"],
                         "PHISTREAM_BATCHED_PREFILL_REAL_GATE_V1")
        with patch("scripts.batched_prefill_matrix.verified_baseline", return_value=fake):
            with self.assertRaises(ValueError):
                process(self.root, "unused", self.root, "mixed_7", target)

    def test_run_always_requires_clearly_separate_cuda_approval(self):
        with patch("scripts.batched_prefill_matrix.verified_baseline",
                   side_effect=AssertionError("should deny before hashing")):
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

    def test_one_token_baseline_never_enters_batch_candidate(self):
        self.assertNotIn("short_1", CASES_WITH_BATCH)
        with self.assertRaises(ValueError):
            process(self.root, "unused", self.root, "short_1",
                    self.root / "no.json")


if __name__ == "__main__":
    unittest.main()
