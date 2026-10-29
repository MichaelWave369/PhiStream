"""PocketRotor E1 CPU synthetic, no pretrained checkpoints or CUDA in CI."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from phistream.safeweights import LocalCheckpoint
from phistream.pocketrotor import pocket_inventory
from scripts.pocketrotor_e1 import (
    METHODS, _read_plan, _write, measure_method, proposed_plan,
    run_study, schedule, summarize,
)

from test_llama_execution import local_llama
from test_reference_gate import make_trace


class PocketRotorE1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.config, cls.weights = local_llama(cls.root)
        cls.ckpt = LocalCheckpoint(cls.root)
        cls.trace_data = make_trace(cls.root, cls.config, cls.weights, steps=32)
        cls.trace = cls.root / "reference.json"
        cls.trace.write_text(json.dumps(cls.trace_data), encoding="utf-8")
        cls.cache_bytes = sum(
            x["bytes_fp32"] for x in pocket_inventory(cls.ckpt) if x["layer"] == 0
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def params(self):
        return {
            "device": "cpu", "cache_mib": self.cache_bytes / (1024 * 1024),
            "capacity_mib": 16, "reserve_mib": 1,
            "max_tensor_mib": 1, "head_rows": 7,
            "rounds": 3, "seed": 369,
        }

    def write_plan(self, path):
        plan, _ = proposed_plan(self.ckpt, self.trace, **self.params())
        self.assertEqual(plan["status"], "READY_FOR_REVIEW")
        _write(plan, path, create=True)
        return plan

    def test_balanced_schedule_reproducible_and_fail_closed(self):
        for rounds in (3, 6):
            orders = schedule(rounds, 369)
            self.assertEqual(orders, schedule(rounds, 369))
            for position in range(3):
                for method in METHODS:
                    self.assertEqual(sum(o[position] == method for o in orders), rounds // 3)
        for rounds in (1, 2, 4, 0, True):
            with self.assertRaises(ValueError):
                schedule(rounds, 369)
        for seed in (-1, 2**32, True):
            with self.assertRaises(ValueError):
                schedule(3, seed)

    def test_read_only_plan_no_model_execution_or_weight_fingerprint(self):
        with patch("scripts.pocketrotor_e1.StreamingLlama",
                   side_effect=AssertionError("plan cannot execute")):
            with patch("scripts.pocketrotor_e1.checkpoint_fingerprint",
                       side_effect=AssertionError("plan cannot hash entire model")):
                plan, trace = proposed_plan(self.ckpt, self.trace, **self.params())
        self.assertFalse(plan["model_execution"])
        self.assertEqual(plan["reference_coverage"]["continuation_steps"], 32)
        self.assertEqual(len(plan["schedule"]), 3)
        self.assertEqual(len(trace["expected_ids"]), 32)
        self.assertTrue(plan["nested_pinned_layer_ids"])
        self.assertFalse(plan["production_promoted"])

    def test_consent_guard_and_plan_tamper_before_engine(self):
        frozen = self.root / "e1-consent-plan.json"
        output = self.root / "e1-consent-run.json"
        self.write_plan(frozen)
        with patch("scripts.pocketrotor_e1.measure_method",
                   side_effect=AssertionError("must not run")):
            with self.assertRaises(PermissionError):
                run_study(self.root, self.trace, frozen, output)
        self.assertFalse(output.exists())

        tampered = _read_plan(frozen)
        tampered["schedule"] = [["stream", "nested-pin", "flat-lru"]] * 3
        frozen.write_text(json.dumps(tampered))
        with patch("scripts.pocketrotor_e1.measure_method",
                   side_effect=AssertionError("must not run")):
            with self.assertRaisesRegex(ValueError, "Plan drift"):
                run_study(self.root, self.trace, frozen, output,
                          approve_local_execution=True)
        self.assertFalse(output.exists())

    def test_failed_reference_hash_blocks_before_weights_execution(self):
        frozen = self.root / "e1-invalid-hash-plan.json"
        output = self.root / "e1-invalid-hash-run.json"
        self.write_plan(frozen)
        original = self.trace.read_text()
        trace = json.loads(original)
        trace["checkpoint_fingerprint"]["files"]["config.json"] = "a" * 64
        self.trace.write_text(json.dumps(trace))
        try:
            with patch("scripts.pocketrotor_e1.measure_method",
                       side_effect=AssertionError("must not run")):
                with self.assertRaisesRegex(ValueError, "Plan drift"):
                    run_study(self.root, self.trace, frozen, output,
                              approve_local_execution=True)
            self.assertFalse(output.exists())
        finally:
            self.trace.write_text(original)

    def test_three_modes_match_full_reference_cold_and_warm(self):
        from phistream.reference import validate_reference_trace
        validated = validate_reference_trace(self.trace_data, self.ckpt)
        runs = {}
        for method in METHODS:
            with self.subTest(method=method):
                result = measure_method(
                    self.ckpt, validated, method,
                    device="cpu", cache_bytes=self.cache_bytes,
                    max_tensor_mib=1, head_rows=7,
                )
                self.assertEqual(result["status"],
                                 "PARITY_MATCH_USER_REFERENCE_UNAUTHENTICATED")
                for phase in ("cold", "warm"):
                    self.assertEqual(result[phase]["greedy_matched"], 32)
                    self.assertEqual(result[phase]["probes_matched"],
                                     result[phase]["probes_total"])
                runs[method] = result
        self.assertGreater(runs["nested-pin"]["warm"]["cache"]["hits"], 0)
        self.assertLess(runs["nested-pin"]["warm"]["staged_bytes"],
                        runs["stream"]["warm"]["staged_bytes"])
        self.assertLessEqual(runs["nested-pin"]["warm"]["cache"]["end_resident_bytes"],
                             self.cache_bytes)

    def test_three_round_full_receipt_no_promotion(self):
        frozen = self.root / "e1-all-plan.json"
        output = self.root / "e1-all-run.json"
        self.write_plan(frozen)
        result = run_study(self.root, self.trace, frozen, output,
                           approve_local_execution=True)
        self.assertEqual(result["status"], "LOCAL_PAIRED_REVIEW_ONLY")
        self.assertTrue(result["model_execution"])
        self.assertEqual(len(result["round_results"]), 3)
        self.assertFalse(result["summary"]["production_promoted"])
        self.assertFalse(result["reference_source_authenticated"])
        self.assertEqual(json.loads(output.read_text())["status"], result["status"])
        with self.assertRaises(FileExistsError):
            run_study(self.root, self.trace, frozen, output,
                      approve_local_execution=True)

    def test_summary_refuses_missing_or_incorrect_data(self):
        good = {
            "status": "PARITY_MATCH_USER_REFERENCE_UNAUTHENTICATED",
            "wall_seconds": 10.0,
            "staged_bytes": 100,
        }
        better = {**good, "wall_seconds": 8.0, "staged_bytes": 30}
        rounds = [{
            "complete": True,
            "methods": {
                "stream": {"cold": good, "warm": good},
                "flat-lru": {"cold": better, "warm": better},
                "nested-pin": {"cold": better, "warm": better},
            },
        } for _ in range(3)]
        summary = summarize(rounds)
        self.assertTrue(summary["nested_candidate_for_further_review"])
        self.assertEqual(summary["candidate_results"]["nested-pin"]["warm"]
                         ["median_staged_bytes_avoided"], 70)
        rounds[0]["methods"]["nested-pin"]["warm"] = {
            **better, "status": "FAIL_PARITY",
        }
        with self.assertRaisesRegex(ValueError, "Unmatched"):
            summarize(rounds)

    def test_policy_holds_no_model_execution(self):
        params = self.params()
        params["capacity_mib"] = .00001
        plan, _ = proposed_plan(self.ckpt, self.trace, **params)
        self.assertEqual(plan["status"], "HELD_BY_MEMORY_POLICY")
        with self.assertRaises(ValueError):
            proposed_plan(self.ckpt, self.trace, **{**self.params(), "cache_mib": float("nan")})


if __name__ == "__main__":
    unittest.main()
