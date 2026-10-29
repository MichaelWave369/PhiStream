"""PocketRotor E1 evidence audit tests: CPU synthetic, standard library audit."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from phistream.safeweights import LocalCheckpoint
from phistream.pocketrotor import pocket_inventory
from scripts.pocketrotor_e1 import proposed_plan, run_study
from scripts.pocketrotor_e1_audit import (
    AUDIT_SCHEMA, audit, _read, _replay_summary,
)
from test_llama_execution import local_llama
from test_reference_gate import make_trace


class PocketRotorE1AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.model = cls.root / "checkpoint"
        cls.model.mkdir()
        cfg, weights = local_llama(cls.model)
        ckpt = LocalCheckpoint(cls.model)
        trace = make_trace(cls.model, cfg, weights, steps=32)
        cls.trace = cls.root / "trace.json"
        cls.trace.write_text(json.dumps(trace), encoding="utf-8")
        cache_bytes = sum(
            v["bytes_fp32"] for v in pocket_inventory(ckpt) if v["layer"] == 0
        )
        plan, _ = proposed_plan(
            ckpt, cls.trace, device="cpu", cache_mib=cache_bytes / 1048576,
            capacity_mib=16, reserve_mib=1,
            max_tensor_mib=1, head_rows=7, rounds=3, seed=369,
        )
        cls.plan = cls.root / "plan.json"
        cls.plan.write_text(json.dumps(plan), encoding="utf-8")
        cls.receipt = cls.root / "original-receipt.json"
        run_study(cls.model, cls.trace, cls.plan, cls.receipt,
                  approve_local_execution=True)
        cls.raw = json.loads(cls.receipt.read_text())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def review(self, *, receipt=None, plan=None, trace=None):
        return audit(
            receipt or self.receipt, plan or self.plan, trace or self.trace,
            self.model,
        )

    def modified(self, data, suffix):
        path = self.root / suffix
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_cpu_three_way_real_fixture_receipt_structurally_passes(self):
        with patch("torch.cuda.is_available",
                   side_effect=AssertionError("auditor may not query GPU")):
            result = self.review()
        self.assertEqual(result["schema"], AUDIT_SCHEMA)
        self.assertEqual(result["status"], "STRUCTURE_AND_ARITHMETIC_MATCH_UNAUTHENTICATED")
        self.assertFalse(result["model_executed_by_auditor"])
        self.assertFalse(result["performance_certified"])
        self.assertEqual(result["reviewed_trial_phases"], 18)
        self.assertEqual(result["recomputed_summary"], self.raw["summary"])

    def test_tampered_timing_and_summary_rejected(self):
        bad = copy.deepcopy(self.raw)
        bad["round_results"][0]["methods"]["nested-pin"]["warm"]["wall_seconds"] /= 2
        with self.assertRaisesRegex(ValueError, "Summary arithmetic"):
            self.review(receipt=self.modified(bad, "bad-timing.json"))
        bad = copy.deepcopy(self.raw)
        bad["summary"]["nested_candidate_for_further_review"] = not bad["summary"]["nested_candidate_for_further_review"]
        with self.assertRaisesRegex(ValueError, "Summary arithmetic"):
            self.review(receipt=self.modified(bad, "bad-conclusion.json"))

    def test_invalid_parity_and_missing_trial_rejected(self):
        bad = copy.deepcopy(self.raw)
        bad["round_results"][0]["methods"]["nested-pin"]["cold"]["probes_matched"] -= 1
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.review(receipt=self.modified(bad, "bad-probe.json"))
        bad = copy.deepcopy(self.raw)
        bad["round_results"].pop()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.review(receipt=self.modified(bad, "missing-round.json"))
        bad = copy.deepcopy(self.raw)
        bad["status"] = "FAIL_CLOSED_INCOMPLETE"
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.review(receipt=self.modified(bad, "failed-run.json"))

    def test_hash_binding_and_checkpoint_change_rejected(self):
        bad = copy.deepcopy(self.raw)
        bad["trace_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "binding"):
            self.review(receipt=self.modified(bad, "stale-trace.json"))
        bad = copy.deepcopy(self.raw)
        bad["plan_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "binding"):
            self.review(receipt=self.modified(bad, "stale-plan.json"))
        model_config = self.model / "config.json"
        original = model_config.read_bytes()
        try:
            model_config.write_bytes(original + b" ")
            with self.assertRaisesRegex(ValueError, "Checkpoint bytes"):
                self.review()
        finally:
            model_config.write_bytes(original)

    def test_cache_limit_and_fake_gpu_memory_rejected(self):
        bad = copy.deepcopy(self.raw)
        bad["round_results"][0]["methods"]["flat-lru"]["warm"]["cache"]["end_resident_bytes"] = 1000000000
        with self.assertRaisesRegex(ValueError, "budget"):
            self.review(receipt=self.modified(bad, "cache-overflow.json"))
        bad = copy.deepcopy(self.raw)
        bad["round_results"][0]["methods"]["stream"]["cold"]["peak_cuda_allocated_per_step_bytes"] = 42
        with self.assertRaisesRegex(ValueError, "CUDA"):
            self.review(receipt=self.modified(bad, "fake-gpu.json"))

    def test_schedule_drift_and_json_duplicate_rejected(self):
        bad = copy.deepcopy(self.raw)
        bad["schedule"] = [list(reversed(x)) for x in bad["schedule"]]
        with self.assertRaisesRegex(ValueError, "scheduler"):
            self.review(receipt=self.modified(bad, "bad-order.json"))
        dup = self.root / "duplicate-keys.json"
        dup.write_text('{"schema":"x","schema":"y"}')
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            _read(dup, 1048576)

    def test_independent_replay_requires_complete_data(self):
        recomputed = _replay_summary(self.raw["round_results"])
        self.assertEqual(recomputed, self.raw["summary"])
        for method in ("flat-lru", "nested-pin"):
            for phase in ("cold", "warm"):
                self.assertIn("median_relative_gain", recomputed["candidate_results"][method][phase])


if __name__ == "__main__":
    unittest.main()
