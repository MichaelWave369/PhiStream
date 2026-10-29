"""CPU-only bounded PocketRotor E0 tests; no downloads or pretrained weights."""
from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from phistream.llama import LAYER_NAMES, StreamingLlama
from phistream.pocketrotor import (
    PocketRotorCache, PocketRotorLlama, _quad, pocket_inventory,
)
from phistream.safeweights import LocalCheckpoint
from scripts.pocketrotor_lab import plan_experiment, run_experiment
from test_llama_execution import local_llama


class PocketRotorE0Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        local_llama(self.path)
        self.ckpt = LocalCheckpoint(self.path)
        self.layer_bytes = sum(
            pocket["bytes_fp32"] for pocket in pocket_inventory(self.ckpt)
            if pocket["layer"] == 0
        )
        self.params = dict(
            cache_bytes=self.layer_bytes, device="cpu",
            capacity_mib=16, reserve_mib=1,
            max_tensor_mib=1, head_rows=7,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_finite_addresses_unique_and_stable(self):
        inventory = pocket_inventory(self.ckpt)
        self.assertEqual(len(inventory), 2 * len(LAYER_NAMES))
        self.assertEqual(len({p["address"] for p in inventory}), len(inventory))
        self.assertEqual(inventory[0]["address"], "0/00")
        self.assertEqual(inventory[9]["address"], "1/00")
        self.assertEqual(_quad(15, 2), "33")
        for illegal in (-1, 16):
            with self.assertRaises(ValueError):
                _quad(illegal, 2)

    def test_nested_pin_accounted_fp32_and_no_cache_overrun(self):
        nested = PocketRotorCache(self.ckpt, budget_bytes=self.layer_bytes, policy="nested-pin")
        self.assertEqual(nested.pinned_layers, [0])
        self.assertEqual(nested.pinned_planned_bytes, self.layer_bytes)
        decoder = PocketRotorLlama(self.ckpt, cache_bytes=self.layer_bytes,
                                    policy="nested-pin", max_tensor_mib=1, head_rows=7)
        for token in [1, 7, 3, 8]:
            decoder.step(token)
        stats = decoder.pocket_cache.stats()
        self.assertEqual(stats["bytes_resident"], self.layer_bytes)
        self.assertLessEqual(stats["peak_bytes_resident"], stats["budget_bytes"])
        self.assertGreater(stats["hits"], 0)
        self.assertEqual(stats["evictions"], 0)
        self.assertEqual(stats["pinned_layer_ids"], [0])

    def test_flat_lru_can_evict_but_never_exceed_budget(self):
        decoder = PocketRotorLlama(self.ckpt, cache_bytes=self.layer_bytes,
                                    policy="flat-lru", max_tensor_mib=1, head_rows=7)
        for token in [1, 7, 3]:
            decoder.step(token)
        stats = decoder.pocket_cache.stats()
        self.assertLessEqual(stats["peak_bytes_resident"], self.layer_bytes)
        self.assertGreater(stats["evictions"], 0)

    def test_tiny_budget_bypasses_everything_no_faulty_pin(self):
        decoder = PocketRotorLlama(self.ckpt, cache_bytes=1,
                                    policy="nested-pin", max_tensor_mib=1, head_rows=7)
        baseline = StreamingLlama(self.ckpt, max_tensor_mib=1, head_rows=7)
        torch.testing.assert_close(decoder.step(2), baseline.step(2), rtol=0, atol=0)
        self.assertEqual(decoder.pocket_cache.stats()["pinned_layer_ids"], [])
        self.assertEqual(decoder.pocket_cache.stats()["bytes_resident"], 0)

    def test_same_model_three_mode_parity_and_staging(self):
        receipt = run_experiment(
            self.ckpt, [1, 7, 3, 8], approve_local_execution=True, **self.params
        )
        self.assertEqual(receipt["status"], "LOCAL_EXPERIMENT_PARITY_PASSED")
        self.assertFalse(receipt["production_promoted"])
        self.assertFalse(receipt["independent_model_reference"])
        self.assertEqual([r["policy"] for r in receipt["runs"]],
                         ["stream", "flat-lru", "nested-pin"])
        baseline, lru, nested = receipt["runs"]
        self.assertEqual(baseline["greedy_ids"], lru["greedy_ids"])
        self.assertEqual(baseline["greedy_ids"], nested["greedy_ids"])
        self.assertEqual(nested["max_absolute_logit_delta"], 0)
        self.assertLess(nested["weight_bytes_staged_total"],
                        baseline["weight_bytes_staged_total"])
        self.assertLessEqual(nested["cache"]["peak_bytes_resident"], self.layer_bytes)
        json.dumps(receipt, allow_nan=False)

    def test_unapproved_run_never_executes_or_hashes_weights(self):
        with patch("scripts.pocketrotor_lab.checkpoint_fingerprint",
                   side_effect=AssertionError("never hash during unauthorized run")):
            with patch("scripts.pocketrotor_lab.StreamingLlama",
                       side_effect=AssertionError("never execute")):
                with self.assertRaises(PermissionError):
                    run_experiment(self.ckpt, [1, 2], **self.params)

    def test_hard_policy_hold_and_invalid_inputs(self):
        with self.assertRaises(MemoryError):
            run_experiment(self.ckpt, [1, 2], approve_local_execution=True,
                           **{**self.params, "capacity_mib": 0.00001})
        for ids in ([], [1], [1, 2, True], [1, 32], [1] * 33):
            with self.subTest(ids=str(ids)[:30]):
                with self.assertRaises(ValueError):
                    plan_experiment(self.ckpt, ids, **self.params)
        for budget in (0, -1, True, 1.4):
            with self.subTest(budget=budget):
                with self.assertRaises(ValueError):
                    PocketRotorCache(self.ckpt, budget_bytes=budget, policy="nested-pin")
        with self.assertRaises(ValueError):
            PocketRotorCache(self.ckpt, budget_bytes=1024, policy="unknown")

    def test_cli_defaults_read_only(self):
        from scripts.pocketrotor_lab import main
        with patch("sys.argv", ["pocketrotor", str(self.path), "--ids", "1,2",
                                 "--cache-mib", "0.002",
                                 "--capacity-mib", "16", "--reserve-mib", "1"]):
            with patch("scripts.pocketrotor_lab.StreamingLlama",
                       side_effect=AssertionError("plan may not run decoder")):
                output = StringIO()
                with redirect_stdout(output):
                    main()
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["status"], "REVIEW_REQUIRED")
        self.assertFalse(plan["model_execution"])
        self.assertFalse(plan["production_promoted"])


if __name__ == "__main__":
    unittest.main()
