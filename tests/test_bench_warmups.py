"""Benchmark warmups must stay separate from recorded fixture measurements.

All executions use the local deterministic untrained CPU-only fixture.
There are no model downloads, real weights or CUDA requirements.
"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from phistream.cli import benchmark
from phistream.format import ModelConfig, ModelStore, build_fixture


class BenchmarkWarmupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "fixture"
        build_fixture(self.root, ModelConfig(
            vocab_size=16, dim=8, n_layers=2, n_heads=2, n_kv_heads=1,
            ffn_dim=16, max_context=32), seed=369)
        self.store = ModelStore(self.root)

    def bench(self, mode: str, warmups: int, pin_budget_mib: float = 0.0):
        return benchmark(self.store, mode, "cpu", [1, 2, 3], 4,
                         pin_budget_mib, warmups=warmups)

    def test_two_warmups_not_counted_as_recorded_stream_loads(self):
        cold = self.bench("stream", 0)
        warm = self.bench("stream", 2)
        self.assertEqual(cold["warmup_passes_unscored"], 0)
        self.assertEqual(warm["warmup_passes_unscored"], 2)
        self.assertFalse(cold["warmup_generation_parity_checked"])
        self.assertTrue(warm["warmup_generation_parity_checked"])
        self.assertEqual(cold["generated_token_ids"], warm["generated_token_ids"])
        self.assertEqual(cold["weight_load_events"], warm["weight_load_events"])
        self.assertEqual(cold["weight_bytes_loaded_total"], warm["weight_bytes_loaded_total"])
        self.assertEqual(cold["pinned_layer_count"], warm["pinned_layer_count"])
        self.assertEqual(len(warm["generated_token_ids"]), 4)
        self.assertEqual(warm["provenance"], "MEASURED_ON_THIS_MACHINE_SYNTHETIC_UNTRAINED")
        self.assertEqual(
            warm["timing_scope"],
            "ONE_FRESH_ENGINE_GENERATION_ONLY_AFTER_UNSCORED_WARMUPS",
        )
        self.assertGreater(warm["total_seconds"], 0)
        self.assertGreater(warm["generated_tokens_per_second_end_to_end"], 0)

    def test_resident_and_pinned_modes_preserve_semantics(self):
        for mode, budget in (("resident", 0.0), ("stream", 0.1)):
            with self.subTest(mode=mode):
                original = self.bench(mode, 0, budget)
                warmed = self.bench(mode, 1, budget)
                self.assertEqual(original["generated_token_ids"],
                                 warmed["generated_token_ids"])
                self.assertEqual(original["weight_load_events"],
                                 warmed["weight_load_events"])
                self.assertEqual(original["weight_bytes_loaded_total"],
                                 warmed["weight_bytes_loaded_total"])
                self.assertEqual(original["pinned_layer_parameter_bytes"],
                                 warmed["pinned_layer_parameter_bytes"])

    def test_out_of_range_and_bool_warmups_fail_before_execution(self):
        for value in (-1, 6, True, 0.5, "1", None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "warmups"):
                    self.bench("stream", value)

    def test_nonpositive_tokens_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            benchmark(self.store, "stream", "cpu", [1], 0, 0.0, warmups=1)


if __name__ == "__main__":
    unittest.main()
