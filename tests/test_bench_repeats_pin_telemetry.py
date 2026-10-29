"""Independent CPU checks for Hunter's repeated fixture-benchmark repairs.

Original benchmark experiment/patch: Claude (Hunter's Xi table), commit
2cd5c83; integration reviewed against supplied 5060 evidence ZIP.
No pretrained weights, networks or CUDA needed by these tests.
"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

import torch
from phistream.cli import benchmark, benchmark_repeated, _gpu_telemetry
from phistream.format import ModelConfig, ModelStore, build_fixture


class RepeatsPinTelemetryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name) / "fixture"
        build_fixture(root, ModelConfig(
            vocab_size=16, dim=8, n_layers=2, n_heads=2, n_kv_heads=1,
            ffn_dim=16, max_context=32), seed=369)
        self.store = ModelStore(root)

    def test_default_single_receipt_keeps_historic_fields(self):
        r = benchmark(self.store, "stream", "cpu", [1, 2, 3], 4, 0.0)
        for key in ("generated_tokens_per_second_end_to_end", "pinned_layer_count",
                    "weight_bytes_loaded_total", "warmup_passes_unscored"):
            self.assertIn(key, r)
        self.assertIsNone(r["pin_warning"])
        self.assertIsNone(r["gpu_telemetry_before"])
        self.assertIsNone(r["gpu_telemetry_after"])

    def test_budget_below_one_layer_warns_and_reports_minimum(self):
        smallest = benchmark(self.store, "stream", "cpu", [1], 2, 0.0)["layer_parameter_bytes_min"]
        under = (smallest - 1) / (1024 * 1024)
        r = benchmark(self.store, "stream", "cpu", [1], 2, under)
        self.assertEqual(r["pinned_layer_count"], 0)
        self.assertIn("PIN_BUDGET_PINS_ZERO_LAYERS", r["pin_warning"])
        self.assertAlmostEqual(r["min_pin_mib_for_one_layer"],
                               smallest / (1024 * 1024), places=6)
        exact = benchmark(self.store, "stream", "cpu", [1], 2, smallest / (1024 * 1024))
        self.assertEqual(exact["pinned_layer_count"], 1)
        self.assertIsNone(exact["pin_warning"])

    def test_resident_mode_never_warns(self):
        r = benchmark(self.store, "resident", "cpu", [1], 2, 0.0001)
        self.assertIsNone(r["pin_warning"])

    def test_repeats_report_spread_and_identical_tokens(self):
        r = benchmark_repeated(self.store, "stream", "cpu", [1, 2, 3], 4, 0.0,
                               warmups=1, repeats=5)
        self.assertEqual(len(r["trials"]), 5)
        self.assertEqual(r["trials"][0]["warmup_passes_unscored"], 1)
        self.assertTrue(all(t["warmup_passes_unscored"] == 0 for t in r["trials"][1:]))
        s = r["summary"]["generated_tokens_per_second_end_to_end"]
        self.assertEqual(s["n"], 5)
        self.assertLessEqual(s["min"], s["median"])
        self.assertLessEqual(s["median"], s["max"])
        self.assertGreaterEqual(s["stdev"], 0)
        self.assertEqual(len({t["weight_bytes_loaded_total"] for t in r["trials"]}), 1)

    def test_repeats_out_of_range_fail_closed(self):
        for value in (0, 1, 51, True, 2.0, "5"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "repeats"):
                    benchmark_repeated(self.store, "stream", "cpu", [1], 2, 0.0, repeats=value)

    def test_gpu_telemetry_cpu_explicitly_absent(self):
        self.assertIsNone(_gpu_telemetry(torch.device("cpu")))

    def test_nvidia_smi_metadata_shape_fail_closed(self):
        # Invalid CSV field counts must never be treated as a real GPU reading.
        with patch("torch.cuda.mem_get_info", return_value=(1000, 2000)), \
             patch("torch.cuda.current_device", return_value=0), \
             patch("shutil.which", return_value="/usr/bin/nvidia-smi"), \
             patch("subprocess.run") as shell:
            shell.return_value.stdout = "bad,incomplete"
            result = _gpu_telemetry(torch.device("cuda"))
        self.assertEqual(result["gpu_free_bytes"], 1000)
        self.assertEqual(result["gpu_total_bytes"], 2000)
        self.assertIsNone(result["nvidia_smi"])


if __name__ == "__main__":
    unittest.main()
