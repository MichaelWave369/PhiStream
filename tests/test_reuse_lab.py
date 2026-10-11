"""Bounded scratch host-buffer reuse tests; no physical GPU required in CI."""
from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from phistream import reuse
from phistream.reuse import (
    SCHEMA, PROVENANCE, CUDA_METHODS, CPU_METHODS,
    _orders, parse_sizes, run_reuse_lab,
)


class ReuseLabTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_size_contract_rejects_invalid_and_huge(self):
        self.assertEqual(parse_sizes("4,16,64"), (4,16,64))
        for value in ("", "0", "129", "4,4", "16,64,128",
                      "4,16,64,1", "1.5", "01", "4,-1", " 4"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_sizes(value)

    def test_balanced_cuda_method_positions(self):
        for rounds in (6, 12):
            sequences = _orders(CUDA_METHODS, rounds, 369)
            self.assertEqual(len(sequences), rounds)
            for name in CUDA_METHODS:
                self.assertEqual(sum(row[0] == name for row in sequences), rounds // 3)
                self.assertEqual(sum(row[1] == name for row in sequences), rounds // 3)
                self.assertEqual(sum(row[2] == name for row in sequences), rounds // 3)
            self.assertEqual(sequences, _orders(CUDA_METHODS, rounds, 369))

    def test_cpu_safe_synthetic_and_cleanup(self):
        before = set(self.workspace.iterdir())
        receipt = run_reuse_lab(self.workspace, gpu="none", sizes_mib="1,2",
                                rounds=6, warmups=1)
        self.assertEqual(receipt["schema"], SCHEMA)
        self.assertEqual(receipt["provenance"], PROVENANCE)
        self.assertEqual(receipt["status"], "PASS_CPU_REUSE_BASELINE")
        self.assertEqual(receipt["design"]["methods"], list(CPU_METHODS))
        self.assertFalse(receipt["design"]["pinned_reuse_buffer_preallocated"])
        self.assertEqual(len(receipt["size_reports"]), 2)
        for report in receipt["size_reports"]:
            self.assertEqual(len(report["fixture_sha256"]), 64)
            self.assertTrue(report["scratch_deleted"])
            self.assertEqual(sum(report["round_winner_counts"].values()), 6)
            for name in CPU_METHODS:
                self.assertEqual(
                    len(report["methods"][name]["total"]["samples_seconds"]), 6
                )
        self.assertFalse(receipt["model_execution"])
        self.assertFalse(receipt["automatic_layer_placement"])
        self.assertEqual(set(self.workspace.iterdir()), before)
        json.dumps(receipt, allow_nan=False)

    def test_invalid_inputs_have_no_side_effects(self):
        for kwargs in (
            {"sizes_mib": "0"}, {"sizes_mib": "128,128"},
            {"gpu": "unknown"}, {"rounds": 5}, {"rounds": 18},
            {"rounds": True}, {"warmups": 0}, {"warmups": 5},
            {"warmups": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                run_reuse_lab(self.workspace, **kwargs)
        self.assertEqual(list(self.workspace.iterdir()), [])

    def test_cuda_missing_refused_before_scratch(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "no CPU fallback"):
                run_reuse_lab(self.workspace, gpu="cuda", sizes_mib="1")
        self.assertEqual(list(self.workspace.iterdir()), [])

    def test_corrupt_scratch_fails_and_cleans(self):
        original = reuse._write_fixture

        def tamper(path, size):
            digest, duration = original(path, size)
            with path.open("r+b") as fh:
                fh.seek(0)
                fh.write(b"\x00\x01\x00\x01")
            return digest, duration

        with patch("phistream.reuse._write_fixture", side_effect=tamper):
            with self.assertRaisesRegex(RuntimeError, "integrity"):
                run_reuse_lab(self.workspace, gpu="none",
                              sizes_mib="1", rounds=6, warmups=1)
        self.assertEqual(list(self.workspace.iterdir()), [])

    def test_cuda_methods_accounting_using_stubbed_gpu(self):
        # Run full scratch/ordering pipeline using mock timings. No GPU.
        gpu_info = {
            "name":"synthetic-mock","free_before":1024,"total":2048,
            "version":"fake","cuda_build":"none"
        }
        def fake_trial(path, count, digest, method, state):
            self.assertIn(method, CUDA_METHODS)
            return {
                "total": 0.035 if method == "pageable" else (
                    0.034 if method == "pinned_reuse" else 0.038),
                "read":.014,"prepare":.014,
                "pinned_allocation": .0 if method != "pinned_new" else .002,
                "pinned_staging": .0 if method == "pageable" else .001,
                "pre_h2d_sync": .0001,
                "h2d": .004 if method == "pageable" else .001,
                "release_after_timing": .0018,
            }
        with patch("torch.cuda.is_available", return_value=True):
            with patch("phistream.reuse._cuda_state", return_value=gpu_info):
                with patch("phistream.reuse._cuda_trial", side_effect=fake_trial):
                    result = run_reuse_lab(
                        self.workspace, gpu="cuda", sizes_mib="1",
                        rounds=6, warmups=1,
                    )
        self.assertEqual(result["status"], "PASS_CUDA_REUSE_LAB")
        report = result["size_reports"][0]
        self.assertEqual(report["round_winner_counts"]["pinned_reuse"], 6)
        self.assertEqual(report["methods"]["pinned_new"]["pinned_allocation"]["median_seconds"], .002)
        self.assertEqual(len(report["orders"]), 6)
        self.assertEqual(set(self.workspace.iterdir()), set())

    def test_cli_creates_new_local_file_and_refuses_overwrite(self):
        from phistream.cli import main
        out = self.workspace / "local" / "receipt.json"
        argv = ["phistream", "reuse-lab", "--workspace", str(self.workspace),
                "--sizes-mib", "1", "--rounds", "6", "--warmups", "1",
                "--output", str(out)]
        with patch("sys.argv", argv), redirect_stdout(StringIO()):
            main()
        self.assertEqual(json.loads(out.read_text())["status"],
                         "PASS_CPU_REUSE_BASELINE")
        with patch("sys.argv", argv), redirect_stdout(StringIO()):
            with self.assertRaises(FileExistsError):
                main()


if __name__ == "__main__":
    unittest.main()
