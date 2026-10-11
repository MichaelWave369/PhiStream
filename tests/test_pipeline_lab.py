"""CPU CI verifies paired scratch pipeline integrity, bounds and read-only limits."""
from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from phistream.pipeline import (
    SCHEMA, PROVENANCE, balanced_orders, run_pipeline_lab, MIB, _aggregate, _duration_stats,
)


class PipelineLabTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_balanced_order_deterministic_reproducible(self):
        first = balanced_orders(8)
        self.assertEqual(first, balanced_orders(8))
        self.assertEqual(len(first), 8)
        self.assertEqual(sum(x[0] == "pinned" for x in first), 4)
        self.assertEqual(sum(x[0] == "pageable" for x in first), 4)
        for pair in first:
            self.assertEqual(set(pair), {"pinned", "pageable"})

    def test_cpu_pipeline_repeated_and_cleanup(self):
        before = set(self.folder.iterdir())
        r = run_pipeline_lab(self.folder, size_mib=1, gpu="none", repeats=4, warmups=1)
        self.assertEqual(r["schema"], SCHEMA)
        self.assertEqual(r["provenance"], PROVENANCE)
        self.assertEqual(r["status"], "PASS_CPU_PIPELINE")
        self.assertEqual(r["fixture"]["bytes"], MIB)
        self.assertTrue(r["fixture"]["owned_scratch_deleted"])
        self.assertEqual(r["trial_design"]["paired_repetitions"], 4)
        self.assertEqual(r["trial_design"]["warmups_per_mode"], 1)
        self.assertFalse(r["trial_design"]["os_file_cache_controlled"])
        self.assertTrue(r["trial_design"]["checksum_checked_outside_timing"])
        self.assertEqual(r["methods"]["pinned"]["kind"], "FILE_READ_TO_REUSABLE_BYTEARRAY")
        for mode in ("pageable", "pinned"):
            measurements = r["methods"][mode]["measurements"]
            self.assertEqual(len(measurements["end_to_end"]["samples_seconds"]), 4)
            self.assertEqual(len(measurements["file_read"]["samples_seconds"]), 4)
            self.assertEqual(len(measurements["host_buffer_prepare"]["samples_seconds"]), 4)
            self.assertGreater(measurements["end_to_end"]["median_seconds"], 0)
        self.assertFalse(r["model_execution"])
        self.assertFalse(r["automatic_layer_placement"])
        self.assertEqual(set(self.folder.iterdir()), before)
        json.dumps(r, allow_nan=False)

    def test_per_trial_residual_is_not_difference_of_medians(self):
        # Pin staging and pre-transfer CUDA sync are both charged to the full
        # end-to-end path. Any remaining time must be explicitly disclosed.
        trials = [
            {"total": .015, "read": .005, "prepare": .003,
             "pinned_stage": .002, "pre_h2d_sync": .003, "h2d": .001},
            {"total": .018, "read": .005, "prepare": .004,
             "pinned_stage": .003, "pre_h2d_sync": .004, "h2d": .001},
        ]
        result = _aggregate(trials, 1048576, True)
        self.assertEqual(result["unattributed_wall_time"]["samples_seconds"],
                         [.001, .001])
        self.assertAlmostEqual(result["pre_h2d_sync"]["median_seconds"], .0035)
        self.assertAlmostEqual(result["unattributed_wall_time"]["median_seconds"], .001)

    def test_nonnegative_time_accounting_allows_zero_and_rejects_nan(self):
        self.assertEqual(_duration_stats([0., .1, 0.])["median_seconds"], 0.)
        with self.assertRaises(RuntimeError):
            _duration_stats([float("nan")])
        with self.assertRaises(RuntimeError):
            _duration_stats([-1.])

    def test_cpu_trials_have_unattributed_timing_without_gpu_fields(self):
        receipt = run_pipeline_lab(self.folder, size_mib=1, repeats=4, warmups=1)
        for mode in ("pageable", "pinned"):
            m = receipt["methods"][mode]["measurements"]
            self.assertEqual(len(m["unattributed_wall_time"]["samples_seconds"]), 4)
            self.assertNotIn("pre_h2d_sync", m)

    def test_invalid_inputs_no_files(self):
        for kwargs in (
            {"gpu": "auto"}, {"size_mib": 0}, {"size_mib": 129},
            {"size_mib": True}, {"size_mib": 2.5},
            {"repeats": 3}, {"repeats": 21}, {"repeats": True},
            {"warmups": 0}, {"warmups": 6}, {"warmups": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                run_pipeline_lab(self.folder, **kwargs)
        self.assertEqual(list(self.folder.iterdir()), [])

    def test_workspace_missing_rejected_without_creating(self):
        path = self.folder / "missing"
        with self.assertRaisesRegex(ValueError, "existing"):
            run_pipeline_lab(path)
        self.assertFalse(path.exists())

    def test_explicit_cuda_refuses_no_gpu_before_writes(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CPU fallback"):
                run_pipeline_lab(self.folder, gpu="cuda", size_mib=1, repeats=4)
        self.assertEqual(list(self.folder.iterdir()), [])

    def test_fake_corrupted_file_rejected_and_cleaned(self):
        from phistream import pipeline
        saved = pipeline._write_fixture

        def broken_fixture(path, count):
            digest, duration = saved(path, count)
            with path.open("r+b") as fh:
                fh.seek(0)
                fh.write(b"\xff\xff\xff\xff")
            return digest, duration

        with patch("phistream.pipeline._write_fixture", side_effect=broken_fixture):
            with self.assertRaisesRegex(RuntimeError, "SHA256"):
                run_pipeline_lab(self.folder, size_mib=1, repeats=4, warmups=1)
        self.assertEqual(list(self.folder.iterdir()), [])

    def test_cli_local_receipt_no_overwrite(self):
        from phistream.cli import main
        target = self.folder / "receipts" / "pipeline.json"
        args = ["phistream", "pipeline-lab", "--workspace", str(self.folder),
                "--gpu", "none", "--size-mib", "1", "--repeats", "4",
                "--warmups", "1", "--output", str(target)]
        with patch("sys.argv", args), redirect_stdout(StringIO()) as out:
            main()
        self.assertEqual(json.loads(out.getvalue())["status"], "PASS_CPU_PIPELINE")
        self.assertEqual(json.loads(target.read_text())["schema"], SCHEMA)
        with patch("sys.argv", args), redirect_stdout(StringIO()):
            with self.assertRaises(FileExistsError):
                main()


if __name__ == "__main__":
    unittest.main()
