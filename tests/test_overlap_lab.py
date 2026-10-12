"""CPU-only overlap scheduler tests, bounds and advisory fail-closed cases."""
from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from phistream import overlap
from phistream.overlap import (
    SCHEMA, PROVENANCE, _order, recommend, run_overlap_lab,
)


class OverlapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_full_permutation_balance_is_repeatable(self):
        for count in (6, 12):
            seqs = _order(count)
            self.assertEqual(len(seqs), count)
            self.assertEqual(seqs, _order(count))
            for name in overlap.METHODS:
                for position in range(3):
                    self.assertEqual(sum(s[position] == name for s in seqs), count // 3)
        with self.assertRaises(ValueError):
            _order(4)

    def test_cpu_pipeline_receipt_and_full_scratch_cleanup(self):
        existing = self.workspace / "real-file.dat"
        existing.write_bytes(b"DO NOT CHANGE")
        original_files = set(self.workspace.iterdir())
        data = run_overlap_lab(self.workspace, gpu="none",
                               chunk_mib=1, chunks=4, rounds=6, warmups=1)
        self.assertEqual(data["schema"], SCHEMA)
        self.assertEqual(data["provenance"], PROVENANCE)
        self.assertEqual(data["status"], "PASS_CPU_OVERLAP_SCHEDULER")
        self.assertEqual(data["recommendation"]["decision"], "NO_GPU_EVIDENCE")
        self.assertTrue(data["fixture"]["scratch_deleted"])
        self.assertFalse(data["model_execution"])
        self.assertFalse(data["automatic_layer_placement"])
        self.assertFalse(data["design"]["gpu_clock_aligned_with_cpu"])
        self.assertTrue(data["design"]["host_buffers_reused_only_after_cuda_completion_event"])
        self.assertEqual(data["validation"]["audit_chunks"], 4)
        for name in overlap.METHODS:
            measures = data["methods"][name]
            self.assertEqual(len(measures["end_to_end"]["samples_seconds"]), 6)
            self.assertEqual(len(measures["first_measured_trial_host_timeline"]), 4)
            for item in measures["first_measured_trial_host_timeline"]:
                self.assertLessEqual(item["read_start_ms"], item["read_end_ms"])
                self.assertLessEqual(item["enqueue_start_ms"], item["completion_observed_ms"])
        self.assertEqual(set(self.workspace.iterdir()), original_files)
        self.assertEqual(existing.read_bytes(), b"DO NOT CHANGE")
        json.dumps(data, allow_nan=False)

    def test_policy_conservative_and_explains_minimum_gain(self):
        single = [10.] * 6
        self.assertEqual(recommend(single, [9.8] * 6, gpu=True)["decision"], "INCONCLUSIVE")
        self.assertEqual(recommend(single, [9.0] * 6, gpu=True)["decision"], "DOUBLE_CANDIDATE")
        self.assertEqual(recommend(single, [11.0] * 6, gpu=True)["decision"], "SINGLE_CANDIDATE")
        noisy = [9.0, 9.0, 9.0, 9.0, 11.0, 11.0]
        self.assertEqual(recommend(single, noisy, gpu=True)["decision"], "INCONCLUSIVE")
        self.assertEqual(recommend(single, noisy, gpu=False)["decision"], "NO_GPU_EVIDENCE")
        self.assertFalse(recommend(single, [9.0] * 6, gpu=True)["policy_applied"])

    def test_no_result_if_input_incomplete(self):
        with self.assertRaises(ValueError):
            recommend([10.] * 6, [10.] * 5, gpu=True)
        with self.assertRaises(ValueError):
            recommend([10.] * 6, [float("nan")] * 6, gpu=True)
        with self.assertRaises(ValueError):
            recommend([10.] * 5, [9.] * 5, gpu=True)

    def test_invalid_inputs_no_scratch(self):
        for args in (
            {"gpu": "auto"}, {"chunk_mib": 3}, {"chunk_mib": True},
            {"chunks": 3}, {"chunks": 40}, {"chunks": True},
            {"chunk_mib": 16, "chunks": 32}, {"rounds": 4},
            {"rounds": 8}, {"rounds": True}, {"warmups": 0},
            {"warmups": 4}, {"warmups": True},
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                run_overlap_lab(self.workspace, **args)
        self.assertEqual(list(self.workspace.iterdir()), [])

    def test_low_disk_ram_and_cuda_missing_fail_before_writes(self):
        from collections import namedtuple
        Usage = namedtuple("Usage", "total used free")
        with patch("phistream.overlap.shutil.disk_usage",
                   return_value=Usage(1000,999,1)):
            with self.assertRaisesRegex(RuntimeError, "disk space"):
                run_overlap_lab(self.workspace, gpu="none", chunk_mib=1, chunks=4)
        with patch("phistream.overlap.physical_memory",
                   return_value={"available_bytes": 10}):
            with self.assertRaisesRegex(RuntimeError, "system RAM"):
                run_overlap_lab(self.workspace, gpu="none", chunk_mib=1, chunks=4)
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "no fallback"):
                run_overlap_lab(self.workspace, gpu="cuda", chunk_mib=1, chunks=4)
        self.assertEqual(list(self.workspace.iterdir()), [])

    def test_corrupted_fixture_detected_in_replay_and_cleaned(self):
        original = overlap._fixture
        def corrupt(path, chunk_bytes, chunks):
            checksum, hashes, seconds = original(path,chunk_bytes,chunks)
            with path.open("r+b") as out:
                out.seek(0)
                out.write(b"bad!")
            return checksum, hashes, seconds
        with patch("phistream.overlap._fixture", side_effect=corrupt):
            with self.assertRaisesRegex(RuntimeError, "integrity|SHA256|modified"):
                run_overlap_lab(self.workspace, gpu="none",
                                chunk_mib=1, chunks=4, warmups=1)
        self.assertEqual(list(self.workspace.iterdir()), [])

    def test_cli_exclusive_receipt_preflight(self):
        from phistream.cli import main
        out = self.workspace / "receipt" / "overlap.json"
        args = ["phistream", "overlap-lab", "--workspace", str(self.workspace),
                "--chunk-mib", "1", "--chunks", "4",
                "--rounds", "6", "--warmups", "1", "--output", str(out)]
        with patch("sys.argv", args), redirect_stdout(StringIO()):
            main()
        receipt = json.loads(out.read_text())
        self.assertEqual(receipt["status"], "PASS_CPU_OVERLAP_SCHEDULER")
        with patch("sys.argv", args), redirect_stdout(StringIO()):
            with patch("phistream.overlap.run_overlap_lab", side_effect=AssertionError("should not rerun")):
                with self.assertRaises(FileExistsError):
                    main()
        self.assertEqual(out.read_text(), json.dumps(receipt, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    unittest.main()
