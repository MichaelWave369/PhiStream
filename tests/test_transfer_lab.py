"""CPU CI for bounded temp fixture, correct labels, failures and cleanup."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
from io import StringIO

from phistream.transfer import (
    SCHEMA, PROVENANCE, _stats, run_transfer_lab, MIB,
)


class TransferLabTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_cpu_lab_measures_repeats_and_cleans_own_files(self):
        before = set(self.folder.iterdir())
        receipt = run_transfer_lab(self.folder, gpu="none", size_mib=1, repeats=3)
        self.assertEqual(receipt["schema"], SCHEMA)
        self.assertEqual(receipt["provenance"], PROVENANCE)
        self.assertEqual(receipt["status"], "PASS_CPU_BASELINE")
        self.assertIsNone(receipt["cuda"])
        self.assertEqual(receipt["fixture"]["size_bytes"], MIB)
        self.assertEqual(len(receipt["fixture"]["sha256"]), 64)
        self.assertTrue(receipt["fixture"]["temporary_owned_file_deleted"])
        self.assertEqual(receipt["storage"]["cache_control"], "UNCONTROLLED_OS_FILESYSTEM_CACHE")
        self.assertFalse(receipt["storage"]["first_read_is_cold_cache"])
        self.assertEqual(len(receipt["storage"]["warm_reads"]["samples_seconds"]), 3)
        self.assertGreater(receipt["storage"]["warm_reads"]["median_mib_per_second"], 0)
        self.assertEqual(receipt["host"]["bytearray_copy"]["repetitions"], 3)
        self.assertFalse(receipt["model_execution"])
        self.assertFalse(receipt["automatic_layer_placement"])
        self.assertEqual(set(self.folder.iterdir()), before)
        json.dumps(receipt, allow_nan=False)

    def test_out_of_bounds_and_types_rejected_without_writes(self):
        for args in (
            {"size_mib": 0}, {"size_mib": 257},
            {"size_mib": 1.1}, {"size_mib": True},
            {"repeats": 2}, {"repeats": 16},
            {"repeats": True}, {"gpu": "auto"},
        ):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    run_transfer_lab(self.folder, **args)
        self.assertEqual(list(self.folder.iterdir()), [])

    def test_unavailable_cuda_refuses_before_any_scratch_file(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "no CPU fallback"):
                run_transfer_lab(self.folder, gpu="cuda", size_mib=1, repeats=3)
        self.assertEqual(list(self.folder.iterdir()), [])

    def test_missing_workspace_refused_no_creation(self):
        missing = self.folder / "missing-dir"
        with self.assertRaisesRegex(ValueError, "existing"):
            run_transfer_lab(missing, size_mib=1, repeats=3)
        self.assertFalse(missing.exists())

    def test_symlink_workspace_refused(self):
        symlink = self.folder / "link"
        try:
            symlink.symlink_to(self.folder, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("No symlink permission")
        with self.assertRaisesRegex(ValueError, "nonsymlink"):
            run_transfer_lab(symlink, size_mib=1)

    def test_disk_space_guard_fails_before_writes(self):
        from collections import namedtuple
        Usage = namedtuple("Usage", "total used free")
        with patch("phistream.transfer.shutil.disk_usage",
                   return_value=Usage(512 * MIB, 511 * MIB, MIB)):
            with self.assertRaisesRegex(RuntimeError, "disk space"):
                run_transfer_lab(self.folder, size_mib=1)
        self.assertEqual(list(self.folder.iterdir()), [])

    def test_stats_denies_nonpositive_or_nonfinite_time(self):
        for times in ([], [0.0], [float("nan")], [float("inf")]):
            with self.assertRaises(RuntimeError):
                _stats(times, MIB)

    def test_cli_creates_new_receipt_and_refuses_overwrite(self):
        from phistream.cli import main
        dest = self.folder / "out" / "transfer.json"
        argv = ["phistream", "transfer-lab",
                "--workspace", str(self.folder),
                "--gpu", "none", "--size-mib", "1", "--repeats", "3",
                "--output", str(dest)]
        with patch("sys.argv", argv), redirect_stdout(StringIO()) as capture:
            main()
        output = json.loads(capture.getvalue())
        self.assertEqual(output["status"], "PASS_CPU_BASELINE")
        self.assertEqual(json.loads(dest.read_text())["schema"], SCHEMA)
        with patch("sys.argv", argv), redirect_stdout(StringIO()):
            with self.assertRaises(FileExistsError):
                main()
        self.assertEqual(
            [p.name for p in self.folder.iterdir()], ["out"]
        )


if __name__ == "__main__":
    unittest.main()
