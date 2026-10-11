"""Read-only host counter and advisory policy tests, without a GPU or network."""
from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from collections import namedtuple

from phistream.hardware import (
    SCHEMA as HARDWARE_SCHEMA, hardware_snapshot, parse_linux_meminfo,
    physical_memory,
)
from phistream.advisor import SCHEMA as ADVISOR_SCHEMA, recommend_residency
from phistream.safeweights import LocalCheckpoint
from test_llama_execution import local_llama

RAM = {"total_bytes": 32 * 2**30, "available_bytes": 17 * 2**30,
       "source": "TEST_FAKE_RAM"}
GPU = {"device": "cuda", "name": "TEST GPU", "total_bytes": 12 * 2**30,
       "free_bytes": 11 * 2**30, "source": "TEST_FAKE_CUDA"}


class HardwareSnapshotTests(unittest.TestCase):
    def test_linux_memavailable_not_free_memory(self):
        text = "MemTotal: 32768000 kB\nMemFree: 1024 kB\nMemAvailable: 17920000 kB\n"
        out = parse_linux_meminfo(text)
        self.assertEqual(out["total_bytes"], 32768000 * 1024)
        self.assertEqual(out["available_bytes"], 17920000 * 1024)
        self.assertNotEqual(out["available_bytes"], 1024 * 1024)

    def test_linux_missing_or_bad_memory_fails_closed(self):
        for text in ("MemTotal: 4 kB\n", "MemTotal: 4 kB\nMemAvailable: 7 kB\n"):
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                parse_linux_meminfo(text)

    def test_windows_global_memory_status_selection(self):
        with patch("phistream.hardware.platform.system", return_value="Windows"):
            with patch("phistream.hardware._windows_physical_memory",
                       return_value={"total_bytes": 32, "available_bytes": 17}):
                r = physical_memory()
        self.assertEqual(r["source"], "WIN32_GLOBAL_MEMORY_STATUS_EX")
        self.assertEqual(r["available_bytes"], 17)

    def test_unknown_os_does_not_guess(self):
        with patch("phistream.hardware.platform.system", return_value="UnknownOS"):
            with self.assertRaisesRegex(RuntimeError, "refusing to guess"):
                physical_memory()

    def test_snapshot_measures_only_selected_directory_capacity(self):
        Disk = namedtuple("Disk", "total used free")
        with tempfile.TemporaryDirectory() as tmp:
            with patch("phistream.hardware.physical_memory", return_value=RAM):
                with patch("phistream.hardware.shutil.disk_usage",
                           return_value=Disk(2 * 2**40, 2**40, 2**40)) as usage:
                    r = hardware_snapshot(gpu="none", drives=(tmp,))
        usage.assert_called_once()
        self.assertEqual(r["schema"], HARDWARE_SCHEMA)
        self.assertEqual(r["status"], "OBSERVED_READ_ONLY")
        self.assertEqual(r["ram"]["available_bytes"], 17 * 2**30)
        self.assertIsNone(r["gpu"])
        self.assertFalse(r["drives"][0]["throughput_measured"])
        self.assertEqual(r["drives"][0]["free_bytes"], 2**40)
        self.assertIn("observed_utc", r)

    def test_snapshot_cuda_data_is_explicit_opt_in(self):
        with patch("phistream.hardware.physical_memory", return_value=RAM):
            with patch("phistream.hardware.cuda_memory", return_value=GPU) as sample:
                r = hardware_snapshot(gpu="cuda")
        sample.assert_called_once()
        self.assertEqual(r["gpu"]["free_bytes"], GPU["free_bytes"])
        self.assertEqual(r["gpu_requested"], "cuda")

    def test_absent_cuda_never_falls_back_to_fake_gpu(self):
        with patch("phistream.hardware.physical_memory", return_value=RAM):
            with patch("phistream.hardware.cuda_memory", side_effect=RuntimeError("CUDA unavailable")):
                with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
                    hardware_snapshot(gpu="cuda")

    def test_bad_path_or_duplicate_drives_rejected(self):
        with self.assertRaises(ValueError):
            hardware_snapshot(gpu="invalid")
        with self.assertRaises(ValueError):
            hardware_snapshot(drives=("x",) * 9)
        with self.assertRaises(ValueError):
            hardware_snapshot(drives=("x", "x"))
        with patch("phistream.hardware.physical_memory", return_value=RAM):
            with self.assertRaises(ValueError):
                hardware_snapshot(drives=("/path/that/does/not/exist/phi369",))


class AdvisorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        local_llama(self.root, dtype="BF16")
        self.ckpt = LocalCheckpoint(self.root)
        self.snapshot = {
            "schema": HARDWARE_SCHEMA,
            "status": "OBSERVED_READ_ONLY",
            "observed_utc": "2026-10-10T20:00:00+00:00",
            "ram": dict(RAM), "gpu": dict(GPU),
            "drives": [],
        }

    def tearDown(self):
        self.tmp.cleanup()

    def advise(self, **kwargs):
        return recommend_residency(
            self.ckpt, self.snapshot, context_tokens=kwargs.pop("context_tokens", 12),
            gpu_reserve_mib=kwargs.pop("gpu_reserve_mib", 1024),
            ram_reserve_mib=kwargs.pop("ram_reserve_mib", 4096),
            max_tensor_mib=kwargs.pop("max_tensor_mib", 1), **kwargs,
        )

    def test_policy_does_not_execute_or_claim_fit(self):
        r = self.advise()
        self.assertEqual(r["schema"], ADVISOR_SCHEMA)
        self.assertEqual(r["status"], "REVIEW_REQUIRED")
        self.assertFalse(r["auto_placement_enabled"])
        self.assertFalse(r["execution_approved"])
        self.assertTrue(r["no_fit_guarantee"])
        self.assertEqual(r["model_layer_count"], 2)
        self.assertEqual(r["proposed_gpu"]["candidate_pinned_layer_indices"], [0, 1])
        self.assertIn("not a layer allocation", r["proposed_gpu"]["interpretation"])
        json.dumps(r, allow_nan=False)

    def test_low_cuda_free_memory_rejects(self):
        self.snapshot["gpu"]["free_bytes"] = 16 * 2**20
        r = self.advise(gpu_reserve_mib=1024)
        self.assertEqual(r["status"], "BLOCKED_BY_POLICY")
        self.assertIn("ESTIMATED_WORKING_SET_EXCEEDS_POLICY_BUDGET", r["blocked_reasons"])
        self.assertEqual(r["proposed_gpu"]["candidate_pinned_layer_indices"], [])

    def test_low_host_ram_rejects_staging(self):
        self.snapshot["ram"]["available_bytes"] = 1024
        r = self.advise(ram_reserve_mib=0)
        self.assertEqual(r["status"], "BLOCKED_BY_POLICY")
        self.assertIn("HOST_LAYER_STAGING_EXCEEDS_RAM_POLICY", r["blocked_reasons"])

    def test_observed_gpu_required(self):
        self.snapshot["gpu"] = None
        with self.assertRaisesRegex(ValueError, "measured CUDA"):
            self.advise()

    def test_invalid_and_nonfinite_reserves_rejected(self):
        for val in (float("nan"), float("inf"), -1, True):
            with self.subTest(val=val), self.assertRaises(ValueError):
                self.advise(gpu_reserve_mib=val)

    def test_cli_snapshot_never_overwrites_local_json(self):
        from phistream.cli import main
        output = self.root / "local" / "hardware.json"
        with patch("phistream.hardware.physical_memory", return_value=RAM):
            with patch("sys.argv", ["phistream", "hardware-snapshot",
                                    "--output", str(output)]), redirect_stdout(StringIO()):
                main()
        content = json.loads(output.read_text())
        self.assertIsNone(content["gpu"])
        with patch("phistream.hardware.physical_memory", return_value=RAM):
            with patch("sys.argv", ["phistream", "hardware-snapshot",
                                    "--output", str(output)]), redirect_stdout(StringIO()):
                with self.assertRaises(FileExistsError):
                    main()

    def test_cli_recommend_read_only_and_blocked_nonzero(self):
        from phistream.cli import main
        argv = ["phistream", "recommend-residency", str(self.root),
                "--context-tokens", "8", "--gpu-reserve-mib", "1024", "--strict"]
        fake_gpu = dict(GPU, free_bytes=8 * 2**20)
        with patch("phistream.hardware.physical_memory", return_value=RAM):
            with patch("phistream.hardware.cuda_memory", return_value=fake_gpu):
                with patch("sys.argv", argv), redirect_stdout(StringIO()) as data:
                    with self.assertRaises(SystemExit) as error:
                        main()
        self.assertEqual(error.exception.code, 2)
        r = json.loads(data.getvalue())
        self.assertEqual(r["status"], "BLOCKED_BY_POLICY")
        self.assertFalse(r["execution_approved"])


if __name__ == "__main__":
    unittest.main()
