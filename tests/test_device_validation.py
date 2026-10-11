"""CPU-only regression tests for hardware receipt format and fail-closed guards."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from phistream import ModelStore, build_fixture
from phistream.cli import main
from phistream.diagnostics import (
    SCHEMA, PROVENANCE, device_information, validate_device,
)


class DeviceValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.fixture = self.root / "toy"
        build_fixture(self.fixture)
        self.store = ModelStore(self.fixture)

    def tearDown(self):
        self.temp.cleanup()

    def test_cpu_diagnostics_reports_no_fake_gpu(self):
        info = device_information("cpu")
        self.assertEqual(info["device"], "cpu")
        self.assertIsNone(info["gpu_name"])
        self.assertIsNone(info["gpu_total_bytes"])
        self.assertIn("torch_version", info)

    def test_cuda_request_never_falls_back(self):
        with patch("phistream.diagnostics.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "no silent CPU fallback"):
                device_information("cuda")

    def test_cpu_validator_checks_parity_and_three_modes(self):
        r = validate_device(self.store, "cpu", tokens=3, pin_budget_mib=0.125)
        self.assertEqual(r["schema"], SCHEMA)
        self.assertEqual(r["provenance"], PROVENANCE)
        self.assertEqual(r["status"], "PASS")
        self.assertTrue(r["parity"]["passed"])
        self.assertTrue(r["fixture"]["verified_sha256"])
        self.assertEqual(set(r["runs"]), {"stream", "pinned", "resident"})
        self.assertEqual(r["runs"]["stream"]["generated_token_ids"],
                         r["runs"]["resident"]["generated_token_ids"])
        self.assertEqual(len(r["runs"]["stream"]["generated_token_ids"]), 3)
        self.assertIsNone(r["runs"]["stream"]["peak_cuda_allocated_bytes"])
        json.dumps(r, allow_nan=False)

    def test_rejects_negative_budget_and_zero_tokens(self):
        with self.assertRaises(ValueError):
            validate_device(self.store, "cpu", tokens=0)
        with self.assertRaises(ValueError):
            validate_device(self.store, "cpu", pin_budget_mib=-2)

    def test_rejects_insufficient_context(self):
        with self.assertRaisesRegex(ValueError, "context"):
            validate_device(self.store, "cpu", tokens=10000)

    def test_rejects_fake_provenance(self):
        self.store.manifest_provenance = "UNKNOWN"
        with self.assertRaisesRegex(ValueError, "synthetic fixtures"):
            validate_device(self.store, "cpu", tokens=2)

    def test_cli_written_receipt_and_no_overwrite(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from unittest.mock import patch

        dest = self.root / "local" / "gpu-validation.json"
        args = ["phistream", "validate-device", str(self.fixture),
                "--device", "cpu", "--tokens", "2", "--output", str(dest)]
        with patch("sys.argv", args), redirect_stdout(StringIO()):
            main()
        data = json.loads(dest.read_text(encoding="utf-8"))
        self.assertEqual(data["status"], "PASS")
        with patch("sys.argv", args), redirect_stdout(StringIO()):
            with self.assertRaises(FileExistsError):
                main()


if __name__ == "__main__":
    unittest.main()
