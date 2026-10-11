"""CPU and mock-CUDA tests for controlled direct readinto into tensor storage."""
from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from phistream import direct
from phistream.direct import (
    SCHEMA, PROVENANCE, CUDA_METHODS, CPU_METHODS,
    _readinto_exact, _summarize, run_direct_lab,
)


class DirectLabTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_direct_read_uses_writable_preallocated_bytearray(self):
        scratch = self.root / "own.bin"
        scratch.write_bytes(b"PhiStream-369")
        before = bytearray(b"_" * len(b"PhiStream-369"))
        duration = _readinto_exact(scratch, memoryview(before), len(before))
        self.assertEqual(before, b"PhiStream-369")
        self.assertGreater(duration, 0)
        with self.assertRaises(ValueError):
            _readinto_exact(scratch, memoryview(bytes(len(before))), len(before))
        with self.assertRaises(ValueError):
            _readinto_exact(scratch, memoryview(before)[:4], len(before))

    def test_exact_direct_read_rejects_short_or_oversized_file(self):
        path = self.root / "scratch"
        path.write_bytes(b"short")
        with self.assertRaisesRegex(RuntimeError, "changed size"):
            _readinto_exact(path, memoryview(bytearray(8)), 8)
        path.write_bytes(b"longlongmore")
        with self.assertRaisesRegex(RuntimeError, "changed size"):
            _readinto_exact(path, memoryview(bytearray(8)), 8)

    def test_cpu_lab_checks_checksum_all_sizes_and_cleans(self):
        before = set(self.root.iterdir())
        receipt = run_direct_lab(self.root, gpu="none",
                                 sizes_mib="1,2", rounds=6, warmups=1)
        self.assertEqual(receipt["schema"], SCHEMA)
        self.assertEqual(receipt["provenance"], PROVENANCE)
        self.assertEqual(receipt["status"], "PASS_CPU_DIRECT_READ")
        self.assertFalse(receipt["model_execution"])
        self.assertFalse(receipt["automatic_layer_placement"])
        self.assertEqual(receipt["design"]["methods"], list(CPU_METHODS))
        self.assertFalse(receipt["design"]["readinto_writes_into_tensor_host_storage"])
        self.assertEqual(len(receipt["size_reports"]), 2)
        for report in receipt["size_reports"]:
            self.assertTrue(report["temporary_scratch_deleted"])
            self.assertEqual(len(report["fixture_sha256"]), 64)
            self.assertEqual(sum(report["round_winner_counts"].values()), 6)
            for mode in CPU_METHODS:
                self.assertEqual(len(report["methods"][mode]["total"]["samples_seconds"]), 6)
        self.assertEqual(set(self.root.iterdir()), before)
        json.dumps(receipt, allow_nan=False)

    def test_invalid_size_round_and_warmup_no_io(self):
        for settings in (
            {"sizes_mib": "0"}, {"sizes_mib": "4,4"},
            {"sizes_mib": "128,64,16"}, {"sizes_mib": "1,2,3,4"},
            {"gpu": "default"}, {"rounds": 3}, {"rounds": True},
            {"warmups": 0}, {"warmups": 5}, {"warmups": True},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                run_direct_lab(self.root, **settings)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_missing_cuda_refuses_without_scratch_write(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "no CPU fallback"):
                run_direct_lab(self.root, gpu="cuda", sizes_mib="1")
        self.assertEqual(list(self.root.iterdir()), [])

    def test_low_host_ram_rejected_before_writes(self):
        with patch("phistream.direct.physical_memory",
                   return_value={"available_bytes": 1}):
            with self.assertRaisesRegex(RuntimeError, "host RAM"):
                run_direct_lab(self.root, sizes_mib="1")
        self.assertEqual(list(self.root.iterdir()), [])

    def test_corrupted_synthetic_scratch_is_rejected_and_cleaned(self):
        original = direct._write_fixture
        def corrupt(path, count):
            digest, elapsed = original(path, count)
            with path.open("r+b") as stream:
                stream.seek(0)
                stream.write(b"NOPE")
            return digest, elapsed
        with patch("phistream.direct._write_fixture", side_effect=corrupt):
            with self.assertRaisesRegex(RuntimeError, "SHA256"):
                run_direct_lab(self.root, sizes_mib="1", warmups=1)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_true_tensor_readinto_mock_cuda_copies_match(self):
        # CPU torch tensors stand in for the CUDA/pinned buffers. Real
        # PyTorch/NumPy memoryviews exercise direct readinto with NO device.
        import torch
        size = 8192
        scratch = self.root / "byte-test.dat"
        scratch.write_bytes(bytes(range(256)) * (size // 256))
        digest = hashlib.sha256(scratch.read_bytes()).hexdigest()
        pageable = torch.empty(size, dtype=torch.uint8)
        pinned_stub = torch.empty(size, dtype=torch.uint8)
        gpu_stub = torch.empty(size, dtype=torch.uint8)
        state = {
            "torch": torch, "cpu": pageable, "pinned": pinned_stub, "gpu": gpu_stub,
            "cpu_view": memoryview(pageable.numpy()).cast("B"),
            "pinned_view": memoryview(pinned_stub.numpy()).cast("B"),
        }
        with patch("torch.cuda.synchronize", return_value=None):
            for name in CUDA_METHODS:
                r = direct._trial_cuda(scratch, size, digest, name, state)
                self.assertTrue(r["checksum_verified"])
                self.assertTrue(r["cuda_sample_verified"])
                self.assertGreater(r["total"], 0)
                self.assertGreater(r["h2d"], 0)
                if name.startswith("direct_"):
                    self.assertEqual(r["temporary_host_copy"], 0)
                    self.assertEqual(r["host_to_pinned"], 0)
                else:
                    self.assertGreater(r["temporary_host_copy"], 0)
                    self.assertGreater(r["host_to_pinned"], 0)
        self.assertEqual(gpu_stub.tolist(), list(scratch.read_bytes()))
        self.assertFalse(pageable.is_pinned())

    def test_cuda_rounds_order_balanced_and_no_disk_leak_mocked(self):
        import torch
        def fake_state(count):
            cpu = torch.empty(count, dtype=torch.uint8)
            pinned = torch.empty(count, dtype=torch.uint8)
            gpu = torch.empty(count, dtype=torch.uint8)
            return {
                "torch": torch, "cpu": cpu, "pinned": pinned, "gpu": gpu,
                "cpu_view": memoryview(cpu.numpy()).cast("B"),
                "pinned_view": memoryview(pinned.numpy()).cast("B"),
                "name": "CI_TEST_ONLY_NO_CUDA", "free_before": 8 * 1024**3,
                "total": 12 * 1024**3, "torch_version": str(torch.__version__),
                "cuda_build": None,
            }
        with patch("torch.cuda.is_available", return_value=True):
            with patch("phistream.direct._state_cuda", side_effect=fake_state):
                with patch("torch.cuda.synchronize", return_value=None):
                    receipt = run_direct_lab(self.root, gpu="cuda",
                                             sizes_mib="1", rounds=6, warmups=1)
        report = receipt["size_reports"][0]
        self.assertEqual(receipt["status"], "PASS_CUDA_DIRECT_READ")
        for mode in CUDA_METHODS:
            self.assertEqual(len(report["methods"][mode]["total"]["samples_seconds"]), 6)
            self.assertEqual(sum(group[0] == mode for group in report["orders"]), 2)
        self.assertEqual(sum(report["round_winner_counts"].values()), 6)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_cli_receipt_written_exclusively(self):
        from phistream.cli import main
        dest = self.root / "receipts" / "direct.json"
        args = ["phistream", "direct-read-lab", "--workspace", str(self.root),
                "--sizes-mib", "1", "--rounds", "6", "--warmups", "1",
                "--output", str(dest)]
        with patch("sys.argv", args), redirect_stdout(StringIO()) as output:
            main()
        self.assertEqual(json.loads(output.getvalue())["status"], "PASS_CPU_DIRECT_READ")
        self.assertEqual(json.loads(dest.read_text())["schema"], SCHEMA)
        with patch("sys.argv", args), redirect_stdout(StringIO()):
            with self.assertRaises(FileExistsError):
                main()


if __name__ == "__main__":
    unittest.main()
