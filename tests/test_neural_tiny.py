"""Offline tiny-Transformer pilot tests; no downloads or unattended GPU access."""
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from phistream import neural_protocol
from phistream.neural_tiny import (
    Config, ExperimentError, checked_dataset, main, preflight, run,
)


class NeuralTinyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        self.addCleanup(self.root.cleanup)
        self.dir = Path(self.root.name)
        self.data = self.dir / "sample.txt"
        self.data.write_text(("phi measures evidence. neural research needs fixed tests.\n" * 25), encoding="utf-8")
        self.cfg = Config(context=8, dim=16, heads=2, layers=1,
                          batch=2, steps=2, seconds=30, device="cpu")

    def test_preflight_real_data_hash_and_no_train(self):
        result = preflight(self.data, context=8)
        self.assertFalse(result["training_started"])
        self.assertEqual(result["dataset_sha256"], hashlib.sha256(self.data.read_bytes()).hexdigest())
        self.assertEqual(result["evaluator_sha256"], neural_protocol.evaluator_digest(8))

    def test_approval_required_without_side_effect(self):
        dest = self.dir / "unapproved"
        with self.assertRaisesRegex(ExperimentError, "approve-training"):
            run(self.data, dest, self.cfg)
        self.assertFalse(dest.exists())

    def test_cpu_training_emits_integrity_inputs(self):
        dest = self.dir / "cpu-baseline"
        outcome = run(self.data, dest, self.cfg, approved=True)
        self.assertEqual(outcome["status"], "LOCAL_TRAINING_COMPLETED_UNVERIFIED")
        self.assertTrue(outcome["final_val_bpb"] > 0)
        self.assertEqual(outcome["steps"], 2)
        self.assertFalse(outcome["production_authorized"])
        self.assertFalse(outcome["checkpoint_saved"])
        self.assertEqual(outcome["plan"]["engine"], "phi_tiny")
        self.assertEqual(outcome["plan"]["evaluator_sha256"], neural_protocol.evaluator_digest(8))
        self.assertIn("peak_vram_mb: 0.000000", (dest / "metrics.log").read_text())
        self.assertEqual(json.loads((dest / "plan.json").read_text()), outcome["plan"])
        self.assertFalse((dest / "model.safetensors").exists())

    def test_data_stays_local_no_text_in_receipt(self):
        dest = self.dir / "local"
        run(self.data, dest, self.cfg, approved=True)
        self.assertNotIn("phi measures evidence", (dest / "run-meta.json").read_text())
        self.assertNotIn(str(self.data), (dest / "run-meta.json").read_text())

    def test_no_overwrite(self):
        dest = self.dir / "existing"
        dest.mkdir()
        with self.assertRaisesRegex(ExperimentError, "must not exist"):
            run(self.data, dest, self.cfg, approved=True)

    def test_input_symlink_rejected(self):
        symlink = self.dir / "link.txt"
        try:
            symlink.symlink_to(self.data)
        except OSError:
            self.skipTest("symlinks unavailable")
        with self.assertRaisesRegex(ExperimentError, "symlink"):
            checked_dataset(symlink, 8)

    def test_invalid_text_rejected(self):
        self.data.write_bytes(bytes([0xff] * 500))
        with self.assertRaisesRegex(ExperimentError, "UTF-8"):
            preflight(self.data, 8)

    def test_invalid_configuration_rejected(self):
        for config in (replace(self.cfg, heads=3),
                       replace(self.cfg, dim=4096),
                       replace(self.cfg, steps=10001),
                       replace(self.cfg, lr=float('nan'))):
            with self.subTest(config=config):
                with self.assertRaises(ExperimentError):
                    config.check()

    def test_same_protocol_context_deterministic(self):
        self.assertEqual(neural_protocol.evaluator_digest(8), neural_protocol.evaluator_digest(8))
        self.assertNotEqual(neural_protocol.evaluator_digest(8), neural_protocol.evaluator_digest(16))
        train, holdout = neural_protocol.split_bytes(self.data.read_bytes())
        self.assertEqual(len(train) + len(holdout), self.data.stat().st_size)
        self.assertEqual(neural_protocol.windows(holdout, 8), neural_protocol.windows(holdout, 8))

    def test_output_inside_git_repository_rejected(self):
        fake_repo = self.dir / "tracked"
        (fake_repo / ".git").mkdir(parents=True)
        with self.assertRaisesRegex(ExperimentError, "outside every Git"):
            run(self.data, fake_repo / "runs" / "pilot", self.cfg, approved=True)

    def test_cuda_refuses_implicit_fallback(self):
        import torch
        if torch.cuda.is_available():
            self.skipTest("CUDA available on host")
        with self.assertRaisesRegex(ExperimentError, "CUDA unavailable"):
            run(self.data, self.dir / "no-gpu", replace(self.cfg, device="cuda"), approved=True)

    def test_cli_help_and_read_only_preflight(self):
        self.assertEqual(main(["preflight", "--data", str(self.data), "--context", "8"]), 0)


if __name__ == "__main__":
    unittest.main()
