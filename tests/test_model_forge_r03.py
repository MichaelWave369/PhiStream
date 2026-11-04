"""CPU-only Phi Model Forge human review and Windows stage boundary tests."""
from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
from phistream.forge_curate import curate
from phistream.model_forge import DEFAULT_DATA, ForgeError, review

class R03CuratorTests(unittest.TestCase):
    def test_explicit_approvals_create_new_reviewed_data(self):
        seen = []
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "approved.jsonl"
            def decide(entry):
                seen.append(entry["id"])
                return "approve"
            result = curate(DEFAULT_DATA, output, decide)
            self.assertEqual(len(seen), 28)
            self.assertEqual(result["approved_train"], 22)
            self.assertEqual(result["approved_eval"], 6)
            self.assertTrue(review(output)["human_review_complete"])
            self.assertFalse(result["training_performed"])
            with self.assertRaisesRegex(ForgeError, "already exists"):
                curate(DEFAULT_DATA, output, decide)

    def test_abort_or_drop_all_eval_does_not_write(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "approved.jsonl"
            with self.assertRaisesRegex(ForgeError, "cancelled"):
                curate(DEFAULT_DATA, output, lambda _: "quit")
            self.assertFalse(output.exists())
            with self.assertRaisesRegex(ForgeError, "held-out"):
                curate(DEFAULT_DATA, output, lambda e: "exclude" if e["split"] == "eval" else "approve")
            self.assertFalse(output.exists())
            with self.assertRaisesRegex(ForgeError, "explicit"):
                curate(DEFAULT_DATA, output, lambda _: "all")
            self.assertFalse(output.exists())

    def test_windows_default_is_readonly_and_training_is_gated(self):
        root = Path(__file__).resolve().parents[1]
        content = (root / "scripts" / "phi_forge_windows_pilot.ps1").read_text()
        self.assertIn("[string]$Stage = 'check'", content)
        self.assertIn("[switch]$ApproveTraining", content)
        self.assertIn("'--execute'", content)
        self.assertIn("'--probe-cuda'", content)
        self.assertIn("'blind-review'", content)
        self.assertNotIn("Invoke-Expression", content)
        self.assertNotIn("ExecutionPolicy Bypass", content)

if __name__ == "__main__":
    unittest.main()
