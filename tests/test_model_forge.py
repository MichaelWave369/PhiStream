"""CPU-only, network-free Phi Model Forge contract tests."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from phistream.model_forge import DEFAULT_DATA, ForgeError, load_examples, review


class ModelForgeTests(unittest.TestCase):
    def test_bundled_seed_is_explicitly_unreviewed_and_separate(self):
        examples, digest = load_examples(DEFAULT_DATA)
        self.assertEqual(len(digest), 64)
        self.assertGreaterEqual(sum(x["split"] == "train" for x in examples), 12)
        self.assertGreaterEqual(sum(x["split"] == "eval" for x in examples), 4)
        receipt = review()
        self.assertEqual(receipt["status"], "REVIEW_REQUIRED")
        self.assertFalse(receipt["training_run"])
        self.assertFalse(receipt["production_ready"])

    def test_approval_does_not_misrepresent_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "approved.jsonl"
            examples, _ = load_examples(DEFAULT_DATA)
            for entry in examples:
                entry["review_status"] = "human-approved"
            path.write_text("".join(json.dumps(x) + "\n" for x in examples), encoding="utf-8")
            report = review(path)
            self.assertEqual(report["status"], "APPROVED_DATASET_ONLY")
            self.assertFalse(report["gpu_measured"])
            self.assertFalse(report["model_quality_proven"])

    def test_duplicate_prompts_and_untrusted_source_fail(self):
        examples, _ = load_examples(DEFAULT_DATA)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "samples.jsonl"
            examples[-1]["messages"][0]["content"] = examples[0]["messages"][0]["content"]
            path.write_text("".join(json.dumps(x) + "\n" for x in examples), encoding="utf-8")
            with self.assertRaisesRegex(ForgeError, "Duplicate prompt"):
                load_examples(path)
            examples[-1]["messages"][0]["content"] = "A unique held-out question"
            examples[-1]["source"] = "private-chat-archive"
            path.write_text("".join(json.dumps(x) + "\n" for x in examples), encoding="utf-8")
            with self.assertRaisesRegex(ForgeError, "provenance"):
                load_examples(path)

    def test_no_blank_files_symlinks_or_excessive_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "examples.jsonl"
            path.write_text("")
            with self.assertRaises(ForgeError):
                load_examples(path)
            path.write_bytes(b" " * (2 * 1024 * 1024 + 1))
            with self.assertRaises(ForgeError):
                load_examples(path)
            link = Path(tmp) / "link.jsonl"
            try:
                link.symlink_to(DEFAULT_DATA)
            except (OSError, NotImplementedError):
                return
            with self.assertRaises(ForgeError):
                load_examples(link)


if __name__ == "__main__":
    unittest.main()
