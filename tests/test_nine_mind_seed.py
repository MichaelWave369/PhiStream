"""Offline R0.4 tests: Nine Mind method seeds and no-approval Forge exports."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from phistream.model_forge import load_examples, review as forge_review
from phistream.nine_mind_seed import SEED, SeedError, _load_seed, export, forge_rows, validate


class NineMindSeedTests(unittest.TestCase):
    def setUp(self):
        self.rows, _ = _load_seed(SEED)

    def _mutated_file(self, rows, tempdir):
        filename = Path(tempdir) / "mutated.jsonl"
        filename.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        return filename

    def test_corpus_has_nine_distinct_methods_with_true_holdouts(self):
        result = validate()
        self.assertEqual(result["specialist_count"], 9)
        self.assertEqual(result["train_count"], 18)
        self.assertEqual(result["eval_count"], 9)
        self.assertEqual(result["councils"], {"invention": 3, "art": 3, "music": 3})
        self.assertEqual(result["status"], "UNREVIEWED_SEED_FIXTURES_ONLY")
        self.assertFalse(result["approval_complete"])
        self.assertFalse(result["training_started"])

    def test_all_nine_profiles_export_into_existing_forge_schema(self):
        profiles = ["tesla", "da-vinci", "feynman", "escher", "dali",
                    "picasso", "bach", "beethoven", "vivaldi"]
        for profile in profiles:
            with self.subTest(profile=profile):
                converted = forge_rows(profile)
                self.assertEqual(len(converted), 3)
                self.assertEqual([r["split"] for r in converted], ["train", "train", "eval"])
                self.assertTrue(all(r["review_status"] == "seed-unreviewed" for r in converted))

    def test_export_is_exclusive_private_unapproved_and_valid_forge_dataset(self):
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / "tesla.jsonl"
            result = export("tesla", dest)
            self.assertEqual(result["examples"], 3)
            self.assertFalse(result["approval_complete"])
            examples, _ = load_examples(dest)
            self.assertEqual(len(examples), 3)
            self.assertEqual(forge_review(dest)["status"], "REVIEW_REQUIRED")
            with self.assertRaisesRegex(SeedError, "exists"):
                export("tesla", dest)

    def test_source_rights_and_human_approval_cannot_be_faked_in_seed(self):
        for key, value in [("license", "UNKNOWN"), ("source", "private-chat"),
                           ("review_status", "human-approved")]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as d:
                rows = copy.deepcopy(self.rows)
                rows[0][key] = value
                with self.assertRaisesRegex(SeedError, "CC0"):
                    validate(self._mutated_file(rows, d))

    def test_duplicate_prompt_or_scenario_family_blocks_leakage(self):
        with tempfile.TemporaryDirectory() as d:
            rows = copy.deepcopy(self.rows)
            rows[2]["prompt"] = rows[0]["prompt"]
            with self.assertRaisesRegex(SeedError, "duplicate prompt"):
                validate(self._mutated_file(rows, d))
        with tempfile.TemporaryDirectory() as d:
            rows = copy.deepcopy(self.rows)
            rows[2]["task_family"] = rows[0]["task_family"]
            with self.assertRaisesRegex(SeedError, "scenario family"):
                validate(self._mutated_file(rows, d))

    def test_missing_heldout_row_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            rows = copy.deepcopy(self.rows)
            rows[2]["split"] = "train"
            with self.assertRaisesRegex(SeedError, "two training and one"):
                validate(self._mutated_file(rows, d))

    def test_unknown_profile_or_bad_council_rejected(self):
        for key, value in [("profile_id", "unknown"), ("council", "music")]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as d:
                rows = copy.deepcopy(self.rows)
                rows[0][key] = value
                with self.assertRaisesRegex(SeedError, "council"):
                    validate(self._mutated_file(rows, d))

    def test_no_duplicate_json_object_keys(self):
        with tempfile.TemporaryDirectory() as d:
            file = Path(d) / "bad.jsonl"
            lines = SEED.read_text(encoding="utf-8").splitlines()
            lines[0] = lines[0][:-1] + ',"id":"nine-999"}'
            file.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(SeedError, "duplicate JSON"):
                validate(file)

    def test_export_rejects_repo_path(self):
        with self.assertRaisesRegex(SeedError, "outside the Git checkout"):
            export("tesla", SEED.parent / "should-never-write.jsonl")

    def test_unrecognized_specialist_denied(self):
        with self.assertRaisesRegex(SeedError, "unknown specialist"):
            forge_rows("made-up-profile")

    def test_bad_rubric_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            rows = copy.deepcopy(self.rows)
            rows[0]["rubric"] = ["yes"]
            with self.assertRaisesRegex(SeedError, "rubric"):
                validate(self._mutated_file(rows, d))


if __name__ == "__main__":
    unittest.main()
