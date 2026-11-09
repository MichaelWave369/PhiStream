"""PhiStream Nine Mind: entirely offline, read-only catalog/mix contract tests."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from phistream.model_atlas import (
    AtlasError, CATALOG_PATH, MIX_SCHEMA, read_json, review_mix, validate_catalog
)


def proposal():
    def adapter(profile, weight):
        return {
            "profile_id": profile,
            "base_model_id": "Qwen/Qwen3-4B-Instruct-2507",
            "base_revision": "a" * 40,
            "tokenizer_sha256": "b" * 64,
            "adapter_sha256": ("c" if profile == "tesla" else "d") * 64,
            "target_modules": ["q_proj", "v_proj"],
            "rank": 8,
            "weight": weight,
        }
    return {
        "schema": MIX_SCHEMA,
        "base_model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "base_revision": "a" * 40,
        "adapters": [adapter("tesla", 0.6), adapter("feynman", 0.4)],
    }


class NineMindModelAtlasTests(unittest.TestCase):
    def setUp(self):
        self.catalog = read_json(CATALOG_PATH)

    def test_catalog_is_review_only_and_nine_in_three_councils(self):
        result = validate_catalog(self.catalog)
        self.assertEqual(result["candidate_count"], 4)
        self.assertEqual(result["profile_count"], 9)
        self.assertEqual(result["councils"], {"invention": 3, "art": 3, "music": 3})
        self.assertFalse(result["weights_downloaded"])
        self.assertFalse(result["routing_activated"])

    def test_review_is_never_execution_or_merge_approval(self):
        result = review_mix(proposal(), self.catalog)
        self.assertEqual(result["status"], "HELD_UNVERIFIED_REVIEW_ONLY")
        self.assertTrue(result["structural_metadata_consistent"])
        self.assertFalse(result["actual_weights_verified"])
        self.assertFalse(result["peft_merge_tested"])
        self.assertFalse(result["routing_activated"])

    def test_mixed_base_and_revision_denied(self):
        plan = proposal()
        plan["adapters"][1]["base_model_id"] = "Qwen/Qwen3.5-4B"
        with self.assertRaisesRegex(AtlasError, "identical base"):
            review_mix(plan, self.catalog)
        plan = proposal()
        plan["adapters"][1]["base_revision"] = "e" * 40
        with self.assertRaisesRegex(AtlasError, "identical base"):
            review_mix(plan, self.catalog)

    def test_target_and_tokenizer_mismatch_denied(self):
        plan = proposal()
        plan["adapters"][1]["target_modules"] = ["k_proj"]
        with self.assertRaisesRegex(AtlasError, "mismatch"):
            review_mix(plan, self.catalog)
        plan = proposal()
        plan["adapters"][1]["tokenizer_sha256"] = "f" * 64
        with self.assertRaisesRegex(AtlasError, "mismatch"):
            review_mix(plan, self.catalog)

    def test_invalid_ratio_or_too_many_specialists_denied(self):
        plan = proposal()
        plan["adapters"][1]["weight"] = 0.3
        with self.assertRaisesRegex(AtlasError, "total"):
            review_mix(plan, self.catalog)
        plan = proposal()
        plan["adapters"] *= 3
        with self.assertRaisesRegex(AtlasError, "at most four"):
            review_mix(plan, self.catalog)

    def test_duplicate_profiles_and_unapproved_catalog_state_denied(self):
        plan = proposal()
        plan["adapters"][1]["profile_id"] = "tesla"
        with self.assertRaisesRegex(AtlasError, "repeated"):
            review_mix(plan, self.catalog)
        candidate = copy.deepcopy(self.catalog)
        candidate["models"][0]["local_qualification"] = "READY"
        with self.assertRaisesRegex(AtlasError, "cannot grant"):
            validate_catalog(candidate)

    def test_bad_json_and_symlink_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.json"
            path.write_text('{"x":1,"x":2}')
            with self.assertRaises(AtlasError):
                read_json(path)
            link = Path(td) / "link.json"
            try:
                link.symlink_to(CATALOG_PATH)
            except (OSError, NotImplementedError):
                return
            with self.assertRaises(AtlasError):
                read_json(link)


if __name__ == "__main__":
    unittest.main()
