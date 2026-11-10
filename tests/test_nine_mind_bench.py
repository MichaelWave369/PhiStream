"""Nine Mind R0.5: no-model, no-network, three-arm blind review unit tests."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from phistream.model_forge import ForgeError, load_examples
from phistream.nine_mind_seed import SEED, forge_rows
from phistream.nine_mind_bench import build_pack, write_pair


def run_receipt(dataset, mode, *, revision="a" * 40):
    examples, dataset_sha256 = load_examples(dataset)
    outputs = [
        {"id": row["id"], "topic": row["topic"],
         "prompt": row["messages"][0]["content"],
         "output": f"Generated {mode} answer for {row['id']}"}
        for row in examples if row["split"] == "eval"
    ]
    result = {
        "schema": "phi.model-forge.holdout-output.v1",
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "model_revision": revision,
        "mode": mode, "dataset_sha256": dataset_sha256,
        "outputs": outputs,
        "adapter_sha256": "d" * 64 if mode == "adapter" else None,
        "review_status": "HUMAN_REVIEW_REQUIRED",
        "quality_proven": False, "deployment_authorized": False,
        "network_upload": False,
    }
    if mode == "prompted":
        result.update({
            "profile_id": "tesla",
            "recipe_sha256": "c" * 64,
            "source_class": "OPERATOR_ASSERTED_NOT_ATTESTED",
            "method_prompt_applied": True,
        })
    return result


class NineMindBenchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dataset = self.root / "tesla-dataset.jsonl"
        self.dataset.write_text("".join(json.dumps(x) + "\n" for x in forge_rows("tesla")),
                                encoding="utf-8")
        self.base, self.prompted, self.adapter = (
            self.root / f"{name}.json" for name in ("base", "prompted", "adapter")
        )
        for path, mode in zip((self.base, self.prompted, self.adapter),
                              ("base", "prompted", "adapter")):
            path.write_text(json.dumps(run_receipt(self.dataset, mode)), encoding="utf-8")

    def build(self, *, adapter=False):
        return build_pack(self.dataset, "tesla", self.base, self.prompted,
                          self.adapter if adapter else None,
                          shuffle=lambda xs: xs.reverse(),
                          session_factory=lambda: "f" * 32)

    def test_two_arm_pack_is_blind_non_authoritative(self):
        pack, key = self.build()
        self.assertEqual(pack["arm_count"], 2)
        self.assertEqual(pack["row_count"], 1)
        self.assertEqual(pack["dataset_review_status"], "UNREVIEWED_MECHANICS_ONLY")
        self.assertEqual(list(pack["rows"][0]["answers"]), ["A", "B"])
        self.assertEqual(key["mapping"][0]["labels"], {"A": "prompted", "B": "base"})
        self.assertNotIn("prompted", pack["rows"][0]["answers"])
        self.assertIsNone(pack["winner"])
        self.assertFalse(pack["quality_improvement_proven"])
        self.assertFalse(pack["production_authorized"])
        self.assertTrue(key["model_outputs_unattested"])

    def test_three_arm_compares_without_training_claim(self):
        pack, key = self.build(adapter=True)
        self.assertEqual(pack["arm_count"], 3)
        self.assertEqual(list(pack["rows"][0]["answers"]), ["A", "B", "C"])
        self.assertEqual(key["mapping"][0]["labels"],
                         {"A": "adapter", "B": "prompted", "C": "base"})
        self.assertEqual(key["adapter_sha256_declared"], "d" * 64)
        self.assertFalse(key["quality_improvement_proven"])

    def test_model_or_revision_mismatch_denied(self):
        receipt = run_receipt(self.dataset, "prompted")
        receipt["model_revision"] = "b" * 40
        self.prompted.write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(ForgeError, "same base"):
            self.build()
        receipt["model_revision"] = "a" * 40
        receipt["model"] = "different/Model"
        self.prompted.write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(ForgeError, "same base"):
            self.build()

    def test_unpinned_base_not_admitted(self):
        receipt = run_receipt(self.dataset, "base")
        receipt["model_revision"] = None
        self.base.write_text(json.dumps(receipt), encoding="utf-8")
        receipt = run_receipt(self.dataset, "prompted")
        receipt["model_revision"] = None
        self.prompted.write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(ForgeError, "exact pinned"):
            self.build()

    def test_malformed_revision_type_is_a_clean_refusal(self):
        receipt = run_receipt(self.dataset, "prompted")
        receipt["model_revision"] = ["invalid", "list"]
        self.prompted.write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(ForgeError, "exact pinned"):
            self.build()

    def test_bad_recipe_or_false_method_application_denied(self):
        for key, value, error in [
            ("recipe_sha256", "not-a-sha", "recipe"),
            ("method_prompt_applied", False, "actual use"),
            ("source_class", "INDEPENDENT_PROOF", "authenticity"),
            ("adapter_sha256", "d" * 64, "must not"),
        ]:
            with self.subTest(key=key):
                receipt = run_receipt(self.dataset, "prompted")
                receipt[key] = value
                self.prompted.write_text(json.dumps(receipt), encoding="utf-8")
                with self.assertRaisesRegex(ForgeError, error):
                    self.build()

    def test_mismatched_dataset_or_tampered_prompt_denied(self):
        receipt = run_receipt(self.dataset, "base")
        receipt["dataset_sha256"] = "0" * 64
        self.base.write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(ForgeError, "different dataset"):
            self.build()
        receipt = run_receipt(self.dataset, "base")
        receipt["outputs"][0]["prompt"] += " injected"
        self.base.write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(ForgeError, "prompt mismatch"):
            self.build()

    def test_invalid_adapter_hash_and_false_quality_denied(self):
        receipt = run_receipt(self.dataset, "adapter")
        receipt["adapter_sha256"] = None
        self.adapter.write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(ForgeError, "requires"):
            self.build(adapter=True)
        receipt = run_receipt(self.dataset, "adapter")
        receipt["quality_proven"] = True
        self.adapter.write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(ForgeError, "must not claim"):
            self.build(adapter=True)

    def test_unknown_profile_rejected(self):
        with self.assertRaisesRegex(ForgeError, "unknown Nine Mind"):
            build_pack(self.dataset, "no-such-council", self.base, self.prompted)

    def test_private_output_pair_written_separately_and_exclusive(self):
        pack, key = self.build(adapter=True)
        a, b = self.root / "review.json", self.root / "key.json"
        result = write_pair(a, b, pack, key)
        self.assertTrue(result["answer_key_separate"])
        public = json.loads(a.read_text(encoding="utf-8"))
        secret = json.loads(b.read_text(encoding="utf-8"))
        self.assertNotIn("mapping", public)
        self.assertIn("mapping", secret)
        self.assertEqual(secret["session"], public["session"])
        with self.assertRaisesRegex(ForgeError, "overwrite"):
            write_pair(a, self.root / "another-key.json", pack, key)

    def test_same_destination_or_repo_output_denied(self):
        pack, key = self.build()
        dest = self.root / "same.json"
        with self.assertRaisesRegex(ForgeError, "separate destinations"):
            write_pair(dest, dest, pack, key)
        with self.assertRaisesRegex(ForgeError, "inside repository"):
            write_pair(SEED.parent / "unsafe-private-review.json", dest, pack, key)
        self.assertFalse(dest.exists())

    def test_bad_shuffle_rejected(self):
        with self.assertRaisesRegex(ForgeError, "randomization"):
            build_pack(self.dataset, "tesla", self.base, self.prompted,
                       shuffle=lambda xs: xs.pop(),
                       session_factory=lambda: "f" * 32)

    def test_duplicate_json_receipt_denied(self):
        self.base.write_text('{"schema":"first","schema":"second"}', encoding="utf-8")
        with self.assertRaisesRegex(ForgeError, "Duplicate receipt JSON"):
            self.build()

    def test_existing_base_receipt_mode_must_match(self):
        receipt = run_receipt(self.dataset, "base")
        receipt["mode"] = "adapter"
        self.base.write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(ForgeError, "schema or mode"):
            self.build()


if __name__ == "__main__":
    unittest.main()
