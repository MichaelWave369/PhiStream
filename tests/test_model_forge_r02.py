"""R0.2 preflight and evidence: deterministic, CPU-only, no model libraries needed."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from phistream.forge_preflight import DEPENDENCIES, inspect
from phistream.forge_review_pack import build_pack, _load_receipt
from phistream.model_forge import DEFAULT_DATA, ForgeError, load_examples


class ForgePreflightTests(unittest.TestCase):
    def test_seed_blocks_training_and_does_not_probe_implicitly(self):
        def unexpected():
            raise AssertionError("CUDA probing must be explicit")
        report = inspect(versions={key: "test-installed" for key in DEPENDENCIES}, gpu_probe=unexpected)
        self.assertEqual(report["reason_code"], "DATASET_NOT_HUMAN_APPROVED")
        self.assertEqual(report["gpu"]["status"], "not_probed")
        self.assertFalse(report["training_started"])
        self.assertFalse(report["fit_guarantee"])

    def test_approved_data_with_missing_packages_stays_blocked(self):
        examples, _ = load_examples(DEFAULT_DATA)
        for example in examples:
            example["review_status"] = "human-approved"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            path.write_text("".join(json.dumps(x) + "\n" for x in examples))
            versions = {key: "1.0" for key in DEPENDENCIES}
            versions["trl"] = None
            report = inspect(path, versions=versions)
            self.assertEqual(report["reason_code"], "DEPENDENCIES_MISSING")
            self.assertIn("trl", report["missing_packages"])
            versions["trl"] = "1.0"
            report = inspect(path, versions=versions)
            self.assertEqual(report["reason_code"], "CUDA_NOT_PROBED")
            report = inspect(path, probe_cuda=True, versions=versions,
                             gpu_probe=lambda: {"status": "observed", "free_mib": 7500})
            self.assertEqual(report["reason_code"], "OPERATOR_TRAINING_EXPERIMENT_CANDIDATE")
            self.assertFalse(report["adapter_created"])
            report = inspect(path, probe_cuda=True, versions=versions,
                             gpu_probe=lambda: {"status": "observed", "free_mib": 3200})
            self.assertEqual(report["reason_code"], "INSUFFICIENT_FREE_VRAM_ADVISORY")


class ForgeReviewPackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        examples, digest = load_examples(DEFAULT_DATA)
        self.eval = [x for x in examples if x["split"] == "eval"]
        self.base = self.root / "base.json"
        self.adapter = self.root / "adapt.json"
        def receipt(mode):
            return {
                "schema": "phi.model-forge.holdout-output.v1", "mode": mode,
                "model": "Qwen/Qwen3-4B-Instruct-2507", "dataset_sha256": digest,
                "review_status": "HUMAN_REVIEW_REQUIRED",
                "quality_proven": False, "deployment_authorized": False, "network_upload": False,
                "outputs": [
                    {"id": x["id"], "topic": x["topic"], "prompt": x["messages"][0]["content"],
                     "output": mode + " result " + x["id"]}
                    for x in self.eval
                ],
            }
        self.base.write_text(json.dumps(receipt("base")))
        self.adapter.write_text(json.dumps(receipt("adapter")))

    def tearDown(self):
        self.tmp.cleanup()

    def test_blinded_pack_keeps_key_separate_and_has_no_auto_winner(self):
        calls = iter([0, 1, 0, 1, 0, 1])
        pack, key = build_pack(DEFAULT_DATA, self.base, self.adapter,
                               coin=lambda: next(calls), identity=lambda: "session-test")
        self.assertEqual(len(pack["rows"]), len(self.eval))
        self.assertEqual(pack["rows"][0]["A"], "base result " + self.eval[0]["id"])
        self.assertEqual(pack["rows"][1]["A"], "adapter result " + self.eval[1]["id"])
        self.assertEqual(key["mapping"][1]["A"], "adapter")
        self.assertNotIn("adapter result", json.dumps(key))
        self.assertEqual(pack["automatic_winner"], None)
        self.assertFalse(pack["deployment_authorized"])

    def test_refuses_mismatched_prompt_and_dataset(self):
        obj = json.loads(self.adapter.read_text())
        obj["outputs"][0]["prompt"] = "tampered"
        self.adapter.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ForgeError, "prompt mismatch"):
            build_pack(DEFAULT_DATA, self.base, self.adapter)
        obj["outputs"][0]["prompt"] = self.eval[0]["messages"][0]["content"]
        obj["dataset_sha256"] = "0" * 64
        self.adapter.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ForgeError, "different dataset"):
            build_pack(DEFAULT_DATA, self.base, self.adapter)

    def test_duplicate_json_keys_refused(self):
        self.base.write_text('{"a":1,"a":2}')
        with self.assertRaisesRegex(ForgeError, "Duplicate"):
            _load_receipt(self.base)


if __name__ == "__main__":
    unittest.main()
