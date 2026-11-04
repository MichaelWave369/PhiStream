"""CPU-only tests of model-to-adapter file binding and shareable field status."""
from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

from phistream.forge_field import field_report, verify_local_adapter
from phistream.model_forge import DEFAULT_DATA, ForgeError, load_examples


class ForgeFieldTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dataset = self.root / "reviewed.jsonl"
        rows, _ = load_examples(DEFAULT_DATA)
        for item in rows:
            item["review_status"] = "human-approved"
        self.dataset.write_text("".join(json.dumps(r) + "\n" for r in rows))
        self.rows, self.dataset_sha = load_examples(self.dataset)
        self.adapter = self.root / "adapter-r03"

    def tearDown(self):
        self.tmp.cleanup()

    def create_adapter(self, *, tamper=False):
        self.adapter.mkdir()
        (self.adapter / "adapter_config.json").write_text(json.dumps({
            "peft_type": "LORA", "task_type": "CAUSAL_LM",
            "base_model_name_or_path": "Qwen/Qwen3-4B-Instruct-2507",
        }))
        header = json.dumps({
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": {
                "dtype": "F32", "shape": [1], "data_offsets": [0, 4]
            }
        }).encode("utf-8")
        raw = struct.pack("<Q", len(header)) + header + b"\x01\x02\x03\x04"
        (self.adapter / "adapter_model.safetensors").write_bytes(raw)
        sha = hashlib.sha256(raw).hexdigest()
        record = {
            "schema": "phi.model-forge.training-receipt.v1",
            "status": "LOCALLY_TRAINED_UNVERIFIED_QUALITY",
            "base_model": "Qwen/Qwen3-4B-Instruct-2507",
            "model_revision": "f" * 40,
            "adapter_sha256": ("0" * 64 if tamper else sha),
            "dataset_sha256": self.dataset_sha,
            "train_example_count": 22, "held_out_examples": 6,
            "steps_requested": 8, "global_step": 8, "max_length": 512,
            "token_budget_audit": {
                "schema": "phi.model-forge.token-budget.v1",
                "status": "TOKEN_BUDGET_PASSED",
                "max_length_configured": 512, "train_count": 22,
                "held_out_count": 6, "max_total_tokens": 410,
            },
            "quality_improvement_proven": False,
            "production_authorized": False, "network_upload": False,
        }
        (self.adapter / "phi-training-receipt.json").write_text(json.dumps(record))
        return sha

    def create_holdout(self, mode, weight_sha=None, *, revision="f"*40):
        data = {
            "schema": "phi.model-forge.holdout-output.v1",
            "mode": mode,
            "model": "Qwen/Qwen3-4B-Instruct-2507",
            "model_revision": revision,
            "dataset_sha256": self.dataset_sha,
            "review_status": "HUMAN_REVIEW_REQUIRED",
            "quality_proven": False,
            "deployment_authorized": False,
            "network_upload": False,
            "adapter_sha256": weight_sha,
            "outputs": [
                {"id": row["id"], "topic": row["topic"],
                 "prompt": row["messages"][0]["content"],
                 "output": "CONFIDENTIAL EVAL OUTPUT " + mode + row["id"]}
                for row in self.rows if row["split"] == "eval"
            ],
        }
        filename = "base-holdout-r03.json" if mode == "base" else "adapter-holdout-r03.json"
        (self.root / filename).write_text(json.dumps(data))

    def test_missing_data_produces_safe_status(self):
        absent = field_report(self.root / "nonexistent.jsonl", self.root)
        self.assertEqual(absent["status"], "WAITING_FOR_CURATED_DATA")
        status = field_report(self.dataset, self.root)
        self.assertEqual(status["status"], "WAITING_FOR_LOCAL_TRAINING")
        self.assertFalse(status["adapter_verified_against_local_receipt"])

    def test_adapters_have_real_file_sha_and_do_not_export_text(self):
        sha = self.create_adapter()
        verified = verify_local_adapter(self.adapter, self.dataset_sha, 22, 6,
                                        base_model="Qwen/Qwen3-4B-Instruct-2507",
                                        model_revision="f"*40)
        self.assertEqual(verified["adapter_sha256"], sha)
        status = field_report(self.dataset, self.root)
        self.assertEqual(status["status"], "WAITING_FOR_HELD_OUT_EVAL")
        self.assertFalse(status["blind_review_ready"])
        self.create_holdout("base")
        self.create_holdout("adapter", sha)
        status = field_report(self.dataset, self.root)
        self.assertEqual(status["status"], "HUMAN_BLIND_REVIEW_REQUIRED")
        serialized = json.dumps(status)
        self.assertNotIn(str(self.root), serialized)
        self.assertNotIn("CONFIDENTIAL", serialized)
        self.assertEqual(status["adapter_sha256"], sha)
        self.assertFalse(status["model_quality_proven"])
        self.assertFalse(status["production_authorized"])

    def test_tampered_adapter_and_mismatched_evaluation_rejected(self):
        self.create_adapter(tamper=True)
        with self.assertRaisesRegex(ForgeError, "changed since"):
            field_report(self.dataset, self.root)
        record = json.loads((self.adapter / "phi-training-receipt.json").read_text())
        weights = self.adapter / "adapter_model.safetensors"
        record["adapter_sha256"] = hashlib.sha256(weights.read_bytes()).hexdigest()
        (self.adapter / "phi-training-receipt.json").write_text(json.dumps(record))
        self.create_holdout("base")
        self.create_holdout("adapter", "0" * 64)
        with self.assertRaisesRegex(ForgeError, "bind to"):
            field_report(self.dataset, self.root)

    def test_partial_training_and_wrong_revision(self):
        self.adapter.mkdir()
        (self.adapter / "adapter_model.safetensors").write_bytes(b"partial")
        status = field_report(self.dataset, self.root)
        self.assertEqual(status["status"], "INCOMPLETE_ADAPTER_ARTIFACT")
        (self.adapter / "adapter_model.safetensors").unlink()
        self.adapter.rmdir()
        self.create_adapter()
        with self.assertRaisesRegex(ForgeError, "revision differs"):
            verify_local_adapter(self.adapter, self.dataset_sha, 22, 6,
                base_model="Qwen/Qwen3-4B-Instruct-2507", model_revision="e"*40)

    def test_windows_stage_available_without_training_approval(self):
        script = (Path(__file__).resolve().parents[1] / "scripts" / "phi_forge_windows_pilot.ps1").read_text()
        self.assertIn("'field-report'", script)
        self.assertIn("phistream.forge_field", script)
        self.assertIn("'--root',$OutputRoot", script)


if __name__ == "__main__":
    unittest.main()
