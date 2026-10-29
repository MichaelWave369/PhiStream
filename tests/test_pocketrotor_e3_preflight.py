"""E3-P preflight offline fixture tests; no Torch/Transformers/model imports."""
from __future__ import annotations

import copy
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.pocketrotor_e3_preflight import (
    SCHEMA, inspect_source, inspect_config, preflight, main,
)

SOURCE = '''
class Qwen3_5MoeTopKRouter:
    def __init__(self, config):
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
    def forward(self, hidden_states):
        router_logits = hidden_states
        router_top_value, router_indices = torch.topk(hidden_states, self.top_k, dim=-1)
        router_scores = router_top_value
        return router_logits, router_scores, router_indices

class Qwen3_5MoeExperts:
    def forward(self, hidden_states, top_k_index, top_k_weights):
        return hidden_states

class Qwen3_5MoeSparseMoeBlock:
    def __init__(self, config):
        self.gate = Qwen3_5MoeTopKRouter(config)
        self.experts = Qwen3_5MoeExperts(config)
    def forward(self, hidden_states):
        _, routing_weights, selected_experts = self.gate(hidden_states)
        expert_output = self.experts(hidden_states, selected_experts, routing_weights)
        return expert_output
'''


def sample_config():
    return {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "num_hidden_layers": 40, "num_experts": 256,
            "num_experts_per_tok": 8, "hidden_size": 2048,
            "moe_intermediate_size": 512,
            "layer_types": ["linear_attention"] * 30 + ["full_attention"] * 10,
        },
    }


class PocketRotorE3PreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.checkpoint = self.root / "hf"
        self.checkpoint.mkdir()
        self.source = self.root / "modeling_qwen3_5_moe.py"
        self.source.write_text(SOURCE, encoding="utf-8")
        (self.checkpoint / "config.json").write_text(json.dumps(sample_config()))
        (self.checkpoint / "model-00001-of-00001.safetensors").write_bytes(b"fixture-not-real-weights")
        (self.checkpoint / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"model.layers.0.x": "model-00001-of-00001.safetensors"}})
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_read_only_positive_labels_are_not_proof_of_real_model(self):
        with patch.dict("sys.modules", {"torch": None, "transformers": None}):
            result = preflight(self.checkpoint, self.source)
        self.assertEqual(result["schema"], SCHEMA)
        self.assertEqual(result["status"], "STATIC_CONFIG_AND_SOURCE_MATCH_OPERATOR_REVIEW_ONLY")
        self.assertEqual(result["config"]["experts"], 256)
        self.assertEqual(result["config"]["top_k"], 8)
        self.assertEqual(result["local_weight_metadata"]["weight_file_count"], 1)
        self.assertFalse(result["source"]["module_hooks_tested_against_installed_runtime"])
        self.assertFalse(result["model_weights_read"])
        self.assertFalse(result["model_execution"])
        self.assertFalse(result["real_native_routing_captured"])
        self.assertFalse(result["cuda_queried"])

    def test_rejects_router_rewrite_or_tuple_contract_change(self):
        mutations = [
            SOURCE.replace("return router_logits, router_scores, router_indices",
                           "return router_logits, router_indices, router_scores"),
            SOURCE.replace("torch.topk(", "torch.sort("),
            SOURCE.replace("self.experts(hidden_states, selected_experts, routing_weights)",
                           "self.experts(hidden_states, routing_weights, selected_experts)"),
            SOURCE.replace("self.gate = Qwen3_5MoeTopKRouter(config)",
                           "self.gate = SomeOtherRouter(config)"),
            SOURCE.replace("def forward(self, hidden_states, top_k_index, top_k_weights)",
                           "def forward(self, hidden_states, indices, top_k_weights)"),
            SOURCE.replace("self.top_k = config.num_experts_per_tok",
                           "self.top_k = 1"),
        ]
        for idx, changed in enumerate(mutations):
            with self.subTest(idx=idx):
                with self.assertRaises(ValueError):
                    inspect_source(changed.encode("utf-8"))

    def test_invalid_config_is_denied(self):
        samples = [
            ("num_experts", 128),
            ("num_experts_per_tok", True),
            ("num_hidden_layers", 39),
            ("hidden_size", 1024),
            ("moe_intermediate_size", 2048),
        ]
        for key, value in samples:
            with self.subTest(key=key):
                config = sample_config()
                config["text_config"][key] = value
                with self.assertRaises(ValueError):
                    inspect_config(json.dumps(config).encode())
        for changed in (
            {"model_type": "qwen3_5_moe_text"},
            {"model_type": "llama", "text_config": sample_config()["text_config"]},
        ):
            with self.assertRaises(ValueError):
                inspect_config(json.dumps(changed).encode())

    def test_refuses_gguf_and_missing_shards(self):
        path = self.checkpoint / "model-00001-of-00001.safetensors"
        path.unlink()
        with self.assertRaisesRegex(ValueError, "No local Hugging Face"):
            preflight(self.checkpoint, self.source)
        (self.checkpoint / "model.gguf").write_bytes(b"GGUF")
        with self.assertRaisesRegex(ValueError, "GGUF checkpoint"):
            preflight(self.checkpoint, self.source)

    def test_unchanged_weight_metadata_cannot_authenticate_contents(self):
        target = self.checkpoint / "model-00001-of-00001.safetensors"
        before = preflight(self.checkpoint, self.source)
        target.write_bytes(b"different bytes but same size?")
        after = preflight(self.checkpoint, self.source)
        self.assertNotEqual(
            before["local_weight_metadata"]["weight_file_total_disk_bytes_unverified"],
            after["local_weight_metadata"]["weight_file_total_disk_bytes_unverified"])
        self.assertFalse(after["local_weight_metadata"]["weight_file_contents_hashed"])

    def test_index_mismatch_rejected(self):
        idx = self.checkpoint / "model.safetensors.index.json"
        idx.write_text(json.dumps({"weight_map": {"tensor": "missing-shard.safetensors"}}))
        with self.assertRaisesRegex(ValueError, "do not match"):
            preflight(self.checkpoint, self.source)

    def test_symlinked_source_and_shard_rejected(self):
        alias = self.root / "alias"
        try:
            alias.symlink_to(self.checkpoint, target_is_directory=True)
        except (NotImplementedError, OSError):
            self.skipTest("Symlinks unavailable")
        with self.assertRaises(ValueError):
            preflight(alias, self.source)
        shard = self.checkpoint / "model-00001-of-00001.safetensors"
        shard.unlink()
        shard.symlink_to(self.source)
        with self.assertRaisesRegex(ValueError, "Symlinked"):
            preflight(self.checkpoint, self.source)

    def test_bad_json_and_size_guard(self):
        config = self.checkpoint / "config.json"
        config.write_text('{"model_type":"one","model_type":"two"}')
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            preflight(self.checkpoint, self.source)
        config.write_text('{"ratio":NaN}')
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            preflight(self.checkpoint, self.source)
        config.write_bytes(b"x" * (512 * 1024 + 1))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            preflight(self.checkpoint, self.source)

    def test_cli_exclusive_and_stdout_no_side_effect(self):
        output = self.root / "report.json"
        args = ["pocketrotor_e3_preflight", "--checkpoint", str(self.checkpoint),
                "--hf-source", str(self.source), "--output", str(output)]
        original = (self.checkpoint / "config.json").read_bytes()
        with patch("sys.argv", args):
            main()
        self.assertEqual(json.loads(output.read_text())["schema"], SCHEMA)
        with patch("sys.argv", args):
            with self.assertRaises(FileExistsError):
                main()
        self.assertEqual(original, (self.checkpoint / "config.json").read_bytes())
        with patch("sys.argv", args[:-2]):
            with redirect_stdout(StringIO()) as stream:
                main()
        self.assertFalse(json.loads(stream.getvalue())["runtime_hook_execution_verified"])


if __name__ == "__main__":
    unittest.main()
