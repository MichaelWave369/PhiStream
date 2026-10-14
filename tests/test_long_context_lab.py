"""PR30 long context lab: pure synthetic metadata-only tests, never inference."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

from phistream.llama import LAYER_NAMES
from phistream.prefill import MAX_BATCH_PREFILL_TOKENS
from scripts.long_context_lab import SCHEMA, TARGETS, make_plan


class FakeCheckpoint:
    def __init__(self, context=2048):
        self.config = {
            "model_type": "llama", "architectures": ["LlamaForCausalLM"],
            "vocab_size": 64, "hidden_size": 16, "num_attention_heads": 4,
            "num_key_value_heads": 2, "num_hidden_layers": 2,
            "intermediate_size": 32, "max_position_embeddings": context,
        }
        self.entries = {}
        d, kvdim, hidden = 16, 8, 32
        shapes = {
            "input_layernorm.weight": (d,),
            "post_attention_layernorm.weight": (d,),
            "self_attn.q_proj.weight": (d, d),
            "self_attn.k_proj.weight": (kvdim, d),
            "self_attn.v_proj.weight": (kvdim, d),
            "self_attn.o_proj.weight": (d, d),
            "mlp.gate_proj.weight": (hidden, d),
            "mlp.up_proj.weight": (hidden, d),
            "mlp.down_proj.weight": (d, hidden),
        }
        for layer in range(2):
            for suffix in LAYER_NAMES:
                shape = shapes[suffix]
                size = 4
                for n in shape:
                    size *= n
                self.entries[f"model.layers.{layer}.{suffix}"] = SimpleNamespace(
                    shape=shape, nbytes=size,
                )

    def validate_llama_layout(self):
        return {"passed": True, "extra_tensors": 0}


class LongContextPlanTests(unittest.TestCase):
    def test_default_all_held_no_cuda_and_context_accounting(self):
        result = make_plan(FakeCheckpoint())
        self.assertEqual(result["schema"], SCHEMA)
        self.assertEqual(result["status"], "ALL_LONG_CONTEXT_TARGETS_HELD_PROTOCOL_ONLY")
        self.assertFalse(result["cuda_queried"])
        self.assertFalse(result["model_execution"])
        self.assertFalse(result["reference_generated"])
        self.assertFalse(result["production_promoted"])
        self.assertFalse(result["speedup_proven"])
        self.assertFalse(result["source_reference_authenticated"])
        self.assertEqual([v["prompt_tokens"] for v in result["targets"]], list(TARGETS))
        self.assertEqual([v["full_context_tokens_required"] for v in result["targets"]],
                         [159, 543, 2079])
        self.assertEqual(result["existing_batched_prefill_max_prompt_tokens"],
                         MAX_BATCH_PREFILL_TOKENS)
        for row in result["targets"]:
            self.assertEqual(row["status"], "HELD_PROTOCOL_ONLY_NO_EXECUTION")
            self.assertFalse(row["execution_approved"])
            self.assertFalse(row["existing_batched_kernel_supports_prompt"])
            self.assertFalse(row["benchmark_measured"])
            self.assertIsNone(result["policy"]["observed_cuda_free_bytes"])
        last = result["targets"][-1]
        self.assertIn("FULL_CONTEXT_EXCEEDS_CHECKPOINT_CONTEXT_LIMIT",
                      last["hold_reasons"])
        self.assertIsNone(last["memory_preflight"])

    def test_supported_model_context_does_not_enable_batch_kernel(self):
        result = make_plan(FakeCheckpoint(context=4096))
        row = result["targets"][-1]
        self.assertTrue(row["full_context_fits_checkpoint_limit"])
        self.assertIsNotNone(row["memory_preflight"])
        self.assertFalse(row["existing_batched_kernel_supports_prompt"])
        self.assertIn("NO_LONG_CONTEXT_BATCHED_IMPLEMENTATION", row["hold_reasons"])

    def test_policy_insufficient_adds_hold_and_never_authorizes(self):
        result = make_plan(FakeCheckpoint(), capacity_mib=1, reserve_mib=1)
        for row in result["targets"][:2]:
            self.assertEqual(row["memory_preflight"]["status"], "BLOCKED_BY_POLICY")
            self.assertIn("ESTIMATED_WORKING_SET_EXCEEDS_POLICY_BUDGET",
                          row["hold_reasons"])
            self.assertFalse(row["execution_approved"])

    def test_bad_budget_and_tensor_policy_fail_closed(self):
        for kwargs in [
            {"capacity_mib": 0},
            {"capacity_mib": float("nan")},
            {"capacity_mib": True},
            {"reserve_mib": -1},
            {"reserve_mib": float("inf")},
            {"max_tensor_mib": 0},
            {"max_tensor_mib": True},
            {"head_rows": 0},
            {"head_rows": 16385},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_plan(FakeCheckpoint(), **kwargs)

    def test_matrix_score_estimate_quadratic_not_peak_claim(self):
        report = make_plan(FakeCheckpoint(context=4096))
        first, _, third = report["targets"]
        self.assertEqual(first["theoretical_one_fp32_attention_score_matrix_bytes"],
                         4 * 128 * 128 * 4)
        self.assertEqual(third["theoretical_one_fp32_attention_score_matrix_bytes"],
                         first["theoretical_one_fp32_attention_score_matrix_bytes"] * 256)
        self.assertNotIn("run", report)
        self.assertEqual(report["checkpoint_binding_scope"],
                         "METADATA_LAYOUT_ONLY_NOT_SHA_BOUND_TO_WEIGHT_BYTES")


if __name__ == "__main__":
    unittest.main()
