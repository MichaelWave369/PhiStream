"""CPU-only regression tests for PhiStream's metadata-derived memory policy.

Uses tiny untrained safetensors fixtures. Assertions cover refusal behavior
and provenance, not GPU capacity, real model compatibility or speed.
"""
from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from phistream.llama import execute_local_tokens
from phistream.memory import memory_preflight, refuse_blocked, SCHEMA
from phistream.safeweights import LocalCheckpoint
from test_llama_execution import local_llama


class MemoryPreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        local_llama(self.root, dtype="BF16")
        self.checkpoint = LocalCheckpoint(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def plan(self, **kwargs):
        return memory_preflight(
            self.checkpoint,
            context_tokens=kwargs.pop("context_tokens", 16),
            capacity_mib=kwargs.pop("capacity_mib", 16.0),
            reserve_mib=kwargs.pop("reserve_mib", 1.0),
            max_tensor_mib=kwargs.pop("max_tensor_mib", 1),
            head_rows=kwargs.pop("head_rows", 7),
            **kwargs,
        )

    def test_estimate_does_not_claim_safe_gpu_fit(self):
        result = self.plan()
        self.assertEqual(result["schema"], SCHEMA)
        self.assertEqual(result["status"], "REVIEW_REQUIRED")
        self.assertTrue(result["no_fit_guarantee"])
        self.assertEqual(result["provenance"], "DERIVED_FROM_CHECKPOINT_METADATA_NOT_MEASURED")
        self.assertEqual(result["capacity_source"], "USER_SPECIFIED_SCENARIO")
        self.assertIsNone(result["cuda_observed_free_bytes"])
        self.assertEqual(len(result["layer_estimates"]), 2)
        self.assertGreater(result["estimated_working_set_bytes"], 0)
        json.dumps(result, allow_nan=False)

    def test_low_budget_blocked_by_policy(self):
        result = self.plan(capacity_mib=0.0001, reserve_mib=0.0)
        self.assertEqual(result["status"], "BLOCKED_BY_POLICY")
        self.assertIn("ESTIMATED_WORKING_SET_EXCEEDS_POLICY_BUDGET", result["blocked_reasons"])
        with self.assertRaisesRegex(MemoryError, "blocked execution"):
            refuse_blocked(result)

    def test_missing_cuda_free_capacity_can_reduce_budget(self):
        result = self.plan(observed_free_bytes=2048)
        self.assertEqual(result["capacity_source"], "USER_POLICY_CAPPED_BY_CUDA_FREE")
        self.assertEqual(result["effective_capacity_bytes"], 2048)
        self.assertEqual(result["status"], "BLOCKED_BY_POLICY")

    def test_fp16_weights_count_as_fp32_gpu_staging(self):
        result = self.plan()
        layer = result["layer_estimates"][0]
        self.assertGreater(layer["fp32_execution_weight_bytes"],
                           layer["checkpoint_weight_bytes"])
        self.assertEqual(result["execution_dtype"], "float32")
        self.assertGreater(result["kv_cache_fp32_bytes"], 0)

    def test_context_change_increases_kv_residency_estimate(self):
        short = self.plan(context_tokens=4)
        long = self.plan(context_tokens=48)
        self.assertGreater(long["kv_cache_fp32_bytes"], short["kv_cache_fp32_bytes"])
        self.assertGreater(long["estimated_working_set_bytes"],
                           short["estimated_working_set_bytes"])

    def test_invalid_inputs_rejected(self):
        for arguments in [
            {"capacity_mib": float("nan")}, {"capacity_mib": True},
            {"reserve_mib": -1}, {"context_tokens": 0},
            {"context_tokens": 65}, {"max_tensor_mib": 0},
            {"head_rows": 0}, {"observed_free_bytes": -2},
            {"observed_free_bytes": 7.5},
        ]:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.plan(**arguments)

    def test_tensor_cap_detected_from_checkpoint_metadata(self):
        # A large synthetic checkpoint isn't needed to exercise this rule.
        name = "model.layers.0.self_attn.q_proj.weight"
        orig = self.checkpoint.entries[name]
        from dataclasses import replace
        self.checkpoint.entries[name] = replace(orig, shape=(4096, 4096))
        # Deliberately mutate the metadata in this unit test to simulate a large
        # validly-sized layer, without allocating a giant real tensor on CI.
        from phistream.llama import LlamaSettings
        with patch("phistream.memory.LlamaSettings.from_checkpoint",
                   return_value=LlamaSettings.from_checkpoint(LocalCheckpoint(self.root))):
            result = self.plan(max_tensor_mib=1)
        self.assertIn("SINGLE_TENSOR_READ_CAP_EXCEEDED", result["blocked_reasons"])
        self.assertGreater(result["oversize_tensor_count"], 0)

    def test_execution_guard_blocks_before_weight_read(self):
        with patch("phistream.llama.StreamingLlama", side_effect=AssertionError("execution began")):
            with self.assertRaises(MemoryError):
                execute_local_tokens(
                    self.checkpoint, [1, 2, 3], 3,
                    capacity_mib=0.0001, reserve_mib=0,
                    max_tensor_mib=1,
                )

    def test_execution_guard_retains_provenance(self):
        receipt = execute_local_tokens(
            self.checkpoint, [1, 2, 3], 2,
            capacity_mib=16, reserve_mib=1, max_tensor_mib=1,
        )
        self.assertEqual(receipt["status"], "EXECUTED_UNVERIFIED_REFERENCE")
        self.assertEqual(receipt["memory_preflight"]["status"], "REVIEW_REQUIRED")
        self.assertTrue(receipt["memory_preflight"]["no_fit_guarantee"])

    def test_cli_plan_offline_and_strict_nonzero(self):
        from phistream.cli import main
        argv = [
            "phistream", "plan-memory", str(self.root), "--capacity-mib", "0.0001",
            "--context-tokens", "16", "--reserve-mib", "0",
        ]
        with patch("sys.argv", argv), redirect_stdout(StringIO()) as out:
            main()
        self.assertEqual(json.loads(out.getvalue())["status"], "BLOCKED_BY_POLICY")
        with patch("sys.argv", argv + ["--strict"]), redirect_stdout(StringIO()):
            with self.assertRaises(SystemExit) as e:
                main()
        self.assertEqual(e.exception.code, 2)

    def test_cli_plan_creates_local_receipt_without_overwrite(self):
        from phistream.cli import main
        output = self.root / "receipts" / "plan.json"
        argv = [
            "phistream", "plan-memory", str(self.root), "--capacity-mib", "16",
            "--context-tokens", "8", "--reserve-mib", "1",
            "--output", str(output),
        ]
        with patch("sys.argv", argv), redirect_stdout(StringIO()):
            main()
        self.assertEqual(json.loads(output.read_text())["status"], "REVIEW_REQUIRED")
        with patch("sys.argv", argv), redirect_stdout(StringIO()):
            with self.assertRaises(FileExistsError):
                main()

    def test_cli_cuda_preflight_fails_without_gpu(self):
        from phistream.cli import main
        with patch("phistream.cli.torch.cuda.is_available", return_value=False):
            with patch("sys.argv", [
                "phistream", "plan-memory", str(self.root),
                "--capacity-mib", "128", "--context-tokens", "4",
                "--device", "cuda",
            ]):
                with self.assertRaisesRegex(RuntimeError, "no CPU fallback"):
                    main()


if __name__ == "__main__":
    unittest.main()
