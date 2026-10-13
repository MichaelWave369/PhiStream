"""PR23 CPU-only synthetic batched-prefill candidate tests.

No pretrained checkpoint, GPU, network, user model or production switch.
"""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from phistream.llama import StreamingLlama
from phistream.prefill import (
    MAX_BATCH_PREFILL_TOKENS,
    approved_batched_prefill,
    estimated_extra_workspace_bytes,
)
from phistream.safeweights import LocalCheckpoint
from test_llama_execution import independent_batched_logits, local_llama


class BatchedPrefillCandidateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config, self.weights = local_llama(self.root)
        self.ckpt = LocalCheckpoint(self.root)
        self.policy = dict(capacity_mib=16, reserve_mib=1, max_tensor_mib=1, head_rows=7)

    def tearDown(self):
        self.temp.cleanup()

    def run_candidate(self, ckpt, ids):
        return approved_batched_prefill(
            ckpt, ids, approve_local_execution=True, device="cpu", **self.policy
        )

    def test_last_logit_and_kv_match_serial_and_independent_reference(self):
        for ids in ([1, 7], [9, 28, 7, 13, 4, 3], [42 % 32] * 12,
                    list(range(MAX_BATCH_PREFILL_TOKENS))):
            with self.subTest(prompt_length=len(ids)):
                baseline = StreamingLlama(self.ckpt, device="cpu", max_tensor_mib=1, head_rows=7)
                for token in ids:
                    serial_logits = baseline.step(token)
                batched, last_logits, receipt = self.run_candidate(self.ckpt, list(ids))
                torch.testing.assert_close(last_logits, serial_logits, atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(
                    last_logits, independent_batched_logits(
                        self.weights, self.config, list(ids))[-1],
                    atol=1e-5, rtol=1e-5,
                )
                self.assertEqual(batched.position, len(ids))
                for layer in range(self.config["num_hidden_layers"]):
                    self.assertEqual(len(batched.keys[layer]), len(ids))
                    self.assertEqual(len(batched.values[layer]), len(ids))
                    for j in range(len(ids)):
                        torch.testing.assert_close(
                            batched.keys[layer][j], baseline.keys[layer][j],
                            atol=1e-5, rtol=1e-5,
                        )
                        torch.testing.assert_close(
                            batched.values[layer][j], baseline.values[layer][j],
                            atol=1e-5, rtol=1e-5,
                        )
                self.assertLess(batched.weight_load_count, baseline.weight_load_count)
                self.assertLess(batched.weight_bytes_staged, baseline.weight_bytes_staged)
                self.assertEqual(receipt["reference_parity"], "NOT_ESTABLISHED_FOR_THIS_EXECUTION")
                self.assertFalse(receipt["production_promoted"])
                self.assertFalse(receipt["single_baseline_modified"])

                # Decode still uses exactly the original StreamingLlama.step().
                next_token = int(serial_logits.argmax().item())
                torch.testing.assert_close(
                    batched.step(next_token), baseline.step(next_token),
                    atol=1e-5, rtol=1e-5,
                )

    def test_future_token_cannot_leak_into_prior_kv(self):
        a, _, _ = self.run_candidate(self.ckpt, [1, 2, 3, 4])
        b, _, _ = self.run_candidate(self.ckpt, [1, 2, 13, 17])
        for layer in range(self.config["num_hidden_layers"]):
            for pos in (0, 1):
                torch.testing.assert_close(a.keys[layer][pos], b.keys[layer][pos],
                                           atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(a.values[layer][pos], b.values[layer][pos],
                                           atol=1e-6, rtol=1e-6)

    def test_tied_and_bf16_layout_without_reference_promotion(self):
        with tempfile.TemporaryDirectory() as path:
            cfg, weights = local_llama(Path(path), dtype="BF16", tied=True)
            ckpt = LocalCheckpoint(path)
            ids = [2, 5, 1, 7]
            _, last, evidence = self.run_candidate(ckpt, ids)
            torch.testing.assert_close(
                last, independent_batched_logits(weights, cfg, ids)[-1],
                atol=1e-5, rtol=1e-5,
            )
            self.assertIn("UNVERIFIED", evidence["status"])

    def test_must_opt_in_no_execution_without_approval(self):
        with patch("phistream.prefill.StreamingLlama", side_effect=AssertionError("no model")):
            with self.assertRaises(PermissionError):
                approved_batched_prefill(self.ckpt, [1, 2], **self.policy)
            with self.assertRaises(PermissionError):
                approved_batched_prefill(self.ckpt, [1, 2], approve_local_execution=1, **self.policy)

    def test_shape_context_and_id_validation_fails_before_execution(self):
        cases = [[], [1], [1, -1], [1, True], [1, 32],
                 [1] * (MAX_BATCH_PREFILL_TOKENS + 1), (1, 2)]
        with patch("phistream.prefill.StreamingLlama", side_effect=AssertionError("no decoder")):
            for tokens in cases:
                with self.subTest(tokens=str(tokens)[:32]):
                    with self.assertRaises(ValueError):
                        approved_batched_prefill(
                            self.ckpt, tokens, approve_local_execution=True, **self.policy
                        )

    def test_memory_policy_block_is_prior_to_engine_creation(self):
        with patch("phistream.prefill.StreamingLlama", side_effect=AssertionError("no decoder")):
            with self.assertRaises(MemoryError):
                approved_batched_prefill(
                    self.ckpt, [1, 2], approve_local_execution=True,
                    capacity_mib=0.001, reserve_mib=0,
                    max_tensor_mib=1, head_rows=7,
                )
        self.assertGreater(estimated_extra_workspace_bytes(
            StreamingLlama(self.ckpt, max_tensor_mib=1).cfg, 24
        ), 0)
        with self.assertRaises(ValueError):
            estimated_extra_workspace_bytes(
                StreamingLlama(self.ckpt, max_tensor_mib=1).cfg, 25
            )

    def test_no_cuda_fallback(self):
        with patch("phistream.prefill.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
                approved_batched_prefill(
                    self.ckpt, [1, 2], device="cuda",
                    approve_local_execution=True, **self.policy
                )


if __name__ == "__main__":
    unittest.main()
