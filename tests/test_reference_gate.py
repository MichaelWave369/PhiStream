"""Offline reference protocol tests with independently batched synthetic logits.

No downloaded checkpoints, no language models, and no inferred real-model
parity. All weights and traces are created locally during each test.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from phistream.reference import (
    TRACE_SCHEMA, checkpoint_fingerprint, compare_reference,
    read_reference_trace, validate_reference_trace,
)
from phistream.safeweights import LocalCheckpoint
from test_llama_execution import independent_batched_logits, local_llama


def make_trace(root: Path, cfg: dict, weights: dict, *, steps: int = 32) -> dict:
    ckpt = LocalCheckpoint(root)
    prompt = [1, 4, 3]
    ids = prompt.copy()
    next_ids, probes = [], []
    # These logits come from the independent full-sequence reference rather
    # than the PhiStream StreamingLlama decoder.
    for _ in range(steps):
        logits = independent_batched_logits(weights, cfg, ids)[-1]
        new_id = int(logits.argmax().item())
        next_ids.append(new_id)
        probes.append({
            str(new_id): float(logits[new_id].item()),
            "0": float(logits[0].item()),
            "1": float(logits[1].item()),
        })
        ids.append(new_id)
    return {
        "schema": TRACE_SCHEMA,
        "reference": {
            "runtime": "Independent full-sequence PyTorch test baseline",
            "version": "synthetic-v1", "execution_device": "cpu",
            "asserted_independent": True,
        },
        "checkpoint_fingerprint": checkpoint_fingerprint(ckpt),
        "prompt_token_ids": prompt,
        "expected_generated_token_ids": next_ids,
        "logit_probes": probes,
        "atol": 1e-5,
        "rtol": 1e-5,
    }


class ReferenceGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg, self.weights = local_llama(self.root)
        self.ckpt = LocalCheckpoint(self.root)
        self.trace = make_trace(self.root, self.cfg, self.weights)

    def tearDown(self):
        self.temp.cleanup()

    def test_synthetic_32_token_reference_match_is_unverified(self):
        receipt = compare_reference(self.ckpt, self.trace, max_tensor_mib=1, head_rows=7)
        self.assertEqual(receipt["status"], "MATCH_USER_REFERENCE_UNVERIFIED")
        self.assertEqual(receipt["exact_greedy_tokens_matched"], 32)
        self.assertEqual(receipt["steps"], 32)
        self.assertEqual(receipt["logit_probes_matched"], receipt["total_logit_probes"])
        self.assertFalse(receipt["independent_reference_authenticated"])
        self.assertEqual(receipt["logit_coverage"], "SPARSE_COORDINATES_ONLY")
        json.dumps(receipt, allow_nan=False)

    def test_too_short_never_claims_complete_candidate(self):
        short = dict(self.trace)
        short["expected_generated_token_ids"] = self.trace["expected_generated_token_ids"][:4]
        short["logit_probes"] = self.trace["logit_probes"][:4]
        receipt = compare_reference(self.ckpt, short, max_tensor_mib=1)
        self.assertEqual(receipt["status"], "MATCH_TOO_SHORT_USER_REFERENCE_UNVERIFIED")

    def test_wrong_expected_token_is_failure(self):
        wrong = dict(self.trace)
        wrong["expected_generated_token_ids"] = self.trace["expected_generated_token_ids"].copy()
        wrong["expected_generated_token_ids"][0] = (
            wrong["expected_generated_token_ids"][0] + 1
        ) % self.cfg["vocab_size"]
        receipt = compare_reference(self.ckpt, wrong, max_tensor_mib=1)
        self.assertEqual(receipt["status"], "MISMATCH")
        self.assertLess(receipt["exact_greedy_tokens_matched"], 32)

    def test_wrong_sampled_logit_is_failure(self):
        wrong = dict(self.trace)
        wrong["logit_probes"] = [dict(x) for x in self.trace["logit_probes"]]
        wrong["logit_probes"][4]["0"] += 0.5
        receipt = compare_reference(self.ckpt, wrong, max_tensor_mib=1)
        self.assertEqual(receipt["status"], "MISMATCH")
        self.assertLess(receipt["logit_probes_matched"], receipt["total_logit_probes"])
        self.assertGreater(receipt["max_abs_sampled_logit_error"], 0.4)

    def test_stale_fingerprint_fails_before_model_execution(self):
        wrong = dict(self.trace)
        wrong["checkpoint_fingerprint"] = dict(self.trace["checkpoint_fingerprint"])
        files = dict(wrong["checkpoint_fingerprint"]["files"])
        files["config.json"] = "a" * 64
        wrong["checkpoint_fingerprint"]["files"] = files
        with patch("phistream.reference.StreamingLlama", side_effect=AssertionError("execution forbidden")):
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                compare_reference(self.ckpt, wrong)

    def test_reference_assertion_is_required(self):
        wrong = dict(self.trace)
        wrong["reference"] = dict(self.trace["reference"])
        wrong["reference"]["asserted_independent"] = False
        with self.assertRaisesRegex(ValueError, "independent implementation"):
            validate_reference_trace(wrong, self.ckpt)

    def test_requires_all_shard_hashes(self):
        wrong = dict(self.trace)
        wrong["checkpoint_fingerprint"] = {
            "algorithm": "sha256", "files": {"config.json": "a" * 64}
        }
        with self.assertRaisesRegex(ValueError, "all weight shards"):
            validate_reference_trace(wrong, self.ckpt)

    def test_boolean_token_id_is_rejected(self):
        wrong = dict(self.trace)
        wrong["prompt_token_ids"] = [True, 1, 2]
        with self.assertRaisesRegex(ValueError, "prompt token"):
            validate_reference_trace(wrong, self.ckpt)

    def test_invalid_nonfinite_or_missing_probes_are_rejected(self):
        wrong = dict(self.trace)
        wrong["logit_probes"] = [dict(x) for x in self.trace["logit_probes"]]
        wrong["logit_probes"][0]["0"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_reference_trace(wrong, self.ckpt)
        wrong["logit_probes"][0] = {}
        with self.assertRaisesRegex(ValueError, "1-128"):
            validate_reference_trace(wrong, self.ckpt)

    def test_duplicate_json_keys_and_nonfinite_json_rejected(self):
        path = self.root / "reference.json"
        path.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate JSON"):
            read_reference_trace(path)
        path.write_text('{"logits":NaN}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            read_reference_trace(path)

    def test_local_trace_size_cap(self):
        path = self.root / "reference.json"
        with path.open("wb") as fh:
            fh.truncate(8 * 1024 * 1024 + 1)
        with self.assertRaisesRegex(ValueError, "8 MiB"):
            read_reference_trace(path)

    def test_reference_trace_is_not_signed(self):
        self.assertEqual(
            compare_reference(self.ckpt, self.trace, max_tensor_mib=1)["provenance"],
            "USER_SUPPLIED_REFERENCE_NOT_INDEPENDENTLY_AUTHENTICATED",
        )


if __name__ == "__main__":
    unittest.main()
