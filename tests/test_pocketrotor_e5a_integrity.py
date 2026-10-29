"""E5-A public pretrained checkpoint integrity gate, offline CPU-only tests.

These tests validate the pinned digest check with tiny scratch files. They
never load or download IBM Granite checkpoint weights or import Transformers.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.pocketrotor_e5_pilot import (
    EXPECTED_CHECKPOINT_SHA256, MODEL_ID, MODEL_REVISION,
    _sha_file, require_pinned_checkpoint_sha,
)


class PocketRotorE5AIntegrityTests(unittest.TestCase):
    def test_researched_digest_frozen_to_exact_public_revision(self):
        self.assertEqual(MODEL_ID, "ibm-granite/granite-3.0-1b-a400m-instruct")
        self.assertEqual(
            MODEL_REVISION, "23d6629dbac8c0a0652f28991f62bb75157c02e3")
        self.assertEqual(
            EXPECTED_CHECKPOINT_SHA256,
            "f7ae1cee56a9ea6c5360437b1c0407f8d84816b2cc75470f4e7e5236fa2a07dc")
        self.assertEqual(len(EXPECTED_CHECKPOINT_SHA256), 64)

    def test_exact_pinned_digest_is_accepted(self):
        self.assertIsNone(require_pinned_checkpoint_sha(EXPECTED_CHECKPOINT_SHA256))

    def test_any_modified_or_unverified_hash_fails_closed(self):
        bad = [
            "0" * 64, "", None, 0, True,
            EXPECTED_CHECKPOINT_SHA256.upper(),
            EXPECTED_CHECKPOINT_SHA256[:-1] + "0",
            EXPECTED_CHECKPOINT_SHA256 + " ",
        ]
        for idx, value in enumerate(bad):
            with self.subTest(idx=idx):
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    require_pinned_checkpoint_sha(value)

    def test_hash_reads_actual_contents_and_detects_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "small-synthetic.safetensors"
            target.write_bytes(b"fake-mock-weights-not-a-real-model")
            first = _sha_file(target)
            self.assertEqual(first, sha256(target.read_bytes()).hexdigest())
            self.assertNotEqual(first, EXPECTED_CHECKPOINT_SHA256)
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                require_pinned_checkpoint_sha(first)
            target.write_bytes(b"fake-mock-weights-not-a-real-mOdel")
            second = _sha_file(target)
            self.assertNotEqual(first, second)
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                require_pinned_checkpoint_sha(second)

    def test_no_symlink_checkpoint_hashing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = root / "synthetic.safetensors"
            alias = root / "alias.safetensors"
            original.write_bytes(b"synthetic")
            try:
                alias.symlink_to(original)
            except (OSError, NotImplementedError):
                self.skipTest("Symlinks unsupported")
            with self.assertRaisesRegex(ValueError, "symlinked"):
                _sha_file(alias)

    def test_no_unpinned_bypass_even_if_an_operator_changes_runtime_digest(self):
        # This is a pure test of the binding contract, not real model proof.
        with patch("scripts.pocketrotor_e5_pilot.EXPECTED_CHECKPOINT_SHA256",
                   "2" * 64):
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                require_pinned_checkpoint_sha(
                    "f7ae1cee56a9ea6c5360437b1c0407f8d84816b2cc75470f4e7e5236fa2a07dc")


if __name__ == "__main__":
    unittest.main()
