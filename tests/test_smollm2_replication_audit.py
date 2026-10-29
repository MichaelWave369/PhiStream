"""CPU-only negative controls for read-only SmolLM2 replication receipt audit."""
from __future__ import annotations

import hashlib
import io
import zipfile
import unittest

from scripts.smollm2_replication_audit import (
    audit_bytes, approximately, check_pair, validate_archive,
)


def observation(method, *, matched=True, seconds=10.0, ttft=3.0):
    return {
        "status": "PARITY_MATCH_USER_REFERENCE_UNAUTHENTICATED",
        "greedy_matched": 32 if matched else 31, "greedy_total": 32,
        "logit_matched": 9, "logit_total": 9,
        "decoder_seconds": seconds, "time_to_first_logits_seconds": ttft,
        "comparisons": [
            {"step": i, "token_match": matched, "probe_match": True}
            for i in range(32)
        ],
    }


class ExternalReplicaAuditTests(unittest.TestCase):
    def test_recomputes_paired_gain_from_full_observations(self):
        g, t = check_pair({
            "methods": {
                "SINGLE": observation("SINGLE", seconds=10.0, ttft=3.0),
                "BATCHED": observation("BATCHED", seconds=8.0, ttft=0.6),
            }
        }, 9)
        self.assertAlmostEqual(g, 0.2)
        self.assertAlmostEqual(t, 0.8)

    def test_rejects_mismatched_tokens_and_unmatched_probes(self):
        for bad in (observation("SINGLE", matched=False),
                    {**observation("SINGLE"), "logit_matched": 8}):
            with self.subTest(bad=bad["greedy_matched"]):
                with self.assertRaisesRegex(ValueError, "parity"):
                    check_pair({
                        "methods": {
                            "SINGLE": bad,
                            "BATCHED": observation("BATCHED"),
                        }
                    }, 9)

    def test_rejects_invalid_timing(self):
        with self.assertRaisesRegex(ValueError, "timing"):
            check_pair({
                "methods": {
                    "SINGLE": observation("SINGLE", seconds=-1.0),
                    "BATCHED": observation("BATCHED"),
                }
            }, 9)

    def test_summary_rounded_precision(self):
        self.assertTrue(approximately(0.1449706424, 0.144971))
        self.assertFalse(approximately(0.15, 0.144971))
        self.assertFalse(approximately(0.1, "0.1"))

    def test_full_archive_sha_required(self):
        with self.assertRaisesRegex(ValueError, "Full expected SHA-256"):
            audit_bytes(b"not a zip", "5fde6fa6")
        with self.assertRaisesRegex(ValueError, "Archive SHA-256 mismatch"):
            audit_bytes(b"not a zip", "0" * 64)

    def test_refuses_zip_traversal_and_duplicate_entries(self):
        def make(entries):
            out = io.BytesIO()
            with zipfile.ZipFile(out, "w") as z:
                for path in entries:
                    z.writestr(path, b"{}")
            return out.getvalue()

        with zipfile.ZipFile(io.BytesIO(make([
            "matrix/manifest.json", "matrix/matrix-review.json", "../escape.json"
        ]))) as z:
            with self.assertRaisesRegex(ValueError, "Unsafe ZIP path"):
                validate_archive(z)

        with zipfile.ZipFile(io.BytesIO(make([
            "matrix/manifest.json", "matrix/matrix-review.json",
            "matrix/manifest.json"
        ]))) as z:
            with self.assertRaisesRegex(ValueError, "Duplicate ZIP entry"):
                validate_archive(z)


if __name__ == "__main__":
    unittest.main()
