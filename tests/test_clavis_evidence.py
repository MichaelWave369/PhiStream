"""CLAVIS R0.2 research bridge: synthetic contracts only, never real model evidence."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from phistream.clavis import ClavisError
from phistream.clavis_evidence import (
    BINDING_SCHEMA, SCHEMA, _load_private, audit_blind_review,
    method_bindings, review_route,
)
from phistream.nine_mind_bench import KEY_SCHEMA, MANUAL_RUBRIC, PACK_SCHEMA
from phistream.nine_mind_prompt_run import recipe


def mock_blind_rows(profile="tesla", count=8):
    """Entirely invented numbers: test the plumbing, not model quality."""
    session = "f" * 32
    rows, mapping = [], []
    for n in range(count):
        ident = f"mock-{n}"
        # Alternating positions prevents label A from being the prompted arm.
        labels = {"A": "prompted", "B": "base"} if n % 2 else {
            "A": "base", "B": "prompted",
        }
        scores = {label: 4 if arm == "prompted" else 2
                  for label, arm in labels.items()}
        rows.append({
            "id": ident,
            "topic": "synthetic",
            "prompt": f"PRIVATE MOCK PROMPT {n}",
            "answers": {"A": f"PRIVATE MOCK ANSWER A {n}",
                        "B": f"PRIVATE MOCK ANSWER B {n}"},
            "reviewer_scores": scores,
            "rubric": list(MANUAL_RUBRIC),
        })
        mapping.append({"id": ident, "labels": labels})
    pack = {
        "schema": PACK_SCHEMA, "session": session,
        "status": "HUMAN_REVIEW_REQUIRED", "dataset_sha256": "a" * 64,
        "arm_count": 2, "row_count": count, "rows": rows,
        "dataset_review_status": "ALL_ROWS_HUMAN_APPROVED",
        "public_seed_status": "POSSIBLE_PUBLIC_EXPOSURE_NOT_INDEPENDENT_BENCHMARK",
        "quality_improvement_proven": False, "winner": None,
        "model_execution_by_this_tool": False, "production_authorized": False,
    }
    key = {
        "schema": KEY_SCHEMA, "session": session,
        "dataset_sha256": "a" * 64,
        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "model_revision": "b" * 40,
        "profile_id_asserted": profile,
        "source_receipt_content_sha256": {"base": "c" * 64, "prompted": "d" * 64},
        "adapter_sha256_declared": None,
        "prompt_recipe_sha256_declared": hashlib.sha256(
            recipe(profile).encode("utf-8")
        ).hexdigest(),
        "mapping": mapping,
        "private_answer_key": True,
        "model_outputs_unattested": True,
        "quality_improvement_proven": False,
        "deployment_authorized": False,
    }
    return pack, key


class ClavisEvidenceBridgeTests(unittest.TestCase):
    def test_nine_mind_source_fingerprints_bound_without_inference(self):
        report = review_route("engineering", ["electrical", "physics", "prototype"])
        self.assertEqual(report["nine_mind_bindings"]["schema"], BINDING_SCHEMA)
        self.assertEqual(report["blind_review"], None)
        self.assertFalse(report["inference_authorized"])
        self.assertFalse(report["deployment_authorized"])
        bindings = report["nine_mind_bindings"]["specialist_bindings"]
        self.assertEqual({b["profile_id"] for b in bindings},
                         {"tesla", "feynman", "da-vinci"})
        for binding in bindings:
            self.assertEqual(
                binding["recipe_sha256"],
                hashlib.sha256(recipe(binding["profile_id"]).encode("utf-8")).hexdigest(),
            )
        self.assertFalse(report["nine_mind_bindings"]["model_loaded"])

    def test_audits_scored_review_without_proclaiming_winner(self):
        pack, key = mock_blind_rows()
        report = audit_blind_review(pack, key)
        self.assertEqual(report["schema"], SCHEMA)
        self.assertEqual(report["status"], "CANDIDATE_FOR_INDEPENDENT_MANUAL_REVIEW")
        scores = report["score_summary"]
        self.assertEqual(scores["unverified_average_score_by_arm"],
                         {"base": 2, "prompted": 4})
        self.assertEqual(scores["prompted_minus_base_average_unverified"], 2)
        self.assertEqual(scores["paired_prompted_vs_base_unverified"]["prompted_higher"], 8)
        self.assertFalse(report["model_quality_proven"])
        self.assertFalse(report["deployment_authorized"])
        self.assertFalse(report["model_execution_independently_attested"])
        exported = json.dumps(report)
        self.assertNotIn("PRIVATE MOCK", exported)
        self.assertNotIn("mapping", exported)
        self.assertNotIn("answers", exported)

    def test_unreviewed_short_and_ungraded_holds(self):
        original, key = mock_blind_rows()
        cases = [
            ("HELD_UNREVIEWED_DATASET", lambda p: p.update(
                dataset_review_status="UNREVIEWED_MECHANICS_ONLY")),
            ("HELD_INSUFFICIENT_HELDOUT_ROWS", lambda p: (
                p.update(row_count=1), p.update(rows=p["rows"][:1]))),
            ("HELD_UNGRADED_RESPONSES", lambda p: p["rows"][0][
                "reviewer_scores"].update(A=None)),
        ]
        for expected, mutate in cases:
            with self.subTest(expected=expected):
                pack = copy.deepcopy(original)
                mutate(pack)
                if expected == "HELD_INSUFFICIENT_HELDOUT_ROWS":
                    test_key = copy.deepcopy(key)
                    test_key["mapping"] = test_key["mapping"][:1]
                else:
                    test_key = key
                report = audit_blind_review(pack, test_key)
                self.assertEqual(report["status"], expected)
                self.assertIsNone(report["score_summary"])

    def test_detects_malformed_or_tampered_blinding_contract(self):
        original, original_key = mock_blind_rows()
        cases = [
            (lambda p, k: k.update(session="0" * 32), "session"),
            (lambda p, k: k.update(prompt_recipe_sha256_declared="0" * 64),
             "fingerprint"),
            (lambda p, k: p["rows"][0]["reviewer_scores"].update(A=True),
             "scores"),
            (lambda p, k: p["rows"][0]["reviewer_scores"].update(A=float("nan")),
             "scores"),
            (lambda p, k: k["mapping"][0]["labels"].update(A="prompted", B="prompted"),
             "permutation"),
            (lambda p, k: k["mapping"][0].update(id=k["mapping"][1]["id"]),
             "repeated"),
            (lambda p, k: p.update(quality_improvement_proven=True), "quality"),
            (lambda p, k: k.update(model_outputs_unattested=False), "unverified"),
            (lambda p, k: k.update(adapter_sha256_declared="e" * 64),
             "two-arm"),
            (lambda p, k: k["source_receipt_content_sha256"].update(base="bad"),
             "digest"),
        ]
        for mutate, message in cases:
            with self.subTest(message=message):
                pack, key = copy.deepcopy(original), copy.deepcopy(original_key)
                mutate(pack, key)
                with self.assertRaisesRegex(ClavisError, message):
                    audit_blind_review(pack, key)

    def test_key_profile_must_be_in_proposed_route(self):
        pack, key = mock_blind_rows("feynman")
        review = review_route("engineering", ["electrical"], pack, key)
        self.assertEqual(review["blind_review"]["status"],
                         "HELD_PROFILE_NOT_SELECTED_BY_PLAN")
        self.assertIsNone(review["blind_review"]["score_summary"])
        self.assertFalse(review["evidence_can_override_route"])

    def test_proposal_rejects_lone_pack_or_key(self):
        pack, key = mock_blind_rows()
        with self.assertRaisesRegex(ClavisError, "both"):
            review_route("engineering", ["electrical"], pack=pack)
        with self.assertRaisesRegex(ClavisError, "both"):
            review_route("engineering", ["electrical"], key=key)

    def test_only_local_bounded_private_json_read(self):
        pack, key = mock_blind_rows()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "blind.json"
            path.write_text(json.dumps(pack), encoding="utf-8")
            self.assertEqual(_load_private(path)["schema"], PACK_SCHEMA)
            bad = Path(td) / "duplicate.json"
            bad.write_text('{"schema": "x", "schema": "y"}', encoding="utf-8")
            with self.assertRaisesRegex(ClavisError, "invalid"):
                _load_private(bad)
            link = Path(td) / "alias.json"
            try:
                link.symlink_to(path)
            except (OSError, NotImplementedError):
                pass
            else:
                with self.assertRaisesRegex(ClavisError, "outside"):
                    _load_private(link)


if __name__ == "__main__":
    unittest.main()
