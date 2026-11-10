"""CLAVIS R0.3: wholly synthetic offline evidence consistency contracts.

These fixtures are invented protocol data, not observed model performance.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from phistream.clavis import ClavisError
from phistream.clavis_qualification import SCHEMA, _sign_test, qualify
from phistream.model_forge import load_examples
from phistream.nine_mind_bench import _digest, MANUAL_RUBRIC, PACK_SCHEMA, KEY_SCHEMA
from phistream.nine_mind_prompt_run import recipe
import hashlib


def synthetic_review(dataset: Path, *, count: int = 24, approved: bool = True,
                     three_arms: bool = False, graded: bool = True):
    """Build internally consistent invented Forge examples and model receipts."""
    examples = []
    for i in range(count + 1):
        examples.append({
            "schema": "phi.model-forge.example.v1",
            "id": f"phi-{i + 1:03d}",
            "split": "train" if i == 0 else "eval",
            "topic": "hardware",
            "source": "project-authored-fixture",
            "license": "CC0-1.0",
            "review_status": "human-approved" if approved else "seed-unreviewed",
            "messages": [
                {"role": "user", "content": f"Fake private hardware experiment {i}"},
                {"role": "assistant", "content": f"Mock reference answer for {i}"},
            ],
        })
    dataset.write_text("".join(json.dumps(row) + "\n" for row in examples), encoding="utf-8")
    original, digest = load_examples(dataset)
    evals = [row for row in original if row["split"] == "eval"]
    arms = ("base", "prompted", "adapter") if three_arms else ("base", "prompted")
    receipts = {}
    for arm in arms:
        receipt = {
            "schema": "phi.model-forge.holdout-output.v1",
            "model": "Qwen/Qwen3-4B-Instruct-2507",
            "model_revision": "a" * 40,
            "mode": arm,
            "dataset_sha256": digest,
            "outputs": [
                {"id": row["id"], "topic": row["topic"],
                 "prompt": row["messages"][0]["content"],
                 "output": f"SYNTHETIC SECRET {arm} output for {row['id']}"}
                for row in evals
            ],
            "adapter_sha256": "e" * 64 if arm == "adapter" else None,
            "review_status": "HUMAN_REVIEW_REQUIRED",
            "quality_proven": False,
            "deployment_authorized": False,
            "network_upload": False,
        }
        if arm == "prompted":
            receipt.update({
                "profile_id": "tesla",
                "recipe_sha256": hashlib.sha256(recipe("tesla").encode("utf-8")).hexdigest(),
                "source_class": "OPERATOR_ASSERTED_NOT_ATTESTED",
                "method_prompt_applied": True,
            })
        receipts[arm] = receipt
    mappings, rows = [], []
    for i, row in enumerate(evals):
        arrangement = list(arms) if i % 2 == 0 else list(reversed(arms))
        labels = {chr(65+j): arm for j, arm in enumerate(arrangement)}
        mappings.append({"id": row["id"], "labels": labels})
        answer = {
            label: receipts[arm]["outputs"][i]["output"]
            for label, arm in labels.items()
        }
        scores = {
            label: (None if not graded else {"base": 2, "prompted": 4, "adapter": 3}[arm])
            for label, arm in labels.items()
        }
        rows.append({
            "id": row["id"], "topic": row["topic"],
            "prompt": row["messages"][0]["content"],
            "answers": answer, "reviewer_scores": scores,
            "rubric": list(MANUAL_RUBRIC),
        })
    pack = {
        "schema": PACK_SCHEMA, "session": "f" * 32,
        "status": "HUMAN_REVIEW_REQUIRED",
        "dataset_sha256": digest, "arm_count": len(arms),
        "row_count": count, "rows": rows,
        "dataset_review_status": (
            "ALL_ROWS_HUMAN_APPROVED" if approved else "UNREVIEWED_MECHANICS_ONLY"
        ),
        "public_seed_status": "POSSIBLE_PUBLIC_EXPOSURE_NOT_INDEPENDENT_BENCHMARK",
        "quality_improvement_proven": False,
        "winner": None, "model_execution_by_this_tool": False,
        "production_authorized": False,
    }
    key = {
        "schema": KEY_SCHEMA, "session": "f" * 32,
        "dataset_sha256": digest, "model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "model_revision": "a" * 40, "profile_id_asserted": "tesla",
        "source_receipt_content_sha256": {arm: _digest(receipts[arm]) for arm in arms},
        "adapter_sha256_declared": "e" * 64 if three_arms else None,
        "prompt_recipe_sha256_declared": hashlib.sha256(
            recipe("tesla").encode("utf-8")
        ).hexdigest(),
        "mapping": mappings,
        "private_answer_key": True, "model_outputs_unattested": True,
        "quality_improvement_proven": False, "deployment_authorized": False,
    }
    return pack, key, receipts


class ClavisR03QualificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dataset = Path(self.tmp.name) / "private.jsonl"

    def test_reconciles_all_local_inputs_without_quality_claim(self):
        pack, key, receipts = synthetic_review(self.dataset)
        report = qualify(self.dataset, pack, key, receipts)
        self.assertEqual(report["schema"], SCHEMA)
        self.assertEqual(report["status"], "SCREENED_FOR_FURTHER_INDEPENDENT_REVIEW_ONLY")
        for item in ("dataset_sha256_verified_locally", "source_receipt_digests_verified_locally",
                     "blind_answers_matched_to_receipts"):
            self.assertTrue(report[item])
        self.assertEqual(report["heldout_rows"], 24)
        self.assertFalse(report["quality_proven"])
        self.assertFalse(report["routing_activated"])
        self.assertFalse(report["deployment_authorized"])
        self.assertFalse(report["model_execution_attested"])
        self.assertFalse(report["human_grading_attested"])
        stats = report["analysis"]
        self.assertEqual(stats["prompted_minus_base_mean_unverified"], 2.0)
        self.assertEqual(stats["paired_outcomes"]["prompted_higher"], 24)
        self.assertAlmostEqual(stats["exploratory_exact_two_sided_sign_p"], 0.00000012)
        self.assertFalse(stats["statistically_qualified"])
        serialized = json.dumps(report)
        self.assertNotIn("SYNTHETIC SECRET", serialized)
        self.assertNotIn("Fake private", serialized)
        self.assertNotIn('"mapping"', serialized)

    def test_holds_sample_floor_unapproved_and_ungraded(self):
        for kwargs, status in [
            ({"count": 8}, "HELD_R03_SCREENING_SAMPLE_FLOOR"),
            ({"approved": False}, "HELD_UNREVIEWED_DATASET"),
            ({"graded": False}, "HELD_UNGRADED_RESPONSES"),
        ]:
            with self.subTest(status=status):
                pack, key, receipts = synthetic_review(self.dataset, **kwargs)
                report = qualify(self.dataset, pack, key, receipts)
                self.assertEqual(report["status"], status)
                self.assertIsNone(report["analysis"])

    def test_three_arm_original_adapter_receipt_reconciliation(self):
        pack, key, receipts = synthetic_review(self.dataset, three_arms=True)
        report = qualify(self.dataset, pack, key, receipts)
        self.assertEqual(report["arm_count"], 3)
        self.assertEqual(report["analysis"]["prompted_minus_base_mean_unverified"], 2)
        self.assertFalse(report["deployment_authorized"])

    def test_tampered_receipt_or_pack_fails_closed(self):
        pack, key, receipts = synthetic_review(self.dataset)
        mutated = copy.deepcopy(receipts)
        mutated["base"]["outputs"][0]["output"] += " changed"
        with self.assertRaisesRegex(ClavisError, "digest"):
            qualify(self.dataset, pack, key, mutated)
        new_key = copy.deepcopy(key)
        new_key["source_receipt_content_sha256"]["base"] = _digest(mutated["base"])
        with self.assertRaisesRegex(ClavisError, "blinded answer"):
            qualify(self.dataset, pack, new_key, mutated)
        altered = copy.deepcopy(pack)
        altered["rows"][0]["prompt"] += " tampered"
        with self.assertRaisesRegex(ClavisError, "prompt/topic"):
            qualify(self.dataset, altered, key, receipts)
        with self.assertRaisesRegex(ClavisError, "receipt arms"):
            qualify(self.dataset, pack, key, {"base": receipts["base"]})
        changed_key = copy.deepcopy(key)
        changed_key["model_revision"] = "c" * 40
        with self.assertRaisesRegex(ClavisError, "identity"):
            qualify(self.dataset, pack, changed_key, receipts)

    def test_original_dataset_and_review_status_rechecked(self):
        pack, key, receipts = synthetic_review(self.dataset)
        changed = self.dataset.read_text().replace("Mock reference answer", "Changed reference answer")
        self.dataset.write_text(changed)
        with self.assertRaisesRegex(ClavisError, "dataset SHA"):
            qualify(self.dataset, pack, key, receipts)
        pack, key, receipts = synthetic_review(self.dataset)
        forged_pack = copy.deepcopy(pack)
        forged_pack["dataset_review_status"] = "UNREVIEWED_MECHANICS_ONLY"
        with self.assertRaisesRegex(ClavisError, "review status"):
            qualify(self.dataset, forged_pack, key, receipts)

    def test_exploratory_sign_statistic_boundaries(self):
        self.assertEqual(_sign_test(0, 0), 1)
        self.assertEqual(_sign_test(3, 3), 1)
        self.assertEqual(_sign_test(10, 0), 2 / (2 ** 10))
        self.assertEqual(_sign_test(0, 10), 2 / (2 ** 10))


if __name__ == "__main__":
    unittest.main()
