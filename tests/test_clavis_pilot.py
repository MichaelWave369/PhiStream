"""CLAVIS R0.6: fake protocol fixtures, not a physical GPU pilot."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from phistream.clavis import ClavisError
from phistream.clavis_pilot import SCHEMA, REPORT_SCHEMA, preflight, review_completed_pilot
from phistream.model_forge import load_examples
from phistream.nine_mind_bench import _digest, KEY_SCHEMA, MANUAL_RUBRIC, PACK_SCHEMA
from phistream.nine_mind_prompt_run import prepare, recipe


def create_fake_study(path: Path, *, source="OPERATOR_ASSERTED_PRIVATE_PILOT",
                      approved=True, count=24, groups=4):
    """All constructed records are synthetic, including approval assertions."""
    rows = []
    for n in range(count + 1):
        rows.append({
            "schema": "phi.model-forge.example.v1",
            "id": f"phi-{n+1:04d}", "split": "train" if n == 0 else "eval",
            "topic": "hardware", "source": "project-authored-fixture",
            "license": "CC0-1.0",
            "review_status": "human-approved" if approved else "seed-unreviewed",
            "messages": [
                {"role": "user", "content": f"SYNTHETIC private task {n}"},
                {"role": "assistant", "content": f"SYNTHETIC reference {n}"},
            ],
        })
    path.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
    examples, digest = load_examples(path)
    manifest = {
        "schema": SCHEMA,
        "source_class": source,
        "profile_id": "tesla",
        "dataset_sha256": digest,
        "case_families": [
            {"id": row["id"], "family": (
                "train-only" if n == 0 else f"holdout-family-{(n - 1) % groups}"
            )}
            for n, row in enumerate(examples)
        ],
        "reviewer_separation_declared": True,
        "holdout_frozen_before_generation_declared": True,
    }
    return manifest, examples


def fake_receipts_and_pack(path: Path, examples: list[dict]):
    plan = prepare(path, "tesla")
    evals = [x for x in examples if x["split"] == "eval"]
    receipts = {}
    for arm in ("base", "prompted"):
        rec = {
            "schema": "phi.model-forge.holdout-output.v1",
            "model": plan["model"],
            "model_revision": plan["model_revision"],
            "mode": arm,
            "dataset_sha256": plan["dataset_sha256"],
            "outputs": [
                {"id": row["id"], "topic": row["topic"],
                 "prompt": row["messages"][0]["content"],
                 "output": f"SYNTHETIC secret {arm} answer {n}"}
                for n, row in enumerate(evals)
            ],
            "adapter_sha256": None,
            "review_status": "HUMAN_REVIEW_REQUIRED",
            "quality_proven": False, "deployment_authorized": False,
            "network_upload": False,
        }
        if arm == "prompted":
            rec.update({
                "profile_id": "tesla", "recipe_sha256": plan["recipe_sha256"],
                "source_class": "OPERATOR_ASSERTED_NOT_ATTESTED",
                "method_prompt_applied": True,
            })
        receipts[arm] = rec
    pack_rows, mappings = [], []
    for n, row in enumerate(evals):
        labels = ({"A": "base", "B": "prompted"} if n % 2 == 0 else
                  {"A": "prompted", "B": "base"})
        pack_rows.append({
            "id": row["id"], "topic": row["topic"],
            "prompt": row["messages"][0]["content"],
            "answers": {
                label: receipts[arm]["outputs"][n]["output"]
                for label, arm in labels.items()
            },
            "reviewer_scores": {
                label: 4 if arm == "prompted" else 3
                for label, arm in labels.items()
            },
            "rubric": list(MANUAL_RUBRIC),
        })
        mappings.append({"id": row["id"], "labels": labels})
    pack = {
        "schema": PACK_SCHEMA, "session": "c" * 32,
        "status": "HUMAN_REVIEW_REQUIRED", "dataset_sha256": plan["dataset_sha256"],
        "arm_count": 2, "row_count": len(evals), "rows": pack_rows,
        "dataset_review_status": "ALL_ROWS_HUMAN_APPROVED",
        "public_seed_status": "POSSIBLE_PUBLIC_EXPOSURE_NOT_INDEPENDENT_BENCHMARK",
        "quality_improvement_proven": False, "winner": None,
        "model_execution_by_this_tool": False, "production_authorized": False,
    }
    key = {
        "schema": KEY_SCHEMA, "session": "c" * 32,
        "dataset_sha256": plan["dataset_sha256"],
        "model_id": plan["model"], "model_revision": plan["model_revision"],
        "profile_id_asserted": "tesla",
        "source_receipt_content_sha256": {arm: _digest(rec) for arm, rec in receipts.items()},
        "adapter_sha256_declared": None,
        "prompt_recipe_sha256_declared": plan["recipe_sha256"],
        "mapping": mappings, "private_answer_key": True,
        "model_outputs_unattested": True,
        "quality_improvement_proven": False, "deployment_authorized": False,
    }
    return pack, key, receipts


class ClavisPilotTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "study.jsonl"

    def test_approved_declared_study_is_only_operator_review_eligible(self):
        manifest, _ = create_fake_study(self.path)
        report = preflight(self.path, manifest)
        self.assertEqual(report["schema"], REPORT_SCHEMA)
        self.assertEqual(report["status"], "OPERATOR_REVIEW_ELIGIBLE_NO_EXECUTION")
        self.assertEqual(report["heldout_rows"], 24)
        self.assertEqual(report["heldout_task_family_count"], 4)
        self.assertEqual(len(report["declared_revision"]), 40)
        self.assertFalse(report["checkpoint_bytes_verified"])
        self.assertFalse(report["gpu_fit_measured"])
        self.assertFalse(report["inference_executed"])
        self.assertFalse(report["model_quality_proven"])
        self.assertFalse(report["production_routing_authorized"])
        self.assertNotIn("SYNTHETIC private", json.dumps(report))
        self.assertNotIn("holdout-family-0", json.dumps(report))

    def test_unreviewed_and_synthetic_states_remain_held(self):
        for opts, expected in [
            ({"approved": False}, "HELD_UNREVIEWED_SOURCE_EXAMPLES"),
            ({"source": "SYNTHETIC_FIXTURE"}, "SYNTHETIC_MECHANICS_ONLY"),
            ({"count": 20}, "HELD_INSUFFICIENT_INDEPENDENT_FAMILY_COVERAGE"),
            ({"groups": 6}, "HELD_INSUFFICIENT_INDEPENDENT_FAMILY_COVERAGE"),
        ]:
            with self.subTest(expected=expected):
                manifest, _ = create_fake_study(self.path, **opts)
                self.assertEqual(preflight(self.path, manifest)["status"], expected)
        manifest, _ = create_fake_study(self.path)
        manifest["reviewer_separation_declared"] = False
        self.assertEqual(preflight(self.path, manifest)["status"],
                         "HELD_REVIEWER_SEPARATION_NOT_DECLARED")
        manifest["reviewer_separation_declared"] = True
        manifest["holdout_frozen_before_generation_declared"] = False
        self.assertEqual(preflight(self.path, manifest)["status"],
                         "HELD_HOLDOUT_FREEZE_NOT_DECLARED")

    def test_family_overlap_and_hash_tamper_fail_closed(self):
        manifest, _ = create_fake_study(self.path)
        manifest["case_families"][1]["family"] = "train-only"
        with self.assertRaisesRegex(ClavisError, "leakage"):
            preflight(self.path, manifest)
        manifest, _ = create_fake_study(self.path)
        manifest["dataset_sha256"] = "0" * 64
        with self.assertRaisesRegex(ClavisError, "SHA"):
            preflight(self.path, manifest)
        manifest, _ = create_fake_study(self.path)
        manifest["case_families"][-1]["id"] = "phi-9999"
        with self.assertRaisesRegex(ClavisError, "inventory"):
            preflight(self.path, manifest)
        manifest, _ = create_fake_study(self.path)
        manifest["private_prompt"] = "should not be in protocol"
        with self.assertRaisesRegex(ClavisError, "schema"):
            preflight(self.path, manifest)

    def test_duplicate_json_keys_fail_closed(self):
        manifest, _ = create_fake_study(self.path)
        lines = self.path.read_text(encoding="utf-8").splitlines()
        lines[0] = lines[0].replace('"schema":', '"schema":"other","schema":', 1)
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ClavisError, "duplicate JSON"):
            preflight(self.path, manifest)

    def test_completed_review_reuses_r03_validation_without_claiming_execution(self):
        manifest, examples = create_fake_study(self.path)
        pack, key, receipts = fake_receipts_and_pack(self.path, examples)
        report = review_completed_pilot(self.path, manifest, pack, key, receipts)
        self.assertEqual(report["status"],
                         "SOURCE_RECEIPTS_RECONCILED_HUMAN_REVIEW_STILL_REQUIRED")
        self.assertEqual(report["evidence_status"],
                         "SCREENED_FOR_FURTHER_INDEPENDENT_REVIEW_ONLY")
        self.assertEqual(report["r03_analysis"]["prompted_minus_base_mean_unverified"], 1)
        self.assertFalse(report["model_execution_independently_attested"])
        self.assertFalse(report["inference_executed"])
        self.assertFalse(report["model_quality_proven"])
        self.assertNotIn("SYNTHETIC secret", json.dumps(report))
        self.assertNotIn('"mapping"', json.dumps(report))

    def test_private_key_mismatch_does_not_invoke_model(self):
        manifest, examples = create_fake_study(self.path)
        pack, key, receipts = fake_receipts_and_pack(self.path, examples)
        changed = copy.deepcopy(key)
        changed["profile_id_asserted"] = "feynman"
        with self.assertRaisesRegex(ClavisError, "frozen"):
            review_completed_pilot(self.path, manifest, pack, changed, receipts)
        changed = copy.deepcopy(key)
        changed["prompt_recipe_sha256_declared"] = "0" * 64
        with self.assertRaisesRegex(ClavisError, "frozen"):
            review_completed_pilot(self.path, manifest, pack, changed, receipts)
        changed_rec = copy.deepcopy(receipts)
        changed_rec["base"]["outputs"][0]["output"] = "tampered"
        with self.assertRaisesRegex(ClavisError, "digest"):
            review_completed_pilot(self.path, manifest, pack, key, changed_rec)

    def test_held_preflight_never_summarizes_fake_performance(self):
        manifest, examples = create_fake_study(self.path, source="SYNTHETIC_FIXTURE")
        pack, key, receipts = fake_receipts_and_pack(self.path, examples)
        report = review_completed_pilot(self.path, manifest, pack, key, receipts)
        self.assertEqual(report["status"], "SYNTHETIC_MECHANICS_ONLY")
        self.assertEqual(report["evidence_status"], "HELD_PREFLIGHT")
        self.assertIsNone(report["r03_analysis"])
        self.assertFalse(report["model_quality_proven"])


if __name__ == "__main__":
    unittest.main()
