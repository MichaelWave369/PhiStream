"""CLAVIS R0.5: synthetic calibration/holdout counterfactuals only."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from phistream.clavis import ClavisError
from phistream.clavis_adaptive import (
    SCHEMA, _adaptive_select, _learn, adaptive_shadow,
)
from phistream.clavis_shadow import MANIFEST_SCHEMA, _validate_manifest


def make_manifest(source="SYNTHETIC_FIXTURE", calibration=8, holdout=24):
    # Creative + visual trains preference for Dali; holdout + creative
    # compares Dali against R0.1's Picasso preference for equal coverage.
    cases = [
        {"id": f"cal-{i:03}", "split": "calibration", "mode": "creative",
         "needs": ["visual"], "reference_profiles": ["dali"]}
        for i in range(calibration)
    ]
    cases += [
        {"id": f"out-{i:03}", "split": "holdout", "mode": "creative",
         "needs": ["creative"], "reference_profiles": ["dali"]}
        for i in range(holdout)
    ]
    return {"schema": MANIFEST_SCHEMA, "source_class": source, "cases": cases}


def r03(profile="dali"):
    # Self-asserted mock R0.3 metadata; not authenticated.
    return {
        "schema": "phistream.clavis.qualification-review.v0.3",
        "status": "SCREENED_FOR_FURTHER_INDEPENDENT_REVIEW_ONLY",
        "profile_id": profile, "arm_count": 2,
        "heldout_rows": 24, "minimum_screening_rows": 24,
        "dataset_sha256_verified_locally": True,
        "source_receipt_digests_verified_locally": True,
        "blind_answers_matched_to_receipts": True,
        "model_and_recipe_declarations_consistent": True,
        "dataset_independence_attested": False,
        "human_grading_attested": False,
        "model_execution_attested": False,
        "checkpoint_bytes_attested": False,
        "latency_or_cost_measured": False,
        "quality_proven": False,
        "routing_activated": False,
        "deployment_authorized": False,
        "private_prompts_answers_or_mappings_exported": False,
        "analysis": {
            "source": "OPERATOR_SUPPLIED_UNATTESTED_MANUAL_SCORES",
            "prompted_minus_base_mean_unverified": 0.3,
            "prompted_minus_base_median_unverified": 0.5,
            "paired_outcomes": {"prompted_higher": 12, "base_higher": 8, "tie": 4},
            "exploratory_exact_two_sided_sign_p": 0.5,
            "null_assumption_verified": False,
            "independence_verified": False,
            "multiplicity_adjusted": False,
            "statistically_qualified": False,
            "quality_claim": "NONE",
        },
    }


class ClavisAdaptiveTests(unittest.TestCase):
    def test_calibrated_tiebreak_improves_label_agreement_not_quality(self):
        report = adaptive_shadow(make_manifest(
            source="OPERATOR_ASSERTED_REVIEW_NOT_ATTESTED"
        ), [r03()])
        self.assertEqual(report["schema"], SCHEMA)
        self.assertEqual(report["status"], "EXPLORATORY_ADAPTIVE_SHADOW_REVIEW_ONLY")
        m = report["holdout_budget_comparison"]["1"]
        self.assertEqual(m["fixed_r01"]["reference_exact_set_rate"], 0)
        self.assertEqual(m["adaptive_r05"]["reference_exact_set_rate"], 1)
        self.assertEqual(m["paired_delta"]["adaptive_higher_reference_recall"], 24)
        self.assertEqual(m["paired_delta"]["adaptive_minus_fixed_macro_recall"], 1)
        self.assertEqual(m["paired_delta"]["fixed_higher_reference_recall"], 0)
        self.assertEqual(report["mode_calibration_support"]["creative"]["cases"], 8)
        for name in (
            "data_independence_attested", "model_quality_proven",
            "statistically_qualified", "live_router_modified", "inference_executed",
            "training_weights_modified", "deployment_authorized",
            "holdout_labels_used_for_fitting",
        ):
            self.assertFalse(report[name], name)
        for private in ("cal-001", "out-001", '"reference_profiles"'):
            self.assertNotIn(private, json.dumps(report))

    def test_heldout_labels_do_not_affect_fitted_policy(self):
        a = make_manifest()
        b = copy.deepcopy(a)
        for case in b["cases"]:
            if case["split"] == "holdout":
                case["reference_profiles"] = ["picasso"]
        _, ca = _validate_manifest(a)
        _, cb = _validate_manifest(b)
        self.assertEqual(_learn(ca), _learn(cb))
        report_a = adaptive_shadow(a)
        report_b = adaptive_shadow(b)
        self.assertEqual(report_a["mode_calibration_support"],
                         report_b["mode_calibration_support"])
        self.assertEqual(report_a["holdout_budget_comparison"]["1"][
            "adaptive_r05"]["reference_exact_set_rate"], 1)
        self.assertEqual(report_b["holdout_budget_comparison"]["1"][
            "adaptive_r05"]["reference_exact_set_rate"], 0)

    def test_no_calibration_preserves_static_tiebreak(self):
        _, cases = _validate_manifest(make_manifest(calibration=0))
        counts, support = _learn(cases)
        selected, covered = _adaptive_select("creative", ("creative",), 1, counts, support)
        self.assertEqual(selected, {"picasso"})
        self.assertTrue(covered)
        short = make_manifest(calibration=7)
        _, short_cases = _validate_manifest(short)
        counts, support = _learn(short_cases)
        selected, _ = _adaptive_select("creative", ("creative",), 1, counts, support)
        self.assertEqual(selected, {"picasso"})
        self.assertEqual(adaptive_shadow(short)["status"], "SYNTHETIC_MECHANICS_ONLY")

    def test_coverage_is_never_sacrificed_for_calibrated_preference(self):
        data = make_manifest()
        _, cases = _validate_manifest(data)
        counts, support = _learn(cases)
        # Dali is preferred by calibration but has no "physics" capability.
        selected, covered = _adaptive_select(
            "creative", ("physics",), 1, counts, support
        )
        self.assertEqual(selected, {"feynman"})
        self.assertTrue(covered)
        selected, covered = _adaptive_select(
            "creative", ("physics", "symbolism"), 1, counts, support
        )
        self.assertFalse(covered)
        self.assertEqual(len(selected), 1)

    def test_synthetic_and_missing_evidence_cannot_promote(self):
        synthetic = adaptive_shadow(make_manifest(), [r03()])
        self.assertEqual(synthetic["status"], "SYNTHETIC_MECHANICS_ONLY")
        real_declared = adaptive_shadow(make_manifest(
            source="OPERATOR_ASSERTED_REVIEW_NOT_ATTESTED"
        ))
        self.assertEqual(real_declared["status"], "HELD_NO_R03_METADATA")
        short_cal = adaptive_shadow(make_manifest(
            source="OPERATOR_ASSERTED_REVIEW_NOT_ATTESTED", calibration=5
        ), [r03()])
        self.assertEqual(short_cal["status"], "HELD_INSUFFICIENT_CALIBRATION")
        short_hold = adaptive_shadow(make_manifest(
            source="OPERATOR_ASSERTED_REVIEW_NOT_ATTESTED", holdout=3
        ), [r03()])
        self.assertEqual(short_hold["status"], "HELD_INSUFFICIENT_HOLDOUT")

    def test_absent_holdout_and_unseen_mode_metrics(self):
        data = make_manifest(holdout=0)
        result = adaptive_shadow(data)
        self.assertIsNone(result["holdout_budget_comparison"]["2"][
            "fixed_r01"]["reference_macro_recall"])
        self.assertIsNone(result["holdout_budget_comparison"]["2"][
            "paired_delta"]["adaptive_minus_fixed_macro_recall"])
        self.assertFalse(result["mode_calibration_support"]["engineering"][
            "calibrated_tiebreak_active"])

    def test_rejects_bad_manifests_and_forged_r03_claims(self):
        data = make_manifest()
        invalid = [
            lambda m: m["cases"][0].update(prompt="secret"),
            lambda m: m["cases"][0].update(reference_profiles=[]),
            lambda m: m["cases"][1].update(id=m["cases"][0]["id"]),
            lambda m: m.update(source_class="INDEPENDENTLY_VERIFIED"),
            lambda m: m["cases"][0].update(needs=["unlisted"]),
        ]
        for mutate in invalid:
            changed = copy.deepcopy(data)
            mutate(changed)
            with self.assertRaises(ClavisError):
                adaptive_shadow(changed)
        bad = r03()
        bad["deployment_authorized"] = True
        with self.assertRaises(ClavisError):
            adaptive_shadow(data, [bad])
        with self.assertRaises(ClavisError):
            adaptive_shadow(data, [r03(), r03()])
        with self.assertRaises(ClavisError):
            adaptive_shadow(data, [r03("tesla")])

    def test_bounded_input_loader_is_offline(self):
        from phistream.clavis_evidence import _load_private
        with tempfile.TemporaryDirectory() as td:
            a = Path(td) / "manifest.json"
            a.write_text(json.dumps(make_manifest()), encoding="utf-8")
            self.assertEqual(adaptive_shadow(_load_private(a))["schema"], SCHEMA)
            bad = Path(td) / "duplicate.json"
            bad.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
            with self.assertRaises(ClavisError):
                _load_private(bad)


if __name__ == "__main__":
    unittest.main()
