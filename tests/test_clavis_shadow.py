"""CLAVIS R0.4 shadow-routing experiments: no inference and no real results."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from phistream.clavis import ClavisError
from phistream.clavis_shadow import (
    MANIFEST_SCHEMA, REPORT_SCHEMA, _load_private, shadow_compare,
)

def manifest(count=24, source="OPERATOR_ASSERTED_REVIEW_NOT_ATTESTED"):
    """Categorical synthetic tests, not real human-reviewed routing outcomes."""
    cases = []
    for n in range(count):
        cases.append({
            "id": f"case-{n:03d}",
            "split": "holdout",
            "mode": "engineering",
            "needs": ["electrical", "physics", "prototype"],
            "reference_profiles": ["tesla", "feynman", "da-vinci"],
        })
    cases.append({
        "id": "calibration-one",
        "split": "calibration", "mode": "creative",
        "needs": ["recursion"],
        "reference_profiles": ["escher"],
    })
    return {"schema": MANIFEST_SCHEMA, "source_class": source, "cases": cases}


def r03(profile="tesla"):
    # Deliberately self-declared mock metadata, not actual qualification.
    return {
        "schema": "phistream.clavis.qualification-review.v0.3",
        "status": "SCREENED_FOR_FURTHER_INDEPENDENT_REVIEW_ONLY",
        "profile_id": profile,
        "arm_count": 2,
        "heldout_rows": 24,
        "minimum_screening_rows": 24,
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


class ClavisShadowTests(unittest.TestCase):
    def test_budget_counterfactuals_are_exact_and_no_execution(self):
        report = shadow_compare(manifest(), [r03()])
        self.assertEqual(report["schema"], REPORT_SCHEMA)
        self.assertEqual(report["status"],
                         "SHADOW_COMPARISON_FOR_INDEPENDENT_REVIEW_ONLY")
        self.assertEqual(report["r03_reconciled_profile_count_asserted"], 1)
        self.assertFalse(report["routing_activated"])
        self.assertFalse(report["inference_executed"])
        self.assertFalse(report["deployment_authorized"])
        self.assertFalse(report["reviewed_model_quality_proven"])
        self.assertFalse(report["reference_labels_independently_attested"])
        budgets = report["budget_comparison"]["holdout"]
        self.assertEqual(budgets["1"]["cases"], 24)
        self.assertEqual(budgets["1"]["reference_macro_recall"], 0.333333)
        self.assertEqual(budgets["1"]["reference_exact_set_rate"], 0)
        self.assertEqual(budgets["3"]["reference_macro_recall"], 1)
        self.assertEqual(budgets["3"]["reference_exact_set_rate"], 1)
        self.assertEqual(budgets["3"]["capability_abstentions"], 0)
        self.assertEqual(budgets["4"]["reference_exact_set_rate"], 1)
        self.assertEqual(report["budget_comparison"]["calibration"]["1"]["reference_any_hit_rate"], 1)
        private = json.dumps(report)
        self.assertNotIn("case-001", private)
        self.assertNotIn("calibration-one", private)
        self.assertNotIn('"reference_profiles"', private)

    def test_synthetic_never_qualifies_and_missing_r03_is_held(self):
        synthetic = shadow_compare(manifest(source="SYNTHETIC_FIXTURE"), [r03()])
        self.assertEqual(synthetic["status"], "SYNTHETIC_MECHANICS_ONLY")
        assert not synthetic["routing_activated"]
        missing = shadow_compare(manifest())
        self.assertEqual(missing["status"], "HELD_NO_R03_RECONCILIATION_METADATA")
        short = shadow_compare(manifest(count=3), [r03()])
        self.assertEqual(short["status"], "HELD_HOLDOUT_SAMPLE_FLOOR")

    def test_no_fabricated_quality_claim_even_if_reference_matches(self):
        report = shadow_compare(manifest(count=24), [r03()])
        self.assertEqual(report["budget_comparison"]["holdout"]["3"]["reference_exact_set_rate"], 1)
        self.assertFalse(report["reviewed_model_quality_proven"])
        self.assertFalse(report["data_independence_verified"])
        self.assertFalse(report["r03_report_contents_independently_authenticated"])

    def test_uncovered_capabilities_trigger_shadow_abstention(self):
        data = manifest(source="SYNTHETIC_FIXTURE", count=1)
        data["cases"][0]["needs"] = ["electrical", "physics", "prototype"]
        report = shadow_compare(data)
        self.assertEqual(report["budget_comparison"]["holdout"]["1"]["capability_abstentions"], 1)
        self.assertEqual(report["budget_comparison"]["holdout"]["3"]["capability_abstentions"], 0)

    def test_empty_split_has_null_not_misleading_zero_rate(self):
        data = manifest(count=1, source="SYNTHETIC_FIXTURE")
        data["cases"] = data["cases"][:1]
        report = shadow_compare(data)
        self.assertIsNone(report["budget_comparison"]["calibration"]["2"]["reference_macro_recall"])
        self.assertEqual(report["budget_comparison"]["calibration"]["2"]["cases"], 0)

    def test_rejects_bad_or_injected_manifests(self):
        original = manifest()
        mutations = [
            lambda d: d.update(schema="fake"),
            lambda d: d.update(source_class="REAL_EVIDENCE"),
            lambda d: d["cases"][0].update(prompt="secret"),
            lambda d: d["cases"][0].update(id="with private spaces"),
            lambda d: d["cases"][1].update(id=d["cases"][0]["id"]),
            lambda d: d["cases"][0].update(split="production"),
            lambda d: d["cases"][0].update(mode="run_tool"),
            lambda d: d["cases"][0].update(needs=["unknown"]),
            lambda d: d["cases"][0].update(needs=["physics", "physics"]),
            lambda d: d["cases"][0].update(reference_profiles=["unknown"]),
            lambda d: d["cases"][0].update(reference_profiles=[]),
            lambda d: d.update(cases=[]),
            lambda d: d.update(cases=d["cases"] * 5),
        ]
        for i, mutate in enumerate(mutations):
            with self.subTest(i=i):
                value = copy.deepcopy(original)
                mutate(value)
                with self.assertRaises(ClavisError):
                    shadow_compare(value)

    def test_rejects_unsafe_r03_claims_and_duplicates(self):
        data = manifest()
        cases = [
            lambda r: r.update(status="PROMOTED"),
            lambda r: r.update(quality_proven=True),
            lambda r: r.update(routing_activated=True),
            lambda r: r.update(model_execution_attested=True),
            lambda r: r.update(statistically_qualified=True),
            lambda r: r.update(heldout_rows=3),
            lambda r: r["analysis"].update(statistically_qualified=True),
            lambda r: r.update(profile_id="bach"),
            lambda r: r.update(source_receipt_digests_verified_locally=False),
        ]
        for i, mutate in enumerate(cases):
            with self.subTest(i=i):
                value = copy.deepcopy(r03())
                mutate(value)
                with self.assertRaises(ClavisError):
                    shadow_compare(data, [value])
        with self.assertRaisesRegex(ClavisError, "duplicate R0.3"):
            shadow_compare(data, [r03(), r03()])
        with self.assertRaises(ClavisError):
            shadow_compare(data, [r03()] * 10)

    def test_invalid_private_json_and_symlink_refused(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            manifest_file = base / "manifest.json"
            manifest_file.write_text(json.dumps(manifest()), encoding="utf-8")
            self.assertEqual(_load_private(manifest_file)["schema"], MANIFEST_SCHEMA)
            duplicate = base / "duplicates.json"
            duplicate.write_text('{"schema":"one","schema":"two"}', encoding="utf-8")
            with self.assertRaises(ClavisError):
                _load_private(duplicate)
            alias = base / "alias.json"
            try:
                alias.symlink_to(manifest_file)
            except (OSError, NotImplementedError):
                return
            with self.assertRaises(ClavisError):
                _load_private(alias)


if __name__ == "__main__":
    unittest.main()
