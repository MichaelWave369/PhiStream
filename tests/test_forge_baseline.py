"""Φ Forge Baseline R0.3: CPU-only, network-free evidence gate tests."""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

from phistream.forge_baseline import (
    PIN, BaselineError, check_scout_receipt, inspect_pin, review_baseline,
)
from phistream.model_atlas import CATALOG_PATH, read_json
from phistream.model_forge import DEFAULT_DATA


def scout_receipt(pin):
    return {
        "schema": "phistream.hf-model-scout.receipt.v0.2",
        "status": "UPSTREAM_METADATA_ONLY_HUMAN_REVIEW_REQUIRED",
        "candidate_count": 1,
        "weights_downloaded": False,
        "training_run": False,
        "inference_run": False,
        "routing_changed": False,
        "vessie_authority_granted": False,
        "observations": [{
            "model_id": pin["model_id"],
            "selected_revision": pin["revision"],
            "observed_model_card_license": pin["reported_license"],
            "license_matches_catalog": True,
            "architectures_reported": [pin["expected_architecture"]],
            "review_status": "REVIEW_REQUIRED",
            "model_weight_downloaded": False,
            "weights_authenticated": False,
            "hardware_fit_verified": False,
            "adapter_compatibility_verified": False,
            "quality_verified": False,
            "activation_authorized": False,
        }],
    }


class ForgeBaselineTests(unittest.TestCase):
    def setUp(self):
        self.pin = read_json(PIN)
        self.catalog = read_json(CATALOG_PATH)

    def test_offline_baseline_reports_only_next_gate(self):
        result = review_baseline(self.pin, self.catalog, dataset=DEFAULT_DATA)
        self.assertEqual(result["scout_receipt_status"], "SCOUT_RECEIPT_NOT_PROVIDED")
        self.assertEqual(result["status"], "HUMAN_REVIEW_REQUIRED")
        self.assertEqual(result["next_gate"], "DATASET_CURATION_AND_HUMAN_APPROVAL_REQUIRED")
        for key in ("model_downloaded","model_weights_verified","hardware_fit_verified",
                    "tokenizer_audit_completed","training_started","adapter_quality_proven",
                    "production_activation_authorized"):
            self.assertIs(result[key],False)

    def test_matching_scout_still_cannot_approve_training(self):
        result = review_baseline(
            self.pin,self.catalog,scout_receipt=scout_receipt(self.pin),dataset=DEFAULT_DATA
        )
        self.assertEqual(result["scout_receipt_status"],"MATCHED_DECLARED_METADATA_NOT_ATTESTED")
        self.assertFalse(result["training_started"])
        self.assertFalse(result["model_weights_verified"])

    def test_non_pinned_revision_rejected(self):
        bad=copy.deepcopy(self.pin)
        bad["revision"]="f"*40
        # A valid alternative SHA cannot be called invalid merely from the pin
        # unless a verified local scout observation disagrees.
        with self.assertRaisesRegex(BaselineError,"mismatch"):
            check_scout_receipt(scout_receipt(self.pin),bad)

    def test_invalid_sha_and_model_identity_rejected(self):
        bad=copy.deepcopy(self.pin)
        bad["revision"]="not-a-sha"
        with self.assertRaisesRegex(BaselineError,"commit SHA"):
            inspect_pin(bad,self.catalog)
        bad=copy.deepcopy(self.pin)
        bad["model_id"]="unknown/not-reviewed"
        with self.assertRaisesRegex(BaselineError,"existing Forge-pilot"):
            inspect_pin(bad,self.catalog)

    def test_model_card_license_mismatch_rejected(self):
        rec=scout_receipt(self.pin)
        rec["observations"][0]["observed_model_card_license"]="other"
        with self.assertRaisesRegex(BaselineError,"license"):
            check_scout_receipt(rec,self.pin)

    def test_missing_architecture_rejected(self):
        rec=scout_receipt(self.pin)
        rec["observations"][0]["architectures_reported"]=[]
        with self.assertRaisesRegex(BaselineError,"architecture"):
            check_scout_receipt(rec,self.pin)

    def test_access_hold_not_promoted(self):
        rec=scout_receipt(self.pin)
        rec["observations"][0]["review_status"]="HELD_ACCESS_REVIEW"
        with self.assertRaisesRegex(BaselineError,"held"):
            check_scout_receipt(rec,self.pin)

    def test_false_execution_or_authentication_not_admitted(self):
        rec=scout_receipt(self.pin)
        rec["observations"][0]["weights_authenticated"]=True
        with self.assertRaisesRegex(BaselineError,"review-only"):
            check_scout_receipt(rec,self.pin)
        rec=scout_receipt(self.pin)
        rec["training_run"]=True
        with self.assertRaisesRegex(BaselineError,"operational action"):
            check_scout_receipt(rec,self.pin)

    def test_metadata_pin_cannot_claim_training(self):
        bad=copy.deepcopy(self.pin)
        bad["training_approved"]=True
        with self.assertRaisesRegex(BaselineError,"cannot assert"):
            inspect_pin(bad,self.catalog)
        bad=copy.deepcopy(self.pin)
        bad["source_reported_not_independently_attested"]=False
        with self.assertRaisesRegex(BaselineError,"unattested"):
            inspect_pin(bad,self.catalog)

    def test_catalog_license_mismatch_rejected(self):
        bad=copy.deepcopy(self.catalog)
        bad["models"][0]["license"]="mit"
        with self.assertRaisesRegex(BaselineError,"license disagree"):
            inspect_pin(self.pin,bad)


if __name__ == "__main__":
    unittest.main()
