"""Φ Forge Baseline R0.3: deterministic model identity/dataset handoff review.

This module never performs a network request, model/tokenizer load, GPU probe,
training or publishing. A public Hugging Face metadata pin is not an authenticated
weight digest, even when a local Model Scout receipt matches it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from .hf_model_scout import SCHEMA as SCOUT_SCHEMA
from .model_atlas import CATALOG_PATH, AtlasError, read_json, validate_catalog
from .model_forge import DEFAULT_DATA, review as review_dataset

PIN = Path(__file__).resolve().parents[1] / "model_atlas" / "forge_baseline_r03.json"
SCHEMA = "phistream.forge-baseline-pin.v0.3"
REPORT = "phistream.forge-baseline-review.v0.3"
_SHA = re.compile(r"[0-9a-f]{40}\Z")
_ALLOWED = {"schema", "model_id", "revision", "reported_license", "expected_architecture",
            "observed_on", "source", "source_class", "source_reported_not_independently_attested",
            "model_weights_downloaded", "local_cuda_fit_verified", "training_approved"}


class BaselineError(ValueError):
    """Handoff input failed a bounded provenance or consistency check."""


def inspect_pin(pin: dict, catalog: dict) -> None:
    validate_catalog(catalog)
    if not isinstance(pin, dict) or set(pin) != _ALLOWED or pin.get("schema") != SCHEMA:
        raise BaselineError("invalid baseline pin schema")
    candidates = {m["id"]: m for m in catalog["models"]}
    ident, revision = pin["model_id"], pin["revision"]
    if ident not in candidates or candidates[ident]["lane"] != "EXISTING_FORGE_PILOT":
        raise BaselineError("baseline must use existing Forge-pilot candidate")
    if not isinstance(revision, str) or not _SHA.fullmatch(revision):
        raise BaselineError("pin must contain a 40-character Hugging Face commit SHA")
    candidate = candidates[ident]
    if candidate["revision"] is not None and candidate["revision"] != revision:
        raise BaselineError("pin differs from catalog's declared revision")
    if pin["reported_license"] != candidate["license"] or pin["reported_license"] != "apache-2.0":
        raise BaselineError("pin and catalog license disagree")
    if pin["expected_architecture"] != "Qwen3ForCausalLM":
        raise BaselineError("baseline architecture mismatch")
    if not isinstance(pin["observed_on"], str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}",pin["observed_on"]):
        raise BaselineError("invalid observation date")
    if pin["source"] != "https://huggingface.co/api/models/" + ident:
        raise BaselineError("invalid source URI")
    if pin["source_class"] != "PUBLIC_HF_API_METADATA_NOT_CHECKPOINT_VERIFICATION":
        raise BaselineError("invalid source evidence class")
    if any(pin[k] is not False for k in ("model_weights_downloaded","local_cuda_fit_verified","training_approved")):
        raise BaselineError("metadata pin cannot assert download, fit or approval")
    if pin["source_reported_not_independently_attested"] is not True:
        raise BaselineError("public metadata must be labeled unattested")


def check_scout_receipt(receipt: dict, pin: dict) -> str:
    if not isinstance(receipt, dict) or receipt.get("schema") != SCOUT_SCHEMA:
        raise BaselineError("unexpected model scout receipt schema")
    if receipt.get("status") != "UPSTREAM_METADATA_ONLY_HUMAN_REVIEW_REQUIRED":
        raise BaselineError("scout receipt is not read-only metadata")
    if any(receipt.get(k) is not False for k in
           ("weights_downloaded","training_run","inference_run","routing_changed","vessie_authority_granted")):
        raise BaselineError("scout receipt cannot claim an operational action")
    observations = receipt.get("observations")
    if not isinstance(observations,list) or len(observations) != 1 or receipt.get("candidate_count") != 1:
        raise BaselineError("require exactly one model observation")
    item = observations[0]
    if not isinstance(item,dict):
        raise BaselineError("invalid observation")
    if item.get("model_id") != pin["model_id"] or item.get("selected_revision") != pin["revision"]:
        raise BaselineError("scout identity/revision mismatch")
    if item.get("observed_model_card_license") != pin["reported_license"] or item.get("license_matches_catalog") is not True:
        raise BaselineError("scout license mismatch or unavailable")
    if pin["expected_architecture"] not in (item.get("architectures_reported") or []):
        raise BaselineError("scout architecture absent or mismatched")
    if item.get("review_status") != "REVIEW_REQUIRED":
        raise BaselineError("scout metadata is held for additional review")
    if any(item.get(k) is not False for k in
           ("model_weight_downloaded","weights_authenticated","hardware_fit_verified",
            "adapter_compatibility_verified","quality_verified","activation_authorized")):
        raise BaselineError("scout observation is not review-only")
    return "MATCHED_DECLARED_METADATA_NOT_ATTESTED"


def review_baseline(pin: dict, catalog: dict, *, scout_receipt: dict | None = None, dataset: Path = DEFAULT_DATA) -> dict:
    inspect_pin(pin, catalog)
    scout_status = "SCOUT_RECEIPT_NOT_PROVIDED"
    if scout_receipt is not None:
        scout_status = check_scout_receipt(scout_receipt, pin)
    dataset_receipt = review_dataset(dataset)
    if dataset_receipt["human_review_complete"]:
        next_gate = "TOKENIZER_AUDIT_AND_CUDA_PREFLIGHT_REQUIRED"
    else:
        next_gate = "DATASET_CURATION_AND_HUMAN_APPROVAL_REQUIRED"
    return {
        "schema": REPORT, "status": "HUMAN_REVIEW_REQUIRED",
        "model_id": pin["model_id"], "revision": pin["revision"],
        "source_class": pin["source_class"], "scout_receipt_status": scout_status,
        "dataset_sha256": dataset_receipt["dataset_sha256"],
        "dataset_status": dataset_receipt["status"],
        "training_examples": dataset_receipt["train_examples"],
        "held_out_examples": dataset_receipt["eval_examples"],
        "next_gate": next_gate,
        "model_downloaded": False, "model_weights_verified": False,
        "hardware_fit_verified": False, "tokenizer_audit_completed": False,
        "training_started": False, "adapter_quality_proven": False,
        "production_activation_authorized": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin",type=Path,default=PIN)
    parser.add_argument("--catalog",type=Path,default=CATALOG_PATH)
    parser.add_argument("--dataset",type=Path,default=DEFAULT_DATA)
    parser.add_argument("--scout-receipt",type=Path,default=None)
    args = parser.parse_args()
    try:
        pin = read_json(args.pin)
        catalog = read_json(args.catalog)
        scout_receipt = read_json(args.scout_receipt) if args.scout_receipt else None
        result = review_baseline(pin,catalog,scout_receipt=scout_receipt,dataset=args.dataset)
        print(json.dumps(result,indent=2,sort_keys=True))
    except (BaselineError,AtlasError,OSError,ValueError) as exc:
        parser.exit(2,"Forge Baseline HELD: "+str(exc)+"\n")


if __name__ == "__main__":
    main()
