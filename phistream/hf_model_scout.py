"""Φ Model Atlas R0.2: opt-in Hugging Face metadata scout.

Never downloads files or model weights, loads tensors, trains, runs inference,
writes a registry, promotes a model, or modifies Vessie routing.
All model-card information is untrusted upstream metadata for human review.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import re
import sys

from phistream.model_atlas import AtlasError, CATALOG_PATH, read_json, validate_catalog

SHA = re.compile(r"[0-9a-f]{40}\Z")
SCHEMA = "phistream.hf-model-scout.receipt.v0.2"
MAX_CANDIDATES = 8
MAX_FILES = 4096
MAX_MODEL_FILES_BYTES = 1 << 54

class ScoutError(ValueError):
    """An upstream record is unavailable, malformed, or incompatible."""


def _text(value, maximum=200):
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or len(value) > maximum or any(ord(c) < 32 or ord(c) == 127 for c in value):
        return None
    return value


def _card_license(info):
    card = getattr(info, "card_data", None)
    if isinstance(card, dict):
        raw = card.get("license")
    else:
        raw = getattr(card, "license", None)
    return _text(raw, 80)


def inspect_info(candidate, info, requested_revision=None):
    """Normalize a bounded ModelInfo. No dynamic code, downloads, or network here."""
    if not isinstance(candidate, dict):
        raise ScoutError("invalid candidate")
    expected_id = candidate.get("id")
    observed_id = _text(getattr(info, "id", None), 160)
    if observed_id != expected_id:
        raise ScoutError("upstream repository identity differs from selected candidate")
    sha = getattr(info, "sha", None)
    if not isinstance(sha, str) or not SHA.fullmatch(sha):
        raise ScoutError("upstream revision has no valid 40-char commit SHA")
    if requested_revision is not None and sha != requested_revision:
        raise ScoutError("upstream revision differs from explicitly requested commit")
    if candidate.get("revision") is not None and candidate["revision"] != sha:
        raise ScoutError("upstream revision differs from pinned catalog revision")
    siblings = getattr(info, "siblings", None)
    if not isinstance(siblings, (list, tuple)) or not 0 < len(siblings) <= MAX_FILES:
        raise ScoutError("upstream file list absent or outside bounded range")

    all_bytes = 0
    known_sizes = 0
    safetensors_bytes = 0
    safetensors_count = 0
    risky_weight_count = 0
    for entry in siblings:
        name = _text(getattr(entry, "rfilename", None), 512)
        if name is None or name.startswith("/") or "\\" in name or ".." in name.split("/"):
            raise ScoutError("invalid upstream file name")
        size = getattr(entry, "size", None)
        if size is not None:
            if type(size) is not int or size < 0 or size > MAX_MODEL_FILES_BYTES:
                raise ScoutError("invalid upstream file size metadata")
            known_sizes += 1
            all_bytes += size
            if all_bytes > MAX_MODEL_FILES_BYTES:
                raise ScoutError("upstream aggregate file size too large")
        if name.endswith(".safetensors"):
            safetensors_count += 1
            if size is not None:
                safetensors_bytes += size
        if name.lower().endswith((".bin", ".pkl", ".pickle", ".pt", ".pth", ".ckpt")):
            risky_weight_count += 1

    card_license = _card_license(info)
    expected_license = candidate.get("license")
    license_match = card_license == expected_license if card_license is not None else None
    config = getattr(info, "config", None)
    if not isinstance(config, dict):
        config = {}
    architecture = config.get("architectures")
    if not isinstance(architecture, list) or not 0 < len(architecture) <= 8:
        architectures = []
    else:
        architectures = [a for a in (_text(x, 96) for x in architecture) if a]
    gated = getattr(info, "gated", None)
    private = getattr(info, "private", None)
    disabled = getattr(info, "disabled", None)
    restriction = (
        "HELD_LICENSE_CONFLICT" if license_match is False
        else "HELD_ACCESS_REVIEW" if bool(gated) or bool(private) or bool(disabled)
        else "REVIEW_REQUIRED"
    )
    return {
        "model_id": expected_id,
        "selected_revision": sha,
        "declared_catalog_license": expected_license,
        "observed_model_card_license": card_license,
        "license_matches_catalog": license_match,
        "author_reported": _text(getattr(info, "author", None), 160),
        "architectures_reported": architectures,
        "private_reported": bool(private),
        "gated_reported": bool(gated),
        "disabled_reported": bool(disabled),
        "repository_file_count": len(siblings),
        "repository_bytes_reported": all_bytes if known_sizes == len(siblings) else None,
        "repository_bytes_known_files_only": all_bytes,
        "file_sizes_complete": known_sizes == len(siblings),
        "safetensors_file_count": safetensors_count,
        "safetensors_bytes_known_files_only": safetensors_bytes,
        "potential_pickle_weight_file_count": risky_weight_count,
        "review_status": restriction,
        "model_weight_downloaded": False,
        "weights_authenticated": False,
        "hardware_fit_verified": False,
        "adapter_compatibility_verified": False,
        "quality_verified": False,
        "activation_authorized": False,
    }


def inspect_candidates(catalog, selected_ids, *, client, revisions=None):
    """Caller supplies an API client; no API instantiated by this pure function."""
    validate_catalog(catalog)
    models = {x["id"]: x for x in catalog["models"]}
    ids = list(selected_ids)
    if not ids or len(ids) > MAX_CANDIDATES or len(ids) != len(set(ids)):
        raise ScoutError("select one to eight distinct catalog models")
    if any(ident not in models for ident in ids):
        raise ScoutError("requested model not in reviewed catalog")
    revisions = revisions or {}
    for ident, pin in revisions.items():
        if ident not in ids or not isinstance(pin, str) or not SHA.fullmatch(pin):
            raise ScoutError("revisions must be valid pins for selected model IDs")
    observations = []
    for ident in ids:
        pin = revisions.get(ident) or models[ident].get("revision")
        # Deliberately read metadata, even file sizes, without a download call.
        info = client.model_info(
            repo_id=ident, revision=pin, timeout=12,
            files_metadata=True, token=False,
        )
        observations.append(inspect_info(models[ident], info, requested_revision=pin))
    return {
        "schema": SCHEMA,
        "status": "UPSTREAM_METADATA_ONLY_HUMAN_REVIEW_REQUIRED",
        "source": "HUGGING_FACE_HUB_MODEL_INFO",
        "candidate_count": len(observations),
        "observations": observations,
        "weights_downloaded": False,
        "training_run": False,
        "inference_run": False,
        "routing_changed": False,
        "vessie_authority_granted": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default=str(CATALOG_PATH),
                        help="existing locally reviewed Nine Mind candidate catalog")
    parser.add_argument("--inspect", action="store_true",
                        help="request remote metadata only; no model downloads")
    parser.add_argument("--allow-network-metadata", action="store_true",
                        help="explicit opt-in required with --inspect")
    parser.add_argument("--model", action="append",
                        help="catalog model ID to inspect; repeat up to eight times")
    parser.add_argument("--revision", help="40-character commit SHA; only with one --model")
    args = parser.parse_args(argv)
    try:
        catalog = read_json(args.catalog)
        validate_catalog(catalog)
        if not args.inspect:
            if args.allow_network_metadata or args.model or args.revision:
                raise ScoutError("use --inspect and --allow-network-metadata for an opt-in metadata request")
            print(json.dumps({"schema": SCHEMA, "status": "OFFLINE_CATALOG_ONLY",
                              "candidates": len(catalog["models"]),
                              "network_access": False, "weights_downloaded": False}, indent=2))
            return 0
        if not args.allow_network_metadata or not args.model:
            raise ScoutError("--inspect requires --allow-network-metadata and at least one --model")
        if args.revision and len(args.model) != 1:
            raise ScoutError("--revision requires exactly one --model")
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise ScoutError("optional huggingface_hub missing; install it separately") from exc
        # Explicit fixed public endpoint, no credential auto-discovery, no other hosts.
        api = HfApi(endpoint="https://huggingface.co", token=False)
        pins = {args.model[0]: args.revision} if args.revision else {}
        receipt = inspect_candidates(catalog, args.model, client=api, revisions=pins)
        receipt["observed_at_utc"] = datetime.now(timezone.utc).isoformat()
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    except (ScoutError, AtlasError, OSError, ValueError) as exc:
        parser.exit(2, "Phi Model Scout HELD: " + str(exc) + "\n")


if __name__ == "__main__":
    sys.exit(main())
