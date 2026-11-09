"""Offline Φ Model Atlas R0.1: candidates and *asserted* adapter-mix preflight.

Never downloads, loads, trains, merges, executes, or activates models.
GA108 owns specialist identities and Vessie's governed router retains authority.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

CATALOG_SCHEMA = "phistream.model-atlas.catalog.v0.1"
MIX_SCHEMA = "phistream.model-atlas.mix-proposal.v0.1"
MAX_DOCUMENT_BYTES = 128 * 1024
CATALOG_PATH = Path(__file__).resolve().parents[1] / "model_atlas" / "candidates.json"
MODEL_KEYS = {"id", "family", "modality", "license", "lane", "revision",
              "local_qualification", "url"}
PROFILE_KEYS = {"id", "council", "focus"}
ADAPTER_KEYS = {"profile_id", "base_model_id", "base_revision",
                "tokenizer_sha256", "adapter_sha256", "target_modules", "rank", "weight"}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
REVISION = re.compile(r"[0-9a-f]{40}\Z")
MODEL_ID = re.compile(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+\Z")
PROFILE_ID = re.compile(r"[a-z][a-z0-9-]{1,32}\Z")


class AtlasError(ValueError):
    """Input is malformed or cannot safely be considered for review."""


def _unique_dict(items):
    result = {}
    for key, val in items:
        if key in result:
            raise AtlasError("duplicate JSON key")
        result[key] = val
    return result


def read_json(path: Path) -> dict:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise AtlasError("regular local JSON file required; symlinks denied")
    if not 0 < path.stat().st_size <= MAX_DOCUMENT_BYTES:
        raise AtlasError("JSON document size outside allowed range")
    try:
        data = json.loads(path.read_text(encoding="utf-8"),
                          object_pairs_hook=_unique_dict,
                          parse_constant=lambda _: (_ for _ in ()).throw(AtlasError("nonfinite JSON value")))
    except (ValueError, UnicodeError) as exc:
        raise AtlasError("invalid JSON file") from exc
    if not isinstance(data, dict):
        raise AtlasError("JSON document must be an object")
    return data


def validate_catalog(data: dict) -> dict:
    required = {"schema", "reviewed_on", "status", "intended_host", "models", "profiles"}
    if set(data) != required or data.get("schema") != CATALOG_SCHEMA or data.get("status") != "CANDIDATES_ONLY":
        raise AtlasError("unexpected catalog schema or status")
    if not isinstance(data["reviewed_on"], str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", data["reviewed_on"]):
        raise AtlasError("invalid review date")
    if not isinstance(data["intended_host"], str) or not data["intended_host"].strip():
        raise AtlasError("missing intended host")
    models, profiles = data["models"], data["profiles"]
    if not isinstance(models, list) or not 1 <= len(models) <= 32:
        raise AtlasError("invalid model candidate count")
    if not isinstance(profiles, list) or len(profiles) != 9:
        raise AtlasError("Nine Mind requires exactly nine research profiles")
    model_ids, profile_ids, councils = set(), set(), {}
    for model in models:
        if not isinstance(model, dict) or set(model) != MODEL_KEYS:
            raise AtlasError("invalid model candidate fields")
        ident = model["id"]
        if not isinstance(ident, str) or not MODEL_ID.fullmatch(ident) or ident in model_ids:
            raise AtlasError("invalid or duplicate Hugging Face model identifier")
        model_ids.add(ident)
        if model["url"] != "https://huggingface.co/" + ident:
            raise AtlasError("candidate must link to its canonical Hugging Face page")
        if model["license"] not in {"apache-2.0", "mit"}:
            raise AtlasError("license must be explicitly reviewed")
        if model["family"] not in {"qwen3", "qwen3_5", "phi3"}:
            raise AtlasError("unreviewed architecture")
        if model["modality"] not in {"text", "image-text"}:
            raise AtlasError("unreviewed modality")
        if model["lane"] not in {"EXISTING_FORGE_PILOT", "ARCHITECTURE_REVIEW_REQUIRED"}:
            raise AtlasError("unreviewed training lane")
        if model["local_qualification"] != "UNVERIFIED":
            raise AtlasError("catalog cannot grant local qualification")
        if model["revision"] is not None and (
            not isinstance(model["revision"], str) or not REVISION.fullmatch(model["revision"])
        ):
            raise AtlasError("revision must be null or a pinned 40-character commit")
    for profile in profiles:
        if not isinstance(profile, dict) or set(profile) != PROFILE_KEYS:
            raise AtlasError("invalid profile fields")
        ident, council, focus = profile["id"], profile["council"], profile["focus"]
        if not isinstance(ident, str) or not PROFILE_ID.fullmatch(ident) or ident in profile_ids:
            raise AtlasError("invalid or duplicate profile identity")
        if council not in {"invention", "art", "music"} or not isinstance(focus, str) or not 5 <= len(focus) <= 200:
            raise AtlasError("invalid profile council or focus")
        profile_ids.add(ident)
        councils[council] = councils.get(council, 0) + 1
    if councils != {"invention": 3, "art": 3, "music": 3}:
        raise AtlasError("council must have exactly three profiles")
    return {"schema": "phistream.model-atlas.catalog-review.v0.1",
            "status": "CANDIDATES_ONLY", "candidate_count": len(model_ids),
            "profile_count": len(profile_ids), "councils": councils,
            "weights_downloaded": False, "model_loaded": False,
            "adapter_trained": False, "routing_activated": False}


def review_mix(data: dict, catalog: dict) -> dict:
    """Inspect *declared metadata* only, not checkpoint bytes or PEFT compatibility."""
    validate_catalog(catalog)
    if not isinstance(data, dict) or set(data) != {"schema", "base_model_id", "base_revision", "adapters"}:
        raise AtlasError("invalid mix-proposal fields")
    if data["schema"] != MIX_SCHEMA:
        raise AtlasError("unsupported mix-proposal schema")
    available = {item["id"]: item for item in catalog["models"]}
    profiles = {item["id"] for item in catalog["profiles"]}
    base, revision = data["base_model_id"], data["base_revision"]
    if not isinstance(base, str) or base not in available:
        raise AtlasError("base model not in review catalog")
    if not isinstance(revision, str) or not REVISION.fullmatch(revision):
        raise AtlasError("mix proposal must declare a pinned base revision")
    if available[base]["revision"] is not None and available[base]["revision"] != revision:
        raise AtlasError("base revision differs from reviewed catalog pin")
    adapters = data["adapters"]
    if not isinstance(adapters, list) or not 1 <= len(adapters) <= 4:
        raise AtlasError("GA108 research council may propose at most four adapters")
    ids, compatible_signature, total = set(), None, 0.0
    for adapter in adapters:
        if not isinstance(adapter, dict) or set(adapter) != ADAPTER_KEYS:
            raise AtlasError("invalid adapter metadata")
        ident = adapter["profile_id"]
        if not isinstance(ident, str) or ident not in profiles or ident in ids:
            raise AtlasError("unknown or repeated specialist")
        ids.add(ident)
        if adapter["base_model_id"] != base or adapter["base_revision"] != revision:
            raise AtlasError("all adapters must declare identical base ID and revision")
        if not isinstance(adapter["adapter_sha256"], str) or not SHA256.fullmatch(adapter["adapter_sha256"]):
            raise AtlasError("missing or invalid declared adapter SHA-256")
        if not isinstance(adapter["tokenizer_sha256"], str) or not SHA256.fullmatch(adapter["tokenizer_sha256"]):
            raise AtlasError("missing or invalid declared tokenizer SHA-256")
        modules = adapter["target_modules"]
        if (not isinstance(modules, list) or not modules or len(modules) > 32
                or any(not isinstance(x, str) or not re.fullmatch(r"[A-Za-z0-9_.]{1,80}", x) for x in modules)
                or len(set(modules)) != len(modules)):
            raise AtlasError("invalid adapter target modules")
        rank = adapter["rank"]
        if type(rank) is not int or not 1 <= rank <= 256:
            raise AtlasError("invalid LoRA rank")
        weight = adapter["weight"]
        if type(weight) not in (int, float) or not math.isfinite(weight) or not 0 <= weight <= 1:
            raise AtlasError("invalid mix weight")
        total += weight
        signature = (adapter["tokenizer_sha256"], tuple(sorted(modules)), rank)
        if compatible_signature is not None and compatible_signature != signature:
            raise AtlasError("declared adapter tokenizer/modules/rank mismatch")
        compatible_signature = signature
    if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-8):
        raise AtlasError("proposed routing percentages must total 1.0")
    return {"schema": "phistream.model-atlas.mix-review.v0.1",
            "status": "HELD_UNVERIFIED_REVIEW_ONLY",
            "base_model_id": base, "adapter_profiles": sorted(ids),
            "structural_metadata_consistent": True,
            "actual_weights_verified": False, "peft_merge_tested": False,
            "model_loaded": False, "routing_activated": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline PhiStream model/LoRA proposal review; never executes")
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--mix-plan", type=Path)
    args = parser.parse_args()
    try:
        catalog = read_json(args.catalog)
        report = review_mix(read_json(args.mix_plan), catalog) if args.mix_plan else validate_catalog(catalog)
        print(json.dumps(report, indent=2, sort_keys=True))
    except (AtlasError, OSError) as exc:
        parser.exit(2, "Model Atlas HELD: " + str(exc) + "\n")


if __name__ == "__main__":
    main()
