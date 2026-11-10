"""CLAVIS Φ369 R0.2: offline Nine Mind evidence bridge, never production routing.

Binds existing categorical CLAVIS plans to current Nine Mind method fingerprints,
then audits privately graded R0.5 blind packs/answer keys. Metadata and scores
are self-asserted; this module never loads a model, executes tools, uploads
prompts, or authorizes deployment. Do not publish private pack/key files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

from .clavis import ClavisError, SPECIALISTS, plan
from .forge_review_pack import _load_receipt
from .model_atlas import CATALOG_PATH, read_json, validate_catalog
from .model_forge import ForgeError
from .nine_mind_bench import ARMS, KEY_SCHEMA, MANUAL_RUBRIC, PACK_SCHEMA
from .nine_mind_prompt_run import METHODS, recipe
from .nine_mind_seed import REPO_ROOT

SCHEMA = "phistream.clavis.evidence-review.v0.2"
BINDING_SCHEMA = "phistream.clavis.nine-mind-bindings.v0.2"
MIN_REVIEW_ROWS = 8
MAX_REVIEW_ROWS = 256
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
REVISION = re.compile(r"[0-9a-f]{40}\Z")
SESSION = re.compile(r"[0-9a-f]{32}\Z")
LABELS = ("A", "B", "C")


def _sha(value: object) -> bool:
    return type(value) is str and SHA256.fullmatch(value) is not None


def _catalog() -> dict:
    catalog = read_json(CATALOG_PATH)
    validate_catalog(catalog)
    ids = {item["id"] for item in catalog["profiles"]}
    if ids != set(SPECIALISTS) or ids != set(METHODS):
        raise ClavisError("Nine Mind profile registry drift; route held")
    return catalog


def method_bindings(route: dict) -> dict:
    """Read-only binding to exact current Nine Mind R0.6 prompt methods."""
    catalog = _catalog()
    if not isinstance(route, dict) or route.get("schema") != "phistream.clavis.route-plan.v0.1":
        raise ClavisError("unsupported CLAVIS route proposal")
    if route.get("gate", {}).get("inference_authorized") is not False or (
        route["gate"].get("operator_approval_required") is not True
    ):
        raise ClavisError("proposal must retain operator and inference holds")
    profiles = [p.get("profile_id") for p in route.get("specialists", [])]
    if not profiles or len(profiles) > 4 or len(set(profiles)) != len(profiles):
        raise ClavisError("invalid or empty proposed specialists")
    if any(p not in METHODS for p in profiles):
        raise ClavisError("unknown Nine Mind method")
    return {
        "schema": BINDING_SCHEMA,
        "status": "RESEARCH_BINDING_ONLY",
        "specialist_bindings": [
            {"profile_id": ident, "recipe_sha256": hashlib.sha256(
                recipe(ident).encode("utf-8")
            ).hexdigest(), "binding_type": "EXISTING_NINE_MIND_R06_METHOD"}
            for ident in profiles
        ],
        "catalog_schema": catalog["schema"],
        "profile_identity_assertion": "CURRENT_SOURCE_ONLY_NOT_RUNTIME_ATTESTATION",
        "prompts_exported": False,
        "model_loaded": False,
        "routing_activated": False,
    }


def _exact(value, fields: set[str], description: str) -> None:
    if not isinstance(value, dict) or set(value) != fields:
        raise ClavisError(f"unexpected {description} fields")


def _validate_pack_key(pack: dict, key: dict) -> tuple[str, tuple[str, ...], list[dict]]:
    """Check consistency and all privacy/provenance flags before any scoring."""
    _exact(pack, {
        "schema", "session", "status", "dataset_sha256", "arm_count",
        "row_count", "rows", "dataset_review_status", "public_seed_status",
        "quality_improvement_proven", "winner", "model_execution_by_this_tool",
        "production_authorized",
    }, "blind pack")
    _exact(key, {
        "schema", "session", "dataset_sha256", "model_id", "model_revision",
        "profile_id_asserted", "source_receipt_content_sha256",
        "adapter_sha256_declared", "prompt_recipe_sha256_declared",
        "mapping", "private_answer_key", "model_outputs_unattested",
        "quality_improvement_proven", "deployment_authorized",
    }, "private answer key")
    if pack["schema"] != PACK_SCHEMA or key["schema"] != KEY_SCHEMA:
        raise ClavisError("unsupported Nine Mind blind review schema")
    if (not isinstance(pack["session"], str) or not SESSION.fullmatch(pack["session"]) or
        pack["session"] != key["session"]):
        raise ClavisError("blind session/key mismatch")
    if not _sha(pack["dataset_sha256"]) or pack["dataset_sha256"] != key["dataset_sha256"]:
        raise ClavisError("dataset digest mismatch")
    if (pack["status"] != "HUMAN_REVIEW_REQUIRED" or
        pack["quality_improvement_proven"] is not False or pack["winner"] is not None or
        pack["model_execution_by_this_tool"] is not False or
        pack["production_authorized"] is not False):
        raise ClavisError("blind pack must not claim quality or deployment")
    if (key["private_answer_key"] is not True or key["model_outputs_unattested"] is not True or
        key["quality_improvement_proven"] is not False or
        key["deployment_authorized"] is not False):
        raise ClavisError("answer key must preserve unverified/private status")
    if pack["dataset_review_status"] not in (
        "ALL_ROWS_HUMAN_APPROVED", "UNREVIEWED_MECHANICS_ONLY"
    ) or pack["public_seed_status"] != "POSSIBLE_PUBLIC_EXPOSURE_NOT_INDEPENDENT_BENCHMARK":
        raise ClavisError("unsupported dataset provenance declarations")

    count = pack["arm_count"]
    if type(count) is not int or count not in (2, 3):
        raise ClavisError("expected two or three blind review arms")
    arms = ARMS[:count]
    labels = LABELS[:count]
    row_count = pack["row_count"]
    rows, mappings = pack["rows"], key["mapping"]
    if (type(row_count) is not int or not 1 <= row_count <= MAX_REVIEW_ROWS or
        not isinstance(rows, list) or len(rows) != row_count or
        not isinstance(mappings, list) or len(mappings) != row_count):
        raise ClavisError("blind row count/mapping mismatch")

    if (not isinstance(key["model_id"], str) or
        key["model_id"] not in {entry["id"] for entry in _catalog()["models"]} or
        not isinstance(key["model_revision"], str) or
        not REVISION.fullmatch(key["model_revision"])):
        raise ClavisError("unrecognized/unpinned Nine Mind base model")
    profile = key["profile_id_asserted"]
    if type(profile) is not str or profile not in METHODS:
        raise ClavisError("unknown Nine Mind profile")
    if key["prompt_recipe_sha256_declared"] != hashlib.sha256(
        recipe(profile).encode("utf-8")
    ).hexdigest():
        raise ClavisError("method fingerprint differs from current Nine Mind R0.6")
    hashes = key["source_receipt_content_sha256"]
    if not isinstance(hashes, dict) or set(hashes) != set(arms) or not all(
        _sha(v) for v in hashes.values()
    ):
        raise ClavisError("invalid source receipt digest declarations")
    if count == 2 and key["adapter_sha256_declared"] is not None:
        raise ClavisError("two-arm review cannot declare adapter weights")
    if count == 3 and not _sha(key["adapter_sha256_declared"]):
        raise ClavisError("three-arm review requires a declared adapter digest")

    seen = set()
    by_id = {}
    for row in rows:
        _exact(row, {"id", "topic", "prompt", "answers", "reviewer_scores", "rubric"}, "review row")
        identifier = row["id"]
        if (type(identifier) is not str or not 1 <= len(identifier) <= 128 or
            identifier in seen):
            raise ClavisError("repeated or invalid blind row ID")
        seen.add(identifier)
        if (type(row["topic"]) is not str or type(row["prompt"]) is not str or
            not isinstance(row["answers"], dict) or
            set(row["answers"]) != set(labels) or
            any(type(answer) is not str or len(answer) > 10000
                for answer in row["answers"].values())):
            raise ClavisError("invalid blind answers")
        if row["rubric"] != list(MANUAL_RUBRIC):
            raise ClavisError("changed scoring rubric")
        scores = row["reviewer_scores"]
        if not isinstance(scores, dict) or set(scores) != set(labels):
            raise ClavisError("review score labels do not match answers")
        for score in scores.values():
            if score is not None and (
                type(score) not in (int, float) or
                not math.isfinite(score) or not 0 <= score <= 5
            ):
                raise ClavisError("scores must be null or finite numeric 0..5")
        by_id[identifier] = row
    mapping_ids = set()
    for entry in mappings:
        _exact(entry, {"id", "labels"}, "private label mapping")
        ident = entry["id"]
        if type(ident) is not str or ident not in by_id or ident in mapping_ids:
            raise ClavisError("private mapping has missing/repeated row IDs")
        mapping_ids.add(ident)
        if (not isinstance(entry["labels"], dict) or
            set(entry["labels"]) != set(labels) or
            set(entry["labels"].values()) != set(arms) or
            any(type(v) is not str for v in entry["labels"].values())):
            raise ClavisError("private mapping must be an arm permutation")
    if mapping_ids != seen:
        raise ClavisError("incomplete private mapping")
    return profile, arms, mappings


def audit_blind_review(pack: dict, key: dict) -> dict:
    """Summarize *operator-entered* blinded scores; never certify quality."""
    profile, arms, mappings = _validate_pack_key(pack, key)
    report = {
        "schema": SCHEMA,
        "status": "HELD",
        "profile_id": profile,
        "model_id_declared": key["model_id"],
        "model_revision_declared": key["model_revision"],
        "arm_count": len(arms),
        "row_count": pack["row_count"],
        "minimum_review_rows": MIN_REVIEW_ROWS,
        "dataset_sha256_declared": pack["dataset_sha256"],
        "source_digest_verified_against_original_receipts": False,
        "human_review_independently_authenticated": False,
        "model_execution_independently_attested": False,
        "model_quality_proven": False,
        "automatic_winner": None,
        "model_loaded": False,
        "router_changed": False,
        "deployment_authorized": False,
        "private_prompts_or_answers_exported": False,
        "private_key_exported": False,
        "score_summary": None,
    }
    if pack["dataset_review_status"] != "ALL_ROWS_HUMAN_APPROVED":
        report["status"] = "HELD_UNREVIEWED_DATASET"
        return report
    if pack["row_count"] < MIN_REVIEW_ROWS:
        report["status"] = "HELD_INSUFFICIENT_HELDOUT_ROWS"
        return report
    if any(score is None for row in pack["rows"] for score in row["reviewer_scores"].values()):
        report["status"] = "HELD_UNGRADED_RESPONSES"
        return report

    rows = {row["id"]: row for row in pack["rows"]}
    totals = {arm: 0.0 for arm in arms}
    outcomes = {"prompted_higher": 0, "base_higher": 0, "tie": 0}
    for entry in mappings:
        graded = {
            arm: rows[entry["id"]]["reviewer_scores"][label]
            for label, arm in entry["labels"].items()
        }
        for arm, score in graded.items():
            totals[arm] += score
        if graded["prompted"] > graded["base"]:
            outcomes["prompted_higher"] += 1
        elif graded["prompted"] < graded["base"]:
            outcomes["base_higher"] += 1
        else:
            outcomes["tie"] += 1

    count = pack["row_count"]
    averages = {arm: round(totals[arm] / count, 6) for arm in arms}
    report["status"] = "CANDIDATE_FOR_INDEPENDENT_MANUAL_REVIEW"
    report["score_summary"] = {
        "unverified_average_score_by_arm": averages,
        "prompted_minus_base_average_unverified": round(
            (totals["prompted"] - totals["base"]) / count, 6
        ),
        "paired_prompted_vs_base_unverified": outcomes,
        "score_scale": "OPERATOR_ENTERED_SINGLE_VALUE_PER_ANSWER_0_TO_5",
        "quality_claim": "NONE_UNATTESTED_NONINDEPENDENT_EVIDENCE",
    }
    return report


def review_route(mode: str, needs: list[str] | tuple[str, ...],
                 pack: dict | None = None, key: dict | None = None) -> dict:
    """Combine existing CLAVIS plan, source-bound recipe IDs and optional review."""
    proposed = plan(mode, needs)
    bindings = method_bindings(proposed)
    if (pack is None) != (key is None):
        raise ClavisError("review requires both blind pack and private answer key")
    evidence = audit_blind_review(pack, key) if pack is not None else None
    if evidence is not None and evidence["profile_id"] not in {
        p["profile_id"] for p in proposed["specialists"]
    }:
        evidence["status"] = "HELD_PROFILE_NOT_SELECTED_BY_PLAN"
        evidence["score_summary"] = None
    return {
        "schema": "phistream.clavis.governed-research-bridge.v0.2",
        "plan": proposed,
        "nine_mind_bindings": bindings,
        "blind_review": evidence,
        "evidence_can_override_route": False,
        "external_authority_unchanged": True,
        "inference_authorized": False,
        "deployment_authorized": False,
    }


def _load_private(path: Path) -> dict:
    source = Path(path).expanduser()
    resolved = source.resolve(strict=True)
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents or source.is_symlink():
        raise ClavisError("private review inputs must be regular files outside the repository")
    try:
        return _load_receipt(source)
    except ForgeError as exc:
        raise ClavisError("invalid or oversized private review JSON") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline CLAVIS/Nine Mind fingerprint binding and private review; never executes models"
    )
    parser.add_argument("--mode", choices=("discovery", "engineering", "creative", "production"),
                        required=True)
    from .clavis import CAPABILITIES
    parser.add_argument("--need", choices=sorted(CAPABILITIES), action="append", required=True)
    parser.add_argument("--pack", type=Path, help="private, locally scored Nine Mind blind pack")
    parser.add_argument("--key", type=Path, help="private, separately stored Nine Mind answer key")
    args = parser.parse_args()
    try:
        if (args.pack is None) != (args.key is None):
            raise ClavisError("supply both --pack and --key")
        if args.pack is not None and args.pack.resolve() == args.key.resolve():
            raise ClavisError("pack and key must be distinct files")
        report = review_route(args.mode, args.need,
                              _load_private(args.pack) if args.pack else None,
                              _load_private(args.key) if args.key else None)
    except (ClavisError, ForgeError, ValueError, OSError) as exc:
        parser.exit(2, "CLAVIS R0.2 HELD: " + str(exc) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
