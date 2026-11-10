"""CLAVIS Φ369 R0.6: frozen private evaluation pilot gate, no model execution.

Reuses Nine Mind's approved local preflight and R0.3 receipt reconciliation.
Never downloads or loads models, authenticates GPU execution, grades model
outputs, trains routing weights, calls the network or promotes a route.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from .clavis import ClavisError
from .clavis_evidence import _load_private
from .clavis_qualification import MIN_EVAL, qualify
from .model_forge import ForgeError, load_examples
from .nine_mind_prompt_run import METHODS, prepare
from .nine_mind_seed import REPO_ROOT

SCHEMA = "phistream.clavis.pilot-protocol.v0.6"
REPORT_SCHEMA = "phistream.clavis.pilot-review.v0.6"
SOURCES = ("SYNTHETIC_FIXTURE", "OPERATOR_ASSERTED_PRIVATE_PILOT")
FAMILY_PATTERN = re.compile(r"[a-z][a-z0-9-]{3,59}\Z")
MIN_EVAL_FAMILIES = 4
MIN_EVAL_PER_FAMILY = 4
MAX_EVAL = 256


def _deny(reason: str) -> None:
    # Refusals avoid printing prompts, task IDs and local paths.
    raise ClavisError(reason)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _deny("duplicate JSON key in private study dataset")
        result[key] = value
    return result


def _private_dataset(path: Path) -> Path:
    """Bounded, non-symlink dataset outside checkout, no automatic creation."""
    source = Path(path).expanduser()
    resolved = source.resolve(strict=True)
    if source.is_symlink() or resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        _deny("private dataset must be a regular non-symlink file outside the repository")
    if not source.is_file() or not 0 < source.stat().st_size <= 2 * 1024 * 1024:
        _deny("invalid private dataset size or type")
    return source


def _check_original_jsonl(path: Path) -> None:
    """Extra R0.6 duplicate-key defense over Forge's existing row validator."""
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            json.loads(line, object_pairs_hook=_unique,
                       parse_constant=lambda _: _deny("nonfinite dataset JSON"))
        except (ValueError, TypeError) as exc:
            _deny("invalid private study dataset JSON")


def preflight(dataset: Path, protocol: dict, *, max_new_tokens: int = 128) -> dict:
    """Validate a declared frozen study BEFORE any separately approved GPU run."""
    if type(protocol) is not dict or set(protocol) != {
        "schema", "source_class", "profile_id", "dataset_sha256",
        "case_families", "reviewer_separation_declared",
        "holdout_frozen_before_generation_declared",
    } or protocol["schema"] != SCHEMA:
        _deny("unsupported R0.6 study protocol")
    source_class, profile = protocol["source_class"], protocol["profile_id"]
    if type(source_class) is not str or source_class not in SOURCES:
        _deny("invalid study source class")
    if type(profile) is not str or profile not in METHODS:
        _deny("unknown Nine Mind study profile")
    for key in ("reviewer_separation_declared", "holdout_frozen_before_generation_declared"):
        if type(protocol[key]) is not bool:
            _deny("study declarations must be explicit booleans")

    path = _private_dataset(dataset)
    _check_original_jsonl(path)
    examples, actual_digest = load_examples(path)
    if (type(protocol["dataset_sha256"]) is not str or
        not re.fullmatch(r"[0-9a-f]{64}", protocol["dataset_sha256"]) or
        protocol["dataset_sha256"] != actual_digest):
        _deny("study protocol and source dataset SHA-256 mismatch")

    entries = protocol["case_families"]
    if type(entries) is not list or len(entries) != len(examples):
        _deny("task family inventory must cover every source dataset row")
    case_map: dict[str, str] = {}
    for entry in entries:
        if type(entry) is not dict or set(entry) != {"id", "family"}:
            _deny("invalid task family row")
        ident, family = entry["id"], entry["family"]
        if (type(ident) is not str or type(family) is not str or
            FAMILY_PATTERN.fullmatch(family) is None or ident in case_map):
            _deny("invalid or duplicated task-family identity")
        case_map[ident] = family
    if set(case_map) != {row["id"] for row in examples}:
        _deny("task family inventory does not match Forge dataset IDs")

    families: dict[str, set[str]] = {"train": set(), "eval": set()}
    eval_counts: Counter = Counter()
    count_eval = count_train = 0
    for row in examples:
        family = case_map[row["id"]]
        split = row["split"]
        families[split].add(family)
        if split == "eval":
            count_eval += 1
            eval_counts[family] += 1
        else:
            count_train += 1
    if families["train"] & families["eval"]:
        _deny("task-family leakage across training and holdout splits")
    # Uses the existing Nine Mind pinned model, topic and human-review gate.
    existing = prepare(path, profile, max_new_tokens=max_new_tokens)
    if existing["dataset_sha256"] != actual_digest or existing["held_out_count"] != count_eval:
        _deny("Nine Mind preflight and frozen dataset disagree")
    if count_eval > MAX_EVAL:
        _deny("R0.6 heldout rows exceed the bounded R0.3 reviewer capacity")

    enough = (
        count_eval >= MIN_EVAL and
        len(eval_counts) >= MIN_EVAL_FAMILIES and
        all(n >= MIN_EVAL_PER_FAMILY for n in eval_counts.values())
    )
    if source_class == "SYNTHETIC_FIXTURE":
        status = "SYNTHETIC_MECHANICS_ONLY"
    elif not existing["dataset_all_rows_approved"]:
        status = "HELD_UNREVIEWED_SOURCE_EXAMPLES"
    elif not enough:
        status = "HELD_INSUFFICIENT_INDEPENDENT_FAMILY_COVERAGE"
    elif not protocol["reviewer_separation_declared"]:
        status = "HELD_REVIEWER_SEPARATION_NOT_DECLARED"
    elif not protocol["holdout_frozen_before_generation_declared"]:
        status = "HELD_HOLDOUT_FREEZE_NOT_DECLARED"
    else:
        status = "OPERATOR_REVIEW_ELIGIBLE_NO_EXECUTION"

    manifest_digest = hashlib.sha256(json.dumps(
        protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")).hexdigest()
    return {
        "schema": REPORT_SCHEMA,
        "status": status,
        "profile_id": profile,
        "source_class": source_class,
        "dataset_sha256_verified_locally": actual_digest,
        "protocol_sha256_locally": manifest_digest,
        "declared_model_id": existing["model"],
        "declared_revision": existing["model_revision"],
        "recipe_sha256_current_source": existing["recipe_sha256"],
        "train_rows": count_train,
        "heldout_rows": count_eval,
        "heldout_task_family_count": len(eval_counts),
        "min_holdout_rows": MIN_EVAL,
        "min_holdout_families": MIN_EVAL_FAMILIES,
        "min_rows_per_family": MIN_EVAL_PER_FAMILY,
        "all_rows_human_approved_declared": existing["dataset_all_rows_approved"],
        "reviewer_separation_operator_declared": protocol["reviewer_separation_declared"],
        "holdout_freeze_operator_declared": protocol["holdout_frozen_before_generation_declared"],
        "independent_review_attested": False,
        "holdout_isolation_attested": False,
        "checkpoint_bytes_verified": False,
        "tokenizer_bytes_verified": False,
        "gpu_fit_measured": False,
        "inference_executed": False,
        "model_quality_proven": False,
        "production_routing_authorized": False,
        "private_prompts_or_answers_exported": False,
    }


def review_completed_pilot(
    dataset: Path,
    protocol: dict,
    pack: dict,
    key: dict,
    source_receipts: dict[str, dict],
    *,
    max_new_tokens: int = 128,
) -> dict:
    """Reconcile existing locally generated outputs, without making them."""
    result = preflight(dataset, protocol, max_new_tokens=max_new_tokens)
    if result["status"] != "OPERATOR_REVIEW_ELIGIBLE_NO_EXECUTION":
        return {**result, "evidence_status": "HELD_PREFLIGHT",
                "r03_analysis": None, "inference_executed": False}
    if (type(key) is not dict or
        key.get("profile_id_asserted") != result["profile_id"] or
        key.get("dataset_sha256") != result["dataset_sha256_verified_locally"] or
        key.get("model_id") != result["declared_model_id"] or
        key.get("model_revision") != result["declared_revision"] or
        key.get("prompt_recipe_sha256_declared") != result["recipe_sha256_current_source"]):
        _deny("private blind key differs from the frozen R0.6 study")
    checked = qualify(Path(dataset), pack, key, source_receipts)
    if checked["status"] == "SCREENED_FOR_FURTHER_INDEPENDENT_REVIEW_ONLY":
        status = "SOURCE_RECEIPTS_RECONCILED_HUMAN_REVIEW_STILL_REQUIRED"
    else:
        status = "HELD_R03_SOURCE_OR_REVIEW_REQUIREMENTS"
    return {
        **result,
        "status": status,
        "evidence_status": checked["status"],
        "source_receipts_consistent_locally": checked["source_receipt_digests_verified_locally"],
        "blinded_answers_consistent_locally": checked["blind_answers_matched_to_receipts"],
        "r03_analysis": checked["analysis"] if status.startswith("SOURCE_RECEIPTS") else None,
        # The script can inspect already present receipts, not run the model.
        "inference_executed": False,
        "model_execution_independently_attested": False,
        "model_quality_proven": False,
        "production_routing_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLAVIS R0.6 local frozen-study preflight and receipt review; never runs models"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--pack", type=Path)
    parser.add_argument("--key", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--prompted", type=Path)
    parser.add_argument("--adapter", type=Path)
    args = parser.parse_args()
    try:
        optional = (args.pack, args.key, args.base, args.prompted)
        if any(x is not None for x in optional) or args.adapter is not None:
            if not all(x is not None for x in optional):
                _deny("completed review requires pack, key, base and prompted receipts")
        paths = [args.dataset, args.protocol, *(
            x for x in (*optional, args.adapter) if x is not None
        )]
        if len({path.expanduser().resolve(strict=True) for path in paths}) != len(paths):
            _deny("all private study files must be distinct")
        protocol = _load_private(args.protocol)
        if args.pack is None:
            result = preflight(args.dataset, protocol,
                               max_new_tokens=args.max_new_tokens)
        else:
            receipts = {"base": _load_private(args.base),
                        "prompted": _load_private(args.prompted)}
            if args.adapter is not None:
                receipts["adapter"] = _load_private(args.adapter)
            result = review_completed_pilot(
                args.dataset, protocol, _load_private(args.pack),
                _load_private(args.key), receipts,
                max_new_tokens=args.max_new_tokens,
            )
    except (ClavisError, ForgeError, OSError, ValueError) as exc:
        parser.exit(2, "CLAVIS R0.6 HELD: " + str(exc) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
