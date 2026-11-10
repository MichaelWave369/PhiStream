"""CLAVIS Φ369 R0.3: reconcile local Nine Mind evidence, not model execution.

Verifies the existing R0.5 blinded pack/key against a local reviewed Forge
dataset and the original local base/prompted/(optional adapter) receipts.
Human scores, run provenance, and checkpoint identity remain unverified.
This module never loads weights, trains, calls a service, or changes routing.
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path
import statistics

from .clavis import ClavisError
from .clavis_evidence import _load_private, _validate_pack_key, audit_blind_review
from .forge_review_pack import _check_receipt
from .model_forge import ForgeError, load_examples
from .nine_mind_bench import _digest, _check_adapter_provenance, _validate_prompted

SCHEMA = "phistream.clavis.qualification-review.v0.3"
MIN_EVAL = 24  # Explicit screening floor, NOT statistical power or validation.
MAX_EVAL = 256


def _refuse(message: str) -> None:
    # Generic error messages do not reveal private example IDs, prompts or files.
    raise ClavisError(message)


def _sign_test(wins: int, losses: int) -> float:
    """Exploratory exact two-sided binomial sign test, discarding ties.

    Assumes independent exchangeable paired comparisons under the null.
    That assumption has NOT been established for the supplied tasks/reviews.
    """
    total = wins + losses
    if total == 0:
        return 1.0
    tail = sum(comb(total, k) for k in range(min(wins, losses) + 1))
    return min(1.0, 2 * tail / (2 ** total))


def qualify(dataset: Path, pack: dict, key: dict, receipts: dict[str, dict]) -> dict:
    """Reconcile *content* and local file hashes, then screen human scores.

    Validates complete source receipts and exact answers, not merely their
    declared digest fields. The input is caller-controlled, thus this proves
    consistency of local files, NOT that a real checkpoint produced them.
    """
    profile, arms, mappings = _validate_pack_key(pack, key)
    if not isinstance(receipts, dict) or set(receipts) != set(arms):
        _refuse("exactly the declared source receipt arms are required")
    examples, dataset_sha = load_examples(dataset)
    if dataset_sha != pack["dataset_sha256"]:
        _refuse("original dataset SHA-256 mismatch")
    heldout = {row["id"]: row for row in examples if row["split"] == "eval"}
    if not heldout or len(heldout) != pack["row_count"]:
        _refuse("original held-out dataset rows mismatch")
    if pack["dataset_review_status"] != (
        "ALL_ROWS_HUMAN_APPROVED" if all(x["review_status"] == "human-approved" for x in examples)
        else "UNREVIEWED_MECHANICS_ONLY"
    ):
        _refuse("review status differs from original dataset rows")

    outputs: dict[str, dict[str, str]] = {}
    for arm in arms:
        receipt = receipts[arm]
        # Also validates matching prompt text, topic, row ID and output counts.
        _check_receipt(receipt, arm, heldout, dataset_sha)
        _check_adapter_provenance(receipt, arm)
        if arm == "prompted":
            _validate_prompted(receipt, heldout, dataset_sha, profile)
            if receipt["recipe_sha256"] != key["prompt_recipe_sha256_declared"]:
                _refuse("prompted source recipe fingerprint differs from private key")
        if receipt.get("model") != key["model_id"] or receipt.get("model_revision") != key["model_revision"]:
            _refuse("source model identity or revision mismatch")
        if _digest(receipt) != key["source_receipt_content_sha256"][arm]:
            _refuse("source receipt content digest differs from private key")
        if arm == "adapter" and receipt.get("adapter_sha256") != key["adapter_sha256_declared"]:
            _refuse("source adapter checksum declaration mismatch")
        outputs[arm] = {row["id"]: row["output"] for row in receipt["outputs"]}

    by_id = {row["id"]: row for row in pack["rows"]}
    if set(by_id) != set(heldout):
        _refuse("blinded row set differs from dataset")
    for entry in mappings:
        row_id = entry["id"]
        row = by_id[row_id]
        original = heldout[row_id]
        if row["prompt"] != original["messages"][0]["content"] or row["topic"] != original["topic"]:
            _refuse("blinded prompt/topic differs from dataset")
        for label, arm in entry["labels"].items():
            if row["answers"][label] != outputs[arm][row_id]:
                _refuse("blinded answer differs from matching source receipt")

    audited = audit_blind_review(pack, key)
    report = {
        "schema": SCHEMA,
        "status": "HELD",
        "profile_id": profile,
        "arm_count": len(arms),
        "heldout_rows": len(heldout),
        "minimum_screening_rows": MIN_EVAL,
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
        "analysis": None,
    }
    if audited["status"] != "CANDIDATE_FOR_INDEPENDENT_MANUAL_REVIEW":
        report["status"] = audited["status"]
        return report
    if len(heldout) < MIN_EVAL:
        report["status"] = "HELD_R03_SCREENING_SAMPLE_FLOOR"
        return report

    differences: list[float] = []
    counts = {"prompted_higher": 0, "base_higher": 0, "tie": 0}
    for entry in mappings:
        scores = by_id[entry["id"]]["reviewer_scores"]
        mapped = {arm: scores[label] for label, arm in entry["labels"].items()}
        diff = float(mapped["prompted"] - mapped["base"])
        differences.append(diff)
        if diff > 0:
            counts["prompted_higher"] += 1
        elif diff < 0:
            counts["base_higher"] += 1
        else:
            counts["tie"] += 1

    report["status"] = "SCREENED_FOR_FURTHER_INDEPENDENT_REVIEW_ONLY"
    report["analysis"] = {
        "source": "OPERATOR_SUPPLIED_UNATTESTED_MANUAL_SCORES",
        "prompted_minus_base_mean_unverified": round(statistics.mean(differences), 6),
        "prompted_minus_base_median_unverified": round(statistics.median(differences), 6),
        "paired_outcomes": counts,
        "exploratory_exact_two_sided_sign_p": round(
            _sign_test(counts["prompted_higher"], counts["base_higher"]), 8
        ),
        "null_assumption_verified": False,
        "independence_verified": False,
        "multiplicity_adjusted": False,
        "statistically_qualified": False,
        "quality_claim": "NONE",
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLAVIS R0.3 offline provenance reconciliation and exploratory paired review"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--prompted", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, help="required only for a three-arm blind pack")
    args = parser.parse_args()
    try:
        paths = [args.dataset, args.pack, args.key, args.base, args.prompted]
        if args.adapter is not None:
            paths.append(args.adapter)
        if len({p.expanduser().resolve(strict=True) for p in paths}) != len(paths):
            _refuse("all private inputs must be distinct files")
        # Same protections as R0.2: no symlinks, local files, size bounds,
        # no inputs inside the repo checkout.
        source = {
            "base": _load_private(args.base),
            "prompted": _load_private(args.prompted),
        }
        if args.adapter is not None:
            source["adapter"] = _load_private(args.adapter)
        from .nine_mind_seed import REPO_ROOT
        dataset = args.dataset.expanduser()
        resolved = dataset.resolve(strict=True)
        if dataset.is_symlink() or resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
            _refuse("private dataset must be outside the repository")
        result = qualify(dataset, _load_private(args.pack), _load_private(args.key), source)
    except (ClavisError, ForgeError, ValueError, OSError) as exc:
        parser.exit(2, "CLAVIS R0.3 HELD: " + str(exc) + "\n")
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
