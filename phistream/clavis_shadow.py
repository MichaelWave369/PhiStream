"""CLAVIS Φ369 R0.4: offline shadow-routing counterfactuals.

Compares existing R0.1 route *proposals* at 1..4 specialist budgets with
operator-supplied categorical reference labels, separated by calibration
and holdout. Neither labels nor optional R0.3 reports are independently
attested. No model inference, network, training or production routing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from .clavis import CAPABILITIES, MODES, SPECIALISTS, ClavisError, plan
from .clavis_evidence import _load_private
from .clavis_qualification import SCHEMA as QUAL_SCHEMA
from .nine_mind_seed import REPO_ROOT

MANIFEST_SCHEMA = "phistream.clavis.shadow-manifest.v0.4"
REPORT_SCHEMA = "phistream.clavis.shadow-report.v0.4"
SOURCE_CLASSES = ("SYNTHETIC_FIXTURE", "OPERATOR_ASSERTED_REVIEW_NOT_ATTESTED")
SPLITS = ("calibration", "holdout")
BUDGETS = (1, 2, 3, 4)
MIN_HOLDOUT = 24
MAX_CASES = 128
MAX_QUALIFICATIONS = 9
CASE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


def _reject(message: str) -> None:
    # Errors never echo user-supplied identifiers, prompts or local paths.
    raise ClavisError(message)


def _required(obj: object, keys: set[str], label: str) -> None:
    if type(obj) is not dict or set(obj) != keys:
        _reject(f"invalid {label} schema")


def _labels(value: object, valid: frozenset[str], *, maximum: int) -> tuple[str, ...]:
    if (type(value) is not list or not 1 <= len(value) <= maximum or
        any(type(v) is not str or v not in valid for v in value)):
        _reject("unknown or missing categorical labels")
    if len(set(value)) != len(value):
        _reject("duplicate categorical labels")
    return tuple(value)


def _validate_manifest(manifest: dict) -> tuple[str, list[dict]]:
    _required(manifest, {"schema", "source_class", "cases"}, "shadow manifest")
    if manifest["schema"] != MANIFEST_SCHEMA or manifest["source_class"] not in SOURCE_CLASSES:
        _reject("unsupported shadow manifest or source classification")
    cases = manifest["cases"]
    if type(cases) is not list or not 1 <= len(cases) <= MAX_CASES:
        _reject("shadow cases must contain 1..128 categorical items")
    seen: set[str] = set()
    checked: list[dict] = []
    for case in cases:
        _required(case, {"id", "split", "mode", "needs", "reference_profiles"}, "shadow case")
        if type(case["id"]) is not str or CASE_ID.fullmatch(case["id"]) is None or case["id"] in seen:
            _reject("invalid or duplicate shadow case ID")
        seen.add(case["id"])
        if type(case["split"]) is not str or case["split"] not in SPLITS:
            _reject("unknown shadow split")
        if type(case["mode"]) is not str or case["mode"] not in MODES:
            _reject("unknown shadow mode")
        needs = _labels(case["needs"], CAPABILITIES, maximum=12)
        references = _labels(case["reference_profiles"], frozenset(SPECIALISTS), maximum=4)
        # Only bounded, operator-declared labels. No chat text, names, paths or tool calls.
        checked.append({
            "split": case["split"],
            "mode": case["mode"],
            "needs": needs,
            "reference_profiles": references,
        })
    return manifest["source_class"], checked


def _validate_qualification(report: dict, reference_profiles: set[str]) -> str:
    """R0.3 report is an *untrusted metadata declaration*, never authorization."""
    if type(report) is not dict or report.get("schema") != QUAL_SCHEMA:
        _reject("unsupported R0.3 qualification report")
    profile = report.get("profile_id")
    if type(profile) is not str or profile not in reference_profiles:
        _reject("R0.3 profile is outside the reference scope")
    if report.get("status") != "SCREENED_FOR_FURTHER_INDEPENDENT_REVIEW_ONLY":
        _reject("R0.3 source reconciliation is not in screened state")
    if type(report.get("heldout_rows")) is not int or report["heldout_rows"] < 24:
        _reject("R0.3 minimum sample hold missing")
    must_be_true = (
        "dataset_sha256_verified_locally",
        "source_receipt_digests_verified_locally",
        "blind_answers_matched_to_receipts",
        "model_and_recipe_declarations_consistent",
    )
    must_be_false = (
        "dataset_independence_attested", "human_grading_attested",
        "model_execution_attested", "checkpoint_bytes_attested",
        "latency_or_cost_measured", "quality_proven",
        "routing_activated", "deployment_authorized",
        "private_prompts_answers_or_mappings_exported",
    )
    if any(report.get(field) is not True for field in must_be_true):
        _reject("R0.3 report missing local consistency declarations")
    if any(report.get(field) is not False for field in must_be_false):
        _reject("R0.3 report makes unsupported attestation or authorization claim")
    analysis = report.get("analysis")
    if (type(analysis) is not dict or
        analysis.get("source") != "OPERATOR_SUPPLIED_UNATTESTED_MANUAL_SCORES" or
        analysis.get("statistically_qualified") is not False or
        analysis.get("quality_claim") != "NONE"):
        _reject("R0.3 statistical evidence status is incompatible")
    return profile


def _summarize(results: list[tuple[set[str], set[str], bool]]) -> dict:
    if not results:
        return {
            "cases": 0,
            "capability_abstentions": 0,
            "reference_any_hit_rate": None,
            "reference_exact_set_rate": None,
            "reference_macro_recall": None,
            "reference_macro_precision": None,
        }
    hits = 0
    exact = 0
    recalls = 0.0
    precisions = 0.0
    abstentions = 0
    for selected, reference, coverage_ok in results:
        overlap = len(selected & reference)
        hits += int(overlap > 0)
        exact += int(selected == reference and coverage_ok)
        recalls += overlap / len(reference)
        precisions += (overlap / len(selected)) if selected else 0.0
        abstentions += int(not coverage_ok)
    total = len(results)
    return {
        "cases": total,
        "capability_abstentions": abstentions,
        "reference_any_hit_rate": round(hits / total, 6),
        "reference_exact_set_rate": round(exact / total, 6),
        "reference_macro_recall": round(recalls / total, 6),
        "reference_macro_precision": round(precisions / total, 6),
    }


def shadow_compare(manifest: dict, qualifications: list[dict] | None = None) -> dict:
    """Counterfactual proposals *only*; R0.1 remains completely untouched."""
    source_class, cases = _validate_manifest(manifest)
    reports = [] if qualifications is None else qualifications
    if type(reports) is not list or len(reports) > MAX_QUALIFICATIONS:
        _reject("too many R0.3 metadata reports")
    reference_profiles = {p for case in cases for p in case["reference_profiles"]}
    qualified: set[str] = set()
    for report in reports:
        profile = _validate_qualification(report, reference_profiles)
        if profile in qualified:
            _reject("duplicate R0.3 profile report")
        qualified.add(profile)

    measurements: dict[str, dict[int, list[tuple[set[str], set[str], bool]]]] = {
        split: {budget: [] for budget in BUDGETS} for split in SPLITS
    }
    for case in cases:
        reference = set(case["reference_profiles"])
        for budget in BUDGETS:
            proposed = plan(case["mode"], list(case["needs"]), max_specialists=budget)
            if (proposed["gate"]["inference_authorized"] is not False or
                proposed["gate"]["operator_approval_required"] is not True or
                proposed["gate"]["verification_required"] is not True):
                _reject("CLAVIS R0.1 execution gate unexpectedly changed")
            selected = {x["profile_id"] for x in proposed["specialists"]}
            coverage_ok = not proposed["uncovered_capabilities"]
            measurements[case["split"]][budget].append((selected, reference, coverage_ok))

    holdout_n = len(measurements["holdout"][1])
    if source_class == "SYNTHETIC_FIXTURE":
        status = "SYNTHETIC_MECHANICS_ONLY"
    elif holdout_n < MIN_HOLDOUT:
        status = "HELD_HOLDOUT_SAMPLE_FLOOR"
    elif not qualified:
        status = "HELD_NO_R03_RECONCILIATION_METADATA"
    else:
        status = "SHADOW_COMPARISON_FOR_INDEPENDENT_REVIEW_ONLY"

    return {
        "schema": REPORT_SCHEMA,
        "status": status,
        "manifest_source_class": source_class,
        "data_independence_verified": False,
        "reference_labels_independently_attested": False,
        "r03_report_contents_independently_authenticated": False,
        "r03_reconciled_profile_count_asserted": len(qualified),
        "r03_profiles_not_automatically_promoted": True,
        "reviewed_model_quality_proven": False,
        "routing_activated": False,
        "weights_changed": False,
        "inference_executed": False,
        "deployment_authorized": False,
        "raw_cases_or_reference_mappings_exported": False,
        "policy": "UNCHANGED_CLAVIS_R01_GREEDY_CAPABILITY_MATCH",
        "budget_comparison": {
            split: {
                str(budget): _summarize(measurements[split][budget])
                for budget in BUDGETS
            }
            for split in SPLITS
        },
        "interpretation": (
            "Reference overlap measures agreement with operator-supplied labels, "
            "not expert quality, model accuracy or production readiness."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline CLAVIS R0.4 shadow comparison; no model execution or runtime routing"
    )
    parser.add_argument("--manifest", type=Path, required=True,
                        help="private categorical reference manifest outside the Git checkout")
    parser.add_argument("--r03", type=Path, action="append", default=[],
                        help="optional untrusted R0.3 metadata report (up to nine)")
    args = parser.parse_args()
    try:
        paths = [args.manifest, *args.r03]
        if len({p.expanduser().resolve(strict=True) for p in paths}) != len(paths):
            _reject("shadow inputs must have distinct file paths")
        result = shadow_compare(_load_private(args.manifest),
                                [_load_private(p) for p in args.r03])
    except (ClavisError, OSError, ValueError) as exc:
        parser.exit(2, "CLAVIS R0.4 HELD: " + str(exc) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
