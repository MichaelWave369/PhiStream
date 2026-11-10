"""CLAVIS Φ369 R0.5: offline adaptive *shadow* policy comparison.

Fits a transparent, mode-conditioned tie-break preference from calibration
reference labels only. Evaluates fixed R0.1 versus adaptive counterfactuals on
holdout labels. No model, training weights, production routing or tool access.
Operator references and optional R0.3 reports are not independently attested.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from .clavis import ClavisError, PREFERENCES, SPECIALISTS, plan
from .clavis_evidence import _load_private
from .clavis_shadow import (
    BUDGETS, MANIFEST_SCHEMA, MIN_HOLDOUT, _validate_manifest,
    _validate_qualification, _summarize,
)

SCHEMA = "phistream.clavis.adaptive-shadow.v0.5"
MIN_MODE_CALIBRATION = 8
MAX_R03_REPORTS = 9


def _reject(message: str) -> None:
    # No private case IDs or file paths in exceptions.
    raise ClavisError(message)


def _learn(cases: list[dict]) -> tuple[dict[str, Counter], dict[str, int]]:
    """Only calibration reference labels are visible to policy fitting."""
    counts: dict[str, Counter] = {}
    support: dict[str, int] = {}
    for case in cases:
        if case["split"] != "calibration":
            continue
        mode = case["mode"]
        support[mode] = support.get(mode, 0) + 1
        counts.setdefault(mode, Counter()).update(case["reference_profiles"])
    return counts, support


def _adaptive_select(mode: str, needs: tuple[str, ...], budget: int,
                     counts: dict[str, Counter], support: dict[str, int]) -> tuple[set[str], bool]:
    """Greedy coverage exactly as R0.1, with calibrated tie breaking only.

    If fewer than eight calibration examples exist for this mode, every tie
    resolves by R0.1's original static preferences. We never rank an
    unrelated specialist above someone who covers more task capabilities.
    """
    if type(budget) is not int or budget not in BUDGETS:
        _reject("invalid shadow specialist budget")
    remaining = set(needs)
    requested = set(needs)
    pool = list(SPECIALISTS)
    chosen: set[str] = set()
    priority = PREFERENCES[mode]
    mode_n = support.get(mode, 0)
    while remaining and pool and len(chosen) < budget:
        def key(profile: str) -> tuple:
            # Smoothed reference inclusion rate is NOT quality/confidence.
            count = counts.get(mode, {}).get(profile, 0)
            rate = ((count + 1) / (mode_n + 2)) if mode_n >= MIN_MODE_CALIBRATION else 0.0
            return (
                -len(SPECIALISTS[profile] & remaining),
                -len(SPECIALISTS[profile] & requested),
                -rate,
                priority.index(profile) if profile in priority else len(priority),
                list(SPECIALISTS).index(profile),
            )

        next_profile = sorted(pool, key=key)[0]
        gained = SPECIALISTS[next_profile] & remaining
        if not gained:
            break
        chosen.add(next_profile)
        remaining -= gained
        pool.remove(next_profile)
    return chosen, not remaining


def _paired(a: list[tuple[set[str], set[str], bool]],
            b: list[tuple[set[str], set[str], bool]]) -> dict:
    if len(a) != len(b):
        _reject("inconsistent paired shadow population")
    if not a:
        return {
            "cases": 0,
            "adaptive_higher_reference_recall": 0,
            "fixed_higher_reference_recall": 0,
            "reference_recall_ties": 0,
            "adaptive_minus_fixed_macro_recall": None,
            "adaptive_minus_fixed_exact_set_rate": None,
        }
    adaptive_wins = fixed_wins = ties = 0
    recall_delta = 0.0
    exact_delta = 0
    for (fixed, reference, fixed_covered), (adaptive, adaptive_reference, adaptive_covered) in zip(a, b):
        if reference != adaptive_reference:
            _reject("paired reference labels changed during evaluation")
        fixed_recall = len(fixed & reference) / len(reference)
        adaptive_recall = len(adaptive & reference) / len(reference)
        recall_delta += adaptive_recall - fixed_recall
        exact_delta += int(adaptive == reference and adaptive_covered) - int(
            fixed == reference and fixed_covered
        )
        if adaptive_recall > fixed_recall:
            adaptive_wins += 1
        elif adaptive_recall < fixed_recall:
            fixed_wins += 1
        else:
            ties += 1
    return {
        "cases": len(a),
        "adaptive_higher_reference_recall": adaptive_wins,
        "fixed_higher_reference_recall": fixed_wins,
        "reference_recall_ties": ties,
        "adaptive_minus_fixed_macro_recall": round(recall_delta / len(a), 6),
        "adaptive_minus_fixed_exact_set_rate": round(exact_delta / len(a), 6),
    }


def adaptive_shadow(manifest: dict, qualifications: list[dict] | None = None) -> dict:
    """Fit from calibration only; evaluate held-out labels without adaptation."""
    source_class, cases = _validate_manifest(manifest)
    reports = [] if qualifications is None else qualifications
    if type(reports) is not list or len(reports) > MAX_R03_REPORTS:
        _reject("invalid R0.3 report list")
    reference_profiles = {p for case in cases for p in case["reference_profiles"]}
    qualified: set[str] = set()
    for report in reports:
        profile = _validate_qualification(report, reference_profiles)
        if profile in qualified:
            _reject("duplicate R0.3 profile metadata")
        qualified.add(profile)

    counts, support = _learn(cases)
    holdouts = [case for case in cases if case["split"] == "holdout"]
    calibrations = [case for case in cases if case["split"] == "calibration"]

    # Keep train/test order and report no per-task predictions or references.
    comparisons: dict[str, dict[str, dict]] = {}
    for budget in BUDGETS:
        fixed_results: list[tuple[set[str], set[str], bool]] = []
        adaptive_results: list[tuple[set[str], set[str], bool]] = []
        for case in holdouts:
            fixed_plan = plan(case["mode"], list(case["needs"]), max_specialists=budget)
            gate = fixed_plan["gate"]
            if not (gate.get("verification_required") is True and
                    gate.get("operator_approval_required") is True and
                    gate.get("inference_authorized") is False and
                    gate.get("deployment_authorized") is False):
                _reject("R0.1 production holds changed unexpectedly")
            reference = set(case["reference_profiles"])
            fixed_results.append((
                {p["profile_id"] for p in fixed_plan["specialists"]}, reference,
                not fixed_plan["uncovered_capabilities"],
            ))
            selected, covered = _adaptive_select(case["mode"], case["needs"], budget, counts, support)
            adaptive_results.append((selected, reference, covered))
        comparisons[str(budget)] = {
            "fixed_r01": _summarize(fixed_results),
            "adaptive_r05": _summarize(adaptive_results),
            "paired_delta": _paired(fixed_results, adaptive_results),
        }

    supported = {mode: n for mode, n in support.items() if n >= MIN_MODE_CALIBRATION}
    if source_class == "SYNTHETIC_FIXTURE":
        status = "SYNTHETIC_MECHANICS_ONLY"
    elif len(calibrations) < MIN_MODE_CALIBRATION:
        status = "HELD_INSUFFICIENT_CALIBRATION"
    elif len(holdouts) < MIN_HOLDOUT:
        status = "HELD_INSUFFICIENT_HOLDOUT"
    elif not supported:
        status = "HELD_NO_SUPPORTED_MODE"
    elif not qualified:
        status = "HELD_NO_R03_METADATA"
    else:
        status = "EXPLORATORY_ADAPTIVE_SHADOW_REVIEW_ONLY"

    # Even "review only" can be supported for merely one mode; unsampled
    # modes always revert to the unchanged fixed tie-break policy.
    return {
        "schema": SCHEMA,
        "status": status,
        "manifest_source_class": source_class,
        "calibration_cases": len(calibrations),
        "holdout_cases": len(holdouts),
        "calibration_minimum_per_mode": MIN_MODE_CALIBRATION,
        "holdout_minimum": MIN_HOLDOUT,
        "mode_calibration_support": {
            mode: {"cases": support.get(mode, 0),
                   "calibrated_tiebreak_active": mode in supported}
            for mode in PREFERENCES
        },
        "r03_profiles_declared": len(qualified),
        "policy_fit_source": "CALIBRATION_REFERENCE_LABELS_ONLY",
        "policy_objective": "HEURISTIC_LABEL_AGREEMENT_NOT_MODEL_QUALITY",
        "holdout_labels_used_for_fitting": False,
        "unseen_modes_use_fixed_r01_tiebreak": True,
        "data_independence_attested": False,
        "reference_labels_independently_attested": False,
        "r03_evidence_independently_authenticated": False,
        "model_quality_proven": False,
        "statistically_qualified": False,
        "live_router_modified": False,
        "inference_executed": False,
        "training_weights_modified": False,
        "deployment_authorized": False,
        "private_case_ids_or_reference_mappings_exported": False,
        "holdout_budget_comparison": comparisons,
        "interpretation": (
            "An offline, potentially biased calibration-label tie-break, "
            "NOT a learned model or evidence of expert accuracy. "
            "Holdout agreement must be independently audited before operational use."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLAVIS R0.5 offline calibrated tie-break shadow evaluation, no inference"
    )
    parser.add_argument("--manifest", required=True,
                        help="private R0.4 categorical manifest JSON, not a model prompt")
    parser.add_argument("--r03", action="append", default=[],
                        help="optional private R0.3 metadata JSON; never used to tune policy")
    args = parser.parse_args()
    try:
        from pathlib import Path
        paths = [Path(args.manifest), *(Path(p) for p in args.r03)]
        if len({p.expanduser().resolve(strict=True) for p in paths}) != len(paths):
            _reject("shadow inputs must be separate files")
        result = adaptive_shadow(_load_private(paths[0]),
                                 [_load_private(p) for p in paths[1:]])
    except (ClavisError, OSError, ValueError) as exc:
        parser.exit(2, "CLAVIS R0.5 HELD: " + str(exc) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
