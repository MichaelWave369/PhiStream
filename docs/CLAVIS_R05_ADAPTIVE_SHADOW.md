# PhiStream CLAVIS Phi369 R0.5: Adaptive Shadow-Policy Lab

**Research only; no live routing, inference, model training or deployment.**

R0.4 compared fixed specialist budgets against operator-supplied reference labels. R0.5 adds a separately evaluated, calibration-only tie-break policy. This is a *categorical label agreement* experiment, not a demonstrated improvement in model quality or cognitive ability.

## Method

Reuse the existing R0.4 manifest schema: phistream.clavis.shadow-manifest.v0.4. Each case has an opaque ID, a calibration/holdout split, a mode, capability needs, and 1–4 reference specialist IDs. No prompt text or source documents are permitted.

The adaptive policy counts how often each specialist appears in calibration reference labels for each CLAVIS mode. A mode requires at least eight calibration cases before learning a tie-break. It estimates the smoothed inclusion rate as (reference_count + 1) / (calibration_cases_for_mode + 2). This is a heuristic, NOT a calibrated success probability, confidence or quality measure.

At 1, 2, 3 and 4 specialist budgets, the adaptive selector preserves R0.1's two first-order objectives: maximum incremental capability coverage, then maximum requested-capability overlap. Only when these values tie can the calibration-derived preference reorder candidates. Remaining ties use the exact unchanged R0.1 preference order.

Modes with insufficient calibration data fall back to R0.1. Holdout references are never used in the fitting step. Reference accuracy, task independence and grader integrity are not independently verified; even a favorable holdout metric never authorizes deployment.

## Offline example

Create a private JSON file outside the checkout using the existing R0.4 manifest fields. Start with synthetic examples (source_class SYNTHETIC_FIXTURE), and keep private task text out of this categorical manifest.

    {
      "schema": "phistream.clavis.shadow-manifest.v0.4",
      "source_class": "SYNTHETIC_FIXTURE",
      "cases": [
        {
          "id": "cal-001", "split": "calibration",
          "mode": "creative", "needs": ["visual"],
          "reference_profiles": ["dali"]
        },
        {
          "id": "hold-001", "split": "holdout",
          "mode": "creative", "needs": ["creative"],
          "reference_profiles": ["dali"]
        }
      ]
    }

Then run:

    python -m phistream.clavis_adaptive --manifest "C:\Private\clavis-shadow.json"

Or, after installing the project:

    phi-clavis-adaptive --manifest "C:\Private\clavis-shadow.json"

Optional local R0.3 source-reconciled **metadata** (never used for fitting):

    phi-clavis-adaptive --manifest "C:\Private\clavis-shadow.json" --r03 "C:\Private\clavis-r03.json"

Local inputs must be separate, regular, size-bounded JSON files outside the checkout. The existing R0.2 private loader rejects symlinks, nonfinite numbers and duplicate keys. The output shows aggregate metrics only, never per-case IDs, prompts or private reference maps.

## Outputs and holds

For each budget 1–4 the report includes fixed R0.1 versus adaptive R0.5 reference any-hit, exact-set, macro precision, macro recall and capability abstentions, plus paired wins/losses/ties on reference recall and adaptive-minus-fixed deltas.

- SYNTHETIC_MECHANICS_ONLY means all observations are invented fixtures.
- HELD_INSUFFICIENT_CALIBRATION means fewer than eight calibration cases.
- HELD_INSUFFICIENT_HOLDOUT means fewer than 24 holdout cases.
- HELD_NO_SUPPORTED_MODE means no mode has eight calibration cases.
- HELD_NO_R03_METADATA means no in-scope source-reconciliation metadata was supplied.
- EXPLORATORY_ADAPTIVE_SHADOW_REVIEW_ONLY means local format and screening checks passed for an operator-asserted dataset, NOT model quality or independently verified deployment readiness.

R0.3 metadata is self-asserted and never used for adaptation. No result is a policy activation token. The old production router, weights and all authorization gates remain untouched.

## Verification

    python -m unittest tests.test_clavis_adaptive -v
    python -m unittest discover -s tests -v
    python -m phistream.clavis_adaptive --help

The CI fixtures deliberately use fabricated reference labels; they test algorithm behavior, not physical inference or improved intelligence. Future meaningful evaluation requires frozen private real tasks, independently reviewed reference labeling, authenticated local checkpoint/recipe provenance, repeated blind grading, measured cost and latency, and explicit operator release gates.

**Ledger Above Ego.**
