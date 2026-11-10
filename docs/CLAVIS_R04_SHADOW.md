# Φ CLAVIS Φ369 R0.4 — Shadow Routing Laboratory

**Research only.** This rung evaluates how the **existing** CLAVIS R0.1
categorical planner agrees with operator-supplied specialist reference
labels, separately for calibration and holdout sets. Nothing here changes
the live route, model weights, GA108/Vessie policy, PocketRotor, inference,
training, network access or PhiOS execution.

The key cannot be promoted to real inference by an R0.4 status string.

## What a shadow comparison measures

Each task is represented by only **categorical** information: an opaque case
ID, a split, a CLAVIS mode, 1–12 existing capability labels, and up to four
operator-provided reference Nine Mind specialist IDs.

The lab runs the unchanged R0.1 \`plan(..., max_specialists=k)\` at
**k = 1, 2, 3 and 4**. It reports only aggregate:

- **Reference any-hit rate:** fraction of cases with at least one selected
  reference profile.
- **Reference exact-set rate:** fraction where selected profiles exactly
  match the reference set *and* all requested capabilities were covered.
- **Reference macro recall and precision:** overlap with categorical
  reference labels, averaged across tasks.
- **Capability abstentions:** how often that budget leaves a capability
  uncovered.

These are agreement and budget-sensitivity metrics, **not model quality,
accuracy, calibrated intelligence, or proof that any specialist is better**.
A human's reference label can be wrong, fabricated or influenced by
seeing the router's predictions. Calibration and holdout case IDs are
distinct, but the program cannot prove semantic task independence.

## Private manifest format

Create a local \`.json\` file outside the Git checkout. Minimal example:

~~~json
{
  "schema": "phistream.clavis.shadow-manifest.v0.4",
  "source_class": "SYNTHETIC_FIXTURE",
  "cases": [
    {
      "id": "example-001",
      "split": "calibration",
      "mode": "engineering",
      "needs": ["electrical", "physics", "prototype"],
      "reference_profiles": ["tesla", "feynman", "da-vinci"]
    },
    {
      "id": "example-002",
      "split": "holdout",
      "mode": "creative",
      "needs": ["recursion"],
      "reference_profiles": ["escher"]
    }
  ]
}
~~~

This example and every CI case are **synthetic mechanics fixtures**.
Use \`SYNTHETIC_FIXTURE\` for invented/test comparisons. Only use
\`OPERATOR_ASSERTED_REVIEW_NOT_ATTESTED\` when the labels were actually
provided by a separately reviewed human protocol. That declaration is
**not independent attestation**.

Do **not** paste real prompts, personal data, file paths, answers, logs,
checkpoint content, or repository tokens into this manifest. Input parsing
is strict; unknown keys and specialist labels fail closed, duplicate case
IDs fail, and the total task set is bounded to 128 cases. Output excludes
all per-task case IDs and reference mappings.

## Run, with no model or GPU

~~~powershell
python -m phistream.clavis_shadow --manifest "C:\Private\clavis-shadow.json"
# optional installed alias:
phi-clavis-shadow --manifest "C:\Private\clavis-shadow.json"
~~~

The results show per-split tables for budgets 1/2/3/4, plus explicit
negative provenance and authorization flags. For empty splits, rates
are \`null\` rather than deceptively claiming 0%.

## Optional: compare with R0.3 source-consistency reports

After R0.3 has genuinely reconciled a private dataset, R0.5 blind pack
and its actual source receipts, supply **only its sanitized metadata
report** as an additional local input:

~~~powershell
phi-clavis-shadow --manifest "C:\Private\clavis-shadow.json" \
  --r03 "C:\Private\clavis-r03-tesla.json"
~~~

On PowerShell, place this on one line or use backticks for line
continuation.

The R0.3 report is accepted only if its schema is exactly the current
R0.3 schema, its source-consistency fields and screening threshold are
present, its counts agree, and it explicitly retains **all negative
attestation and authorization flags**. Its profile must occur in the
operator's reference labels. Distinct profiles cannot be repeated.
Reports with alleged independent execution, deployment authorization,
statistical qualification, drifted fields, or missing evidence are held.

**A valid R0.3 report is still self-declared metadata, not trustworthy
independent evidence.** R0.4 does not re-open the R0.3 source receipts or
claim their authenticity; use R0.3 directly for those checks. It does
not use R0.3 scores to train a ranking model.

The \`SHADOW_COMPARISON_FOR_INDEPENDENT_REVIEW_ONLY\` status requires:
- Source class declared as operator-reviewed (unattested);
- at least **24** holdout tasks (only an operational floor);
- at least one in-scope, internally consistent R0.3 metadata report.

It remains a **review status**, not a deploy permission, statistical
performance guarantee, or independent validation.

Synthetic-only data results in \`SYNTHETIC_MECHANICS_ONLY\` regardless of
the number of cases or attached R0.3 metadata. Other inputs may be held
for missing holdout rows or absent R0.3 receipts. All paths are regular
local files outside the repository, non-symlink, bounded, and distinct.

## Reproduce CPU tests

~~~sh
python -m unittest tests.test_clavis_shadow -v
python -m unittest discover -s tests -v
python -m phistream.clavis_shadow --help
~~~

Future steps require an independent reviewer, frozen/private tasks not
used to choose the reference routing labels, multiple domain splits,
real model output verification and cost/latency measurements. Only a
separate, explicitly approved routing adapter can request live changes.

**Ledger Above Ego. The shadow can measure its outline without pretending
to be the light.**
