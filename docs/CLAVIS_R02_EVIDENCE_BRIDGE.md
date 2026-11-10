# Φ CLAVIS Φ369 R0.2 — Nine Mind Evidence Bridge

**Scope: offline, local, research-only review. No inference, autonomous model
selection, weight changes, production activation, uploads or human-score
fabrication.** This additive bridge uses the existing CLAVIS R0.1 plans,
Model Atlas candidate identities and the exact current Nine Mind R0.6
method-prompt recipes.

CLAVIS R0.1 knew *which profiles could plausibly address a capability*.
R0.2 binds selected profiles to their actual source-level prompt fingerprints
and can read the **existing** Nine Mind R0.5 separate blind pack/private key
to summarize **manually entered** scores. These are still operator assertions,
not independent validation of an actual model run.

## First: inspect the Nine Mind bridge (no private inputs)

```sh
python -m phistream.clavis_evidence \
  --mode engineering --need electrical --need physics --need prototype
# installed CLI alias:
phi-clavis-review --mode creative --need recursion --need symbolism
```

Output includes:
- unchanged CLAVIS R0.1 route proposal and safety gates;
- each proposed profile ID with a SHA-256 fingerprint of its exact current
  Nine Mind R0.6 method recipe;
- `blind_review: null` until an operator supplies private graded evidence;
- explicit `inference_authorized: false` and `deployment_authorized: false`.

**Fingerprints are source bindings, not proof a prompt was actually used or
that any weights were loaded.** A future recipe change deliberately causes
an old key fingerprint to fail review until explicitly reconciled.

## Optional: audit a privately graded R0.5 blind pack

Use the current `phistream.nine_mind_bench` flow to create a private
blinded review pack and a distinct private answer key. The original
comparison is **base vs. method-prompted**, with an optional separately
authorized adapter third arm. Never create fake outputs and pass them off
as physical model inference. See [Nine Mind R0.5](NINE_MIND_BLIND_BENCH_R05.md)
and [R0.6](NINE_MIND_PROMPT_RUNNER_R06.md).

**Before consulting the answer key,** a human reviewer grades each anonymized
answer's existing `reviewer_scores` entry as **one numeric overall score
between 0 and 5**, where 0 means unusable and 5 means excellent. The original
rubric list must remain unchanged. This is a deliberately coarse,
unverified summary score, not five independently scored dimensions. Keep
answers, grades, original receipts, pack and key private and outside
the public Git checkout.

If every source dataset row was independently reviewed and a sufficiently
large holdout exists, run:

```powershell
python -m phistream.clavis_evidence `
  --mode engineering --need electrical --need physics --need prototype `
  --pack "C:\Private\nine-mind-blind-scored.json" `
  --key "C:\Private\nine-mind-answer-key.json"
```

The review checks that the existing pack and key share the same session and
dataset digest; each row has exactly the expected blinded labels and a
permuted private arm mapping; the model is in the current reviewed candidate
catalog and revision is pinned; no input claims proof/deployment; and the
key's declared recipe SHA matches the current Nine Mind R0.6 recipe.

The review **does not** independently reopen or hash the original source
model receipts, authenticate who wrote the grades, prove the model actually
ran, or verify the correctness of the underlying answers. Its source digest
fields are only self-declared correlators.

A report is held, without scores, if:

- original dataset rows were not all human-approved;
- the review has fewer than 8 held-out examples;
- any score is missing;
- the associated profile is not actually selected in the requested CLAVIS plan.

Malformed mappings, altered rubrics, unrecognized profiles, duplicate JSON
keys, invalid scores/hashes, mismatched source fingerprints, or false
authorization flags are **rejected**.

When admissible to further review, the report gives only per-arm unverified
mean scores, a paired prompted-versus-base win/tie/loss count and an
unverified average delta. It never reveals prompt text, answer text,
key mappings or the underlying private path names. Even a positive
unverified difference is **not** a validated model improvement.

## Governance and security

- The data is loaded locally and the input files are bounded, ordinary
  non-symlink JSON files **outside the repository**.
- Data is never written, sent to a service, logged as a raw prompt, or sent to
  CI. The review output is structured metadata; safeguard private source
  files and console outputs according to your environment.
- Invalid data cannot turn on `inference_authorized`, `router_changed`,
  `deployment_authorized` or any executor.
- This bridge does not alter `phistream.cli`, `phistream.engine`,
  PocketRotor native MoE routes, Model Forge training, Nine Mind R0.6,
  Vessie/GA108 authority, or PhiOS.
- The score threshold is a **minimum sample-count guard**, not a statistical
  power calculation. Real conclusions require independent validation,
  evaluator calibration, multiple diverse private scenarios and measured
  runtime and cost/latency evidence.

## Tests

```sh
python -m unittest tests.test_clavis_evidence -v
python -m unittest discover -s tests -v
python -m phistream.clavis_evidence --help
```

Unit tests use entirely fabricated *mechanics-only* score fixtures, never
claiming these are actual model observations.

## Next controlled gate

R0.3 should collect real, human-approved, held-out private tasks on the
operator's own machine; independently verify model and tokenizer bytes,
validate recipe provenance, run blinded comparisons, and add statistical
uncertainty and independent latency measurements. Only after these gates
should a separate authorized component use evidence to rank or activate
runtime experts.

**Ledger above ego. A score is not a mind, and a proposed route is not permission.**
