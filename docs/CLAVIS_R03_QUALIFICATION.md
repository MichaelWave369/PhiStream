# Φ CLAVIS Φ369 R0.3 — Original-Source Qualification Screen

**Status: OFFLINE RESEARCH ONLY.** The R0.3 screen reconciles original
local Nine Mind dataset and source output receipts against the existing R0.5
blind review and R0.2 evidence bridge. It does **not** run models, train
adapters, measure GPU latency, certify model quality, or change Vessie's,
GA108's, PhiOS's or PhiStream's runtime routing decisions.

## Why this rung exists

R0.2 checks self-declared source hashes and blinded review structure, but
cannot determine whether the submitted output receipts are actually the ones
from which the blind pack was constructed. R0.3 closes that **file-consistency
gap** by reopening and validating the underlying original local files.

It **does not** close the larger provenance gap: a malicious or mistaken
operator could fabricate a mutually consistent collection of receipts,
scores, and examples. Only an independently controlled execution and
review protocol can qualify physical model behavior.

## Required private local files

1. A Forge JSONL dataset with an explicit train/eval split and stable
   byte-for-byte SHA-256.
2. The human-scored Nine Mind R0.5 blind-review JSON pack.
3. The separate private answer key from that same blinded review.
4. The original \`base\` and \`prompted\` model output JSON receipts.
5. Optionally, the original \`adapter\` output receipt for a three-arm
   study, but only after adapter training was explicitly approved elsewhere.

Keep **all** inputs outside the Git checkout and CI. This module prints
only bounded metadata and scores, not private prompts, answers or arm maps.

Run:

\`\`\`powershell
python -m phistream.clavis_qualification \`
  --dataset "C:\Private\tesla-study.jsonl" \`
  --pack "C:\Private\tesla-graded-blind.json" \`
  --key "C:\Private\tesla-private-answer-key.json" \`
  --base "C:\Private\tesla-base-output.json" \`
  --prompted "C:\Private\tesla-prompted-output.json"

# CLI alias after pip install -e .
phi-clavis-qualify --help
\`\`\`

For an existing adapter comparison, also pass
\`--adapter "C:\Private\tesla-adapter-output.json"\`.

## Validation chain

The qualification routine uses the repository's **existing** original
validators, not duplicate parsers. It rejects on any of these problems:

- missing, symlinked, oversized, duplicate-key receipt, or input inside
  the Git checkout;
- dataset hash mismatch, train/eval split mismatch, held-out ID mismatch,
  review-status mismatch;
- invalid base/prompted/adapter receipt schema, prompt identity, source
  digest, revision, model ID, prompt-method fingerprint, or adapter digest;
- changed blind-pack prompt, topic, answer, mapping, or source content;
- unreviewed dataset, missing human scores, or fewer than **24** held-out
  rows (stricter than R0.2's eight-row protocol check).

The **24-row screening floor is an operational minimum only**, not a power
calculation, independent sampling proof, or model acceptance threshold.
The validator checks that dataset inputs are locally consistent and review
flags are present. It cannot independently verify whether human review
occurred, whether training data leaked into heldout tasks, or whether
model inference occurred on a specific checkpoint.

## Exploratory paired statistics

For a complete, internally consistent set with at least 24 scored rows:

- Mean and median of per-question prompted-minus-base scores on the
  existing 0-to-5 overall manual grading scale.
- Paired wins, losses, and ties, by **arm identity** from the separate key.
- An **exploratory exact two-sided sign-test p-value**, ignoring ties.

**Do not interpret the p-value as proof of improvement.** Its validity
assumes independent exchangeable non-tied comparisons, assumptions that
R0.3 does not establish. The statistic is unadjusted for multiple
specialists, task families, repeated trials, human-grading bias, or
selection after seeing results. Scores are operator-supplied and do not
measure model latency, security, reliability, or factual accuracy on
their own. Output always declares \`statistically_qualified: false\`
and \`quality_proven: false\`.

Even successful consistency screening is labeled
\`SCREENED_FOR_FURTHER_INDEPENDENT_REVIEW_ONLY\`, not production-ready.
Unapproved or incomplete data remains held.

## Preservation of existing architecture

This is a standalone, opt-in \`phistream.clavis_qualification\` module.
It does not change the R0.1 CLAVIS planner, R0.2 fingerprint bridge,
Nine Mind R0.5/R0.6, Model Forge, PocketRotor, \`phistream.engine\`,
or the normal PhiStream CLI. The next stage needs operator-owned
real-run evidence, held-out dataset independence checks, repeated
blinded grading, model/tokenizer byte verification, and independently
measured hardware cost/latency before any authorized routing gate.

Run CI/CPU-only tests with:

\`\`\`sh
python -m unittest tests.test_clavis_qualification -v
python -m unittest discover -s tests -v
python -m phistream.clavis_qualification --help
\`\`\`

**Ledger Above Ego:** consistent receipts are not authenticated runs.
