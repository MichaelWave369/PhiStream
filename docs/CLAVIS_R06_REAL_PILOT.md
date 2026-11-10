# CLAVIS Phi369 R0.6: Frozen Real-Evaluation Pilot

**Status: offline preflight and local receipt reconciliation, NOT a completed physical GPU test.**

This rung connects the existing Nine Mind R0.6 locally approved inference tools, the R0.3 source reconciliation, and the CLAVIS research program without creating a new model executor. No model is downloaded, loaded, trained, routed or measured by this pull request.

## The study must be frozen before running a model

The operator creates a private Forge JSONL dataset, outside Git and CI. Each example must have an original source and license, human review, and a real train/evaluation split. The protocol requires at least **24 evaluation rows across at least four distinct evaluation task families**, with a minimum of four examples per evaluation family, no train/eval family overlap and no duplicate case IDs. These are screening floors, not proof of independence or statistical power.

The study protocol is a separate bounded JSON object with EXACT keys:

- schema: phistream.clavis.pilot-protocol.v0.6
- source_class: SYNTHETIC_FIXTURE or OPERATOR_ASSERTED_PRIVATE_PILOT
- profile_id: one of the nine existing Nine Mind profiles, e.g. tesla
- dataset_sha256: 64-character lowercase hex SHA256 of the exact source JSONL bytes
- case_families: one object per Forge row, each with id and family (lowercase hyphenated slug)
- reviewer_separation_declared: an explicit boolean
- holdout_frozen_before_generation_declared: an explicit boolean

A human must supply the declarations truthfully. The software cannot authenticate the reviewer or guarantee semantic independence. Public Nine Mind seed fixtures are not real human-approved benchmark evidence.

## Preflight (CPU only)

From a PhiStream checkout, run:

    python -m phistream.clavis_pilot --dataset "C:\Private\tesla-study.jsonl" --protocol "C:\Private\tesla-protocol.json"

Or use the installed CLI alias:

    phi-clavis-pilot --dataset "C:\Private\tesla-study.jsonl" --protocol "C:\Private\tesla-protocol.json"

This calls the existing Nine Mind preflight, including topic-match and pinned-model recipe checks, validates Forge dataset bytes, checks duplicate JSON object keys, verifies task-family separation, and compares source dataset SHA256 with the declared frozen digest. It never starts a model or opens network access.

Possible statuses include SYNTHETIC_MECHANICS_ONLY, HELD_UNREVIEWED_SOURCE_EXAMPLES, HELD_INSUFFICIENT_INDEPENDENT_FAMILY_COVERAGE, HELD_REVIEWER_SEPARATION_NOT_DECLARED, HELD_HOLDOUT_FREEZE_NOT_DECLARED, and OPERATOR_REVIEW_ELIGIBLE_NO_EXECUTION. **The last status is still not permission to run the model.**

## Optional operator-owned real inference

If the operator has reviewed license, privacy, memory fit and dependency compatibility, and has separately authorized local inference, use the existing evaluators. These commands are NOT run by CLAVIS, CI or this PR.

Base arm (existing Forge evaluator, cache-only unless the operator deliberately enables downloads):

    python -m scripts.phi_model_eval --mode base --model Qwen/Qwen3-4B-Instruct-2507 --revision cdbee75f17c01a7cc42f958dc650907174af0554 --dataset "C:\Private\tesla-study.jsonl" --max-new-tokens 128 --output "C:\Private\tesla-base.json"

Prompted arm (existing Nine Mind R0.6 double-consent local runner, no automatic downloads):

    python -m phistream.nine_mind_prompt_run --profile tesla --dataset "C:\Private\tesla-study.jsonl" --max-new-tokens 128 --execute --approve-local-inference --output "C:\Private\tesla-prompted.json"

The physical GPU/cache/VRAM model fit has not been established. The base and prompted runners use separate loading paths, so parity is not automatically proven. Both output receipts contain private prompts and answers and must remain outside version control.

After independently grading a blind comparison made with the already-existing Nine Mind R0.5 bench, reconcile against the frozen study using:

    phi-clavis-pilot --dataset "C:\Private\tesla-study.jsonl" --protocol "C:\Private\tesla-protocol.json" --pack "C:\Private\tesla-graded-pack.json" --key "C:\Private\tesla-private-key.json" --base "C:\Private\tesla-base.json" --prompted "C:\Private\tesla-prompted.json"

An optional adapter arm can be included only if separately reviewed and authorized for a real adapter evaluation. This path calls the existing R0.3 source-reconciliation validator and additionally requires the key to match the frozen R0.6 model, profile, dataset and method fingerprint.

Successful reconciliation means SOURCE_RECEIPTS_RECONCILED_HUMAN_REVIEW_STILL_REQUIRED, **not model accuracy, certified execution, independent review, statistical qualification or deployment permission**. Raw case text, reference responses, and private blind-key mappings are not emitted by the CLI.

## What R0.6 does and does not establish

It establishes *local consistency* of categorical task-family labels, source dataset digest, source-level model/recipe pins, operator declarations and existing blind-review files.

It does not independently authenticate checkpoint/tokenizer bytes, GPU execution, dataset independence, reviewer identity, decoding parity, latency, energy use or result quality. These require separate future controlled experiments. The existing GA108/Vessie authority, PhiOS, PhiStream inference engine and CLAVIS production-hold flags remain untouched.

## Tests

    python -m unittest tests.test_clavis_pilot -v
    python -m unittest discover -s tests -v
    python -m phistream.clavis_pilot --help

All CI tests use completely **synthetic fixtures**, with invented approvals and scores. They do not constitute a physical model evaluation.

**Ledger Above Ego.**
