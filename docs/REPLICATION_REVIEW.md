# PR #28: cross-day replication evidence review

## Scope

Operator-reported results from October 12–13, 2026 Pacific:

- Original spread_24, seed 369, four paired comparisons: median paired decoder-time reduction 39.684%, BATCHED faster 4/4, median TTFT reduction 92.5016%.
- Replication spread_24, seed 70421, eight paired comparisons: median paired decoder-time reduction 40.5861%, BATCHED faster 8/8, median TTFT reduction 92.493%.
- Status from submitted local receipt: MATCHED_CANDIDATE_FOR_FURTHER_REVIEW. This does not establish statistically significant speedup or justify production promotion.

The new run was not independently authenticated or audited at the time PR #28 was authored. The dashboard must distinguish the operator's local result from a later, successfully executed offline review.

## What this PR audits

scripts/replication_review.py is offline-only, read-only and SHA-bound. It:

1. Recomputes PR25's frozen original metadata preflights for all three workloads, rehashing local checkpoint/reference and previous PR24 artifacts.
2. Validates all original PR25 plan/run files using the PR26 full per-step and per-pair checks. Cross-checks the existing local PR26 census and six original file SHA256 digests.
3. Recomputes the new spread_24 metadata preflight with 8 pairs, seed 70421 and one warmup per method; the saved plan must be identical except for the plan-creation timestamp.
4. Examines all 18 replication executions: 2 warmups, 16 measured runs, 32/32 greedy token matches and all 288 sampled logits in each trial, memory policy receipts, timing/staging counters, and 31 continuation steps.
5. Verifies the frozen balanced order and recalculates paired decoder-time and TTFT summaries from every measured pair; summaries and qualification status must agree.
6. Rehashes receipt/reference files after reading. Writes one new exclusive JSON, preserving existing weights and receipts.

Expected combined counts, only **if every reported local receipt passes**: 20 pairs, 20 BATCHED-faster paired comparisons, 48 model executions including warmups, 1,536 matched greedy IDs, 13,774 matched sparse logit probes. The new spread_24 median differs by +0.9021 percentage points from the first spread_24 cohort. These are expectations based on the user's logs, not verified PR28 review evidence until the local script succeeds.

Expected status: REPLICATION_REVIEW_CANDIDATE_NOT_SPEEDUP_CERTIFIED.

## Operator PowerShell: no model execution

Run after PR28 merges, from the existing PhiStream checkout:

~~~powershell
cd "C:\Users\micro\Downloads\PhiStream"
git pull --ff-only
if ($LASTEXITCODE -ne 0) { throw "git pull failed" }

$ckpt = "D:\PhiStream\Models\SmolLM2-135M-Instruct"
$preflight = Get-ChildItem "benchmarks\local\smollm2-cuda-preflight-*.json" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1 -ExpandProperty FullName
$matrixDir = "benchmarks\local\smollm2-matrix-20261012-131912204"
$gateDir = Join-Path $matrixDir "batched-prefill"
$review = Join-Path $gateDir "replication-review-seed70421.json"
if (Test-Path $review) { throw "Receipt exists; preserve original" }

py -m scripts.replication_review --checkpoint "$ckpt" --preflight-receipt "$preflight" --matrix-dir "$matrixDir" --output "$review"

if ($LASTEXITCODE -ne 0) { throw "Review failed closed; do not overwrite receipt" }
$r = Get-Content "$review" -Raw | ConvertFrom-Json
$r | Select-Object status, model_execution, general_speedup_proven | Format-List
$r.original_spread_24 | Select-Object median_paired_relative_gain, candidate_faster_pair_count | Format-List
$r.replication | Select-Object pairs, seed, median_paired_relative_gain, candidate_faster_pair_count, review_thresholds_met | Format-List
$r.combined | Format-List
~~~

The audit reads and hashes full local model bytes. It does not load an inference engine, access CUDA, download files, upload data, overwrite receipts, or enable the BATCHED scheduler.

## Public site

The dashboard presents the submitted replication as operator-reported / awaiting PR28 offline review. The local JSON importer performs only structural and arithmetic checks, **not SHA256 authentication**.

## Limits

The cross-day calendar dates are user-reported, not independent time attestations. GPU clocks, thermals, caches and background load were not tightly controlled. Same model, single RTX 5070, frozen 24-token prompt and 32-step teacher-forced continuation. No longer-context, general-model, production throughput or multi-GPU conclusions. Four and eight pairs are small cohorts, with no statistical significance established.

The original SINGLE path remains the default. No promotion.

**Ledger above ego.**
