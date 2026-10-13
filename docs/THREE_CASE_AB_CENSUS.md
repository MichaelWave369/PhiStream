# PR #26: SHA-bound three-case paired A/B evidence census

The user-reported RTX 5070 results for the three frozen SmolLM2-135M-Instruct workloads are: mixed_7 15.53%, repeat_12 24.77%, spread_24 39.68% median paired decoder-time reductions, with BATCHED winning 4/4 pairs in each case. These are not general speedup or statistical-significance claims.

The standalone read-only audit in scripts/three_case_ab_audit.py validates the existing local evidence, not model execution.

## Verified by the audit

- Recomputes SHA-bound PR25 preflight against the unchanged local checkpoint, original PR21 four-case matrix, original PR24 candidate receipt, and external user-declared reference traces.
- For each case, requires the saved plan to equal fresh preflight metadata apart from timestamp; run metadata must match that same predeclared design and checkpoint/reference hashes.
- Uses exactly four measured balanced pairs with seed 369 and one warmup per method. Confirms all 30 recorded executions have 32 matching greedy IDs and every case-specific sparse probe, complete step flags, accepted CUDA memory policy, finite timing data and positive staging counters.
- Recomputes the original PR25 paired gain/TTFT statistics from each run's observations. Tampered summaries, skipped pairs, missing probes, altered reference hashes, or wrong run statuses fail closed.
- Writes one fresh exclusive SHA-bound local JSON review. No CUDA inference, model downloads, checkpoint edits, network access or scheduler changes.

**Expected if local receipts match the logs:** 3 cases, 12 paired comparisons, 24 measured executions and 6 warmups, with 960 matched greedy checks and 8,590 matched sampled-logit checks. These totals are predictions until the local review actually runs.

Expected status: THREE_CASES_REVIEW_CANDIDATE_NOT_SPEEDUP_CERTIFIED.

## Local PowerShell: offline review ONLY

Run after CI passes and this PR merges. From the existing PhiStream checkout:

~~~powershell
git pull --ff-only
$ckpt = "D:\PhiStream\Models\SmolLM2-135M-Instruct"
$preflight = Get-ChildItem "benchmarks\local\smollm2-cuda-preflight-*.json" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1 -ExpandProperty FullName
$matrixDir = "benchmarks\local\smollm2-matrix-20261012-131912204"
$gateDir = Join-Path $matrixDir "batched-prefill"
$census = Join-Path $gateDir "three-case-paired-ab-census.json"
if (Test-Path $census) { throw "Census exists; preserve the original" }

py -m scripts.three_case_ab_audit --checkpoint "$ckpt" --preflight-receipt "$preflight" --matrix-dir "$matrixDir" --output "$census"
if ($LASTEXITCODE -ne 0) { throw "Census failed closed" }

$c = Get-Content "$census" -Raw | ConvertFrom-Json
$c | Select-Object status,case_count,total_pairs,total_matching_greedy_ids_in_original_receipts,total_matching_sparse_logits_in_original_receipts,model_execution,general_speedup_proven | Format-List
$c.cases | Select-Object case,median_paired_relative_gain,candidate_faster_pair_count,review_thresholds_met | Format-Table
~~~

All original weights and receipts remain unchanged. SHA256 checks authenticate local *consistency*, not the external source publisher or model license. The census is not an inference run or independent replication.

## Research after this review

A later, separately authorized experiment should repeat the pairs with new immutable filenames, new random seed, and another day before expanding to held-out prompts, RTX 3060, deeper contexts and independent comparisons against other engines. No new CUDA runs or automatic BATCHED activation are authorized by this PR.

**Ledger above ego.**
