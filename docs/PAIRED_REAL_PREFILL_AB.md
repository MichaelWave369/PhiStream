# PR #25: Paired real-model A/B prefill benchmark

## Research question and guardrails

The operator-reported RTX 5070 runs of SmolLM2-135M-Instruct matched
96/96 generated greedy IDs and 859/859 sparse logits in the three
batched cases against the user-declared reference. Their earlier
one-shot timings were encouraging but not paired measurements.
This gate measures whether any observed timing benefit replicates
without changing the SINGLE default. The external reference provenance
and checkpoint publisher remain user-declared, not authenticated.

The standalone Python module **scripts/paired_real_prefill_ab.py**
defaults to a **read-only plan**; execution needs an explicit CUDA
run command and approval. It validates the original frozen PR21 four-case
reference/receipt matrix and the PR24 candidate case success receipt.

## Frozen experiment contract

- Only original mixed_7, repeat_12, spread_24 prompt IDs, same weights,
  same externally asserted SHA-bound reference and 32 decoded steps.
- A fresh decoder for each trial; no cross-mode KV reuse.
- Even 4 through 20 pairs, equal AB/BA order, seeded deterministic shuffle.
  Warmups: 0 through 2 per method, default 1, all unscored.
- Every warmup and trial checks all 32 greedy IDs plus every original
  sampled logit, with original per-coordinate tolerances. Fail closed
  on mismatches, untrusted files, missing authorization, or OOM.
- Both modes time engine creation, prompt and all 31 continuation
  steps plus probe comparisons; full checkpoint SHA work is outside
  timers. BATCHED additionally rechecks its memory policy inside
  the timed region, a conservative asymmetry explicitly disclosed.
- Full sequence memory policy preflight plus batched workspace estimate
  required for each trial. Estimate is not a hard VRAM guarantee.
- Records trial order, time-to-first logits, wall clock, per-step time,
  staged weights/operations, peak PyTorch CUDA allocated, and parity.
- Reports median per-pair relative gain, faster-pair fraction and
  staging-byte differences, not a claim of statistical significance.

**Predeclared further-review gates:** median paired wall gain at least
5%, candidate faster in at least 75% of pairs, and no failed parity.
A qualifying result means only CANDIDATE_FOR_FURTHER_REVIEW.
No production promotion, throughput or independent source-validation claim.
Cache, temperature, clocks, GPU background processes and OS contention
remain uncontrolled; replicate on different days and longer prompts.

## Phase A: default read-only plan, no model inference

After CI passes and PR25 is merged, from a PhiStream PowerShell checkout:

~~~powershell
git pull --ff-only
$ckpt = "D:\PhiStream\Models\SmolLM2-135M-Instruct"
$preflight = Get-ChildItem "benchmarks\local\smollm2-cuda-preflight-*.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
$matrixDir = "benchmarks\local\smollm2-matrix-20261012-131912204"
$gateDir = Join-Path $matrixDir "batched-prefill"
$case = "spread_24"
$plan = Join-Path $gateDir "paired-ab-plan_$case.json"
if (Test-Path $plan) { throw "Plan exists; preserve the original" }
py -m scripts.paired_real_prefill_ab plan --checkpoint "$ckpt" --preflight-receipt "$preflight" --matrix-dir "$matrixDir" --case "$case" --pairs 4 --seed 369 --warmups 1 --output "$plan"
if ($LASTEXITCODE -ne 0) { throw "Read-only paired A/B plan failed" }
~~~

Expected: SHA_BOUND_AB_PLAN_NOT_EXECUTED, model_execution false, with
SHA bindings for the prior baseline and candidate and an exact order
schedule. No model execution or GPU work. Original evidence stays intact.

## Phase B: separately authorized local CUDA experiment

This consumes GPU compute and requires review before explicit approval.
Do not launch automatically after merging. Stop if previous evidence
differs or if memory availability is uncertain.

~~~powershell
$run = Join-Path $gateDir "paired-ab-run_$case.json"
if (Test-Path $run) { throw "Run receipt exists; never overwrite" }
py -m scripts.paired_real_prefill_ab run --checkpoint "$ckpt" --preflight-receipt "$preflight" --matrix-dir "$matrixDir" --case "$case" --pairs 4 --seed 369 --warmups 1 --device cuda --capacity-mib 8192 --reserve-mib 1024 --approve-local-execution --output "$run"
$code = $LASTEXITCODE
if (Test-Path $run) {
    $r = Get-Content $run -Raw | ConvertFrom-Json
    $r | Select-Object status,case,model_execution | Format-List
    $r.summary | Format-List
}
if ($code -ne 0) { Write-Warning "Fail closed; preserve the receipt" }
~~~

One approved 4-pair case executes 2 warmups and 8 measured local
pretrained-model runs. Other cases need separate explicit authorization.
Results never change the default SINGLE decoder.

## CI scope

The CI job runs CPU synthetic fixtures and tests schedule balancing,
repeatability, full teacher-forced 32-step parity, negative controls,
receipt tampering, consent refusals and descriptive threshold math.
No pretrained weights, reference downloads or RTX experiments in CI.

**Ledger above ego.** A reproducible staging counter is not a
reproducible speedup.
