# ΦPocketRotor E1-A · Independent Evidence Arithmetic Audit

**Status:** Read-only offline review, no model execution, no CUDA calls,
no network, no package installs, no checkpoint writes.

The E1 balanced study can eventually produce a 3- or 6-round local receipt.
E1-A provides a separate **standard-library-only** tool to check the
evidence, reject malformed/missing/incomplete cases, independently recompute
paired performance arithmetic, and preserve its underlying source files.

## Why this exists

A green CPU test is not proof that a model is faster on an RTX 5070.
Likewise, an E1 JSON reporting a 10% gain might conceal an incomplete trial,
tampered summary, an incorrect reference, an invalid cache-memory budget, or
different weights than the frozen plan. This audit catches those structural
and arithmetic failures. It **cannot** prove a GPU actually ran, independently
authenticate who generated the reference, or certify a speedup.

## One read-only command

Use this **after** completing a separately approved E1 run. This tool does
not initiate an E1 experiment.

~~~powershell
git pull --ff-only
$root = "D:\PhiStream\Experiments"
$model = "D:\PhiStream\Models\SmolLM2-135M-Instruct"
$plan = Join-Path $root "pocketrotor-e1-plan-001.json"
$trace = Join-Path $root "existing-32-step-reference.json"
$run = Join-Path $root "pocketrotor-e1-run-001.json"
$audit = Join-Path $root "pocketrotor-e1-audit-001.json"

if (Test-Path $audit) { throw "Audit exists; preserve original" }
py -m scripts.pocketrotor_e1_audit --checkpoint "$model" --trace "$trace" --plan "$plan" --receipt "$run" --output "$audit"
if ($LASTEXITCODE -ne 0) { throw "PocketRotor E1-A audit failed closed" }
Get-Content "$audit" -Raw | ConvertFrom-Json | Select-Object status,reviewed_rounds,reviewed_trial_phases
~~~

**Without --output**, the audit prints JSON to stdout and creates no file.
With --output, it creates exactly one new local JSON report; it cannot
overwrite an existing file. Every checkpoint source file is read for SHA256
checking; it is never modified.

Expected successful status:

\`STRUCTURE_AND_ARITHMETIC_MATCH_UNAUTHENTICATED\`

It is **not** \`VERIFIED_BENCHMARK\`, \`PROVEN_SPEEDUP\` or any other claim
of independent physical performance attestation.

## Checks performed

1. Validate file sizes, no input symlinks, strict JSON with duplicate-key and
   nonfinite-value rejection.
2. Validate E1 plan is a \`READY_FOR_REVIEW\`, never-executed, unchanged plan.
3. Recompute SHA256 of plan, reference trace and every declared model shard
   plus config. Match across reference, frozen plan and E1 run.
4. Reconstruct seeded balanced policy schedule for 3 or 6 complete rounds.
   Verify exact order, no missing methods, and exactly one cold/warm phase per
   method in each round.
5. Require complete 32-step greedy-token and every sampled-logit probe
   *claimed* parity in all phases; reject missing or failed counts.
6. Require finite, positive wall-clock times and valid load/transfer and
   cache-hit/miss/eviction counters. Verify reported cache residency stays
   within the declared budget and the recorded memory-policy estimate permits
   the active working set plus the full cache budget.
7. Independently recompute per-round gains, median gain, win count,
   bytes-avoided and threshold eligibility against the original streaming
   baseline for flat-LRU and nested-pin, in both logical-cache phases.
8. Reject claims that a real reference was authenticated or that production
   promotion took place. Label the audit's limited authority accurately.

## Limitations and failure behavior

An audit failure exits nonzero and does not issue a success report. It
does not repair, delete, or overwrite E1 evidence; users should preserve
both the original failure receipt and any upstream logs.

A coordinated alteration of plan, reference, checkpoint and run could
still forge a consistent history. SHA256 is a tamper-evidence link, **not
a signature of who ran a model**. The auditor does not recompute model
outputs from actual weights, does not validate external trace provenance,
and does not eliminate OS-cache, thermal or GPU-clock confounds.

CPU CI tests use a generated tiny, untrained local safetensors fixture
with an independent full-sequence synthetic reference and an E1
three-round run. No public pretrained model, real GPU, remote service,
large-model deployment, or asynchronous weight prefetch is involved.

**Next evidence gate:** a genuine operator-controlled RTX 5070 receipt,
paired replication on more than one day, and subsequent comparison to
an equally configured practical inference backend. Do not promote a
scheduler from test-suite success alone.

**Ledger Above Ego.**
