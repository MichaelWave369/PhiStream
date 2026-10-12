# PR #22: Weight staging audit (real-weight receipts, no new model runs)

## Why

In the user's successful four-case SmolLM2-135M-Instruct matrix on RTX 5070,
all 128 greedy IDs and 1,147 sparse logit coordinates matched the
user-declared external Hugging Face reference. The user supplied:

| Case | Prompt tokens | Decoder time | Weight staging timer | Staged bytes | Unique generated IDs |
|---|---:|---:|---:|---:|---:|
| short_1 | 1 | 9.825242 s | 8.040275 s | 8,608,961,664 | 25 |
| mixed_7 | 7 | 11.394797 s | 9.388994 s | 10,223,141,760 | 22 |
| repeat_12 | 12 | 12.555416 s | 10.390424 s | 11,568,291,840 | 28 |
| spread_24 | 24 | 16.142352 s | 13.352226 s | 14,796,652,032 | 6 |

Those values are user-provided receipt extracts, not CI measurements or
a comparison with another engine. The staging timer accounts for about
82% of decoder time in these one-shot cases, but IMPORTANTLY it measures
**file read, tensor materialization, CPU->CUDA transfer, dtype conversion,
synchronization and Python overhead**, NOT exclusively PCIe transfer,
SSD transfer, DRAM speed or GPU computation.

The exact staged byte pattern is:

    staged bytes = (prompt_token_count + 32 - 1) * 269,030,016 + 1,152

That matches how the current StreamingLlama implementation is written:
each supplied prompt token is sent through step(), followed by 31
teacher-forced generated tokens. Each step streams 269,030,016 bytes
of model tensors. A further 1,152 bytes are staged at initialization
(final norm weight).

This is a STRONG, falsifiable structural explanation, not proof that
changing the decoder would yield a proportional runtime speedup.

**Hypothesis:** a carefully implemented batched causal-mask prefill,
processing prompt tokens layer-by-layer rather than processing one token
through all layers at a time, should reduce redundant per-prompt whole-layer
loads while keeping exact causal attention and KV state. It would need its
own reference parity, VRAM cap validation and A/B benchmarks. None of
those improvements have been implemented or claimed here.

## A. Read-only audit against the existing evidence

This command ONLY reads the exact same four existing trace/parity JSON
files, original checkpoint, preflight receipt, frozen manifest and matrix
review JSON. It creates ONE new output JSON under benchmarks/local.
It never runs CUDA, models, Transformers, Torch or pip, and it changes
no checkpoint file. Use the existing PowerShell variables from PR21:

    git pull --ff-only
    $ckpt = "D:\PhiStream\Models\SmolLM2-135M-Instruct"
    $preflight = Get-ChildItem "benchmarks\local\smollm2-cuda-preflight-*.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
    $matrixDir = "benchmarks\local\smollm2-matrix-20261012-131912204"
    $manifest = Join-Path $matrixDir "manifest.json"
    $summary = Join-Path $matrixDir "matrix-review.json"
    $audit = Join-Path $matrixDir "weight-stage-audit.json"

    py -m scripts.weight_stage_audit --checkpoint "$ckpt" --preflight-receipt "$preflight" --manifest "$manifest" --receipts-dir "$matrixDir" --matrix-summary "$summary" --output "$audit"
    if ($LASTEXITCODE -ne 0) { throw "Weight-stage audit did not meet the expected counter pattern" }

Expected when the original receipts remain unchanged:
EXACT_STAGE_COUNTER_LINEARITY_OBSERVED.

This audits each case by independently re-running PR21's exact
SHA-bound trace/receipt checks, including original source declarations
and all 32 token/logit probes; it does NOT re-run model inference.

A pass rederives the observed per-token model-payload byte counter,
constant initialization offset, and the measured number of staged
weight-load operations per single-token decoder step. It also writes
source file SHA256s of the original reference/parity receipts, the
manifest and the matrix review, helping make audit calculations
repeatable. The audit does not independently authenticate the supplied
Hugging Face runtime or its weights' publisher.

An incomplete, forged or altered prior matrix FAILS CLOSED. If the
receipt pattern is genuinely nonlinear (while parity receipts are
otherwise valid), output status is COUNTER_PATTERN_NOT_FULLY_LINEAR and
exit code 2. Do not change counters to force a linear result.

## B. Inspect the derived summary (still no model run)

    $a = Get-Content "$audit" -Raw | ConvertFrom-Json
    [PSCustomObject]@{
        Status = $a.status
        PerStepPayloadBytes = $a.weight_payload_bytes
        ConstantExtraBytes = $a.per_run_fixed_staged_bytes_outside_single_token_steps
        LoadOperationsPerStep = $a.weight_load_operations_per_decoder_step
        MedianStageShare = $a.median_staging_time_fraction_observed
        ModelExecuted = $a.model_execution
    } | Format-List

The raw per-case table lives under the cases array in that same audit.
Do NOT use any staging fraction as a mathematical promise about
potential speedup. The case timings come from a single run per prompt
length and use confounded settings. Weight staging includes distinct
operations and CPU/GPU synchronization; there is not enough evidence to
estimate attainable latency gains.

## Next hypothesis gate (NOT implemented)

A separately opt-in batched-prefill candidate would require:

1. A pinned candidate algorithm with correct causal masks, RoPE
   positions and per-layer KV state identical to the single-token baseline.
2. Explicit check that prompt prefill never exceeds memory-policy
   limits and never silently falls back to a different architecture.
3. New reference IDs and sparse logits for the *same prompts*,
   validated against the same immutable checkpoint hashes.
4. Repeated A/B tests with balanced/randomized method order after
   matching parity, including time-to-first logits, decoding,
   resident GPU allocation, staging bytes and wall-clock receipts.
5. Default baseline stays single-buffer, no production promotion,
   no optimizer activation just from a staging-ratio observation.

The existing PR21 benchmark timing numbers are a baseline observation
with no randomized repetitions. The next algorithm must earn BOTH
correctness AND performance evidence. Do not interpret a 269 MB
counter reduction as 269 MB of VRAM capacity savings; disk-to-host
staging and residency policy remain separate concerns.

**Ledger above ego.** A counted byte is not a measured millisecond
saved, and a mathematically obvious optimization is still unproven
until it reproduces the reference and runs faster in controlled trials.
