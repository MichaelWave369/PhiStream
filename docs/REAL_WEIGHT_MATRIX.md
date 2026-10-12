# PhiStream PR #21: real-weight validation matrix

## Why this gate exists

The user's first SmolLM2-135M-Instruct real-weight comparison on RTX 5070
matched 32/32 greedy IDs and 288/288 sparse logit probes, but 29 of 32
target token IDs were the same token 32. This is real evidence for that
sequence, not evidence that all contexts, token types or architectures work.

This PR freezes FOUR different literal-token prompts, each requiring a
separate, operator-approved 32-step Hugging Face CUDA reference and a
separate, operator-approved 32-step PhiStream CUDA parity receipt.
Everything else is read-only and offline. No tokenizer is needed;
literal IDs do not constitute natural-language input.

Cases are frozen by scripts/real_weight_matrix.py and cannot be edited:
- short_1: one literal token 1
- mixed_7: 9,28,7,139,302,900,4096
- repeat_12: 42 six times, 13 four times, 1 twice
- spread_24: 24 spread-vocabulary IDs, explicitly enumerated in manifest

The frozen manifest binds the full config and single-shard safetensors
SHA256 to the earlier non-execution checkpoint preflight, exact model
source/revision/license declarations, all cases and expected 32 steps.
Original model weights are READ ONLY. No automatic model discovery,
network, install, checkpoint conversion, Ollama access or scheduler
activation is permitted.

**New PR21 reference binding:** real-model-gate receipts store a SHA256
of the actual reference JSON. It verifies its bytes before and after
reading and after decoding. The matrix reviewer verifies each matching
reference file SHA. The historical successful PR20 parity receipt
predates this extra binding, so keep it as history and run NEW matrix cases.

## A. Create a frozen plan (NO inference)

Run the following from the local PhiStream directory in PowerShell.
Run git pull first. Keep the variables alive for phases B through D.

    git pull --ff-only
    $ckpt = "D:\PhiStream\Models\SmolLM2-135M-Instruct"
    $revision = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
    $preflight = Get-ChildItem "benchmarks\local\smollm2-cuda-preflight-*.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
    if (-not $preflight) { throw "Existing CUDA hash preflight receipt not found" }
    $matrixDir = "benchmarks\local\smollm2-matrix-$(Get-Date -Format 'yyyyMMdd-HHmmssfff')"
    New-Item -ItemType Directory -Path $matrixDir | Out-Null
    $manifest = Join-Path $matrixDir "manifest.json"
    py -m scripts.real_weight_matrix plan --checkpoint "$ckpt" --preflight-receipt "$preflight" --output "$manifest"
    if ($LASTEXITCODE -ne 0) { throw "Matrix plan failed" }

Expected: FROZEN_PLAN_NOT_EXECUTED and four case IDs.
This is pure local file reading/hashing; it does not import Transformers.

## B. Four EXPLICITLY APPROVED Hugging Face CUDA reference executions

This section loads and runs the existing pretrained SmolLM2 weights
on the GPU FOUR separate times and saves four new local reference files.
Do not run this section unless you consent to those real executions.

    $cases = (Get-Content "$manifest" -Raw | ConvertFrom-Json).cases
    foreach ($case in $cases) {
        $promptIds = $case.prompt_token_ids -join ","
        $out = Join-Path $matrixDir "reference_$($case.id).json"
        py scripts\export_hf_reference.py --checkpoint "$ckpt" --preflight-receipt "$preflight" --device cuda --prompt "$promptIds" --run-reference --approve-local-execution --output "$out"
        if ($LASTEXITCODE -ne 0) { throw "External reference failed for $($case.id). Preserve traceback." }
    }

Each reference is 32 full-sequence greedy forward steps with no cache,
using the separate installed Transformers runtime. Checkpoint hashes
must match both before and after inference. No new files are downloaded.

## C. Four separately approved real PhiStream comparisons

Each case first undergoes a NON-INFERENCE scenario review and must pass,
then PhiStream's original SINGLE-BUFFER model execution is approved.
Every comparison is SHA-bound to that exact reference file.

    foreach ($case in $cases) {
        $traceCase = Join-Path $matrixDir "reference_$($case.id).json"
        $reviewCase = Join-Path $matrixDir "preflight_$($case.id).json"
        py -m phistream.cli real-model-gate "$ckpt" --capacity-mib 8192 --reserve-mib 1024 --device scenario --trace "$traceCase" --source "HuggingFaceTB/SmolLM2-135M-Instruct" --revision $revision --license-id "Apache-2.0" --output "$reviewCase"
        if ($LASTEXITCODE -ne 0) { throw "Scenario review failed for $($case.id)" }
        $g = Get-Content "$reviewCase" -Raw | ConvertFrom-Json
        if ($g.status -ne "PREFLIGHT_TRACE_REVIEW_REQUIRED") { throw "Trace held: $($case.id)" }
        $parityCase = Join-Path $matrixDir "parity_$($case.id).json"
        py -m phistream.cli real-model-gate "$ckpt" --capacity-mib 8192 --reserve-mib 1024 --device cuda --trace "$traceCase" --source "HuggingFaceTB/SmolLM2-135M-Instruct" --revision $revision --license-id "Apache-2.0" --execute-parity --approve-local-execution --output "$parityCase"
        if ($LASTEXITCODE -ne 0) { Write-Warning "Parity did not pass for $($case.id). Keep receipt." }
    }

This is a separate explicit authorization for each PhiStream model run.
Review your own GPU capacity and do not run multiple jobs concurrently.

## D. Offline four-case evidence review (NO inference)

    $summary = Join-Path $matrixDir "matrix-review.json"
    py -m scripts.real_weight_matrix review --checkpoint "$ckpt" --preflight-receipt "$preflight" --manifest "$manifest" --receipts-dir "$matrixDir" --output "$summary"
    if (Test-Path "$summary") {
        $m = Get-Content "$summary" -Raw | ConvertFrom-Json
        [PSCustomObject]@{
            Status = $m.status
            PassingCases = "$($m.passing_cases)/$($m.required_cases)"
            UniquePredictedIDs = $m.unique_reference_greedy_ids_across_cases
            DistinctFiveTokenPrefixes = $m.distinct_first_five_greedy_sequences
            MedianDecoderSeconds = $m.median_decoder_seconds_descriptive_only
            ProductionReady = $m.production_readiness
        } | Format-List
    }

The reviewer writes the new local summary even on incomplete or failed
matrix evidence, then exits code 2. It never calls a model or CUDA.

## Status vocabulary

MATCHED_LIMITED_FOUR_CASES: all four cases exactly match the
user-declared reference at 32 IDs and all sparse logit samples, with
at least four unique observed greedy token IDs and two distinct
five-token prefixes across cases. This is still only four traces
for one model on one GPU, not general correctness.

MATCHED_LIMITED_LOW_OUTPUT_DIVERSITY: all four parity pairs matched
but the output was repetitive; do not generalize.

INCOMPLETE_EVIDENCE: an expected file is missing.
FAIL_TRUST_CONTRACT: a case, source, manifest, trace SHA256, execution
consent, reference declaration or ledger report cannot be trusted.
FAIL_PARITY: one or more true mismatches or incomplete matched counts.

The PR includes CPU-only negative-control tests for swapped traces,
edited logits, forged matching reports, missing cases, edited plans
and unauthorized double-buffer flags. They use entirely synthetic
untrained fixtures or mocked outputs. NO pretrained-model validation
is done in GitHub CI.

Scope deliberately excluded: full-vocabulary logit parity, natural
language tokenization, external reference cryptographic authentication,
repeated independent days/GPUs/models, real-model double-buffer
speedups, comparison to Hugging Face runtime speed or Ollama, and
production readiness. Model default remains SINGLE.

**Ledger above ego.** A mismatch or low-diversity result is useful
evidence, not a failure of the experiment.
