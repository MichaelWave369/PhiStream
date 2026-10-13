# PR #24: SHA-bound real-weight batched-prefill comparison

## Why this is a separate experiment

PR23 introduced a default-disconnected 2..24 token causal batched-prefill
candidate. The original PR21 four-case RTX5070 matrix and PR22 read-only
stage audit established a useful baseline, **not proof** that the candidate
matches a pretrained reference or runs faster.

This experimental gate has a default **plan**, plus an independently
operator-authorized **run**. It uses the **existing** original checkpoint,
preflight SHA, frozen manifest, matrix review, all four original
reference/parity pairs and exact Hugging Face reference trace SHA256s.
It refuses altered baseline evidence or checkpoint bytes before executing.

The one-token `short_1` case is a control: PR23 requires 2..24
prompt tokens and therefore this gate only accepts `mixed_7`,
`repeat_12`, and `spread_24`. Do not substitute a new prompt.

## 1. Audit and plan, NO model execution

From `C:\Users\micro\Downloads\PhiStream` in PowerShell, after PR24
has passed CI and been merged:

```powershell
git pull --ff-only
$ckpt = "D:\PhiStream\Models\SmolLM2-135M-Instruct"
$preflight = Get-ChildItem "benchmarks\local\smollm2-cuda-preflight-*.json" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1 -ExpandProperty FullName
$matrixDir = "benchmarks\local\smollm2-matrix-20261012-131912204"
$gateDir = Join-Path $matrixDir "batched-prefill"
New-Item -ItemType Directory -Force $gateDir | Out-Null
$case = "mixed_7"
$plan = Join-Path $gateDir "plan_$case.json"

py -m scripts.batched_prefill_matrix plan `
    --checkpoint "$ckpt" `
    --preflight-receipt "$preflight" `
    --matrix-dir "$matrixDir" `
    --case "$case" `
    --output "$plan"
```

Expected plan status: `PLAN_SHA_BOUND_NOT_EXECUTED`.
It hashes the *existing* local checkpoint, verifies all four prior
SHA-bound real-model parity receipts and writes one new local JSON.
No inference, Transformers runtime, downloaded model or GPU execution
occurs. The original checkpoint and prior JSON receipts remain untouched.

The next command will reject attempts to overwrite the same output.
Use a new filename only if a deliberate new plan is needed.

## 2. Separately authorized real CUDA parity execution

Only after inspecting the plan, the code and available GPU resources,
the operator may choose to approve ONE of the three eligible cases.
The run executes a newly created experimental batched-prefill engine
on the existing checkpoint, then teacher-forces the other 31 steps
through the canonical single-token decoder:

```powershell
$case = "mixed_7"
$run = Join-Path $gateDir "candidate_$case.json"
py -m scripts.batched_prefill_matrix run `
    --checkpoint "$ckpt" `
    --preflight-receipt "$preflight" `
    --matrix-dir "$matrixDir" `
    --case "$case" `
    --device cuda `
    --capacity-mib 8192 `
    --reserve-mib 1024 `
    --approve-local-execution `
    --output "$run"
```

**This command executes real pretrained local weights and consumes CUDA
VRAM.** It must not be run as an implicit consequence of planning, CI,
or the PR merge; permission belongs to the local operator per case.
Each new case needs its own reviewed explicit command and output file.
The test gate has a memory-policy estimate for *all 32 steps* as well as
the extra batched-prompt workspace, not a hard GPU allocation guarantee.
It refuses to fall back to CPU.

The comparison requires all **32 exact greedy token IDs** and every
sampled logit probe (tolerances pinned by original trace). Each step
records expected/actual greedy IDs, whether all its probes matched,
and maintains teacher-forced continuation if an early step fails.
The result is labeled `MATCH_USER_REFERENCE_UNAUTHENTICATED` or
`PARITY_MISMATCH`. A mismatch exits 2 and writes a mismatch receipt,
not a success.

The run also records descriptive first-logits time, all 31 continuation
steps, staging counters and measured peak PyTorch CUDA allocation,
plus a full checkpoint and reference SHA recheck *after* execution.
No result may independently authenticate the declared reference source.
Checkpoint/model/license publisher remain user-declared.

## Explicit exclusions

- Does **not** modify `StreamingLlama.step()`, the established
  `real-model-gate`, the production scheduler, or default SINGLE.
- Does **not** run HF reference inference again or download weights.
- Does **not** compare against llama.cpp/Ollama or claim superiority.
- Does **not** constitute an A/B benchmark. Existing baseline times are
  one-shot measurements with uncontrolled cache/system confounders.
- Does **not** enable automatic model routing or production promotion.
- CI uses only synthetic untrained CPU fixtures and fail-closed mocks.

If real-model parity holds, the NEXT separate gate is repeated, paired
A/B timing with balanced/randomized order and predeclared acceptance
criteria, while preserving the baseline. A valid result might still
show zero improvement or even be slower.

**Ledger above ego.** Neither a green test nor fewer staged bytes
counts as a measured real-world speedup.
