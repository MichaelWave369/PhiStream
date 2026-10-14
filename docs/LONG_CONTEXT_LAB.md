# ΦStream Long Context Laboratory · PR #30 (planning only)

**Status:** `ALL_LONG_CONTEXT_TARGETS_HELD_PROTOCOL_ONLY`

This is an original, separately gated research protocol for **128**, **512**, and **2,048** input-token experiments. It is **not** a longer-context inference implementation. The existing tested BATCHED path is still capped at **2–24 prompt tokens** (`phistream/prefill.py`); the PR25 A/B runner still accepts only the previously frozen 7/12/24-token cases. PR #30 changes neither.

## Why a planning gate is needed

Previous local RTX 5070 experiments measured a promising BATCHED benefit on a limited SmolLM2-135M-Instruct checkpoint and prompts no longer than 24 tokens. Those results do not demonstrate memory safety or accuracy for longer prompts. Batching all prompt positions with dense causal attention can require **quadratic** score tensors, in addition to layer activations, KV cache, host RAM, GPU allocator overhead and other workspaces.

The checkpoint's context limit must include continuation positions. For a 32-step teacher-forced trace, the final model input position uses **prompt_length + 31**. Therefore these three proposed inputs need:

| Input prompt tokens | Full context requirement | Gate |
|---:|---:|---|
| 128 | 159 | Held: no candidate implementation/reference approval |
| 512 | 543 | Held: no candidate implementation/reference approval |
| 2,048 | 2,079 | Held; additionally blocked if checkpoint context limit is below 2,079 |

An advertised 2,048-token context is insufficient for a 2,048-token prompt **plus** a 32-step continuation. A larger declared context does not itself make the experimental prefill safe or executable.

## What the read-only planner does

The script opens an **explicit local safetensors directory** using the repository's bounded header inspector and supported narrow Llama layout validator, without materializing weights. It:

1. Reads the checkpoint's model configuration, tensor shapes, advertised context limit and FP32 execution dimensions.
2. Computes the full context requirement for all three targets.
3. Runs the existing **metadata-only** `memory_preflight` for targets **within** the checkpoint context limit. Its result is either `BLOCKED_BY_POLICY` or `REVIEW_REQUIRED`; the latter **does not mean actual GPU fit or approval**.
4. Computes the byte count of **one theoretical** dense FP32 prompt-by-prompt attention-score matrix: `attention_heads × prompt_tokens² × 4`. It is not a CUDA peak, execution recommendation or valid future chunked implementation estimate.
5. Reports all target statuses as `HELD_PROTOCOL_ONLY_NO_EXECUTION`, with explicit missing gates.
6. Writes **one new, never-overwritten** JSON receipt. It includes the SHA256 of `config.json` **only**. It does not hash safetensors weight payload bytes and therefore is **not a SHA-bound full-checkpoint artifact**.

The script has only a `plan` command. There is **no** `run` subcommand, `--approve-local-execution` option, live CUDA query, tokenizer, reference generator, inference engine or scheduler access.

### Operator PowerShell, safe after merge

From the existing local checkout, run this metadata-only command. The selected 8192 MiB capacity and 1024 MiB reserve are **user-declared scenario budgets**, not observed hardware free memory.

~~~powershell
cd "C:\Users\micro\Downloads\PhiStream"
git pull --ff-only
if ($LASTEXITCODE -ne 0) { throw "Git pull failed" }

$ckpt = "D:\PhiStream\Models\SmolLM2-135M-Instruct"
$matrixDir = "benchmarks\local\smollm2-matrix-20261012-131912204"
$planDir = Join-Path $matrixDir "long-context"
$planFile = Join-Path $planDir "held-plan-128-512-2048.json"

if (Test-Path $planFile) { throw "Planning receipt already exists; preserve it" }

py -m scripts.long_context_lab plan `
    --checkpoint "$ckpt" `
    --capacity-mib 8192 `
    --reserve-mib 1024 `
    --output "$planFile"
if ($LASTEXITCODE -ne 0) { throw "Long context metadata plan failed closed" }

$p = Get-Content "$planFile" -Raw | ConvertFrom-Json
$p | Select-Object status, model_execution, cuda_queried, production_promoted | Format-List
$p.targets | Select-Object prompt_tokens, full_context_tokens_required,
    full_context_fits_checkpoint_limit, status | Format-Table -AutoSize
~~~

Expected: `ALL_LONG_CONTEXT_TARGETS_HELD_PROTOCOL_ONLY`, three held rows, `model_execution=False`, `cuda_queried=False`, `production_promoted=False`. This script does not generate measured results. You can open the receipt in the **Long Context Laboratory** panel on the GitPage, which parses it in-browser without uploading or persisting its contents.

## Future gates, **not implemented or authorized by PR #30**

**Gate A: design and correctness.** Choose a chunked, memory-bounded causal prefill plan, with finite segment sizes, explicit absolute RoPE positions and KV-continuation compatibility. This requires a separate PR with CPU-only tests for causal masking, segmentation boundaries, cache shape/position, failure cases and output equivalence against SINGLE where feasible. **Do not remove or increase the existing 24-token cap** before this gate.

**Gate B: immutable reference traces.** Freeze held-out deterministic token-ID prompts and checkpoint/source declarations. Generate independent runtime traces through a separately reviewed/approved offline workflow, with 32 teacher-forced steps and sampled logits; hash and preserve originals. User-declared origin is not independent attestation.

**Gate C: resource and approval.** Recompute policy estimates against the precise full context, define a true bounded workspace, inspect actual free CUDA memory, retain reserves, and introduce an explicit per-case opt-in command. A `REVIEW_REQUIRED` estimate alone never authorizes inference.

**Gate D: controls and replication.** Apply exact greedy ID and sparse-logit parity to every warmup and timed trial; compare SINGLE and candidate on the same checkpoint with balanced randomized AB/BA order. Include cold/warm caches, OS/GPU background load, wall time, TTFT, GPU allocator peaks, staged bytes and negative cases. Preserve all receipts, even failed results.

**Gate E: review, not promotion.** Independent receipt arithmetic and SHA-bound consistency audit first. No automatic production routing, general speedup certification, large-model fit claims or release-state promotion.

**Ledger Above Ego.**
