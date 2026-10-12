# ΦStream PR #19 · Real-Checkpoint Bridge, Phase 1

**This is a conservative opt-in gateway, not a new LLM runtime.** PhiStream
already had a *narrow* canonical Llama safetensors decoder and independent
full-sequence *synthetic fixture* tests. PR #19 connects its existing
safetensors inspection, metadata-based memory preflight, full checkpoint SHA-256
binding and user-supplied reference comparison under one explicit permission
boundary, with diagnostic timings.

No weights or tokenizer are included, downloaded, modified, copied, or
automatically selected. No Ollama/GGUF routing, PhiStream synthetic double
scheduler, or Vessie integration is enabled.

## Supported inputs (fail closed)

An existing, explicitly selected, licensed local directory (or supported
safetensors file plus its config) using canonical \`LlamaForCausalLM\` naming
and F32/F16/BF16 tensors. The existing strict decoder rejects unsupported
RoPE scaling, sliding attention, MoE, tensor-parallel variations, quantized
layouts, additional weight tensors and variant architectures. Many current
models will be **unsupported**; do not force compatibility by renaming them.

The source, pinned revision and license identifier are user declarations,
not authenticated proof of legal permission. PhiStream doesn't fetch a model,
validate a license, know its tokenizer or infer correct special tokens.

## Phase A: metadata preflight (default, no model inference)

Substitute your **actual, locally held, compatible safetensors directory**
for the example placeholder. Do not point this at an Ollama GGUF blob.

\`\`\`powershell
git pull --ff-only
$ckpt = "C:\PATH\TO\EXISTING\LICENSED\LLAMA_SAFETENSORS"
$stamp = Get-Date -Format "yyyyMMdd-HHmmssfff"
py -m phistream.cli real-model-gate "$ckpt" --capacity-mib 8192 --reserve-mib 1024 --context-tokens 40 --device scenario --source "DECLARED_SOURCE" --revision "PINNED_REVISION" --license-id "LICENSE_IDENTIFIER" --output "benchmarks\local\real-gate-preflight-$stamp.json"
\`\`\`

This parses headers and validates executable architecture and estimated
memory only. No CUDA query, tensor staging, SHA hashing or token generation.
A successful preflight returns \`PREFLIGHT_REVIEW_REQUIRED\`; the memory plan
is also \`REVIEW_REQUIRED\` by design, **not a guaranteed fit**.

Use \`--hash-checkpoint\` to additionally stream SHA-256 across every
local weight shard and config, which can take significant time. This
**reads the bytes**, but never changes them. The receipt withholds the
selected absolute model path by default. Hashes and user-declared
metadata may still be sensitive if shared.

For a CUDA availability-capped metadata preflight, use
\`--device cuda\`; it queries actual free VRAM but does not allocate
model weights. A requested unavailable GPU fails rather than switching
devices.

## Phase B: independent reference, only if YOU already have one

The existing [reference protocol](REFERENCE_TRACE.md) describes the local
\`PHISTREAM_REFERENCE_TRACE_V1\` JSON that another identifiable implementation
must produce from the **same exact safetensors bytes and input token IDs**.
At least 32 greedy next-token IDs, with sampled logits for each step, must
be supplied. One complete config-and-shards SHA-256 manifest is required.
The independent origin is user asserted and remains unauthenticated.

**Do not fabricate expected tokens from ΦStream's own output.** A real
reference trace must come from separately executed reference code/model.
Tokenizer and license compatibility are operator responsibilities.

Prevalidate its dimensions and output length *without executing weights*:

\`\`\`powershell
py -m phistream.cli real-model-gate "$ckpt" --capacity-mib 8192 --reserve-mib 1024 --device scenario --source "DECLARED_SOURCE" --revision "PINNED_REVISION" --license-id "LICENSE_IDENTIFIER" --trace "benchmarks\local\independent-reference.json"
\`\`\`

Output \`PREFLIGHT_TRACE_REVIEW_REQUIRED\` does not claim the SHA binding
or external implementation was independently verified. To check the
file's full SHA-256 without decoding, add \`--hash-checkpoint\`.

## Phase C: explicit >=32-token local parity execution

Only with your deliberate approval and a **previously inspected** model
and independent trace. This is real model-weight execution and can be
slow due to FP32 per-layer staging. It will not guarantee an 8 GiB VRAM
ceiling even with the metadata preflight.

\`\`\`powershell
$stamp = Get-Date -Format "yyyyMMdd-HHmmssfff"
py -m phistream.cli real-model-gate "$ckpt" --capacity-mib 8192 --reserve-mib 1024 --device cuda --max-tensor-mib 512 --head-rows 128 --source "DECLARED_SOURCE" --revision "PINNED_REVISION" --license-id "LICENSE_IDENTIFIER" --trace "benchmarks\local\independent-reference.json" --execute-parity --approve-local-execution --output "benchmarks\local\real-parity-$stamp.json"
\`\`\`

The command enforces:
1. Canonical supported architecture and tensor sizes from local safetensors.
2. Source/revision/license declarations (not license verification).
3. An existing local reference trace with >=32 expected greedy token IDs and
   each step's sampled logits.
4. A metadata-derived memory policy capped by currently observed free CUDA
   memory, with a user-defined reserve. Blocked memory plans return
   \`HELD_MEMORY_POLICY\`, record the reason, and **never start the decoder**.
5. Byte-for-byte SHA-256 equality of all checkpoint shards/config with the
   reference trace before executing. Mismatches error before decoder setup.
6. Explicit opt-in execution. Default command does not decode.
7. Teacher-forced reference token/logit comparison; status
   \`PARITY_MATCH_USER_REFERENCE_UNAUTHENTICATED\` or \`PARITY_MISMATCH\`
   (exit status 2 for mismatch).

**All decoding remains the existing single-buffer model path**.
PR #18 \`QUALIFIED_SYNTHETIC\` never authorizes this model decoder to
adopt double buffering. No automatic layer caching or checkpoint rewriting.

## New measurement semantics

The referenced decoder comparison now records:
- **Time to first logits, including engine setup and prompt processing**.
- Total comparison decoder wall time excluding complete checkpoint SHA hashing.
- Individual subsequent teacher-forced step durations and an approximate
  rate of subsequent tokens per second (not a separately measured natural
  autoregressive decode benchmark).
- Weight load count, bytes staged and staging time.
- Maximum observed per-step CUDA tensor allocation (PyTorch allocator only,
  not driver or whole-process VRAM peak).

This is *not* a benchmark against Ollama, AirLLM or llama.cpp, and it doesn't
implement tokenizer/EOS policies. A trace that matches 32 token IDs plus
sparse logits is **not** full-logit equivalence or authenticated reference
independence. No official external checkpoint has been exercised by CI.

## Four-step real-checkpoint release gate (still open)

- [ ] Operator identifies legally held, compatible local Llama safetensors
      and documents license/source/revision.
- [ ] Full fingerprint matches a truly independent reference trace for
      the same weights, tokenizer conventions and >=32 greedy steps.
- [ ] Actual GPU run returns full token/logit parity plus real measured
      first-logit latency, subsequent decode speed and memory allocation.
- [ ] Separate comparable single-vs-double **model execution** test,
      with active-layer and KV budget proof and independent review.

Until then, ΦStream is an **experimental** engine, not verified
real-world Llama inference or a speedup over competitors.
