# ΦPocketRotor E5 · Pretrained Router Pilot

**Status: next evidence rung, no real pretrained run has happened merely
because this PR exists or because the synthetic CI passes.**

E4 proved a read-only router hook can observe *actual Hugging Face
Qwen3.5 library code* running with tiny, untrained weights. E5 adds an
adapter for a substantially smaller **pretrained** Mixture-of-Experts
model, IBM's public \`ibm-granite/granite-3.0-1b-a400m-instruct\`.

Official model sources:
- https://huggingface.co/ibm-granite/granite-3.0-1b-a400m-instruct
- https://huggingface.co/ibm-granite/granite-3.0-1b-a400m-instruct/blob/23d6629dbac8c0a0652f28991f62bb75157c02e3/config.json
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/granitemoe/modeling_granitemoe.py

Unlike Qwen3.5-35B-A3B, Granite 3.0-1B-A400M Instruct is about 1.3B
total parameters, using **24 native routed layers, 32 experts per
layer and top-8 expert selection**. Its publicly released single
safetensors model shard is roughly **2.67 GB**. The actual model is
Apache-2.0 licensed (follow upstream notices).

## What the adapter does

\`scripts/pocketrotor_e5_granite.py\` attaches one read-only hook to
each installed native \`GraniteMoeTopKRouter\` and a matching hook to
each \`GraniteMoeExperts\`, under the conventional
\`model.layers.N.block_sparse_moe\` Python module path.

The gate's third output contains native selected expert IDs, while
the experts module takes the same IDs as its second argument. E5
compares IDs *exactly, in native order*, and refuses unrecognized
classes, missing/duplicated layers, skipped kernels, batch or multi-token
forwards, altered routing, oversized route groups, and incomplete
train/holdout sequences. It does not reroute experts or replace any
model output.

The source format is unchanged from E2-R; the resulting trace can
pass through the E2 finite-cache simulator. E2 still runs **logical
cache replay**, not actual disk or GPU expert paging.

## Two completely different workflows

### 1. Ordinary PR test: small random native CPU blocks

\`.github/workflows/pocketrotor-e5-granite-native.yml\`

Runs real Hugging Face Granite *classes* in a two-layer toy CPU model
with random, tiny weights. It checks exact unhooked-vs-hooked output
parity, 32 complete native routing events, proper source labeling,
and bad-module/changed-IDs guards.

This does NOT download a checkpoint or prove pretrained inference.

### 2. Explicit manual-only pretrained experiment

\`.github/workflows/pocketrotor-e5-pretrained-manual.yml\`

**Never triggered by PR/push/merge.** To launch *after you have
reviewed and merged* E5:

1. Open the GitHub repository's **Actions** tab.
2. Select **PhiPocketRotor E5 Pretrained IBM Granite Route Pilot**.
3. Choose **Run workflow** on \`main\`.
4. Select \`AUTHORIZE_IBM_GRANITE_2_67GB_CPU_DOWNLOAD\` rather than
   \`DO_NOT_RUN\`.
5. Review the result and its \`pocketrotor-e5-pretrained-router-evidence\`
   artifact. If the job fails, do not treat missing evidence as pass.

The self-contained runner pins:
- \`transformers==5.21.0\`;
- IBM model \`ibm-granite/granite-3.0-1b-a400m-instruct\`
  at revision \`23d6629dbac8c0a0652f28991f62bb75157c02e3\`;
- a CPU-only PyTorch installation from the official CPU wheel index;
- \`trust_remote_code=False\`, local-only loading **after** the explicit
  public checkpoint download, and required safetensors format.

It hashes the full downloaded safetensors file locally, verifies
the config has exactly 24 layers / 32 experts / top-8, measures actual
expert tensor sizes per layer, and runs **32 teacher-forced train**
token steps plus **two disjoint 16-token holdouts** on frozen public
English text. The model retains its native KV cache within each
split and resets cache between splits. It executes batch=1,
sequence=1 forwards only, with the native router's selected indices
compared directly against the indices consumed by the native experts.
The first token's unhooked vs hooked full-model logits are compared
bitwise. No model samples, private prompts or token IDs are uploaded.

The recording format stores one event per layer and token:
24 routed layers * 64 steps = **1,536 native event records**. It
replays the E2 policies for layers 0, 6, 12, 18 and 23 separately,
using cache capacity of 12 real-sized experts and one speculative
prefetch per step. The JSON receipt retains demand bytes and **total
transfer bytes** including speculative loads, plus flat-Markov and
ordinary-LRU controls. Results are not filtered to only winning layers.

Only the route IDs and a small JSON evidence summary are uploaded
for 14 days. **No model weights, prompt text or full token stream**
is included in the artifact.

## Evidence labels and limitations

If the manual job succeeds, it can substantiate *an operator-reported
capture of native routing decisions produced by a real pretrained
IBM Granite checkpoint*. That is distinct from an independent
attestation of provenance. The report binds the model revision,
local weight-file SHA256, source, split lengths and trace SHA256.

The following claims are **not** supported:
- Qwen3.5 pretrained routing fidelity: Granite is a different MoE.
- Measured RTX 5070 expert staging speedup or improved tokens/second.
- Model output or accuracy improvement due to nested bubbles: routing
  and inference remain canonical and unchanged.
- Deployment readiness, online/streaming performance or validation
  across many held-out workloads.
- Cross-model comparability from a single descriptive, short dataset.

The test is deliberately *not* a performance benchmark. It is a
limited **real-model selection probe** so we can stop treating
synthetic route patterns as if they describe actual MoE routing.

It downloads several gigabytes on a disposable GitHub runner and may
hit bandwidth, model-runtime, time or memory limits. Failure is a
valuable result and must remain visible. Do not bypass
\`--authorize-download\` or automatically schedule the pretrained job.

**Ledger Above Ego.**
