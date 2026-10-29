# ΦPocketRotor E3 · Qwen3.5 Native Expert Capture Adapter

**Scope:** opt-in read-only post-router observation of **an already loaded**
Hugging Face Transformers Qwen3.5 MoE text decoder. No models, tokens,
weights, local files, or external services are loaded by this adapter.
It is **not** an Ollama/GGUF hook and **does not** execute Qwen3.5 itself.

Hugging Face's Qwen3.5 MoE router, as reviewed on 2026-10-29 in
\`transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py\`, returns
\`(router_logits, router_scores, router_indices)\`. Its
\`Qwen3_5MoeSparseMoeBlock\` then passes \`router_indices\` directly to
\`Qwen3_5MoeExperts\` as the second positional argument. This
implementation attaches an ordinary PyTorch **forward hook** to each
native \`Qwen3_5MoeTopKRouter\` and a **forward prehook** to the corresponding
native \`Qwen3_5MoeExperts\`, verifying that the expert indices that
actually enter computation match those selected by the gate. It never
recomputes top-k, modifies weights, changes routing, changes inputs
or replaces either module's results.

Source: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py
See also the official Qwen3.5 MoE documentation:
https://huggingface.co/docs/transformers/model_doc/qwen3_5_moe

## First 256-expert support

Qwen3.5-35B-A3B's text model uses **256 routed experts and top-k 8**.
The E2 simulator and E2-R capture bridge formerly accepted at most 128.
E3 expands the bounded limit to **256**, without relaxing the per-token
maximum of eight selected routed experts, file caps, or cache budgets.
The shared always-active expert is **not included** in the 256 routed
expert IDs. A native decoder may have a shared-expert path separately;
this E3 adapter observes only routed top-k.

The optional hook constructor defaults to
\`expected_experts=256\`, \`expected_top_k=8\`,
\`expected_layers=40\`, and verifies contiguous layer indices.

## Integrate into a trusted **existing** text-token inference runner

Only a locally controlled runner that already has a compatible
Qwen3.5 MoE **Transformers** model loaded in eval mode should integrate:

~~~python
from scripts.pocketrotor_e3_qwen35 import Qwen35NativeRouterCapture

source = {
    "kind": "operator_asserted_native",
    "runtime": "transformers-version-from-local-environment",
    "model_id": "verified-local-model-revision",
    "checkpoint_sha256": "64-character-lowercase-hex-of-actual-checkpoint",
    "route_semantics": "POST_NATIVE_ROUTER_SELECTED_EXPERT_IDS",
    "token_text_recorded": False,
}

# model is already loaded (outside PhiStream) and must be in eval mode.
# Do not automatically fetch a checkpoint or enable trust_remote_code.
# Reuse your runner's correct model-specific KV-cache and token plumbing.
with Qwen35NativeRouterCapture(model, source=source) as observer:
    # For each split ("train", then a holdout) and each *single* token:
    # result = observer.run_step(
    #     split="train", step=0,
    #     forward=lambda: model(**your_existing_single_token_inputs),
    # )
    # Feed the next token with the runner's native cache/state handling.
    # Repeat for at least 2 tokens in train AND each subsequent holdout.
    pass
capture = observer.export()  # only after context closed and complete splits
# Serialize the resulting capture yourself to a NEW private local JSON file.
~~~

**This code intentionally omits a generic model loader/forward loop.**
Qwen3.5 is a complex multimodal/Gated DeltaNet family with model-specific
cache and text-tower behavior, and a 35B-weight checkpoint cannot be
assumed to fit a 12GB GPU. A fragment that invents a generic KV cache or
silently downloads a huge model would not produce credible evidence.

### Strict capture gates

- \`model.training == False\` (eval); exact native class/module identity,
  one routed gate and one expert computation per layer, and expected
  contiguous layer indices. An unknown model revision fails closed.
- Only one token row and one batch element per native forward.
  Multi-token prefill or batch inference is rejected rather than
  mislabeled as an ordered decode trace.
- Every hook observes a **third native gate output** containing a
  signed-integer tensor of shape \`(1, top_k)\`. It verifies that the
  **second experts input** contains exactly the same IDs, in order.
- Every layer must execute once per \`run_step\`; skipped/replaced
  optimized kernels cause a hard failure, not a silent trace gap.
- Hook failure invalidates the capture. Closing removes all hooks even
  on exceptions. Nothing is exported from an active or failed session.
- Source identity and claimed checkpoint SHA256 stay **operator-asserted
  and unauthenticated**. A source label is not independent provenance.
- Recording IDs on GPU implies small host synchronization overhead.
  Timings from an instrumented model are **not valid inference speed
  benchmarks**.

### After a valid operator-approved run

Serialize the capture to a local file (not a public PR). Then the existing
E2-R CLI can inspect and convert one native routed layer at a time,
and E2 can compare LRU, frequency, flat Markov and nested bubble prediction.

~~~powershell
py -m scripts.pocketrotor_e2_route_bridge inspect --input "D:\PhiStream\Experiments\private-qwen-route.json"
py -m scripts.pocketrotor_e2_route_bridge convert --input "D:\PhiStream\Experiments\private-qwen-route.json" --layer 0 --expert-bytes 4096 --capacity-bytes 32768 --max-prefetch 1 --trace-output "D:\PhiStream\Experiments\qwen-layer0-e2.json" --manifest-output "D:\PhiStream\Experiments\qwen-layer0-manifest.json"
py -m scripts.pocketrotor_e2 evaluate --trace "D:\PhiStream\Experiments\qwen-layer0-e2.json"
~~~

**4096 expert bytes above is an illustrative simulation assumption,
NOT the actual Qwen3.5 expert size.** Real sizes and quantization must
be measured from a validated checkpoint before comparing hardware
performance.

## CI scope and future gate

E3 CI uses **tiny fake native-shaped PyTorch modules** whose test
classes mimic the HF gate and experts signatures. It validates hook
preservation, gate-to-consumer ID parity, privacy boundaries, 256-index
accounting, topology failure, multi-token refusal and conversion into
E2's replay policy. No Transformers dependency is added, no downloaded
checkpoint is executed, and no GPU proof is claimed.

The next real milestone is integrating this hook into an explicitly
reviewed local text inference runner, proving that its exact HF package
revision matches the inspected router, then capturing a small, genuinely
held-out set of native selection events. No performance claim until real
weights and transfers are separately instrumented.

**Ledger Above Ego.**
