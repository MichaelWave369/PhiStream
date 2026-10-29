# ΦPocketRotor E2-R · Native Expert Route Capture Bridge

**Status: opt-in capture interface and offline trace converter.** No real
model is loaded or instrumented by the CLI. Actual selected expert IDs must
be supplied from a separately controlled model runtime. A capture marked
"operator_asserted_native" is **not cryptographically authenticated**.

E2 made deterministic predictions on synthetic per-expert activation traces.
E2-R creates the missing versioned route-capture contract needed to accept
real selections *without modifying any model, tokenizer, or routing result*.

## Integration principle

Do not infer selected experts from logits, recompute top-k in this bridge,
intercept user prompts, or attempt to guess a model's internal routing logic.

A compatible **trusted local runtime** integrates the tiny recording API
at the point **after its native router finalized expert selection**:

~~~python
from scripts.pocketrotor_e2_route_bridge import RouteEventRecorder

recorder = RouteEventRecorder(
    source={
        "kind": "operator_asserted_native",
        "runtime": "local-verified-runtime-name-and-version",
        "model_id": "local-verified-model-revision",
        "checkpoint_sha256": "64-lowercase-hex-digits-from-independent-hash",
        "route_semantics": "POST_NATIVE_ROUTER_SELECTED_EXPERT_IDS",
        "token_text_recorded": False,
    },
    expert_count=16,  # must match this model layer's native expert count
)

# INSIDE the runtime's own post-router instrumentation callback:
# one event per (split, token step, layer), in chronological order
# recorder.record(split=current_split, layer=native_layer_number,
#                 step=current_step_in_split,
#                 experts=list(native_selected_expert_ids))

# After all train and holdout segments have completed:
capture = recorder.export()
# The caller can serialize it to an explicitly chosen local JSON file.
~~~

**This is an integration interface, not a prewired hook for Qwen or other
models.** The runtime owner must verify whether its router uses grouped
top-k, shared experts, expert capacity drops, adjusted scores, batched
tokens, or other model-specific policies. The IDs supplied must reflect
the **actual final expert selection**, not a competing routing algorithm.

No prompt text, tokenizer token IDs, router logits, chat history or private
user data fields are accepted. Metadata are limited to bounded source
labels, claimed checkpoint SHA256, routing stage and categorical events.

## Contract

The root JSON object accepts only:
\`schema\`, \`source\`, \`expert_count\`, \`events\`.

The source has exactly:
\`kind\` (\`synthetic\` or \`operator_asserted_native\`),
\`runtime\`, \`model_id\`, \`checkpoint_sha256\`,
\`route_semantics="POST_NATIVE_ROUTER_SELECTED_EXPERT_IDS"\`,
and \`token_text_recorded=false\`.

Each event has only \`split\`, \`layer\`, \`step\`, \`experts\`.
Events are grouped by chronological split: **train first**, followed
by 1..8 named holdouts. Each split must record the same 1..128
selected layers and contain complete, gapless token-step indices
0..N-1 for every layer. All recorded layers must have the same
N within a split. Each sequence has 2..4096 steps; expert indices
must be 0..127, distinct per step, with at most 8 selected experts
per routed group. This first version supports only uniform
expert_count across captured layers.

This structure deliberately fails closed when a model's output does
not meet the simplified E2 simulation assumptions. The bridge does
not remap unknown architectures or conceal dropped events.

## Run the no-inference demonstration

The standard-library-only CLI works in a normal GitHub Actions CPU runner:

~~~sh
python -m scripts.pocketrotor_e2_route_bridge --help
python -m scripts.pocketrotor_e2_route_bridge demo --output /tmp/pocketrotor-e2r-capture.json
python -m scripts.pocketrotor_e2_route_bridge inspect --input /tmp/pocketrotor-e2r-capture.json
python -m scripts.pocketrotor_e2_route_bridge convert \
  --input /tmp/pocketrotor-e2r-capture.json \
  --layer 1 --expert-bytes 4096 --capacity-bytes 16384 --max-prefetch 1 \
  --trace-output /tmp/pocketrotor-e2r-layer1.json \
  --manifest-output /tmp/pocketrotor-e2r-layer1-manifest.json
python -m scripts.pocketrotor_e2 evaluate \
  --trace /tmp/pocketrotor-e2r-layer1.json
~~~

The demo is explicitly synthetic. No external weights, Ollama, GPU,
model hubs, downloads or connected services are involved.

The converter exports **one routed layer at a time**. Every E2 trace
uses only that layer's chronological training and holdout routes,
preventing accidental cross-layer adjacency. A separate manifest binds:

- SHA256 of the exact source capture file as read from disk;
- canonical content SHA256 of the E2 trace, independently visible in
  E2 \`analyze\` results;
- the selected layer, source-declared runtime and model;
- the caller's **assumed**, not measured, expert size/cache budget;
- explicit negative provenance claims.

Both output files must be new and their directories must exist.
No original capture or checkpoint is modified. If one of two output
writes fails, preserve any partial artifact and do not call the
conversion complete. Avoid copying raw routing captures from private
production chats into public issue trackers.

## Scientific limits and next gate

The bridge does **not** attest that supplied routes came from the
asserted checkpoint. A checksum typed in a manifest is not independent
verification of execution. An attacker could fabricate a well-formed
capture that appears real.

E2's replay still uses one uniform expert-byte size per model, and
there is no actual MoE implementation or GPU prefetch. Real Qwen MoE
routing and quantized weights are not supported by PhiStream's narrow
canonical Llama decoder.

The next rung should instrument one known, explicitly consented,
locally available MoE runtime with a **model-specific** adapter. That
adapter must compare its captured IDs to the runtime's actual native
router outputs, report source/version/hash provenance, freeze a
held-out workload, and never claim model-native parity from trace
formatting alone. Only after that should we implement physical
expert-weight staging and benchmark actual transfer overlap.

**Ledger Above Ego.**
