# ΦPocketRotor E2 · Nested Bubble Expert Trace Laboratory

**Status: offline trace-only simulator. Not a pretrained MoE runtime.**

E0 made finite square pockets addressable in an experimental canonical
Llama layer cache. E1 compared whole-layer pinning against LRU and
streaming with a SHA-bound reference. E1-A audits those receipts.

**E2 asks a different, falsifiable question:** Can a frozen *past-only*
predictor move future MoE expert weight pockets into a finite simulated
cache early enough to reduce on-demand loads, after accounting for
*every* speculative load and eviction? Does grouping experts in nested
quadtree parents add value over flat one-step Markov prediction?

## What exists and what does not

- Four independent replay policies: ordinary LRU without prefetch,
  global-frequency prefetch, flat one-step Markov prefetch, and
  nested-parent / direct-transition bubble prefetch.
- Experts have exact declared fixed byte sizes. Total cache capacity
  is finite, checked every replay step. Every expert loaded on demand
  or speculatively consumes the declared transfer bytes.
- The model's **original required expert IDs are never changed**.
  The learner is frozen on the training sequence. On each holdout
  step, it sees only the current required expert IDs before predicting
  candidates for the *next* step. It never sees future holdout labels.
- The square-pocket address uses the same bounded NW/NE/SW/SE base-4
  digits as E0. Shared parent prefixes supply transition evidence.
- Input hashes, a named same-family holdout, an intentionally shifted
  holdout, and a full breakdown of missed demand bytes, speculative
  bytes, evictions, and useful/wasted prefetches are recorded.
- **NO REAL MODEL EXECUTION, no CUDA, no GPU memory transfer, no
  physical disk streaming, no measured wall-clock performance.**
  Demand bytes are a *critical-path proxy*, not evidence that
  actual inference will speed up.

The in-memory predictor deliberately doesn't train itself on holdout
examples. Future work may compare frozen versus adaptive prediction
but must report its feedback timing and control leakage.

## Run it entirely on GitHub Actions or an existing Python environment

The code is standard library only and has no external packages or
models. GitHub Actions runs the same demo and synthetic tests on each PR.

~~~sh
python -m scripts.pocketrotor_e2 demo
python -m scripts.pocketrotor_e2 demo --output /tmp/pocketrotor-e2-demo.json
~~~

To evaluate a new, **user-owned**, non-sensitive expert-activation
trace instead:

~~~sh
python -m scripts.pocketrotor_e2 evaluate --trace /local/route-trace.json --output /local/new-result.json
~~~

\`--output\` only creates a **new** file and refuses to overwrite an
existing result; without it, JSON is emitted to stdout. Nothing uploads
the trace, fetches models, opens networks, or modifies the source.

## Accepted trace contract (JSON)

~~~json
{
  "schema": "PHISTREAM_POCKETROTOR_E2_TRACE_V1",
  "expert_count": 16,
  "expert_bytes": 4096,
  "capacity_bytes": 16384,
  "max_prefetch": 1,
  "train": [[0, 1], [4, 5], [8, 9], [12, 13]],
  "holdouts": {
    "same_family": [[0, 2], [4, 6]],
    "shifted": [[12, 14], [0, 1]]
  }
}
~~~

This is merely a *schema illustration*, not a trained benchmark.
Valid traces require 4..128 experts, 2..4096 training steps, 2..4096
steps for each of 1..8 holdouts, 1..8 distinct required expert IDs
per step, up to 16 MiB per expert, max 2 GiB simulated cache and
0..4 speculative loads after each step. Demand must fit in the
cache or validation fails. All expert indices must be in range,
and all fields are required with no extras.

The current simulator treats every expert as the same fixed size.
This simplifies the earliest falsification trial. A real MoE adapter
must support heterogeneous expert sizes and native routing metadata.

## What counts as evidence?

Compare every candidate with the same *flat LRU* replay, **and**
compare nested bubbles with flat Markov. Report at least:

- Demand misses and demand bytes (unavoidable synchronous loads in
  this simplified model).
- Speculative prefetch bytes and **total** transferred bytes.
- Useful prefetched blocks, unused blocks and evicted speculation.
- Cache capacity / peak logical occupancy.
- Same-family versus shifted-distribution outcomes.

A policy that reduces demand misses while doubling total transfers may
be physically slower. A model with high cache hits may still suffer
GPU stalls during unexpected expert bursts. No total-byte or cache
benefit can be translated into tokens/second without separate
hardware time measurements and correctly synchronized transfers.

The demo includes a deterministic training sequence and two separately
seeded evaluation sequences. It intentionally preserves failures.
No implied training accuracy, token-generation parity, model IQ,
or MoE correctness is inferred from replaying expert IDs.

## Next real-model gate

Before building a GPU expert loader, export a bounded, trusted **native
routing trace** from a known locally controlled MoE checkpoint through
a supported, independently tested reference runtime. Freeze the trace
and compare these E2 replay policies against a flat cache. Only after
we have genuine traces should we add a compatible quantized expert
storage adapter and measure overlapped prefetch on real hardware.

In particular, canonical \`StreamingLlama\` is **not** a Qwen-MoE or
GGUF executor. E2 does not pretend otherwise.

**Ledger Above Ego.**
