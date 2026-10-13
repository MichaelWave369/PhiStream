# PhiStream PR #23: bounded causal batched-prefill candidate

## Purpose

The user's October 12, 2026 four-case SmolLM2-135M-Instruct offline
weight-stage audit returned EXACT_STAGE_COUNTER_LINEARITY_OBSERVED. Its
existing SINGLE decoder repeatedly staged the same layer weights once per
prompt token: 269,030,016 checkpoint bytes and 655 load operations per
single-token step. This identifies an opportunity, **not a speedup**.

PR23 introduces a **separate**, experimental callable in
phistream/prefill.py. It does not alter StreamingLlama.step(),
generate_ids(), compare_reference(), real-model-gate, CLI defaults,
checkpoint layout, memory advisor, or scheduler policy. Nothing automatically
switches to the candidate.

## How the candidate works

1. Require explicit approval of local execution and 2..24 literal token IDs.
2. Run a preflight of existing model metadata with a user memory policy,
   including a conservative extra batched-workspace estimate. On CUDA,
   also cap the budget with observed free VRAM. This is **not a hard GPU
   allocator or OOM guarantee**.
3. Create a new StreamingLlama instance and read local token embedding
   rows without checkpoint modification.
4. Process every prompt token through one transformer layer at a time,
   staging its nine layer tensors once for the entire prompt.
5. Use split-half absolute-position RoPE, grouped-query expansion, a strict
   upper-triangular causal mask, RMSNorm and SwiGLU. Preserve each layer's
   ordered KV tensors in the exact shape the existing step() expects.
6. Compute only final prompt-token output logits through the existing
   chunked vocabulary head. Leave future generated tokens on the original
   step() implementation.

Explicit developer API (not a public CLI and not a production path):

    from phistream.prefill import approved_batched_prefill
    engine, first_logits, candidate_evidence = approved_batched_prefill(
        checkpoint, [9, 28, 7, 139, 302, 900, 4096],
        approve_local_execution=True,
        device="cpu",
        capacity_mib=8192,
        reserve_mib=1024,
    )

Do not use this on the pretrained SmolLM2 checkpoint just because this
example exists. Each real-model experiment still requires explicit
operator approval after inspecting the installed candidate and the
existing fixed SHA-bound reference evidence.

## Tests and limitations

Run only the tiny untrained CPU synthetic tests:

    py -m unittest discover -s tests -p "test_batched_prefill.py" -v

These test: final prompt-logit parity against the single-token baseline
and separately hand-authored full-sequence reference, preserved per-token
KV state, compatibility with continuing SINGLE decoder steps, tied BF16
weights, forbidden future-token attention, denial without approval,
invalid inputs, budget rejection and CUDA refusal when unavailable.

No pretrained model or GPU runs in CI, and no real-case reference trace is
compared in this PR. Matrix arithmetic on CUDA can produce tiny floating-point
differences from the serial reference; real tolerance and greedy agreement
remain to be measured. The prototype returns an evidence map explicitly
marked UNVERIFIED; it is not a cryptographically bound benchmark receipt.
Its memory estimate is not an exact allocation bound. The candidate does
not make a quality, latency, bandwidth, cost, or VRAM-capacity claim.

## Gates before any performance or promotion claim

- Run synthetic tests and inspect changes.
- Establish full four-case 32-step greedy and sparse-logit agreement on the
  exact same SHA-bound pretrained checkpoint, *separately authorized*.
- Confirm resulting KV continuation and measured peak GPU allocation.
- Run repeated randomized/alternating SINGLE vs BATCHED A/B trials with
  predeclared acceptance thresholds. Record time-to-first logits, wall-clock,
  cache conditions, weight counters, GPU allocation and all failures.
- Fail closed on mismatch, OOM, untrusted trace, or unstable performance.
  Production SINGLE stays the default even if a candidate wins; later
  promotion requires a separate operator-approved change.

**Ledger above ego.** Lower staging counters are not lower measured seconds.
