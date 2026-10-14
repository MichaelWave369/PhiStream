# ΦStream engineering roadmap

## Current: v0.1.0, **experimental**

Original, model-format-constrained and offline. We prove decoder block parity between streamed and fully resident weights using a seeded *untrained* model. The machine executing `bench` generates a JSON receipt. This is not evidence that a 70B model runs, or that it runs faster than AirLLM, llama.cpp, Ollama, or any other engine.

## Gate A: correctness (before adding real model support)

**v0.2 validation-kit candidate:** `doctor --device cuda`, `validate-device`,
a Windows PowerShell script, and a local-only browser JSON receipt viewer are
available for testing. CPU CI validates the receipt contract; **no Nvidia
hardware test has been recorded yet**. These are not completed CUDA parity
checks until a user produces a verified receipt. See
[local device validation](DEVICE_VALIDATION.md).

- [x] Deterministic synthetic transformer checkpoint creation
- [x] File-mapped tensor storage, checksummed manifest
- [x] RoPE, RMSNorm, grouped-query attention, SwiGLU, per-layer KV cache
- [x] CPU stream/resident parity tests and receipt
- [ ] CUDA stream/resident parity on RTX 5070 and RTX 3060
- [ ] Long prefill (chunked) correctness and EOS/tokenizer integration

## Gate B: compatibility, opt-in, licensed local weights

**v0.3 checkpoint-inspection candidate:** independently written, strictly local
safetensors header/shard validation and canonical Llama shape preflight, with
one bounded CPU tensor probe and optional streamed fingerprints. This is NOT
inference, tokenizer integration, or a claim of GGUF/Ollama support.
See [safetensors gate](SAFETENSORS.md).

**Next experimental candidate:** canonical Llama token-ID execution from locally
supplied safetensors, compared on tiny untrained F32/F16/BF16 fixtures against an
independent full-sequence causal reference. Unsupported RoPE scaling, variant
architectures and quantization fail closed. No tokenizer or real-model
independent-reference parity has been established.
See [Llama execution](LLAMA_EXECUTION.md).

**Reference-trace validation candidate:** an offline CLI can compare local
Llama outputs against user-supplied reference greedy token IDs and sampled
logits, bound to SHA256 hashes of the complete checkpoint. The source of the
reference is **asserted, not independently authenticated** and no pretrained
model comparison has been run yet. See [reference protocol](REFERENCE_TRACE.md).

- [x] Implement independently authored *read-only* safetensors inspector and on-demand bounded CPU tensor reader (local format tests; not external fixture or inference parity)
- [x] Preflight one canonical Llama tensor naming/shape layout (structural only)
- [x] Implement narrowly scoped canonical Llama execution adapter with independent **synthetic** causal reference tests
- [x] Build fail-closed **user-supplied** reference-trace comparator (SHA256 bindings, teacher-forced greedy IDs, sampled logits, failure receipts)
- [ ] Establish token/logit parity on a real licensed Llama checkpoint against external reference runtime
- [x] Enforce local-only checkpoint parsing (no auto-download or trust_remote_code)
- [x] Load named local canonical Llama tensors with bounded sequential layer staging
- [ ] Compare token ids/logits against separately recorded licensed reference weights
- [x] No model weights bundled, uploaded or redistributed by preflight
- [ ] Verify source licenses for selected real models and local privacy

**PR #19 Governed Real-Checkpoint Bridge Phase 1:** single entry-point
read-only metadata preflight for explicitly selected canonical Llama
safetensors, source/license/revision declarations, memory-policy review,
optional full SHA-256 and reference-trace shape inspection. Opt-in >=32-step
greedy/sparse-logit parity execution requires a separate authorization and
full shard/config fingerprint match to a user-supplied reference. Decoder-only
timing and per-step peak CUDA allocated observations included. This
is **not** independent source/license authentication, official model parity,
real-model double-buffer activation, tokenizer or benchmark superiority.
See [Real Checkpoint Bridge](REAL_MODEL_BRIDGE.md).

**PR #20 External Reference Laboratory:** independent installed Hugging Face
Transformers is used only as an external offline reference runtime,
not integrated into PhiStream. Default hash-only check of pinned checkpoint,
with separately approved 32-step full-sequence greedy/sparse-logit trace,
before another explicit PhiStream single-buffer comparison. User-declared
provenance remains unauthenticated. No real SmolLM2 parity has been observed
in CI, nor real-model double-buffer performance. See
[External Reference Lab](EXTERNAL_REFERENCE_LAB.md).

## Gate C: measured performance (beat baseline **or do not claim victory**)

**PR #8 advisory candidate:** fresh host RAM + optional CUDA free-memory
snapshots feed a *read-only*, checkpoint-bound layer-placement **proposal**.
Persistent Llama GPU-cache placement has not been implemented, and the
proposal never authorizes execution. A separate bounded, cache-uncontrolled
transfer laboratory now measures synthetic bytes, not model streaming. See
[hardware advisor](HARDWARE_ADVISOR.md).


**Memory guard candidate:** `plan-memory` reports `BLOCKED_BY_POLICY` or
`REVIEW_REQUIRED` with `no_fit_guarantee=true`. An optional
`run-llama-ids --capacity-mib` guard refuses budget-blocked execution
before weight staging. CPU synthetic CI validates arithmetic only; no RTX
GPU results have been measured. See [memory preflight](MEMORY_PREFLIGHT.md).


- [x] Metadata-derived memory policy preflight for active FP32 layer, context KV, workspace and output projection, labeled scenario or observed CUDA free-memory (not a hard VRAM cap)
- [ ] GPU dynamic budget with measured allocator margin, adaptive layer residency and actual OOM recovery
- [x] Read-only Windows/Linux RAM, opt-in CUDA free VRAM and explicit drive capacity snapshots with metadata-derived, **unapproved** pinned-layer advice (not applied by current executor)
- [x] Bounded paired sequential file → host → CUDA pipeline timing with warmups, balanced ordering and whole-path wall-clock receipts (no model tensors)
- [x] Experimental 3-way multi-size pageable/fresh-pinned/reused-pinned CUDA synthetic transfer lab, balanced method ordering, before any real-model scheduler
- [x] Synthetic single- vs double-buffer pinned H2D experiment, CUDA completion-event ownership checks, balanced wall-clock receipts and full out-of-band GPU replay SHA audit (no model compute)
- [x] Synthetic CPU-host timeline and worker-reuse experiment with predeclared 5%/75% evidence gate; not cross-clock GPU overlap proof
- [x] Read-only multi-receipt synthetic evidence gate (three 12-round RTX cohorts), explicit scratch-only CUDA double opt-in, conservative single fallback and forced-serial negative control; NO model activation
- [ ] Async staging and prefetch with synchronization correctness tests
- [x] Controlled scratch file read, host copy and pageable/pinned H2D transfer lab with repeated JSON receipts (OS cache **not** controlled; not a real model benchmark)
- [x] Bounded synthetic `file.readinto()` comparison into preallocated pageable/pinned **host tensors**, with balanced sizes, checksum and GPU sample checks (OS cache uncontrolled; no real model)
- [ ] Pinned-memory and direct disk-read profiling (OS-dependent)
- [ ] Residency policy tuned against trace, not speculative NBG claims
- [ ] Prefill batching, GQA KV budget, quantized storage as optional opt-in
**PR #13:** CUDA reusable-buffer size sweep records per-size timing and
method wins for synthetic bytes; actual RTX hardware receipt pending.
Full model parity and model-layer prefetch remain unverified. See
[reusable buffer lab](REUSE_LAB.md).

- [ ] Compare against Ollama/llama.cpp on same exact weights/tokens/hardware
- [ ] Report TTFT, decode tokens/sec, real peak GPU allocation, RAM, disk reads, errors
- [ ] Record throughput separately for cold/warm caches and context lengths 128/512/2048

**PR #9 transfer laboratory:** bounded local scratch fixture, repeated
post-write disk reads, host bytearray copies and CUDA pageable/pinned
transfers with explicit synchronization and end-to-end staging caution.
User RTX performance is unmeasured until its local receipt is returned.
See [transfer laboratory](TRANSFER_LAB.md).

**PR #10 paired pipeline candidate:** balanced AB/BA pageable and reusable
pinned-host staging for synthetic bytes, whole synchronous file-to-GPU path
measurement, explicit warmups and stage attribution. OS cache uncontrolled;
no compute overlap or live model stream. See [pipeline lab](PIPELINE_LAB.md).

**PR #14 Direct-to-Pinned staging:** opt-in synthetic laboratory compares
multi-copy Python preparation with bounded `file.readinto` into preallocated
CPU tensors (pageable/pinned), followed by synchronized H2D. CUDA validation
on the RTX remains to be recorded. This is **not** disk-to-VRAM DMA, cold-cache
SSD speed or proof of inference acceleration. See [Direct Read Lab](DIRECT_READ_LAB.md).

**PR #15 (bounded streaming candidate):** a background scratch-file `readinto` can be scheduled while CUDA H2D is pending, with event synchronization **before slot reuse**. Buffered tests are measured by complete wall-clock time; pending events do not prove real overlap. No model weights, GPU computation, cold-cache disk reads, or autonomous prefetch. See [chunked lab](CHUNK_LAB.md).

**PR #16 Overlap Intelligence:** new bounded local scratch test compares single,
new-worker and persistent-worker strategies, measures host-clock read/queue/
enqueue/completion-observation intervals, and marks insufficient improvements
INCONCLUSIVE. GPU events are a separate, **uncalibrated** clock domain; no
true physical I/O/GPU overlap or inference-speed improvement claim. See
[Overlap Lab](OVERLAP_LAB.md).

**PR #17 Guarded Scheduler candidate:** 3 independent 12-round synthetic
CUDA receipts, matching platform/PyTorch/GPU identity, raw sample/policy
arithmetic checks and full out-of-band SHA attestations (still user-provided
and unauthenticated). Explicit separate operator flag can enable an owned
scratch-only double-buffer experiment with same-worker serialization
negative control. Default SINGLE; production mode never changed.
See [guarded scheduler contract](GUARDED_SCHEDULER.md).

**PR #18 Qualification candidate:** read-only full guarded receipt census including
borderline 4.97% run; three conservative statuses; minimum three balanced
randomized-order experiments before synthetic qualification; no runtime
promotion. Explicit-only synthetic CUDA control-order shuffle; default
single buffering. See [Qualification Lab](QUALIFICATION_LAB.md).

### PR #21: Four-case real-weight validation matrix (awaiting user execution)

The user locally recorded one SmolLM2-135M-Instruct reference parity run:
32/32 teacher-forced greedy IDs and 288/288 sparse probes on RTX 5070,
with 29/32 greedy outputs repeating ID 32. This is a genuine one-sequence
comparison against a user-declared external runtime, not broad correctness.
A four-case additional prompt-length/token-ID matrix is now defined, with
each reference trace bound by its SHA256 into a new real-model-gate receipt.
Operator-approved local execution is still required separately for each
reference and parity run. CI mocks inputs and verifies fail-closed ledger
logic, not any actual pretrained weights. Four-case results, model-aware
double buffering, performance speedup, all-logit parity, tokenizer support
and production readiness are **not yet demonstrated**. See
[the matrix operator protocol](REAL_WEIGHT_MATRIX.md).

### PR #22: read-only weight stage counter audit (operator receipts pending)

The user-reported 4-case SmolLM2 real-weight matrix matched 128/128
greedy IDs and 1,147/1,147 sampled logits. Existing single-token
streaming timings reveal 8.61 to 14.80 GB cumulative weight staging
with 9.83 to 16.14 seconds observed decoder duration, and about 82%
of that time within the broad weight-loading timer. Every extra
prompt token adds exactly 269,030,016 staged bytes, the checkpoint
payload: clear structural evidence of repeated model streaming
during serial-token prefill. The offline PR22 audit binds prior
receipt hashes and verifies the byte/load-count identity, with
no new model execution. CPU CI uses synthetic mocks only.
Potential causal batched-prefill savings remain a hypothesis requiring
new parity, memory and randomized timing evidence; scheduler SINGLE
remains unchanged. See [weight staging audit](WEIGHT_STAGE_AUDIT.md).

## Gate D: Vessie integration

- [ ] Optional, default-off local process with allowlisted model path
- [ ] Broker performs preflight and selects local Ollama when faster
- [ ] BudgetGenius records disk/time/energy costs, not solely VRAM
- [ ] Anti-M enforces rollback, receipts, timeouts and verified DONE
- [ ] Refuse remote downloads, broad filesystems, autonomous model installation

## Falsifiable milestone

For at least one agreed *real* local model on Windows 11 with the RTX 5070 12GB GPU: generate >= 32 exact matching greedy token IDs relative to an independently loaded reference, fit the specified measured VRAM budget, and publish the complete receipt. For a “better than AirLLM” claim, run both on the same model+revision+hardware and document the comparison with uncertainty. No invented benchmark claims.

### PR #23: separate causal batched-prefill research candidate (no production activation)

The user provided a passing read-only PR22 audit of their prior SHA-bound four-case receipts: `EXACT_STAGE_COUNTER_LINEARITY_OBSERVED`, 269,030,016 staged checkpoint bytes and 655 weight-load operations per existing SINGLE decoder step. This records a structural counter identity, **not a speedup** or independently authenticated execution. PR23 adds a separate 2..24 token causal prefill candidate and CPU-only synthetic agreement/negative-control tests. The canonical SINGLE decoder, parity bridge and CLI stay untouched. Real-checkpoint 32-step parity, GPU memory/timing receipts, repeatability, and promotion remain unapproved and unmeasured. See [batched prefill candidate](BATCHED_PREFILL_CANDIDATE.md).

### PR #24: SHA-bound opt-in pretrained batched-prefill parity gate (operator receipts pending)

The candidate now has a default offline planning path that authenticates **local consistency** of the earlier SHA-bound four-case matrix and a separately approved, fail-closed CUDA comparison for `mixed_7`, `repeat_12`, and `spread_24` (the single-token `short_1` remains control). It reuses immutable trace hashes and requires exact 32-step greedy/sparse-logit parity. It does not auto-run on user hardware; CI uses small synthetic fixtures. A passing candidate result does not independently authenticate source/revision/license or prove performance. Randomized real-weight A/B remains a separate later decision. See [PR24 real gate](BATCHED_PREFILL_REAL_GATE.md).

### PR #25: repeated real-model A/B evidence gate (local execution not recorded in CI)

User-supplied prior real candidate receipts show batched parity across mixed_7, repeat_12, spread_24 (96/96 greedy IDs; 859/859 sparse probes), with descriptive one-shot times below historical SINGLE results. This **does not** establish causal speedup. PR25 adds a default read-only planning CLI, full original baseline/candidate SHA verification, 4–20 paired CUDA trials with balanced AB/BA order and seed, per-trial 32-step parity, explicit approval, conservative full-context memory estimates, fail-closed receipt and human-review-only descriptive performance gates. The decoder and SINGLE default remain unchanged. See [paired A/B protocol](PAIRED_REAL_PREFILL_AB.md).

### PR #26: frozen three-case real A/B evidence census (operator review pending)

User-local PR25 receipts reportedly passed full parity and review thresholds for the three original prompt cases, with 12/12 BATCHED paired wins. The independent read-only PR26 aggregator rechecks source-file SHA bindings, saved plans vs recomputed preflight, all warmups/paired rounds and recorded per-step parity, and independently recalculates each summary. The census writes one new JSON without model execution, GPUs, uploads or promotion. Source/license/reference independence remain user-asserted. Cross-day/new-seed replication, long contexts, RTX 3060, VRAM safety and independent engine comparisons are still **not established**. See [three-case census](THREE_CASE_AB_CENSUS.md).

### PR #28–29: Cross-day replication audit completed locally, public evidence updated

The October 13 RTX 5070 eight-pair, seed-70421 `spread_24` replication is **operator-reported**, with a 40.59% median paired decoder-time reduction and 8/8 BATCHED-faster paired trials, versus the original 39.68% and 4/4. The operator reports the read-only SHA-bound PR28 review **completed successfully**: it revalidated the three original seed-369 plan/run pairs and PR26 census, recomputed the new eight-pair run, and produced a separate local receipt with `REPLICATION_REVIEW_CANDIDATE_NOT_SPEEDUP_CERTIFIED`, 20/20 BATCHED-faster pairs, 1,536/1,536 greedy checks and 13,774/13,774 sampled logits across 48 recorded local executions. The public GitPage now reflects the operator-reported locally audited outcome. The updated GitPage explicitly separates the operator's report from locally SHA-audited results. No longer-context testing, model-publisher authentication, independent time attestation, hardware generalization, production promotion or statistically certified speedup is claimed. See [the offline review procedure](REPLICATION_REVIEW.md).

### PR #30: Long Context Laboratory, all experiments held

A metadata-only planner documents frozen future input lengths of **128, 512, 2,048** tokens with a 32-step continuation, hence model-context requirements of **159, 543, 2,079** positions. Every case is held. The existing experimental batched-prefill implementation still rejects prompts above 24 tokens, and the existing paired A/B harness still uses three previously frozen cases. The PR adds only an offline header/config read, scenario FP32 memory estimate, conservative theoretical score-matrix byte count and local JSON receipt; **no real CUDA free-memory reading, full-checkpoint SHA, inference, oracle generation, benchmark, scheduler change or execution authorization**. A future separately approved and tested chunked implementation plus frozen reference parity and memory monitoring is required. Details in [the long-context laboratory protocol](LONG_CONTEXT_LAB.md).
