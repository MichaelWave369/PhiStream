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

## Gate D: Vessie integration

- [ ] Optional, default-off local process with allowlisted model path
- [ ] Broker performs preflight and selects local Ollama when faster
- [ ] BudgetGenius records disk/time/energy costs, not solely VRAM
- [ ] Anti-M enforces rollback, receipts, timeouts and verified DONE
- [ ] Refuse remote downloads, broad filesystems, autonomous model installation

## Falsifiable milestone

For at least one agreed *real* local model on Windows 11 with the RTX 5070 12GB GPU: generate >= 32 exact matching greedy token IDs relative to an independently loaded reference, fit the specified measured VRAM budget, and publish the complete receipt. For a “better than AirLLM” claim, run both on the same model+revision+hardware and document the comparison with uncertainty. No invented benchmark claims.
