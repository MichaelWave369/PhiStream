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

- [x] Implement independently authored *read-only* safetensors inspector and on-demand bounded CPU tensor reader (local format tests; not external fixture or inference parity)
- [x] Preflight one canonical Llama tensor naming/shape layout (structural only)
- [ ] Support explicit **one** Llama-compatible *execution* architecture/config adapter
- [x] Enforce local-only checkpoint parsing (no auto-download or trust_remote_code)
- [ ] Load local Llama weights into PhiStream's executor
- [ ] Compare token ids/logits against separately recorded licensed reference weights
- [x] No model weights bundled, uploaded or redistributed by preflight
- [ ] Verify source licenses for selected real models and local privacy

## Gate C: measured performance (beat baseline **or do not claim victory**)

- [ ] GPU dynamic budget with safety margin for KV/cache/activations
- [ ] Async staging and prefetch with synchronization correctness tests
- [ ] Pinned-memory and direct disk-read profiling (OS-dependent)
- [ ] Residency policy tuned against trace, not speculative NBG claims
- [ ] Prefill batching, GQA KV budget, quantized storage as optional opt-in
- [ ] Compare against Ollama/llama.cpp on same exact weights/tokens/hardware
- [ ] Report TTFT, decode tokens/sec, real peak GPU allocation, RAM, disk reads, errors
- [ ] Record throughput separately for cold/warm caches and context lengths 128/512/2048

## Gate D: Vessie integration

- [ ] Optional, default-off local process with allowlisted model path
- [ ] Broker performs preflight and selects local Ollama when faster
- [ ] BudgetGenius records disk/time/energy costs, not solely VRAM
- [ ] Anti-M enforces rollback, receipts, timeouts and verified DONE
- [ ] Refuse remote downloads, broad filesystems, autonomous model installation

## Falsifiable milestone

For at least one agreed *real* local model on Windows 11 with the RTX 5070 12GB GPU: generate >= 32 exact matching greedy token IDs relative to an independently loaded reference, fit the specified measured VRAM budget, and publish the complete receipt. For a “better than AirLLM” claim, run both on the same model+revision+hardware and document the comparison with uncertainty. No invented benchmark claims.
