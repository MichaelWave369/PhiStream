# ΦPocketRotor E0 · Bounded Recursive Weight Residency

**Status: opt-in research, not production, not a large-model fit claim.**

ΦPocketRotor E0 tests the specific hypothesis that retaining complete
layer-weight bubbles in a finite GPU/CPU tensor cache can reduce repetitive
weight staging compared with the canonical serial decoder and a conventional
flat LRU cache. The original `StreamingLlama`, all existing inference
commands, PhiOS and Ollama remain unchanged.

## Concrete mapping

- Root = the explicitly selected, unchanged canonical Llama checkpoint.
- Layer bubble = nine canonical tensor names, ordered by layer number.
- Square pocket = one immutable tensor identified by a finite quaternary
  address `<layer-quadtree>/<tensor-quadtree>`; 0=NW, 1=NE, 2=SW, 3=SE.
- Hot tier = an exact-byte-budgeted dictionary of FP32 execution tensors
  on the decoder's explicit CPU/CUDA device.
- Cold tier = the same existing trusted, read-only local safetensors reader.
- Gear = a deterministic cache policy. The E0 nested policy *pins whole
  layer bubbles* in ascending layer order, while flat LRU evicts per-tensor.
  The pin plan is computed before execution and cannot mutate weights.

This is an addressing and *residency* study, not infinite compression,
MoE expert routing, predictive prefetch, asynchronous PCIe transfer or
quantization. A dense decoder still executes every layer. The nested policy
is deliberately simple so it must prove itself against standard LRU.

## Offline plan: NO model execution

Use a **previously obtained, trusted, local canonical LlamaForCausalLM
safetensors** checkpoint supported by the strict existing PhiStream loader.

```powershell
py -m scripts.pocketrotor_lab "D:\PhiStream\Models\MyTrustedLlama" --ids "1,2,3,4" --device cpu --cache-mib 256 --capacity-mib 8192 --reserve-mib 1024
```

Default behavior only parses metadata, validates the model layout and
proposes a deterministic residency plan. It cannot load weights or execute
the decoder. Review the `REVIEW_REQUIRED` / `HELD_BY_POLICY` decision,
available memory and whether any full layer fits the cache.

## Explicit experiment: only after plan review

```powershell
py -m scripts.pocketrotor_lab "D:\PhiStream\Models\MyTrustedLlama" --ids "1,2,3,4" --device cuda --cache-mib 256 --capacity-mib 12288 --reserve-mib 2048 --max-tensor-mib 512 --run --approve-local-execution --output "D:\PhiStream\Experiments\pocketrotor-e0-first.json"
```

The output path's parent must exist, and the file must **not** already exist.
No downloads, tokens, telemetry, checkpoint modification or production
integration occur. `--run` without `--approve-local-execution` fails closed.

Runs one fixed-order comparison with the same loaded checkpoint and token
sequence: `stream`, `flat-lru`, `nested-pin`. It tests entire-token
logit parity against the unchanged streamed decoder, hashes all original
checkpoint bytes before/after, and records timing, staged bytes, cache hits,
misses, evictions, admitted bytes and peak PyTorch allocated bytes when CUDA
is used. Any output mismatch fails the experiment; it cannot be reported as
a successful benchmark.

The memory preflight charges the cache budget *in addition to* the
estimated active decoder footprint. This is conservative policy accounting,
**not a hard GPU allocation guard**. CUDA context, allocator fragmentation,
other processes, CPU copies, temporary allocations and KV growth still matter.
The GPU cap must be inspected on the specific hardware.

## Falsifiability and honest interpretation

- Passing CPU CI tests proves only local fixture parity, policy invariants
  and the pure routing/cache implementation. It does **not** authenticate
  a pretrained model, real hardware performance or an external oracle.
- The E0 local comparison is fixed-order, one-shot and confounded by OS page
  cache, warmups, thermal state, time-to-first-token and running processes.
  It is a **further-review-only** pilot, not a speedup certification.
- Nested pinning can underperform LRU. When the cache fits no complete layer,
  nested E0 intentionally bypasses all layer tensors.
- Head/embedding rows and growing KV cache stay on their existing paths.
  Only the nine named canonical tensors in each decoder layer are cacheable.
- Actual model FP16/BF16 checkpoint bytes are converted to FP32 execution
  tensors; residency accounting uses *FP32 bytes*, not file sizes.
- The mapping is a finite positional quadtree. We make no inference about
  extra spatial dimensions, NBG mathematical closure, or artificial cognition.
- No independent external-reference trace is consulted by this E0 lab.
  Existing real-checkpoint reference gates remain separate requirements.

**Next gate:** independently verified real-checkpoint parity, randomized
balanced cold/warm end-to-end A/B studies, and a predictive prefetch candidate
only if traces justify its complexity. A future MoE adapter needs its own
native-routing preservation, quantization, and accuracy gates.

**Ledger Above Ego.**
