# ΦStream (PhiStream)

**Original memory-tiered inference research engine | v0.1.0 experimental**

Developed for the MichaelWave369 ecosystem. Independent from AirLLM; no code, dependency, architecture adapter or model weights copied from AirLLM. The general technique of layer-by-layer execution is a widely known concept, not a proprietary algorithm.

**Status:** CPU-correctness prototype. The model is a deterministic **untrained** miniature decoder, so generated tokens have **no language meaning**. No support yet for existing Llama/Qwen checkpoints. CUDA operations are implemented using PyTorch but must be verified on actual NVIDIA hardware. **No 70B/405B claim** and no speedup claim.

## Why this exists

Streaming large neural networks layer by layer can make model weights exceed available GPU memory, but it pays a disk/host-to-device transfer penalty at *every* decode step. We want a system that decides when streaming is appropriate, verifies actual outputs, and accounts honestly for time, disk bandwidth and memory. “Better” requires measured comparisons, not marketing.

## Implemented

- Llama-style toy decoder: RMSNorm + RoPE + grouped-query attention + SwiGLU + per-layer KV cache.
- Hand-authored per-layer GPU/CPU streaming code, with fully resident reference mode.
- Memory mapped little-endian FP32 weights, strict manifest validation, optional per-tensor SHA-256 verification, no pickle.
- Static hot-layer pin budget (`--pin-mib`) to reduce repeated IO for a subset of weights. This is not a total VRAM budget.
- CPU fallback with synthetic parity tests; JSON bench receipts labeling hardware and provenance.
- No internet access or external code execution, no model auto-download, no telemetry uploads.

## Quick start

Requires Python 3.10+ and a compatible PyTorch install. On Windows, install the **appropriate official PyTorch build for your CUDA driver** before installing this project; `pip` alone does not guarantee GPU acceleration.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python -m phistream.cli init-fixture artifacts/demo
python -m phistream.cli verify artifacts/demo
python -m phistream.cli parity artifacts/demo --device cpu
python -m phistream.cli bench artifacts/demo --mode stream --device cpu
python -m phistream.cli bench artifacts/demo --mode resident --device cpu
```

On CUDA-ready hardware, use `--device cuda` and compare stream/resident receipts on the **same** checkpoint:

```bash
python -m phistream.cli parity artifacts/demo --device cuda
python -m phistream.cli bench artifacts/demo --mode stream --device cuda --pin-mib 1
python -m phistream.cli bench artifacts/demo --mode resident --device cuda
```

## Important benchmark caveats

- Performance on synthetic toy models does not predict performance on 70B or MoE models.
- For CPU streaming, file mapping is lazy but the OS page cache is outside our control, so `weight_bytes_loaded_total` is not physical disk IO.
- The current engine streams *weights*, not the growing KV cache. KV and activations still consume memory.
- `--pin-mib` limits persistent **layer parameters only**, not embeddings, KV, active layer, temporary GPU allocations, or entire process VRAM.
- `peak_cuda_allocated_bytes` records PyTorch's active tensor allocations, not all GPU memory usage (CUDA context and caching allocator overhead excluded).
- The current code is single-request, single-token step inference; it is not hardened for production, general model loading, or multi-user serving.
- The data files must be trusted and checked before any shared import. `verify` reads all tensor bytes.

## Packaging / next steps

The original on-disk format is fully documented in [docs/FORMAT.md](docs/FORMAT.md). Development gates and acceptance criteria are in [docs/ROADMAP.md](docs/ROADMAP.md).

License: MIT for the PhiStream code. Model checkpoints have their **own** licenses; model weights are not shipped.

## Initial verification evidence (not a speed claim)

On the build environment's **CPU only** run (October 10, 2026 Pacific), the generated 39-tensor fixture passed SHA-256 verification. Streaming and resident modes had a maximum absolute logit difference of **0.0** across four input tokens. Re-run locally because tiny synthetic models do not benchmark disk/PCIe transfer constraints for real models.

## Local RTX validation candidate (not yet a verified CUDA result)

Use `python -m phistream.cli doctor --device cuda` to explicitly check CUDA.
The new `validate-device` command verifies the synthetic fixture and produces
a `PHISTREAM_DEVICE_VALIDATION_V1` JSON receipt with parity and streamed /
pinned / resident measurements. Windows users can run
`./scripts/validate-rtx.ps1 -Device cuda` from PowerShell after installing the
appropriate official CUDA-enabled PyTorch. No auto-downloads or upload.

See [docs/DEVICE_VALIDATION.md](docs/DEVICE_VALIDATION.md) for limitations.
The [live research dashboard](https://michaelwave369.github.io/PhiStream/)
can read a local JSON CUDA receipt in-browser. It does not submit it to a
server or present uploaded, user-provided results as independently verified.

## Safetensors inspection and Llama shape preflight (Gate B candidate)

PhiStream has an **offline, read-only safetensors checkpoint reader**, supporting
explicit single-file and indexed sharded checkpoints. It validates binary
headers, shard maps, tensor ranges, and optionally canonical Llama tensor
shapes. `--probe-tensor` materializes one bounded CPU tensor on demand:

```sh
python -m phistream.cli inspect-checkpoint /path/to/local/checkpoint --llama-layout
python -m phistream.cli inspect-checkpoint /path/to/local/checkpoint --probe-tensor model.norm.weight
```

See [Gate B documentation](docs/SAFETENSORS.md). This **does not yet perform
real-model inference**; local Ollama GGUF weights need a separate adapter.
No checkpoint downloads, deserialization hooks, or public model files.

## Local Llama token-ID execution (Gate B experimental candidate)

PhiStream now has a **narrow canonical Llama-style safetensors inference
adapter**, using independently implemented Llama split-half RoPE, GQA,
RMSNorm, SwiGLU, per-layer weight streaming, and chunked vocabulary logits.
The adapter only accepts local explicitly specified checkpoints. Example:

```sh
python -m phistream.cli inspect-checkpoint /path/to/checkpoint --llama-layout
python -m phistream.cli run-llama-ids /path/to/checkpoint --prompt 1,2,3 --tokens 4 --device cpu
```

**This is experimental token-ID execution, not ready-to-use chat.** It has
no tokenizer, no GGUF adapter, no automatic downloads, no external model
reference parity, and no 70B or RTX performance result. Unsupported Llama
variants and extra tensor fields are rejected. See the [execution safety
and compatibility guide](docs/LLAMA_EXECUTION.md).


## Offline reference-trace comparison (Gate B verification candidate)

PhiStream can now check a **user-supplied independent reference trace** against
the same, **fully SHA256-fingerprinted** local checkpoint. It compares exact
greedy token IDs and explicitly sampled logits under teacher forcing.

```sh
python -m phistream.cli fingerprint-checkpoint /path/to/local-llama
python -m phistream.cli compare-reference /path/to/local-llama --trace benchmarks/local/reference-trace.json --device cpu --output benchmarks/local/reference-result.json
```

Full [reference-trace protocol](docs/REFERENCE_TRACE.md). No model downloads,
reference executor integration, tokenizer or automated upload. Matching
user-provided traces are *not* independent-reference authentication or full
logit parity, and **no pretrained model parity has yet been recorded**.
