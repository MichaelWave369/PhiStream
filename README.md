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

## Memory policy preflight (Gate C research candidate)

Use `plan-memory` to estimate active FP32 layer weights, KV cache,
head chunks and attention workspace against a selected memory budget.
**An estimate is not a GPU fit or performance guarantee.**

```sh
python -m phistream.cli plan-memory /path/to/local-llama --capacity-mib 12288 --reserve-mib 1024 --context-tokens 512 --device scenario
```

With `--device cuda`, the policy is also capped by observed free GPU
bytes. `run-llama-ids` optionally accepts `--capacity-mib` to block
oversized policy scenarios before weight execution (not a hard OOM guard).
See [memory preflight documentation](docs/MEMORY_PREFLIGHT.md).

## Local hardware observation and memory advisor (PR #8 experimental)

The new `hardware-snapshot` command queries actual Windows/Linux available
RAM, optional CUDA free VRAM, and free space on explicitly selected drives.
The separate `recommend-residency` command estimates a possible Llama
layer-placement policy for **human review** from fresh observed free memory.
Neither command runs a model, changes layer residency, downloads weights,
measures disk speed, or guarantees a GPU fit. No execution is approved.

```powershell
py -m phistream.cli hardware-snapshot --gpu cuda --drive "C:\" --drive "D:\" --output benchmarks\local\hardware-rtx.json
```

See the [Hardware-Aware Advisor](docs/HARDWARE_ADVISOR.md) for optional
checkpoint-based recommendations and provenance rules.

## Transfer Laboratory: SSD / RAM / CUDA timings (PR #9)

PhiStream has a **bounded synthetic-byte laboratory** for an opt-in
temporary scratch fixture. It times repeated local file reads, host RAM
copies, and optional pageable/pinned CUDA host-to-device transfers.
**No real model, uncached NVMe benchmark, PCIe raw bandwidth or
adaptive cache is measured.**

```powershell
py -m phistream.cli transfer-lab --workspace "D:\PhiStream\Experiments" --gpu cuda --size-mib 16 --repeats 5 --output "benchmarks\local\transfer-d-rtx5070.json"
```

See [Transfer Laboratory safety and evidence guide](docs/TRANSFER_LAB.md).
The optional GPU mode requires functioning CUDA and never silently falls
back to CPU. The temporary byte fixture is owned by the process and deleted
on exit; only your explicitly requested local JSON receipt remains.

## Paired end-to-end transfer pipeline (PR #10)

The opt-in `pipeline-lab` tests the **complete synchronous** file-read →
host preparation → pageable or preallocated pinned staging → CUDA transfer
path. Methods are compared in reproducible balanced AB/BA pairs with
separate warmups; checksum verification is outside measured durations.

```powershell
py -m phistream.cli pipeline-lab --workspace "D:\PhiStream\Experiments" --gpu cuda --size-mib 64 --repeats 8 --warmups 2 --output "benchmarks\local\pipeline-d-64m-rtx5070.json"
```

See [Paired Pipeline Laboratory](docs/PIPELINE_LAB.md). File reads have
**uncontrolled Windows/OS caching**. This does not benchmark real Llama
layers, model throughput, asynchronous prefetch or adaptive GPU placement.

Pipeline timing attribution: new V1 receipts also report pre-H2D synchronization and the per-trial **unattributed wall-time** remainder. Above-1 pageable/pinned time ratio favors pinned; below-1 favors pageable. Existing V1 receipts remain valid. See [Pipeline Lab](docs/PIPELINE_LAB.md).

**Timing correction:** PR #12 symmetrically retains the temporary pageable tensor through the full-pipeline timer for both CUDA variants, measures buffer-release overhead after timing, and labels the lifecycle in new V1 receipts. Prior pinned-pageable comparisons may include an asymmetric Python memory-release cost; see [pipeline measurement methodology](docs/PIPELINE_LAB.md).

## Reusable Host Buffer Lab (PR #13 experimental)

PhiStream can now compare three **synthetic** CUDA transfer strategies at
multiple sizes: pageable, newly allocated pinned memory and preallocated
reusable pinned memory. The default 4/16/64 MiB lab uses six balanced
method permutations per size. No real model is executed.

```powershell
py -m phistream.cli reuse-lab --workspace "D:\PhiStream\Experiments" --gpu cuda --sizes-mib "4,16,64" --rounds 6 --warmups 2 --output "benchmarks\local\reuse-rtx5070-multi.json"
```

See [Reuse Laboratory](docs/REUSE_LAB.md). Local scratch files are deleted.
CPU-only synthetic tests do not measure pinned GPU transfer speed.

## Direct-to-Pinned Streaming Laboratory (PR #14, synthetic)

The opt-in `direct-read-lab` compares the current copy-heavy `file → bytes → bytearray → pinned → GPU` path against `file.readinto()` writing directly into preallocated pageable or pinned **CPU** tensors, then synchronized H2D. It measures 4/16/64 MiB controlled synthetic bytes by default. It does **not** read directly into GPU VRAM, run Llama checkpoints or control Windows filesystem caching.

```powershell
py -m phistream.cli direct-read-lab --workspace "D:\PhiStream\Experiments" --gpu cuda --sizes-mib "4,16,64" --rounds 6 --warmups 2 --output "benchmarks\local\direct-rtx5070-multi.json"
```

See [Direct Read Lab](docs/DIRECT_READ_LAB.md). Nothing uploads automatically; the scratch fixture is temporary.

## Chunked double-buffer streaming experiment (PR #15)

A bounded, opt-in `chunk-lab` compares sequential pinned CPU → CUDA transfers with **two alternating pinned CPU buffers**, a background next-chunk read and CUDA completion-event ownership checks. Both methods preallocate two host and two GPU buffers, but only the double path attempts to overlap background file reads with pending H2D transfers. Local JSON includes separate full-SHA256 host/GPU replay audit **outside** benchmark timing. No real model, GPU compute overlap, GPUDirect Storage or uncached SSD read claim.

```powershell
py -m phistream.cli chunk-lab --workspace "D:\PhiStream\Experiments" --gpu cuda --chunk-mib 4 --chunks 16 --repeats 6 --warmups 1 --output "benchmarks\local\chunk-4m16-rtx5070.json"
```

See [Chunked Streaming Laboratory](docs/CHUNK_LAB.md). Output collisions are checked before benchmarking and file creation remains exclusive. Local scratch is temporary; receipts never upload automatically.

## Overlap Intelligence (PR #16)

An opt-in synthetic benchmark measures **serial buffering**, **a newly created background worker per trial**, and **a reusable persistent worker** with separately reported startup cost and event-guarded buffer ownership. Per-chunk CPU host timelines disclose read, worker queue, enqueue, and observed CUDA completion intervals, but they are NOT calibrated GPU execution timelines. An evidence gate keeps the single-buffer default and reports INCONCLUSIVE for small or inconsistent gains.

```powershell
py -m phistream.cli overlap-lab --workspace "D:\PhiStream\Experiments" --gpu cuda --chunk-mib 8 --chunks 8 --rounds 6 --warmups 1 --output "benchmarks\local\overlap-8m8-rtx5070.json"
```

See [Overlap Lab](docs/OVERLAP_LAB.md). No automatic scheduler or real model weights are accessed.

## Guarded Synthetic Scheduler (PR #17)

PhiStream now has a **review-only, local** gate for repeated RTX 5070 Overlap Intelligence receipts. Three independent full 12-round, same-hardware 8 MiB × 8 synthetic CUDA receipts must pass both individual and pooled 5%/75% gates. Results remain **unauthenticated, synthetic, default SINGLE**, with no change to Ollama, Vessie or model weights.

```powershell
py -m phistream.cli guarded-scheduler --receipt-glob "benchmarks\local\overlap-8m8-repeat-*.json" --output "benchmarks\local\guarded-review-rtx5070.json"
```

An independently and explicitly operator-approved **scratch-only** CUDA trial can select double buffering when the gate and current GPU preflight pass, or hold/fallback to single. It includes a deliberately serialized same-worker negative control. This is NOT automated production routing. See [Guarded Scheduler](docs/GUARDED_SCHEDULER.md) for opt-in syntax, hard limits and exclusions.
