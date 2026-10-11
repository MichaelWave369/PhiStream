# ΦStream Direct-to-Pinned Streaming Laboratory (PR #14)

## What this tests

An **opt-in synthetic-byte microbenchmark** of **file → reusable host tensor → RTX GPU** versus the current multi-copy preparation route. These are **controlled scratch reads**, NOT actual model layers, pretrained inference, raw SSD throughput, a GPU direct-storage driver, or streaming decode.

Three CUDA paths:

| Method | File read destination | Extra copies | H2D source |
|---|---|---|---|
| `legacy_copy` | Python `bytes` | `bytes` → newly allocated `bytearray` → reusable pinned tensor | Pinned CPU tensor |
| `direct_pageable` | Preallocated pageable CPU tensor via `file.readinto(memoryview(tensor.numpy()))` | No intermediate Python bytes or bytearray | Pageable CPU tensor |
| `direct_pinned` | Preallocated pinned CPU tensor via `file.readinto(memoryview(tensor.numpy()))` | No intermediate Python bytes or bytearray | Pinned CPU tensor |

A **direct read into a CUDA tensor is not performed**. The disk read still goes through the operating system and CPU-addressable host memory. `readinto` avoids extra user-space byte buffers; it does not bypass Windows filesystem cache or prove zero-copy disk-to-GPU DMA.

## Run on RTX 5070

Your existing scratch directory `D:\PhiStream\Experiments` remains the workspace:

```powershell
git pull --ff-only
py -m phistream.cli direct-read-lab --workspace "D:\PhiStream\Experiments" --gpu cuda --sizes-mib "4,16,64" --rounds 6 --warmups 2 --output "benchmarks\local\direct-rtx5070-multi.json"
```

The receipt is local only and can be opened in the site's **Direct Read Laboratory** section. PhiStream never uploads it.

CPU-only structural smoke test (not a pinned-memory or GPU performance result):

```powershell
py -m phistream.cli direct-read-lab --workspace "D:\PhiStream\Experiments" --gpu none --sizes-mib 1 --rounds 6 --warmups 1 --output "benchmarks\local\direct-cpu.json"
```

## Safety and measurement controls

- An exclusive, synthetic temporary scratch file is created inside an explicitly selected existing nonsymlink directory; deleted on exit, including failures.
- Bound sizes to 1–128 MiB each, at most 3 unique sizes and 192 MiB total. Guard available host RAM, free scratch volume capacity and CUDA VRAM **before** allocating work buffers.
- CUDA must be requested explicitly. Missing CUDA is a failure, not an implicit CPU benchmark.
- Six balanced method permutations per size, optional 12 rounds; 1–4 warmups per method excluded.
- Entire synchronous wall clock includes file opening and reading, legacy host copy and pinned staging when applicable, and explicitly synchronized host-to-device copy.
- All three strategies use preallocated CUDA output; direct pageable and pinned tensors are allocated before trial timing. Python temporary sources for the legacy path are retained until the stopwatch has stopped, matching PR #12's correction.
- Check SHA-256 of **each method's destination bytes** and representative GPU bytes **after timing**. A wrong or short file fails, not a PASS receipt. The fixture SHA is recorded.
- All stage times and per-method median/min/max and winner counts are preserved, along with `model_execution: false` and `automatic_layer_placement: false`.
- The CLI never touches external checkpoints, never downloads models, and refuses to overwrite a receipt path.

## Interpretation

Our last corrected 64 MiB experiment measured roughly **18.5 ms file read + 18.2 ms pageable host buffer preparation** out of **40.9 ms** for the reused pinned path. Direct `readinto` is intended to challenge the **extra host copy**, not magically accelerate PCIe transfers.

An improvement here would be evidence about **synthetic host-byte staging**, not proof of Llama/LLM token throughput gains. OS caching is **uncontrolled**. Inference-specific tensor format conversion, layer scheduling, host page lifetimes, optimizer layouts, and KV operations are not modeled. A future real-model adapter requires independent logits/token parity and explicit approval.

The CPU-only path compares `legacy_host_copy` (file bytes then new bytearray) against `direct_host_readinto` (preallocated bytearray), with no PyTorch CUDA claims.

Provenance: `PHISTREAM_DIRECT_READ_LAB_V1`, `MEASURED_LOCAL_SYNTHETIC_READINTO_NO_MODEL`.
