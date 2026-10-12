# Chunked Streaming & Double-Buffer Laboratory (PR #15)

This is a **synthetic, bounded, opt-in transfer experiment**. It does not execute model weights, stream actual Llama layers, benchmark physical SSD cold reads, or authorize automatic scheduling. It investigates whether a background file read into an alternate pinned host buffer can run while an asynchronous GPU transfer is outstanding.

## Windows 11 + RTX 5070: test locally

From your existing PhiStream repo with PyTorch/CUDA available:

```powershell
git pull --ff-only
py -m phistream.cli chunk-lab --workspace "D:\PhiStream\Experiments" --gpu cuda --chunk-mib 4 --chunks 16 --repeats 6 --warmups 1 --output "benchmarks\local\chunk-4m16-rtx5070.json"
```

The workspace must already exist; it receives an exclusive temporary folder of synthetic bytes that is deleted after normal completion and on exceptions. The CLI refuses a previously used output name **before** running, and preserves exclusive-create behavior in case another process creates the file while measuring. JSON receipts stay on your PC.

## Two methods and fairness

- **Single:** read the next chunk directly into pinned host buffer A, enqueue a nonblocking CUDA H2D transfer on a dedicated CUDA stream, synchronize its completion event, and only then read the following chunk.
- **Double:** use the same two preallocated pinned host buffers and two GPU output buffers, but alternate ownership between A/B. A single background worker begins reading the next chunk after checking that its buffer's previous completion event has finished. While that read occurs, the CUDA stream may still be copying from the current buffer.
- Both strategies allocate **two host and two GPU buffers** before the measured series, so the memory footprint is matched. Each strategy uses direct `file.readinto(memoryview(tensor.numpy()))`, no intermediate Python `bytes` staging.
- The sequence uses exclusive temporary fixtures with a different deterministic 256-byte pattern per chunk. SHA-256 digests are recorded per chunk during creation; an independent replay after measured trials checks **full host and full GPU-returned SHA-256** for every chunk. The timed loop itself checks transfer completion via CUDA events before reusing any host buffer. Timed trial destination buffers are **not independently hashed before reuse**; the separate replay validates the same transfer primitive, not each timed transaction.
- 4, 6, or 8 **AB/BA balanced trials**, with 1–3 warmups per method excluded.
- Report all whole-series wall-clock elapsed samples, time spent waiting on events, total host-read duration, sum of GPU CUDA-event copy durations and the number of next reads queued while an event was still pending. A pending event at a scheduling point is **not direct proof of overlap on the device**.

### Parameters

`--chunk-mib`: 1, 2, 4, 8 or 16 MiB. `--chunks`: 4–32. Their product cannot exceed 256 MiB.

`--repeats`: 4, 6 or 8. `--warmups`: 1–3.

`--gpu cuda` must be explicit for GPU measurements and fails closed if CUDA unavailable. Without CUDA the CPU-only fake backend checks scheduler logic and byte integrity but cannot measure PCIe throughput or claim a GPU speedup.

The experiment guards free disk, available RAM and VRAM; these observations are not reservations. CUDA context, buffer allocation, scratch creation and full audit happen outside timed trials.

## Evidence limitations

`schema: PHISTREAM_CHUNK_PIPELINE_V1`, provenance `MEASURED_LOCAL_SYNTHETIC_CHUNK_STREAM_NO_MODEL`, and status `PASS_CUDA_CHUNK_LAB` on actual CUDA, otherwise `PASS_CPU_CHUNK_SCHEDULER`.

- **OS filesystem caching uncontrolled**. A warmed read may come from RAM rather than WD_BLACK NVMe media.
- CUDA transfers are nonblocking from *pinned CPU memory*, not GPUDirect Storage or disk-to-VRAM DMA.
- The CUDA stream does no useful model computation. This tests possible file-read/H2D transfer overlap, **not computation/transfer overlap**.
- Background worker setup and thread scheduling costs are included in the double-path wall-clock unless otherwise changed in a future controlled comparison.
- Short synthetic buffers cannot establish throughput with a real transformer. A future user-approved adapter requires real checkpoint formats, independent logits parity, budget checks and run receipts.
- No automatic upload, network use, modification of model files or autonomous Vessie decisions.

The website's **Chunked Streaming** reader imports a local JSON file in-browser. It never calls CUDA or uploads that file.

## Before claiming a win

Test the RTX 5070 with multiple chunk shapes and at least three repeat runs, with new filenames. Favor a single buffer if double buffering doesn't beat it **in full wall-clock time** under comparable conditions. A pending CUDA event is only an observation; corroborate meaningful overlap independently before attributing speedup to it.
