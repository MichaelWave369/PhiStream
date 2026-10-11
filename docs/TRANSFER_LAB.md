# ΦStream Transfer Laboratory · PR #9 (experimental)

**Measure data movement, not inference.** `transfer-lab` tests a bounded,
synthetic byte fixture in an explicitly selected local workspace. It has no
access to your models, does not modify any checkpoints, and does not alter
the currently running PhiStream layer-placement policy.

## On your RTX 5070 / Windows 11

With an existing PhiStream clone and working CUDA-enabled PyTorch:

```powershell
git pull --ff-only
New-Item -ItemType Directory -Force "D:\PhiStream\Experiments" | Out-Null
py -m phistream.cli transfer-lab --workspace "D:\PhiStream\Experiments" --gpu cuda --size-mib 16 --repeats 5 --output "benchmarks\local\transfer-d-rtx5070.json"
```

You can try `--workspace "C:\Users\micro\Downloads"` for a **separate**
local scratch experiment after verifying the first output. Running these in
sequence is **not** an official SSD A/B benchmark: OS cache, device state and
writeback scheduling can vary. Use distinct receipt filenames.

CPU-only baseline (also run in GitHub Actions with 1 MiB x 3 repeats):

```powershell
py -m phistream.cli transfer-lab --workspace "D:\PhiStream\Experiments" --gpu none --size-mib 16 --repeats 5 --output "benchmarks\local\transfer-d-cpu.json"
```

Scratch files are created in a uniquely named `phistream-transfer-*`
subdirectory beneath the workspace, using exclusive creation. They contain
deterministic synthetic bytes and are **deleted after the run** even when
normal exceptions occur. PhiStream never changes existing model files.
No download, internet request, registry mutation or persistent scratch file.
The output JSON is separate and uses an exclusive new-file write.

## Limits and measurements

- `--size-mib`: integer 1-256, default **16 MiB**. It limits scratch size,
  RAM copies and each CUDA transfer buffer. This is **not** the GPU's
  total allocation cap; other PyTorch/CUDA overhead is possible.
- `--repeats`: integer 3-15, default **5**.
- Refuses insufficient free scratch disk space, invalid directories and
  symlink workspace paths. No automatic CPU fallback when CUDA is requested.
- Checks whole-file SHA-256 on each read and a few device-returned bytes.
- Records a single creation/write/fsync duration, a **first read after write**
  and repeated **warm reads**. `first_read_is_cold_cache` is always `false`.
- Measures repeated Python `bytearray` RAM copies separately.
- CUDA mode measures pageable H2D and pinned H2D copy operations, with
  `torch.cuda.synchronize` surrounding timing; alternates mode order.
- Measures pageable-to-pinned CPU staging separately, because a faster
  pinned H2D copy can be slower end-to-end once staging is included.
- Reports median/min/max and **every** measurement, not a single magic
  number. GPU copy buffers are preallocated; startup is excluded.

Do **not** interpret cached disk read speeds as NVMe raw bandwidth,
or H2D synthetic copy times as real model throughput. The experiment
does **not** test large files uncached, PCIe negotiated link bandwidth,
disk-to-GPU zero-copy, overlapped transfer/compute, GPU layer residency,
CPU graph execution, or large Llama model capability.

## Evidence contract

- `schema`: `PHISTREAM_TRANSFER_LAB_V1`
- `provenance`: `MEASURED_LOCAL_SYNTHETIC_BYTES_NO_MODEL`
- `status`: `PASS_CPU_BASELINE` or `PASS_CUDA_TRANSFERS`
- `storage`: write+fsync, first-after-write, warm read samples and cache caveat
- `host`: bounded RAM-copy samples
- `cuda`: device name, free memory before, pageable/pinned H2D samples,
  pinned-host staging cost, **only when explicitly requested**
- `model_execution: false`; `automatic_layer_placement: false`

Import the local JSON in the **Transfer Laboratory** section of the live
[PhiStream dashboard](https://michaelwave369.github.io/PhiStream/).
The web viewer reads the selected file directly on the browser; it does
not upload it or authenticate its origin. Paths and machine identity may be
visible in a JSON receipt, so review before deliberately publishing one.

## Road to dynamic residency

This milestone isolates the three data-movement stages. The next milestone
would tie observed transfer costs to a separately reviewed static policy,
run randomized/order-controlled repetitions and measure per-layer actual
execution. No autonomous scheduler or Vessie routing change is approved.
