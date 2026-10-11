# PR #10: Paired Transfer Pipeline Laboratory

ΦStream's V1 transfer lab isolates warm file reads, RAM copies and CUDA
host-to-device copies. The new **pipeline-lab** puts those stages inside
**one synchronized, end-to-end wall-clock measurement per trial** and
compares pageable input to preallocated reusable pinned-host staging.

## Run on the RTX 5070 (PowerShell)

```powershell
git pull --ff-only
py -m phistream.cli pipeline-lab --workspace "D:\PhiStream\Experiments" --gpu cuda --size-mib 64 --repeats 8 --warmups 2 --output "benchmarks\local\pipeline-d-64m-rtx5070.json"
```

The directory must already exist. Use a **new** receipt name for any later
run. The CLI refuses overwriting an existing receipt. It creates one
bounded, private scratch file underneath a uniquely named temporary
directory, validates SHA256 content, and deletes the scratch directory on
successful execution or normal exceptions. It never opens a model weight
file, publishes a receipt or modifies an existing checkpoint.

## Method

- Scratch input is deterministic synthetic bytes, 1–128 MiB per run.
- 4–20 **paired** trials, with 1–5 warmups per method (excluded).
- Pair order is seeded for auditability and balanced AB/BA for even pairs.
- For each pageable method trial, time opening/reading the owned file,
  constructing a **fresh pageable buffer**, and copying to a preallocated
  RTX buffer with explicit CUDA synchronization.
- For each pinned method trial, time the same opening/read and pageable
  preparation, then copy to a **preallocated pinned host tensor**, then
  synchronized H2D into the same GPU buffer.
- Entire synchronous pipeline wall time, file-read part, pageable-buffer
  construction, pinned staging, and H2D are **separately reported**. The
  timers include real CPU work, not CUDA-only timing.
- Validate the file's SHA256 and sample returned GPU device bytes **after**
  the timed interval. Hashing/checks aren't included in performance totals.
- Paired outcome reports number of wins and median ratio of each pair's
  pageable full-pipeline time to pinned full-pipeline time.
- Buffers and CUDA context are allocated **before** the measured trials;
  this is *warm reuse*, not cold-start inference.

## Interpretation

**A median ratio over 1 favors the pinned strategy for this synthetic
workload; it does not show that model token throughput would improve.**
With a typical checkpoint, other bottlenecks can dominate: model math,
format conversion, FP32 expansion, KV cache, disk I/O, temporary allocations,
CUDA stream scheduling and time to reuse existing staged layers.

All file reads are **OS-cache uncontrolled** and likely warm. This is NOT
an uncached SSD benchmark, GPUDirect Storage, PCIe peak bandwidth test,
overlapped compute/prefetch benchmark, adaptive placement, or any real
model checkpoint experiment. CPU-only mode measures two ordinary host-buffer
staging strategies and does not call the pinned method hardware-accelerated.

Receipts use `schema: PHISTREAM_PIPELINE_LAB_V1` and provenance
`MEASURED_LOCAL_PAIRED_SYNTHETIC_PIPELINE_NO_MODEL`,
with all samples, ordered pairs, warmups, checkpoint-free fixture hash,
GPU identity if requested, and `automatic_layer_placement: false`.

## Review gate

The code is tested on a CPU runner using a 1 MiB scratch fixture, failure
handling and cleanup. No RTX 5070 GPU measurement exists for this new
pipeline command until you execute it locally. No Vessie routing or model
execution approval is implied by any passing pipeline receipt.

## Interpreting hidden overhead (PR #11)

Your RTX 5070's first 64 MiB paired run recorded **40.378 ms** pageable
versus **41.985 ms** pinned full-pipeline medians, with 0/8 pinned wins.
Although pinned H2D was faster, the pinned path also paid for host staging
and previously unattributed gaps inside the whole-path wall clock.

The next V1 receipts now expose:

- `methods.<mode>.measurements.pre_h2d_sync`: elapsed seconds spent at the
  explicit CUDA synchronization immediately before the H2D timer.
- `methods.<mode>.measurements.unattributed_wall_time`: for each trial,
  complete wall time MINUS file read, host preparation, optional pinned
  stage, pre-H2D sync, and H2D. This is a time **residual**, not a new
  transfer or bandwidth measurement. Other Python bookkeeping contributes.
- `paired_outcome.median_ratio_favors`: `PAGEABLE`, `PINNED` or `TIE`.
  A pageable/pinned ratio **below 1.0** favors pageable, not pinned.

The accounting is per trial because subtracting medians can conceal
variability and is not generally equal to the median residual. Zero
residual is permitted. The older V1 JSON files remain importable;
new timing fields may simply be absent from them.

No GPU performance claim follows from CI. The real RTX 5070 path must be
retested with a new receipt to identify whether explicit synchronization
or some other interpreter/runtime gap explains the ~2 ms remainder.
