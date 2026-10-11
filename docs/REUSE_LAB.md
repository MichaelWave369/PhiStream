# ΦStream Reusable Host Buffer Laboratory · PR #13

**Scope:** synthetic, local, opt-in experiments. No model checkpoints, no
token generation, no automatic GPU layer placement, no network data, and
no claims of production inference improvement.

## Why this experiment

Four corrected 64 MiB pipeline runs on the user’s RTX 5070 favored pinned
staging (29/32 paired wins across them). Median end-to-end gains ranged
from ~0.29 to 0.49 ms in the last three runs, only ~0.7–1.3%.
PhiStream now tests whether preallocating and reusing pinned staging memory
is worth more than newly allocating it on each transfer. This does NOT
attribute improvements to the application until an actual model workload
has been compared to a separately verified reference.

## Local Windows RTX 5070 test

```powershell
git pull --ff-only
py -m phistream.cli reuse-lab --workspace "D:\PhiStream\Experiments" --gpu cuda --sizes-mib "4,16,64" --rounds 6 --warmups 2 --output "benchmarks\local\reuse-rtx5070-multi.json"
```

The CLI requires an existing local workspace, preserves your model folders,
and writes a separate JSON receipt without overwriting an existing file.
Scratch files are exclusively created in private temporary subdirectories
and deleted after measurement. GPU buffers are only temporary test buffers.

## Experimental strategies

- `pageable`: read synthetic file into Python bytes, wrap a newly copied
  pageable CPU tensor, then synchronously copy to the allocated GPU buffer.
- `pinned_new`: same read/pageable preparation, then **allocate** a pinned
  tensor inside every trial, copy into it, and synchronously copy to GPU.
- `pinned_reuse`: same read/pageable preparation, but copy into a pinned
  tensor **allocated once per size before timing**, then copy to GPU.

For CUDA, all temporary pageable source tensors live until the end of
the full pipeline timing. Source cleanup is measured **after** timing.
New pinned allocations happen inside the timed region, while cleanup for
those allocations remains outside it. That asymmetry is intentional to
measure allocation versus reuse, but these timings are **not** process-level
costs including final deallocation or program startup.

Each method's full pipeline time includes file read, pageable host buffer
preparation, pinned allocation (new method only), pinned staging (where
needed), CUDA synchronization, and H2D transfer. Individual stage durations
and an unattributed residual are included for every trial.

Orders are shuffled deterministically across all 6 permutations of the
three CUDA methods; a 12-round run uses two balanced blocks. Warmups per
method are excluded from the reported statistics.

`--sizes-mib` accepts 1–3 unique, strictly decimal integers between 1 and
128 MiB each; their total must be at most 192 MiB. Default 4,16,64.
`--rounds` must be 6 or 12; `--warmups` is 1–4.

Without `--gpu cuda`, the two CPU-only modes compare a fresh bytearray
with a reused bytearray and **are not** pinned-memory measurements.
CI uses this limited CPU mode to check safety/format/ordering; Windows
GPU behavior must be measured by the user. No silent GPU-to-CPU fallback.

## Receipt and limitations

`PHISTREAM_REUSE_LAB_V1` and provenance
`MEASURED_LOCAL_SYNTHETIC_HOST_BUFFER_REUSE_NO_MODEL`.
Per size: SHA256 fixture digest, ownership/cleanup confirmation,
complete per-method stage times, counts of fastest method in each
balanced round, and GPU identity if CUDA was explicitly requested.

Filesystem cache is uncontrolled and reads may come from Windows RAM.
No NVMe cold-read bandwidth, PCIe maximum bandwidth, compute overlap,
model layer decoding, prefetch, storage saturation or large-model VRAM
capacity is measured here. CPU and GPU buffering overhead may vary over
time and with other programs active. Do not publish privately identifying
paths without review. Import resulting JSON in the live browser-only
ΦStream Reuse Laboratory panel. No automatic upload occurs.
