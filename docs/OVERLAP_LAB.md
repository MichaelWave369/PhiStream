# ΦStream Overlap Intelligence Laboratory · PR #16

**Scope: opt-in, local, synthetic scheduling research.** Nothing here reads an
existing checkpoint, executes a language model, tests GPU math overlap or
authorizes adaptive policy changes.

## Why

The first RTX 5070 sweep at a fixed 64 MiB payload returned:

| Chunk shape | Single median | Double median | Double wins |
|---|---:|---:|---:|
| 2 MiB × 32 | 16.146 ms | 17.562 ms | 1/6 |
| 4 MiB × 16 | 14.444 ms | 15.388 ms | 1/6 |
| 8 MiB × 8 | 13.279 ms | 13.184 ms | 3/6 |
| 16 MiB × 4 | 12.377 ms | 12.327 ms | 2/6 |

At 8 and 16 MiB the advantage in unpaired medians was tiny and the
**median paired ratios still favored single**. We should not authorize a
double-buffer scheduler from that evidence.

## Run on Windows + RTX 5070

The D: workspace must already exist:

\`\`\`powershell
git pull --ff-only
py -m phistream.cli overlap-lab --workspace "D:\PhiStream\Experiments" --gpu cuda --chunk-mib 8 --chunks 8 --rounds 6 --warmups 1 --output "benchmarks\local\overlap-8m8-rtx5070.json"
\`\`\`

A 16 MiB × 4 follow-up should use a **different** output filename. The
CLI preflights existing receipt names *before measuring*, and creates new
receipts exclusively. Scratch files are created in a private temporary
folder beneath your selected workspace, then deleted even on normal errors.
No automatic upload.

CPU-only scheduling check, not GPU evidence:

\`\`\`powershell
py -m phistream.cli overlap-lab --workspace "D:\PhiStream\Experiments" --gpu none --chunk-mib 1 --chunks 4 --rounds 6 --warmups 1 --output "benchmarks\local\overlap-cpu.json"
\`\`\`

## Three methods with equal GPU/host buffer budgets

1. \`single\`: sequential readinto pinned host buffer A, H2D enqueue and
   CUDA event wait before the next read.
2. \`double_new_worker\`: alternating two pinned buffers with **new**
   worker-pool creation included in each trial's clock, then shutdown.
3. \`double_reuse_worker\`: alternating identical buffers, but keep one
   persistent worker alive and warm **outside trial clocks**. Its measured
   startup cost is reported separately, not hidden.

All methods use two preallocated pinned host and two GPU buffers for a
matched allocation footprint. Both double methods block before refilling
a host buffer that has a still-outstanding CUDA completion event. On
errors, the scheduler attempts to drain active events. Full SHA-256 of
every host/GPU-returned synthetic chunk is checked in a separate untimed
replay, not during each timed GPU copy. Chunk content varies by index.

Six-round or twelve-round **all-permutation** AB/BC/CA-style method
order, each strategy occupies every position equally. Warmups are
excluded. The test measures real elapsed end-to-end wall-clock for
each 64 MiB series, not separate, misleading headline copy rates.

## What the new host timeline actually means

One full **first measured trial** per method records CPU-monotonic
timestamps for each chunk:

- queue submission, worker read start and end, queue delay
- GPU copy enqueue start and return on the CPU
- CUDA completion **observed after waiting** on its event
- duration of waiting for completion and event pending when read starts

CUDA event elapsed time is recorded **separately**. A CUDA event's
device-clock duration cannot be projected onto CPU perf-counter
timestamps without independent cross-clock calibration. A timestamp
on the CPU saying "event pending" is **not evidence the device was
physically copying concurrently with the disk read**. The chart shows
enqueue-to-observation *brackets*, not GPU-execution bars.

The next stage could use an external CUDA profiler or calibrated
cross-device trace if we require physical overlap evidence. We do
not install a profiler, change Windows settings or claim calibrated
overlap in this PR.

## Recommendation policy

For the paired **single** versus **double_reuse_worker** paths only:

- at least **six** paired measured trials
- at least **75%** of those pairs must favor the candidate
- the median **per-pair relative time improvement** must be >= **5%**

A winning double result is labeled \`DOUBLE_CANDIDATE\`; a corresponding
single result is \`SINGLE_CANDIDATE\`. Anything else is \`INCONCLUSIVE\`.
CPU-only receipts are \`NO_GPU_EVIDENCE\`. All receipts say
\`default: SINGLE\` and \`policy_applied: false\`.

**Candidate is not a production activation**. Additional independent
runs and a real checkpoint with reference logits/token-ID parity are
needed before changing the actual executor, even if 5% is exceeded.

## Budget and provenance

Uses existing safe bounds from PR #15: 1/2/4/8/16 MiB chunks, 4–32
chunks, total <=256 MiB, with disk/free RAM/free VRAM checks.
\`--rounds\` is 6 or 12; \`--warmups\` 1–3. CUDA requires explicit
\`--gpu cuda\`, no silent CPU fallback.

JSON schema: \`PHISTREAM_OVERLAP_LAB_V1\`. Provenance:
\`MEASURED_LOCAL_SYNTHETIC_HOST_TIMELINE_NO_MODEL\`.
The live React site can import it **locally in the browser**; no server
upload, automatic cache scheduler, downloads, or Vessie decisions.

Limitations include Windows filesystem caching, host worker contention,
different thread startup accounting (disclosed), CUDA-device clock
uncalibrated against CPU clocks, synthetic bytes rather than actual
layers, no useful GPU compute, and untimed independent replay integrity.
