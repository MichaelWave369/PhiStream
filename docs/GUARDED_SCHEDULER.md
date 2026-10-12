# ΦStream PR #17 · Guarded Synthetic Scheduler

**No production execution.** ΦStream's single-buffer policy remains the
default and this prototype can execute **only its own bounded synthetic**
CUDA scratch fixture. It cannot open, alter or route any Ollama checkpoint,
cannot change Vessie, and cannot automatically promote a measured candidate
to the real inference executor.

## Why there is now a stronger evidence gate

The RTX 5070's 8 MiB × 8 chunk results include three independent
12-pair repeats with 9/12 double-buffer wins each, and median paired gains
9.29%, 9.71% and 6.98%. This is repeatable *synthetic* evidence, not proof
of GPU/SSD physical overlap, better model token throughput, or a universally
better double-buffer scheduler.

PR #17 requires a specific evidence cohort:

- **At least three distinct 12-round CUDA overlap-lab V1 receipts**, with
  8 MiB × 8 chunks, balanced method ordering and full out-of-band GPU SHA
  integrity audits.
- Exact schema/provenance, synthetic and checkpoint-free execution flags.
- Identical platform, reported GPU name, PyTorch version and CUDA build.
- UTC run timestamps, no duplicates/overlaps; all raw elapsed samples present
  and positive, summary medians and original policy recomputed.
- Each run must independently pass PR #16's **>=5% paired median improvement
  and >=75% double wins**, and the pooled 36-or-more pairs must meet both.
- A missing, invalid, duplicated or mixed-hardware receipt **holds the gate**.
- This is **NOT cryptographic authentication**: a JSON could have been edited.
  The resulting report is labeled user-supplied, unauthenticated evidence.

A passing gate does not execute anything or change default scheduling.

## 1. Review the receipts only (safe default, no CUDA workload)

From your existing repository in PowerShell:

\`\`\`powershell
git pull --ff-only
py -m phistream.cli guarded-scheduler --receipt-glob "benchmarks\local\overlap-8m8-repeat-*.json" --output "benchmarks\local\guarded-review-rtx5070.json"
\`\`\`

This reads only the matching local JSON receipts and creates one local
review report. It never runs benchmarks or GPU computation. It returns
\`ELIGIBLE_FOR_SYNTHETIC_OPT_IN\` when all checks pass; otherwise
\`HELD_SINGLE_DEFAULT\` with individual rejection reasons. The review
does not claim an authenticated hardware measurement or authorize use of
production files.

Existing output files are rejected *before* anything expensive starts and
all writes are exclusive-create. Use fresh filenames for repeated reviews.

## 2. Optional: run a local, operator-approved CUDA experiment

This **requires a separate deliberate action**. It is NOT performed by the
review command or by opening a receipt in the website.

\`\`\`powershell
py -m phistream.cli guarded-scheduler --receipt-glob "benchmarks\local\overlap-8m8-repeat-*.json" --run-synthetic --workspace "D:\PhiStream\Experiments" --request-double --approve-synthetic-double --output "benchmarks\local\guarded-double-rtx5070.json"
\`\`\`

The executor re-checks local CUDA/GPU identity, platform, PyTorch/CUDA build,
available VRAM and bounded scratch resources before selecting the double
candidate. If the evidence gate or approval is missing, or the GPU
environment/free VRAM has materially changed, it **falls back to single**,
subject to the single path's own capacity constraints. Missing CUDA or
insufficient safe memory prevents the run entirely.

Only a temporary 64 MiB synthetic fixture is opened. The scratch files
are created inside an exclusive temporary subfolder and removed afterward.
The local JSON explicitly reports requested/selected strategy, reason for
fallback, six wall-clock trials and an independent full SHA256 host/GPU
replay audit. No original files move or change.

A non-overlapping **negative control** is run beside approved double
buffering: it uses the same persistent worker and buffers, but waits for
each CUDA copy to complete before submitting the next file read.
The two modes are alternated by round. This tests whether scheduled
read-ahead is plausibly useful without claiming independently calibrated
GPU/CPU concurrent execution. A candidate can still lose to the control.

To run **single only**, omit both flags:

\`\`\`powershell
py -m phistream.cli guarded-scheduler --receipt-glob "benchmarks\local\overlap-8m8-repeat-*.json" --run-synthetic --workspace "D:\PhiStream\Experiments" --output "benchmarks\local\guarded-single-rtx5070.json"
\`\`\`

## Evidence receipts and scope boundaries

Review receipt:
- \`PHISTREAM_GUARDED_SCHEDULER_V1\`
- provenance \`LOCAL_UNAUTHENTICATED_SYNTHETIC_RECEIPT_REVIEW\`
- never sets production policy or performs model execution.

Synthetic test receipt:
- \`PHISTREAM_GUARDED_SYNTHETIC_RUN_V1\`
- provenance \`LOCALLY_EXECUTED_SYNTHETIC_GPU_SCRATCH_NO_MODEL\`
- explicitly reports the selected mode and serialized negative control.
- never touches installed models, production scheduling, Vessie or Ollama.

**Important methodological limits:**

- Windows filesystem cache uncontrolled. CPU and GPU clocks are
  uncalibrated. No useful CUDA model-compute kernel is running.
- These runs measure a controlled staging microbenchmark, NOT real
  transformer layer prefetch, model inference, VRAM fit, or token/sec.
- Receipt review checks internal consistency, not secure attestation;
  SHA audits are self-reported by incoming unauthenticated JSON.
- The guard is deliberately restricted to 8 × 8 MiB until **other**
  independently qualified shapes have their own gates and tests.
- A passing operator experiment is not permission for automatic
  activation. Model-adapter promotion needs separate checkpoint
  provenance, reference-token/logit parity, bounded RAM/VRAM, independent
  GPU validation, and a human-reviewed release gate.

The live web panel can read local V1 review and synthetic-run JSON files,
but it cannot create approvals or upload the files.
