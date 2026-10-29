# ΦStream · SmolLM2-135M Day-Two Replication Preregistration

**Status: FROZEN DESIGN / PLAN ONLY. No GPU experiment or service pause authorized.**

This is the next rung after the independently audited October 29 RTX 5060
SmolLM2-135M run. Day one's evidence remains unchanged and `spread_24`
remains labeled **externally reported contended**, regardless of future outcomes.
This plan does not imply that the model has been proven faster.

## 1. Frozen research question

Does the candidate BATCHED prefill implementation consistently reduce
end-to-end elapsed time versus canonical SINGLE on the **same pinned
SmolLM2-135M-Instruct**, with each method reproducing **all 32 exact
greedy IDs** and the independently provided sampled logit probes,
across two separately controlled GPUs?

This experiment tests a narrow pretrained workload and reports
descriptive paired differences. It cannot by itself certify arbitrary
model speedups, cross-machine rankings, statistical significance,
source authenticity, or a production scheduler.

## 2. Fixed before observing day-two outcomes

- Master randomization seed **4277796034** (different from day one's 369).
  It was selected and committed **before the next GPU window**.
  This seed is public and must not be changed if a result looks unfavorable.
- Frozen original pretrained revision:
  `HuggingFaceTB/SmolLM2-135M-Instruct`,
  `12fd25f77366fa6b3b4b768ec3050bf629380bac`.
  Expected file SHA-256:
  - `config.json`:
    `8eb740e8bbe4cff95ea7b4588d17a2432deb16e8075bc5828ff7ba9be94d982a`.
  - `model.safetensors`:
    `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`.
- **Exactly four** frozen literal-token prompt cases from
  `scripts.real_weight_matrix.frozen_cases`:
  `short_1`, `mixed_7`, `repeat_12`, `spread_24`.
  Each has 32 greedy continuation steps; `short_1` is baseline
  correctness control only, and **not** included in paired timings.
- Three paired A/B cases, with **eight balanced pairs per case**:
  4 SINGLE-first, 4 BATCHED-first in a preregistered pseudorandom
  order. One **unscored warmup per method** per case. Every scored
  method must match all reference greedy IDs and sparse logit probes.
- Use exactly the same frozen case seeds and method ordering on the
  5060 and a future 5070 replication; collect distinct per-machine
  receipts. No silent retry, deletion, time-based exclusion or
  re-randomization.

The per-case seed is the first four bytes (big-endian unsigned) of
`sha256(utf8("PhiStream:SmolLM2:day2:v1:" + str(4277796034) + ":" + case))`.
The reference implementation is
`scripts/smollm2_day2_prereg.py`, with the actual method schedule
matching `scripts.paired_real_prefill_ab.schedule(8, derived_seed)`.

| Case | Derived uint32 seed | Pairs | SINGLE-first | BATCHED-first |
| --- | ---: | ---: | ---: | ---: |
| `mixed_7` | 1389626234 | 8 | 4 | 4 |
| `repeat_12` | 757448 | 8 | 4 | 4 |
| `spread_24` | 3796163371 | 8 | 4 | 4 |

The generated manifest also freezes the exact order of all eight
pairs per case. **Do not alter the generator or seed after approval.**

For reproducibility, use the offline generator:

```bash
python -m scripts.smollm2_day2_prereg \
  --output benchmarks/local/smollm2-day2-frozen-prereg.json
```

The file is created *exclusively*, never overwritten. Save its full
SHA-256 and the Git commit hash **before** the first actual experiment,
then make the manifest immutable in the evidence bundle. This script
does not import a model or touch CUDA.

## 3. Memory/permissions gate (no new action authorized)

Day one's reported scenario selected
`--capacity-mib 1600 --reserve-mib 512`, which is a **policy setting,
not actual GPU free memory**. Independently sample both torch and
nvidia-smi readings and preserve their contradiction. Require a
separate, owner-approved GPU window and a conservative check of at
least **2,200 MiB actually reported free by nvidia-smi** before
pretrained execution groups. If the sensor is unavailable, or a
credible lower free-memory bound fails, **do not start**. A policy
pass alone is not a safe-fit guarantee.

Owner Hunter must explicitly authorize any model download, pause
of the local DeepSeek/Twins/Local Table services, unloading, restart
or GPU time. Never automatically unload/kill anything. Snapshot
every involved model/service status first, restore the *same* state
afterward and log the restoration outcome. Pause **all known callers
that can reload models** before the run, not just the DeepSeek seat.
A previously unidentified reload source remains a known risk.

The independent Hugging Face reference must be separately run in
a controlled environment, without importing PhiStream, with the
exact pinned checkpoint and SHA-bound preflight. Stop at *any*
checkpoint mismatch, parity failure or unresolved memory condition.
The user-provided day-two manifest **does not grant consent** to
download or execute models, nor does merging this PR.

## 4. Timestamped GPU contention requirement

**Required before execution: implementation and tests.** The
existing paired A/B runner does **not** timestamp every
SINGLE/BATCHED method boundary and has **no built-in 1-second GPU
process watcher**. Merely writing `nvidia-smi` output next to a
case receipt does **not** make a per-trial contention label reliable.
Those integration gaps must be closed and CPU-tested in a separate
reviewed change before day two is authorized.

The intended collector will:

1. Timestamp trial start/end with monotonic time **and** a recorded
   UTC mapping. Keep each trial's case, pair number, method and
   order in an immutable ledger.
2. Poll a read-only `nvidia-smi --query-compute-apps` process inventory
   at approximately **one-second** intervals, including PIDs, process
   names, available VRAM and query health; log missing/unsupported
   fields and time gaps, not just successful readings.
3. Mark a trial **CONTENDED** if any visible non-allowlisted
   GPU process overlaps its measured window; preserve the full
   trial and its paired counterpart. If the process list is unsupported
   (possible on WSL2), missing, or gaps make overlap ambiguous,
   mark the trial **UNKNOWN**, never `CLEAN`.
4. If monitoring is complete and no unexpected process is detected,
   call that **OBSERVED_CLEAR** only; one-second sampling cannot
   prove physical GPU exclusivity or detect every short process.
5. **Fail closed before starting the next trial** if memory is too
   low, a known reload is in progress, or the watcher is unhealthy.
   If any run fails, preserve partial receipts; restart only with
   an explicitly new run identifier and documented reason.
6. Seal each final receipt with SHA-256 and the source commit; post
   the final archive digest before examining aggregate results.
   Never cherry-pick, delete, reclassify or replace the contended
   historical day-one sample.

Observer overhead is another potential timing perturbation and
must be reported. Maintain explicit provenance distinguishing
observed process overlap from operator-reported service activity.

## 5. Reporting and stopping rules

Report **all 8 pairs per case**, including the raw SINGLE/BATCHED
wall times and time-to-first-logits, per-pair relative gains, median,
spread, recorded bytes staged, temperature/clock/VRAM metadata,
contamination labels and failed/partial trial status. Summarize all
trials first and only then a separately labeled observed-clear
subset, **without changing which data are in the full ledger**.
Always preserve the initial predeclared ordering and trial IDs.

If a process reload, missing watcher data or parity problem occurs,
do not re-label timings as uncontended. Do not infer speedup
certification from a further-review gate or 8 pairs. Stop on a
failed correctness gate; preserve its evidence.

**Implementation compatibility:** the existing
`scripts.paired_real_prefill_ab` accepts 8 pairs and uint32 seeds,
but its older four-pair **external ZIP auditor** expects precisely
4 pairs and seed 369. It **cannot** validate the new study unchanged.
Update/extend the versioned auditor *before* analyzing any
day-two ZIP; never silently weaken the day-one frozen contract.
This preregistration is ready for review now, not a runnable GPU
automation.

**Ledger Above Ego.**
