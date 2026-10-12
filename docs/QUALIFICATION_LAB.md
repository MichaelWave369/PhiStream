# ΦStream PR #18 · Synthetic Scheduler Qualification

**Research classification only.** All evidence is locally loaded, unauthenticated JSON.
No Ollama models, Vessie routes, weights or production scheduler are read or changed.

## Why a strict gate

The prior 8 MiB × 8 RTX 5070 guarded synthetic runs each beat their
same-worker deliberately serialized negative control on reported median times:

| Captured trial group | Double | Serialized | Reported gain |
| --- | ---: | ---: | ---: |
| Original | 12.445 ms | 14.025 ms | 11.27% |
| Repeat 1 | 15.486 ms | 16.296 ms | **4.97%** |
| Repeat 2 | 12.393 ms | 13.997 ms | 11.46% |
| Repeat 3 | 13.023 ms | 14.073 ms | 7.46% |

**All four are included.** We will not silently discard the borderline
4.97% run. CUDA/PyTorch/Windows OS filesystem caching and thermal
conditions remain uncontrolled. CPU and CUDA clocks remain uncalibrated.

## Step 1: review every existing guarded receipt (read-only)

From the repo directory on your Windows PC:

\`\`\`powershell
git pull --ff-only
$stamp = Get-Date -Format "yyyyMMdd-HHmmssfff"
py -m phistream.cli qualify-scheduler --receipt-glob "benchmarks\local\guarded-double-*.json" --output "benchmarks\local\qualification-baseline-$stamp.json"
\`\`\`

The broad nonrecursive pattern covers **both** the original
\`guarded-double-TIMESTAMP.json\` and the three
\`guarded-double-repeat-N-TIMESTAMP.json\` files. If there are other matching
files, they are included too and visible in the SHA-256 ledger.

**Expected outcome if these are exactly the four valid runs:** \`INCONCLUSIVE\`,
because repeat 1 did not achieve a full 5% and the earlier protocol used
a deterministic alternating control order. No GPU work runs in review mode.

The qualification report includes every filename, local JSON SHA-256,
per-run paired-wins, gain computed from method medians, median paired gain
and whether the per-run thresholds pass. Receipt hashing detects **exact
duplicates only**, not fabrication or malicious editing.

## Three outcomes

- \`REJECTED\`: bad/incomplete/duplicate receipts, mixed reported
  GPU/software identity, overlapping capture windows or fewer than three
  valid runs. **Any** invalid input rejects the entire supplied cohort.
- \`INCONCLUSIVE\`: valid receipts, but any included run is below either
  5% median reduction against its serialized control or 5/6 paired wins;
  fewer than three balanced randomized-order runs; or pooled gains below
  5% median paired improvement / 75% paired win share.
- \`QUALIFIED_SYNTHETIC\`: >=3 valid independently timed reports, >=3
  reports with valid balanced randomized order, **every included run**
  beats the serialized control by >=5% using its two medians and wins
  >=5 of 6 pairs, and pooled pairs satisfy >=5% median gain / >=75%
  wins. All hardware identity fields must match.

Even \`QUALIFIED_SYNTHETIC\` says \`default: SINGLE\`,
\`production_policy_applied: false\`, \`review_only: true\`.

For research comparison, treat the original deterministic protocol
as the baseline cohort, then assess separately preregistered new
randomized-order cohorts. Never silently claim the new cohort supersedes
or erases the older borderline evidence.

## Step 2: next experimental protocol (requires explicit approval)

PR #18 adds an optional balanced shuffled **control order** to the
existing guarded synthetic-only runner. It still executes 8 × 8 MiB
temporary owned synthetic bytes, on the same worker, and waits for CUDA
transfer completion before buffer reuse.

\`\`\`powershell
$stamp = Get-Date -Format "yyyyMMdd-HHmmssfff"
py -m phistream.cli guarded-scheduler --receipt-glob "benchmarks\local\overlap-8m8-repeat-*.json" --run-synthetic --workspace "D:\PhiStream\Experiments" --request-double --approve-synthetic-double --randomize-control-order --output "benchmarks\local\guarded-randomized-$stamp.json"
\`\`\`

The \`--randomize-control-order\` switch requires **all three** of
\`--run-synthetic\`, \`--request-double\`, and
\`--approve-synthetic-double\`. Three rounds run control first and three
run double first, shuffled once **before measurements**, with a
64-bit pseudorandom seed recorded in the receipt for reproducibility
and retrospective audit. This does not alter the actual serialized
control or transfer kernels; it changes trial presentation order only.
If the live CUDA/evidence check fails, fallback still selects single
or blocks on resource failure.

To assess a **new preregistered randomized cohort** after three
operator-approved runs, use a separate specific pattern:

\`\`\`powershell
py -m phistream.cli qualify-scheduler --receipt-glob "benchmarks\local\guarded-randomized-*.json" --output "benchmarks\local\qualification-randomized-cohort.json"
\`\`\`

Keep the older baseline review for comparison. These qualification
assessments are not equivalent to cold-cache storage throughput, physical
GPU/SSD overlap, checkpoint parity or token-per-second improvements.

## Guards and limitations

- JSON path glob: local only, nonrecursive, 1–16 files of <=512 KiB each.
- JSON and sample lengths are bounded. Every included series must have
  six positive finite samples with internally consistent min/median/max,
  including the serialized negative control.
- The experimental shape is **exactly** 8 MiB × 8 chunks. All matching
  receipts must claim synthetic mode, CUDA matching, operator approval,
  no fallback, out-of-band full host/GPU replay integrity and unchanged
  production policy.
- All claims are self-reported. A JSON value can be edited and passing
  these checks does not imply cryptographic provenance.
- Local scratch and receipts only; no network upload. No production
  model execution, adaptive model scheduler or autonomous permission.
