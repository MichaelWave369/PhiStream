# PhiStream · Second-Machine Reproduction Protocol

**Purpose:** Compare the same **experiment, weights, token-ID inputs, timing
scope and controls** across RTX 5070 and RTX 5060, not two unrelated
"tokens/second" numbers. No checkpoint downloads or execution by default.

Hunter's collaborator reported an RTX 5060 / WSL2 / CUDA 12.8 run of
the deterministic **four-layer synthetic** fixture, 312 passing tests,
and exact CUDA streamed-vs-resident logit parity. These numbers are
**externally generated evidence**. The accompanying 41-entry 5060 ZIP
was supplied for this review; its three original JSONs match
`originals.sha256`, and the new repeated-trial summaries were
independently recomputed from their trial JSON values. These checks
establish internal consistency, **not an independently witnessed run,
authenticated publisher, or controlled hardware benchmark**.

Their reported results: STREAM 68.5, PINNED 85.9 and RESIDENT 150.9
tokens/s, with corresponding weight-load times 32.6, 28.5 and 3.3 ms.
These are descriptive **tiny untrained fixture** measurements, not
SmolLM2 model-generation or expert-cache benchmarks.

**Corrections from the October 29 evidence audit:**
- The original `validate-cuda-5060.json` shows
  `gpu_free_bytes_before = 6989107200` of `8546484224` bytes,
  implying **about 1.5 GB occupied by torch's count**. No
  `nvidia-smi` snapshot was preserved with that original receipt.
- On the later warmed retest, the DeepSeek r1:8b Ollama model was
  loaded and `nvidia-smi` reported about **6.45 GB used**. Those
  are different experimental sessions; neither VRAM account should
  be silently substituted for the other.
- Subsequent patched runs showed about **6.96 GB free via torch**
  at the same time `nvidia-smi` reported **about 6.6 GB in use**.
  WSL2 GPU memory reporting is contradictory here: report both
  sources with their units and timestamps, never infer reliable
  actual free VRAM from one reading alone.
- The original nominally "pinned" run with `--pin-mib 0.125`
  pinned **zero layers**, since each fixture layer is 147,968 bytes
  (0.141113 MiB). Its 85.9 vs 68.5 tok/s difference is noise, not
  a real pinning benefit. Use `--pin-mib 0.15` for one actual layer.
- The original standalone `bench` JSONs measured four tokens;
  `validate-device` measured eight. Do not compare these as equal
  workloads.

**Patched independent seven-repeat synthetic fixture evidence**
(Claude, Hunter's RTX 5060, with shared WSL2 contention): median
stream 54.9, 0.125 MiB/no-pin 45.9, 0.15 MiB/one-pin 53.3, resident
77.1 tok/s. Coefficients of variation ranged from **16% to 26%**.
The measured **5,984,512 vs 4,652,800 vs 657,664 weight-load bytes**
are a more stable counter distinction than noisy token rates.
The original unmodified files were SHA-checked against their manifest;
the repeated JSON medians and CV were recalculated, but hardware
execution was not independently witnessed.

## A. Repeat the same synthetic workload with unscored warm-ups

All CLI commands are offline and use only a **fresh locally created
untrained fixture**. Change paths as necessary. These examples use
WSL2 Bash and Python rather than copying Windows drive paths verbatim.

~~~bash
cd /path/to/PhiStream
python -m unittest discover -s tests
python -m phistream.cli init-fixture artifacts/hunter-5060-fixture
python -m phistream.cli verify artifacts/hunter-5060-fixture
python -m phistream.cli parity artifacts/hunter-5060-fixture --device cuda
python -m phistream.cli validate-device artifacts/hunter-5060-fixture --device cuda --tokens 8 --pin-mib 0.15 --output artifacts/hunter-5060-validation.json

# Generation-only bench: 2 fresh-engine warmup passes per mode,
# excluded from measured counter values and wall-clock time.
python -m phistream.cli bench artifacts/hunter-5060-fixture --device cuda --mode stream --prompt 1,2,3 --tokens 8 --warmups 2 > artifacts/hunter-5060-stream-warm.json
python -m phistream.cli bench artifacts/hunter-5060-fixture --device cuda --mode stream --prompt 1,2,3 --tokens 8 --pin-mib 0.15 --warmups 2 > artifacts/hunter-5060-pinned-warm.json
python -m phistream.cli bench artifacts/hunter-5060-fixture --device cuda --mode resident --prompt 1,2,3 --tokens 8 --warmups 2 > artifacts/hunter-5060-resident-warm.json

# For descriptive comparisons, request at least five fresh measured engines
# in one process. Inspect the full trial series, not merely its median.
python -m phistream.cli bench artifacts/hunter-5060-fixture --device cuda --mode stream --prompt 1,2,3 --tokens 8 --warmups 2 --repeats 7 > artifacts/hunter-5060-stream-r7.json
python -m phistream.cli bench artifacts/hunter-5060-fixture --device cuda --mode stream --pin-mib 0.15 --prompt 1,2,3 --tokens 8 --warmups 2 --repeats 7 > artifacts/hunter-5060-pin-r7.json
python -m phistream.cli bench artifacts/hunter-5060-fixture --device cuda --mode resident --prompt 1,2,3 --tokens 8 --warmups 2 --repeats 7 > artifacts/hunter-5060-resident-r7.json

# Inspect contention before every group of runs; no script controls clocks:
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,temperature.gpu,utilization.gpu --format=csv
~~~

In the standalone `bench`, `--warmups 0` is still the default for
**historic receipt compatibility**; request `--warmups 1` or
`--warmups 2` explicitly in both machines' new runs.
Each warm-up creates a fresh engine and runs the **same prompt** and
token count. Each trial uses a separate new engine. CUDA work is
synchronized outside the timed region before and after generation.
Warm-up loads, CPU/GPU startup and unscored generation are not included
in the reported per-trial counters. New receipts declare
`warmup_passes_unscored`, `timing_scope`, and parity of generated IDs
between unscored and measured runs.

**Caution:** The `validate-device` command retains its historical
no-explicit-warm-up methodology, so its three mode numbers must NOT be
compared directly with new warm `bench` numbers. Record the exact
`--pin-mib`, number of tokens, and any GPU contention. The **0.15 MiB** budget pins one layer of this exact fixture;
`0.125 MiB` pins none. Repeat-mode receipts include each trial,
mean/min/max/median/sample standard deviation and CV. Each has
torch free/total VRAM and best-effort `nvidia-smi` readings before
and after its run. `--repeats 1` maintains the historic single receipt
shape; `--repeats 2..50` activates the new descriptive series.
Only one two-pass warm-up sequence happens before the first measured
trial; do not assume warm-ups are repeated for each subsequent trial.
The optional `nvidia-smi` subprocess happens outside the measured
generation wall-clock window and may still perturb device clocks.
Record interference and measure on the same machine under matching
conditions.

## B. The *different* pretrained SmolLM2-135M experiment

The RTX 5070 study uses **a real local BF16 safetensors model**:
`HuggingFaceTB/SmolLM2-135M-Instruct`, revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`.
Original 5070 local path (not valid on WSL without adjusting):
`D:\PhiStream\Models\SmolLM2-135M-Instruct`.

The PR21 frozen **literal-token** workloads (not English text):
- `short_1`: `1`
- `mixed_7`: `9,28,7,139,302,900,4096`
- `repeat_12`: `42,42,42,42,42,42,13,13,13,13,1,1`
- `spread_24`: `495,871,1604,2361,3141,4004,5127,6203,7129,8261,9347,10589,11827,13219,14641,15883,17257,18631,19937,21247,22621,23917,25127,26393`

All cases evaluate **32 greedy continuation tokens**, with a
separately executed Hugging Face reference and a SHA-bound PhiStream
parity receipt for each case. See [four-case exact PowerShell protocol]
(REAL_WEIGHT_MATRIX.md).

The 5070's *later* balanced paired comparison is different again:
three nontrivial cases (`mixed_7`, `repeat_12`, `spread_24`);
four paired repetitions per case, seed 369, one unscored warmup
per mode, with randomized balanced order and per-trial reference
parity. The original code and exact parameterized CLI form are in
[paired A/B protocol](PAIRED_REAL_PREFILL_AB.md) and
[three-case audit](THREE_CASE_AB_CENSUS.md).

**STOP: RTX 5070 HISTORICAL COMMAND ONLY. Do not copy or execute the
following 8192 MiB example on Hunter's 8 GB RTX 5060, especially with
DeepSeek running.** This is an archived illustration of the original
5070 protocol, not a safe recipe for a second machine. Real-model runs
need separate approval, machine-specific preflight, and explicit
memory-policy review *before* any copyable execution command.

The following is an **illustrative PowerShell command from the
published RTX 5070 protocol**, NOT a verified transcript of what was
actually typed on Hunter's machine:

~~~powershell
$ckpt = "D:\PhiStream\Models\SmolLM2-135M-Instruct"
$preflight = Get-ChildItem "benchmarks\local\smollm2-cuda-preflight-*.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
$matrixDir = "benchmarks\local\smollm2-matrix-20261012-131912204"
$case = "spread_24"
$run = Join-Path (Join-Path $matrixDir "batched-prefill") "paired-ab-run_$case.json"
py -m scripts.paired_real_prefill_ab run --checkpoint "$ckpt" --preflight-receipt "$preflight" --matrix-dir "$matrixDir" --case "$case" --pairs 4 --seed 369 --warmups 1 --device cuda --capacity-mib 8192 --reserve-mib 1024 --approve-local-execution --output "$run"
~~~

**The historical 8192 MiB sample above is not 5060-ready.** It was designed
for the existing 5070 evidence and paths; the new machine must first
obtain the weights voluntarily, create its own SHA-bound preflight,
independently execute all references, pass the four-case baseline,
build new batched-candidate receipts, and pass the local planning
gate. In particular, `--capacity-mib 8192` does not make 8GB free
when other processes occupy ~6GB. No automated model download, CLI
run, path assumption or memory policy bypass is authorized here.



### Required SHA-bound SmolLM2 preflight writer (not model execution)

The original 5060 plan lacked the exact preflight writer. After
obtaining the pinned checkpoint **with separate operator approval**
and reviewing free VRAM, this example writes a **non-executing**
preflight receipt. It does not load or run model tensors:

```bash
python -m phistream.cli real-model-gate "$CKPT" --device cuda \
  --capacity-mib 1600 --reserve-mib 512 --hash-checkpoint \
  --source HuggingFaceTB/SmolLM2-135M-Instruct \
  --revision 12fd25f77366fa6b3b4b768ec3050bf629380bac \
  --license-id Apache-2.0 \
  --output "benchmarks/local/smollm2-cuda-preflight-$(date +%Y%m%d-%H%M%S).json"
```

The output name is illustrative. **Acceptance requires every one** of
the following, as enforced by `scripts/export_hf_reference.py`'s
`check_checkpoint` routine:

- Receipt schema matches the expected real-model preflight schema,
  `status == PREFLIGHT_REVIEW_REQUIRED`, and the memory preflight status
  is `REVIEW_REQUIRED`.
- `model_execution == false`, `double_buffer_enabled == false`,
  and `scheduler_policy_applied == false`.
- Checkpoint `format == SAFETENSORS` and `model_type == llama`.
  The local `config.json` must also conform to the bounded Llama
  architecture contract.
- A nonempty, user-declared model **source, revision, and license_id**
  are present, marked as **not independently verified**.
- `checkpoint_fingerprint.algorithm == sha256` and the fingerprint
  names **exactly** `config.json` and `model.safetensors` with valid
  SHA-256 digests. The independent reference helper rehashes both
  existing local files and rejects mismatches.

For SmolLM2-135M-Instruct revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`, the published 5070
receipt lists these comparison values:

- `config.json`:
  `8eb740e8bbe4cff95ea7b4588d17a2432deb16e8075bc5828ff7ba9be94d982a`
- `model.safetensors`:
  `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`

Matching both digests establishes identical **file bytes** between
receipts; it does not authenticate the publisher, model training
history, or any real-model execution.
A preflight receipt is not consent to the separately gated reference
or PhiStream real-weight runs. The 1600 MiB capacity is **only an operator-chosen memory policy
scenario**, not a guarantee of fit on a contested WSL2 GPU. It may
even exceed the available amount from nvidia-smi when DeepSeek is
loaded: re-measure and apply the **more conservative credible**
free-memory limit before any separately authorized execution.
The CUDA policy reads `torch.cuda.mem_get_info`; on this WSL2 host
that reading contradicts nvidia-smi, so neither the reported pass
nor the displayed free bytes guarantee safety. The
published 5070 hashes are documented in the external-reference lab.
Preserve every receipt. No real SmolLM2 model run is claimed here.

## B2. Independent review of PR #42 on RTX 5060, October 29, 2026

Hunter's Claude supplied a separate **14,698-byte verification ZIP**
(`20261029-phistream-pr42-verify-claude.zip`) after merging PR #42.
Its SHA-256 was independently checked against the reported digest:

`934ceba7119e8351d817ccb810980c0bf3ea2415891a55b06de62c08d5345de8`

The archive's `receipts.sha256` matches all six listed JSON files.
The seven individual throughput trial values in each of the four
mode receipts were independently recomputed; their medians and
sample-stdev / mean coefficients of variation match the stored
summaries. All seven trials of each mode have unchanged generated
token IDs. Each receipt records one two-pass warm-up before the first
measured trial, then six more fresh-engine measured generations.
These are **checks of internally consistent third-party artifacts**,
not independently witnessed GPU execution.

| Mode (all synthetic) | Median tok/s | CV | Pinned layers | Measured loaded bytes |
| --- | ---: | ---: | ---: | ---: |
| STREAM | 117.0 | 31% | 0 | 5,984,512 |
| STREAM with 0.125 MiB | 105.4 | 32% | 0 (warned) | 5,984,512 |
| STREAM with 0.15 MiB | 99.7 | 46% | 1 | 4,652,800 |
| RESIDENT | 163.2 | 30% | 4 | 657,664 |

The supplied `verify.json` reports checksum verification of 39
fixture tensors; `parity.json` reports exact synthetic CUDA output
parity (max absolute logit delta 0.0). The accompanying report says
**323 tests passed** on the merged commit
`b5c4b32343d1de0b1cad07df067b48722f0b50c4`. That count comes
from the collaborator's report, not an independently witnessed test
run on the 5060. The archived `fixture.sha256` records the synthetic
seed-369 fixture's manifest and weight hashes.

Hardware context was unstable: the new receipt's
`torch.cuda.mem_get_info` reported around 6.99 GB free while
`nvidia-smi` reported around 6,628 MiB used of 8,151 MiB. On this
shared WSL2 machine those readings conflict and **must not** be
combined into a fabricated estimate. The relative throughput
differences varied strongly from the previous session, including
a one-layer-pinned median **below** plain STREAM. Record the transfer
counters as descriptive and do not certify a speedup.

Claude's patch comparison reports no missing functional fixes after
our integration, with one hardened field-count check in
`nvidia-smi` parsing. The full ZIP is **operator-supplied evidence
held outside the public repository**; we do not publish the remote
system's raw metadata or imply independent checkpoint provenance.

## C. What receipts to exchange

Share **copies** of `validate-cuda-5060.json`, both standalone bench
JSON files, plus subsequent cold/warm JSONs. Include the fixture
manifest hash, exact commit SHA, Torch/CUDA build, WSL2/driver details,
VRAM free before each timed group, and complete CUDA error logs if
any. Keep old receipt filenames immutable.

Treat synthetic parity, pretrained model parity, and comparative
performance as **three separate evidence categories**. Repeat exact
workloads on both machines before quoting cross-device gains. Avoid
claiming 5060 vs 5070 speedup until the same experiment has been
run with comparable contention, clocks and warm-up methodology.

**Ledger Above Ego.**
