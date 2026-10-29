# PhiStream · Second-Machine Reproduction Protocol

**Purpose:** Compare the same **experiment, weights, token-ID inputs, timing
scope and controls** across RTX 5070 and RTX 5060, not two unrelated
"tokens/second" numbers. No checkpoint downloads or execution by default.

Hunter's collaborator reported an RTX 5060 / WSL2 / CUDA 12.8 run of
the deterministic **four-layer synthetic** fixture, 312 passing tests,
and exact CUDA streamed-vs-resident logit parity. These numbers are
**external, operator-reported**; the local JSON receipts and the second
machine's GPU telemetry have **not been ingested or authenticated**.

Their reported results: STREAM 68.5, PINNED 85.9 and RESIDENT 150.9
tokens/s, with corresponding weight-load times 32.6, 28.5 and 3.3 ms.
About 6 GB of 8 GB VRAM was reportedly used by unrelated processes.
These are descriptive **tiny untrained fixture** measurements, not
SmolLM2 model-generation or expert-cache benchmarks.

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
python -m phistream.cli validate-device artifacts/hunter-5060-fixture --device cuda --tokens 8 --pin-mib 0.125 --output artifacts/hunter-5060-validation.json

# Generation-only bench: 2 fresh-engine warmup passes per mode,
# excluded from measured counter values and wall-clock time.
python -m phistream.cli bench artifacts/hunter-5060-fixture --device cuda --mode stream --prompt 1,2,3 --tokens 8 --warmups 2 > artifacts/hunter-5060-stream-warm.json
python -m phistream.cli bench artifacts/hunter-5060-fixture --device cuda --mode stream --prompt 1,2,3 --tokens 8 --pin-mib 0.125 --warmups 2 > artifacts/hunter-5060-pinned-warm.json
python -m phistream.cli bench artifacts/hunter-5060-fixture --device cuda --mode resident --prompt 1,2,3 --tokens 8 --warmups 2 > artifacts/hunter-5060-resident-warm.json

# Inspect contention before every group of runs; no script controls clocks:
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,temperature.gpu,utilization.gpu --format=csv
~~~

In the standalone \`bench\`, \`--warmups 0\` is still the default for
**historic receipt compatibility**; request \`--warmups 1\` or
\`--warmups 2\` explicitly in both machines' new runs.
Each warm-up creates a fresh engine and runs the **same prompt** and
token count. Each trial uses a separate new engine. CUDA work is
synchronized outside the timed region before and after generation.
Warm-up loads, CPU/GPU startup and unscored generation are not included
in the reported per-trial counters. New receipts declare
\`warmup_passes_unscored\`, \`timing_scope\`, and parity of generated IDs
between unscored and measured runs.

**Caution:** The \`validate-device\` command retains its historical
no-explicit-warm-up methodology, so its three mode numbers must NOT be
compared directly with new warm \`bench\` numbers. Record the exact
\`--pin-mib\`, number of tokens, and any GPU contention. The
pinned setting here is **illustrative**; replay the exact budget
from the first JSON receipt when comparing a previously run test.

## B. The *different* pretrained SmolLM2-135M experiment

The RTX 5070 study uses **a real local BF16 safetensors model**:
\`HuggingFaceTB/SmolLM2-135M-Instruct\`, revision
\`12fd25f77366fa6b3b4b768ec3050bf629380bac\`.
Original 5070 local path (not valid on WSL without adjusting):
\`D:\PhiStream\Models\SmolLM2-135M-Instruct\`.

The PR21 frozen **literal-token** workloads (not English text):
- \`short_1\`: \`1\`
- \`mixed_7\`: \`9,28,7,139,302,900,4096\`
- \`repeat_12\`: \`42,42,42,42,42,42,13,13,13,13,1,1\`
- \`spread_24\`: \`495,871,1604,2361,3141,4004,5127,6203,7129,8261,9347,10589,11827,13219,14641,15883,17257,18631,19937,21247,22621,23917,25127,26393\`

All cases evaluate **32 greedy continuation tokens**, with a
separately executed Hugging Face reference and a SHA-bound PhiStream
parity receipt for each case. See [four-case exact PowerShell protocol]
(REAL_WEIGHT_MATRIX.md).

The 5070's *later* balanced paired comparison is different again:
three nontrivial cases (\`mixed_7\`, \`repeat_12\`, \`spread_24\`);
four paired repetitions per case, seed 369, one unscored warmup
per mode, with randomized balanced order and per-trial reference
parity. The original code and exact parameterized CLI form are in
[paired A/B protocol](PAIRED_REAL_PREFILL_AB.md) and
[three-case audit](THREE_CASE_AB_CENSUS.md).

The following is an **illustrative PowerShell command from the
published 5070 protocol**, NOT a verified transcript of what was
actually typed on Hunter's machine:

~~~powershell
$ckpt = "D:\PhiStream\Models\SmolLM2-135M-Instruct"
$preflight = Get-ChildItem "benchmarks\local\smollm2-cuda-preflight-*.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
$matrixDir = "benchmarks\local\smollm2-matrix-20261012-131912204"
$case = "spread_24"
$run = Join-Path (Join-Path $matrixDir "batched-prefill") "paired-ab-run_$case.json"
py -m scripts.paired_real_prefill_ab run --checkpoint "$ckpt" --preflight-receipt "$preflight" --matrix-dir "$matrixDir" --case "$case" --pairs 4 --seed 369 --warmups 1 --device cuda --capacity-mib 8192 --reserve-mib 1024 --approve-local-execution --output "$run"
~~~

**Do not paste this command onto a busy 8GB 5060.** It was designed
for the existing 5070 evidence and paths; the new machine must first
obtain the weights voluntarily, create its own SHA-bound preflight,
independently execute all references, pass the four-case baseline,
build new batched-candidate receipts, and pass the local planning
gate. In particular, \`--capacity-mib 8192\` does not make 8GB free
when other processes occupy ~6GB. No automated model download, CLI
run, path assumption or memory policy bypass is authorized here.

## C. What receipts to exchange

Share **copies** of \`validate-cuda-5060.json\`, both standalone bench
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
