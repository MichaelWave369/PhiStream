# ΦPocketRotor E1 · Balanced Real-Checkpoint Residency Gate

**Scope: opt-in, offline, experimental.** This work adds a new standalone
`scripts.pocketrotor_e1` research workflow. It does not change
`StreamingLlama`, E0, the reference comparator, model checkpoints, PhiOS,
Vessie, Ollama or the production model scheduler.

## Research question

Does E0's finite nested whole-layer pinning give repeatable *end-to-end*
benefit over canonical serial streaming **and** flat LRU on identical
checkpoint weights, with output parity against the existing 32-step
user-declared external reference trace?

This is a **dense Llama** experiment. Each layer executes every token. The
nested quadtree addresses are immutable identities, *not* a learned MoE expert
router, asynchronous weight prefetch or a novel compression technique.

## Frozen protocol

- Supported checkpoint: existing local trusted canonical
  `LlamaForCausalLM` safetensors with the strict PhiStream shape check.
  **GGUF/Ollama/Qwen-MoE formats remain unsupported in this executor.**
- A full 32-step reference trace in the existing
  `PHISTREAM_REFERENCE_TRACE_V1` JSON contract is mandatory.
- The plan parses checkpoint metadata, verifies reference structure, records
  the SHA256 of the trace file, and fixes the scheduler design. It does **not**
  read/hash full model weights, create an executor, query CUDA or run inference.
- A run requires an unchanged plan JSON, explicit `--approve-local-execution`,
  and full checkpoint hash verification **before** decoder creation.
  Checkpoint/trace/plan hashes are checked again after execution.
- Policies: `stream`, `flat-lru`, `nested-pin`. All use the same original
  checkpoint, prompt, 32 teacher-forced continuation IDs, sparse reference
  logit coordinates and tolerance rules. Each trial must match every token
  and every reference probe. A mismatch stops and preserves a failure receipt.
- Three or six **balanced rounds**. A seeded shuffled cyclic schedule makes
  every method occupy every trial position equally often. This improves
  order confounding; it does not eliminate thermal or OS-cache confounding.
- Per method, run a fresh decoder (cold **logical** cache), then reset its
  KV state while retaining the same weight cache and repeat the same prompt
  (warm **logical** cache).
- Cold timing includes engine construction. Warm timing excludes engine
  construction and includes whole prompt + 31 continuation steps. Probe
  validation work is included consistently. Never infer SSD cold-cache
  timings or a GPU hardware bandwidth figure from these phases.
- Records elapsed wall time, TTFT, correctly matched tokens/probes, load
  operations, staged tensor bytes, cache hit/miss/eviction deltas, and
  observed per-step PyTorch CUDA peak allocated bytes when available.
  These peak figures omit CUDA context, allocator reservations and other
  processes; policy estimates do not guarantee physical fit.
- Review threshold (for nested caching only): a descriptive median gain
  of at least **5%** versus streaming and faster outcomes in **2/3 rounds**
  in **both** cold and warm logical phases. Flat-LRU results remain visible.
  This status means **further review**, not certified speedup.

## Run locally after code review

All commands are explicit. Nothing automatically loads/downloads a model.

**Step A: read-only plan (CPU scenario, without touching CUDA)**

~~~powershell
git pull --ff-only
$ckpt = "D:\PhiStream\Models\SmolLM2-135M-Instruct"
$trace = "D:\PhiStream\Experiments\existing-32-step-reference.json"
$folder = "D:\PhiStream\Experiments"
$plan = Join-Path $folder "pocketrotor-e1-plan-001.json"
if (Test-Path $plan) { throw "Plan exists; do not overwrite" }
py -m scripts.pocketrotor_e1 plan --checkpoint "$ckpt" --trace "$trace" --output "$plan" --device cuda --cache-mib 128 --capacity-mib 12288 --reserve-mib 2048 --max-tensor-mib 512 --rounds 3 --seed 369
~~~

The selected device is just a scenario label during the plan. Inspect
`READY_FOR_REVIEW`, `memory_policy`, pinned layer IDs and the exact
reference/checkpoint identifiers before authorizing any run. A missing
checkpoint or incompatible trace is a hard stop.

**Step B: separately approved CUDA trial**

~~~powershell
$run = Join-Path $folder "pocketrotor-e1-run-001.json"
if (Test-Path $run) { throw "Run exists; preserve it" }
py -m scripts.pocketrotor_e1 run --checkpoint "$ckpt" --trace "$trace" --plan "$plan" --output "$run" --approve-local-execution
$exit = $LASTEXITCODE
Get-Content "$run" -Raw | ConvertFrom-Json | Select-Object status, model_execution
if ($exit -ne 0) { Write-Warning "Research execution held or failed; inspect receipt" }
~~~

`--run` is NOT used here: the `run` **subcommand** plus the explicit approval
flag is required. Missing approval, plan drift, over-budget memory policy, an
unavailable CUDA device, checkpoint hash mismatch or a destination collision
blocks execution. In-run failures leave an `INCOMPLETE` or `FAIL_CLOSED`
receipt to investigate rather than pretending partial work passed.

Three rounds × three policies × two logical temperature phases =
**18 complete prompt/32-step evaluations**, plus model setup and hashing.
This is nontrivial hardware work, especially if the model is too large.
Use a previously verified, small checkpoint first. Don't automatically launch
a 70B model on a 12 GB RTX 5070.

## Evidence boundaries

- Source authenticity, third-party weight licensing and true independence
  of a supplied reference remain **user-declared, unauthenticated**.
- Sparse logit coordinates are checked, not every vocabulary coordinate.
  The existing E0 study separately checks full-logit agreement against
  PhiStream's own streaming executor, **not an external runtime**.
- The trial is serial, in one process, with OS/NVMe page cache, power limits,
  GPU clocks, background processes and thermal behavior uncontrolled.
- No system cache flush, remote model download, weight rewrite or telemetry
  upload; the local `--output` receipt is exclusive-create.
- No speed claim without replicated, reviewed hardware receipts and comparison
  against a practical backend such as llama.cpp/Ollama with equivalent weights.

**CI runs only synthetic CPU tests**, including the reference parity checks,
schedule arithmetic, consent refusal, plan tampering, budget holds and receipt
preservation. CI results cannot substitute for your RTX 5070 local trial.

**Ledger Above Ego.**
