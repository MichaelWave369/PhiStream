# PR20: Offline Hugging Face Reference Laboratory

This standalone experimental script uses installed Transformers as an external
validation oracle, NOT a dependency of PhiStream's production decoder. It
imports no phistream modules, downloads no model, never modifies original
checkpoint files, uses no tokenizer and never changes scheduling. It does
not copy or reuse Transformers' implementation inside PhiStream.

## Already inspected

Local existing checkpoint: D:\PhiStream\Models\SmolLM2-135M-Instruct
Declared source: HuggingFaceTB/SmolLM2-135M-Instruct
Declared revision: 12fd25f77366fa6b3b4b768ec3050bf629380bac
Declared license: Apache-2.0 (source and license not verified)
Structure: 30 canonical Llama layers, 272 BF16 tensors, one shard.
Previously observed full-file SHA256:
config.json = 8eb740e8bbe4cff95ea7b4588d17a2432deb16e8075bc5828ff7ba9be94d982a
model.safetensors = 5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c

Local hashes prove local byte identity only, not official publisher origin.

## 1. Hash check (safe and read-only; no model execution)

From PowerShell in the PhiStream repository:

    git pull --ff-only
    $ckpt = "D:\PhiStream\Models\SmolLM2-135M-Instruct"
    $preflight = Get-ChildItem "benchmarks\local\smollm2-cuda-preflight-*.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
    if (-not $preflight) { throw "Missing existing preflight receipt" }
    py scripts\export_hf_reference.py --checkpoint "$ckpt" --preflight-receipt "$preflight"

Expected: HASH_VERIFIED_REFERENCE_NOT_EXECUTED. This reads the local config
and safetensors bytes to recompute their SHA256 hashes against the
previous real-model-gate receipt, without even importing Transformers.

## 2. External reference generation (real model load, explicit consent)

Only after the hash check succeeds, review and then run this command.
This DOES execute the pretrained SmolLM2 weights with Hugging Face on CUDA:

    $stamp = Get-Date -Format "yyyyMMdd-HHmmssfff"
    $trace = "benchmarks\local\smollm2-hf-reference-$stamp.json"
    py scripts\export_hf_reference.py --checkpoint "$ckpt" --preflight-receipt "$preflight" --device cuda --prompt "1,4,3" --run-reference --approve-local-execution --output "$trace"

The exporter requires both --run-reference and
--approve-local-execution. It sets HF_HUB_OFFLINE=1,
TRANSFORMERS_OFFLINE=1 and HF_HUB_DISABLE_TELEMETRY=1 before importing
Transformers. The external runtime uses a local-only safetensors
AutoModelForCausalLM, trust_remote_code=False, FP32 weights, eager
attention and no KV cache. No tokenizer, prompt template or EOS policy.

It produces exactly 32 full-sequence forward greedy token steps,
samples top-two and distributed vocabulary logits, verifies that all
checkpoint SHA256 values remain unchanged and writes a new JSON file
exclusively into gitignored benchmarks/local. Literal prompt token IDs
1,4,3 are not encoded English. No model files are uploaded.

Success status:
LOCAL_HF_REFERENCE_TRACE_CREATED_NOT_YET_COMPARED.
No PhiStream parity has been measured by that step.

Do not install, upgrade or silently change numerical tolerances if
your currently installed Transformers build reports an exception.
Save the error for investigation.

## 3. Inspect reference metadata without executing PhiStream

    py -m phistream.cli real-model-gate "$ckpt" --capacity-mib 8192 --reserve-mib 1024 --device scenario --trace "$trace" --source "HuggingFaceTB/SmolLM2-135M-Instruct" --revision "12fd25f77366fa6b3b4b768ec3050bf629380bac" --license-id "Apache-2.0"

Expected: PREFLIGHT_TRACE_REVIEW_REQUIRED. That is NOT parity.

## 4. PhiStream real-checkpoint parity execution (different approval)

Only after the external trace is inspected and you explicitly approve
the real PhiStream decoder run, execute:

    $stamp = Get-Date -Format "yyyyMMdd-HHmmssfff"
    $receipt = "benchmarks\local\smollm2-parity-$stamp.json"
    py -m phistream.cli real-model-gate "$ckpt" --capacity-mib 8192 --reserve-mib 1024 --device cuda --max-tensor-mib 512 --head-rows 128 --trace "$trace" --source "HuggingFaceTB/SmolLM2-135M-Instruct" --revision "12fd25f77366fa6b3b4b768ec3050bf629380bac" --license-id "Apache-2.0" --execute-parity --approve-local-execution --output "$receipt"

The real-model-gate validates the trace, refuses a memory-policy block,
checks full checkpoint file SHA256 against the reference, then compares
32 teacher-forced next-token greedy IDs plus sparse sampled logits.
It records timings and weight-stage counts for the existing single-buffer
decoder, not a multi-engine throughput benchmark.

Possible statuses include PARITY_MATCH_USER_REFERENCE_UNAUTHENTICATED,
PARITY_MISMATCH and HELD_MEMORY_POLICY. All have useful scientific
meaning. No independent cryptographic authentication of the reference
runtime, license or source revision is claimed.

## Boundaries

CI only verifies this script's hash, consent, safety and output contracts
using untrained local fixtures and fake test reference responses. CI does
not run SmolLM2 or a true external runtime. The external implementation
is used solely for validation; never integrated into PhiStream's inference
path. No automatic downloading, tokenizer integration, Ollama/GGUF access,
double-buffer promotion or Vessie settings changes.
