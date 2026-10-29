# ΦPocketRotor E3-P · Local Qwen3.5 Compatibility Preflight

**Status: offline, read-only, static source/config gate.** The script
does **not import Transformers or PyTorch**, query your GPU, deserialize
weights, run the router, load tokens, download a model, or interact with
Ollama. It adds no new packages to PhiStream.

The merged E3 adapter uses post-router and pre-experts hooks to observe
native 256-expert, top-8 selections, while preserving native execution.
E3-P is the next constraint: before attempting that pilot on a real model,
verify that the **locally available** Hugging Face runtime *source
structure* is still compatible and that a compatible Hugging Face
checkpoint directory actually exists.

This avoids a common costly failure: trying a Transformers hook against
Ollama GGUF weights, or assuming 35B weights fit on a 12GB GPU because
only 3B parameters activate each token. Those are different problems.

## What is checked

1. A named local \`modeling_qwen3_5_moe.py\` file is parsed as Python
   AST **without executing or importing it**.
2. The source declares the expected native router, experts and sparse
   block classes. The gate returns native \`router_indices\` in the third
   tuple position from a \`torch.topk\` assignment, and the block passes
   those selected IDs in the second argument to native experts.
3. Router expert count/top-k derive from native model config fields,
   not hard-coded hidden defaults.
4. The explicitly supplied local checkpoint's \`config.json\` identifies
   \`qwen3_5_moe\` with \`qwen3_5_moe_text\` nested text config,
   40 layers, 256 experts, top-8, hidden=2048 and expert width=512,
   with 30 linear-attention and 10 full-attention layers.
5. The checkpoint directory contains nonempty local safetensors file
   **names** and, where present, an index whose referenced shard names
   match local shard names. GGUF files, symlinks, mismatched indices and
   malformed configs are refused.
6. SHA256 of the exact source **file** and config is included in the
   report. Model shards themselves are **not hashed or authenticated**.

Nothing in the preflight proves model weights are valid, installed
source is identical to code executed at runtime, the code would fit in
memory, the hooks would fire after kernel optimization, or native routing
would be captured. The result is a **compatibility lead, not an
execution authorization or performance benchmark**.

## How to run

Run the same code from an existing PhiStream checkout with a **separate
locally available Hugging Face checkpoint** and installed Transformers
source file. The paths below are illustrative and must be changed to
actual existing paths; they are **not** a download command.

~~~powershell
git pull --ff-only
$checkpoint = "D:\Models\Qwen3.5-35B-A3B-HF"
$hfSource = "C:\path\to\your\installed\transformers\models\qwen3_5_moe\modeling_qwen3_5_moe.py"
$receipt = "D:\PhiStream\Experiments\qwen35-e3-preflight-001.json"

if (Test-Path $receipt) { throw "Preserve previous report" }
py -m scripts.pocketrotor_e3_preflight --checkpoint "$checkpoint" --hf-source "$hfSource" --output "$receipt"
if ($LASTEXITCODE -ne 0) { throw "Compatibility preflight held; inspect local paths/config" }
Get-Content "$receipt" -Raw | ConvertFrom-Json | Select-Object status,model_execution,real_native_routing_captured
~~~

Without \`--output\`, the script writes JSON to standard output only.
With \`--output\`, it exclusively creates one new report and will not
overwrite previous evidence.

Expected compatible status:
\`STATIC_CONFIG_AND_SOURCE_MATCH_OPERATOR_REVIEW_ONLY\`

A GGUF checkpoint typically supplied by Ollama is *not* compatible
with the E3 Hugging Face PyTorch hook, even if the model architecture
and human-visible name match. E3-P reports that limitation instead of
pretending Ollama exposes the same Python router modules.

## Online and synthetic CI testing

The automated suite creates a tiny fake source text, synthetic config
and fake nonzero shard filenames. It does **not** claim a pretrained
model or authentic checkpoint. Tests verify AST changes, source drift,
top-k changes, GGUF rejection, broken indexes, symlinks, duplicate
JSON keys, nonfinite JSON and exclusive output protection.

No pretrained model files or local user paths are uploaded to CI.

## Evidence required for the next rung

If a genuine locally installed HF checkpoint and source both pass,
the next independently reviewed stage is a *separately authorized*
native single-token capture, including verification that the exact
installed package revision is the runtime's one, hooks are not bypassed
by optimized kernels, the native gate-to-expert ID comparison holds
for each layer and token, and all output traces remain private.
Only then can E2 replay true expert routes.

A successful source/config check is not permission to load a 35B model
onto a 12GB GPU, and it cannot prove speedup.

**Ledger Above Ego.**
