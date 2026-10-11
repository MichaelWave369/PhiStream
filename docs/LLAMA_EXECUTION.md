# Gate B candidate: Llama token-ID execution (experimental)

PhiStream now has an original, explicit, narrow **LlamaForCausalLM**
decoder for **locally held safetensors** files. It can execute sequential
token IDs with on-demand per-layer weights, canonical Llama split-half
RoPE, GQA, RMSNorm and SwiGLU, then produce **token IDs**.

This is not a chat service, tokenizer, GGUF importer, 70B benchmark,
production model support, or verified agreement with an external model.
The independent-reference tests are small **untrained synthetic fixtures**.

## Example (no models are downloaded)

From the checked-out repository root, after installing a suitable Python
3.10+ environment and official compatible PyTorch:

```powershell
python -m phistream.cli inspect-checkpoint "C:\models\local-llama" --llama-layout
python -m phistream.cli run-llama-ids "C:\models\local-llama" --prompt "1,2,3" --tokens 4 --device cpu --max-tensor-mib 512 --output "benchmarks\local\llama-example.json"
```

**Important:** The numbers `1,2,3` are arbitrary token IDs, *not*
encoded English words. Only use a **correct, separately provided tokenizer
with an appropriate license** if you want meaningful text. PhiStream does not
load or execute tokenizers, download weights, or automatically select files.

For explicit CUDA testing on supported NVIDIA hardware, change
`--device cuda` after running the CUDA doctor and prior synthetic device
validation. CUDA requests fail if unavailable and are **not yet independently
validated** on the RTX 5070 or RTX 3060.

## What actually runs

- `model.embed_tokens.weight` is read as individual rows for input IDs,
  rather than loaded entirely into RAM/VRAM.
- Each decoder block's nine parameter tensors are loaded into FP32 execution
  memory for one layer and released before the next. Weight staging has a
  per-tensor size limit (default **512 MiB**, maximum **4096 MiB**).
- Canonical Llama `rotate_half` applies RoPE (not the separate toy engine's
  adjacent-pair version); causal attention uses grouped KV heads.
- The output projection uses bounded batches of vocabulary rows
  (`--head-rows`, default 128) to compute logits.
- Deterministic greedy token IDs are generated, bounded to 128 new tokens.
- JSON receipts include measured **prefill-inclusive** wall time, load
  counters, GPU PyTorch peak allocations if running CUDA, and explicitly
  `reference_parity: NOT_ESTABLISHED_FOR_THIS_CHECKPOINT`.
- Decoder must fit its active layer, KV cache, activations, and runtime in
  local RAM / VRAM. Streaming weights does not eliminate these limits.

## Deliberate refusal conditions

The input requires a local `config.json` with `model_type: llama`,
canonical Llama projection names/shapes, supported F32/F16/BF16 weights,
and no extra tensor keys. Only standard head dimensions, SwiGLU, full RoPE
and ordinary (non-sliding) causal attention are supported.

RoPE scaling variants including Llama 3.1 extended-context configurations,
attention/MLP biases, Q/K norm variants, mixture-of-experts models, custom
architectures, tensor-parallel checkpoint layouts, and quantized tensors
are rejected. *A checkpoint whose metadata passes is not yet proven to
match an official implementation's numeric or token output.*

Safetensors index and data ranges are validated before executing.
`--output` refuses to overwrite an existing JSON file. Output is
local-only and deliberately **not picked up by the public dashboard as
a verified CUDA receipt**.

## Verification boundaries

The suite builds tiny F32, FP16 and BF16 safetensors checkpoints locally,
then compares ΦStream's **one-token sequential decoder** against a
separately coded **full-sequence causal attention reference** for every
token logit and greedy token ID. This is significant internal correctness
evidence, but **not third-party model-reference parity**.

Next: choose a suitably small, explicitly licensed local Llama-style
checkpoint, record its source/revision/architecture, validate its
tokenizer and compare 32 generated tokens against a named reference
runtime. This is an approval gate; no automatic downloading or publishing
of copyrighted model weights is implied.
