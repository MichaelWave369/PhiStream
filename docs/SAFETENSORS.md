# Gate B: local safetensors preflight (v0.3 candidate)

PhiStream can now inspect local safetensors files **without executing a
checkpoint, importing custom model code, downloading anything, or loading all
weights into RAM**. This is a metadata-first compatibility gate, not a language
generation feature and **not an AirLLM / Ollama / llama.cpp replacement**.

## Commands (Python 3.10+)

Run from the checked-out repository root after installing PhiStream:

```powershell
python -m phistream.cli inspect-checkpoint "C:\models\llama-local"
python -m phistream.cli inspect-checkpoint "C:\models\llama-local" --llama-layout
python -m phistream.cli inspect-checkpoint "C:\models\llama-local" --probe-tensor model.norm.weight
```

An explicit `.safetensors` file works as well. For multiple shards, pass a
folder containing `model.safetensors.index.json` with a `weight_map` of
tensor names to local shard basenames. All referenced shards must exist.

`--llama-layout` checks canonical Llama decoder projection tensor names,
shapes, GQA dimensions and FP32/FP16/BF16 weight dtypes against a local
`config.json` with `model_type: llama`. It refuses incompatible or missing
core tensors. That flag says **shape-compatible**, not executable.

Use `--probe-tensor <name>` to actually read **one** named CPU tensor into
owned memory. Default limit 64 MiB (`--max-mib N`, max 4096). This proves the
on-disk binary can be read and materialized but **does not execute inference**.
For exact data integrity fingerprints (expensive on huge checkpoints), use
`--hash`; hashing reads the entirety of all selected tensor bytes locally.

Output is JSON, provenance
`LOCAL_METADATA_INSPECTED_NOT_BENCHMARKED`, and the optional layout check
has `STRUCTURE_VALID` or `LAYOUT_MISMATCH` status. No GPU throughput,
token counts, or claimed model load timings are produced.

## Deliberate safety boundaries

- Local explicit path only. No URLs, model hub downloads, `trust_remote_code`,
  pickle, dynamic model imports, or automatic checkpoint conversion.
- Shard paths must be basenames with `.safetensors` suffix; path traversal
  and symlink files are rejected.
- Bounded JSON/index metadata, strict duplicate-key checks, dtype/shape/offset
  validation, fully indexed payload, and no overlapping tensor regions.
- Index maps must match shard contents exactly; no missing or duplicate tensors.
- Supported metadata dtypes are an explicit subset. CPU materialization supports
  listed standard PyTorch numeric types; float8 decoding isn't implemented.
- Probe is bounded per tensor and uses the requested RAM, not the GPU.
- Inspecting headers does not independently authenticate weights or filenames;
  opt-in SHA256 computes a fingerprint, **not** a signature or license proof.
- This loader reads **safetensors**, not GGUF. Your existing Ollama models
  will not be inspectable through this path without a separate, intentional
  format adapter.
- Do not publish checkpoint files, proprietary configuration, tokenizers, or
  generated hash inventories without checking source licenses and privacy.

## Following gates

A later, separately validated change must implement explicit one-model
Llama tensor mapping, layerwise math and CPU numerical parity against a
licensed reference. Only after that should we attempt CUDA residency,
predictive prefetch, token sampling or throughput comparisons. Until then,
the public React dashboard must **not** assert successful real-model inference.
