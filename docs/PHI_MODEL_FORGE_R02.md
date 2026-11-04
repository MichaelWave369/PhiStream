# Φ Model Forge R0.2 · local readiness + blinded holdout review

**Status:** CPU-only tested protocol and optional local commands. This is
not a trained model, CUDA success, verified speedup, or deployment approval.

## 1. Inspect the training environment without loading model weights

```powershell
# From the PhiStream checkout, with your desired Python environment activated
python -m phistream.forge_preflight
# ONLY when ready to let PyTorch initialize the local CUDA device:
python -m phistream.forge_preflight --probe-cuda
```

The first command reads an existing JSONL dataset, Python package metadata,
and host platform only. No CUDA initialization, download, train, serve, or
process spawn. The optional second command imports torch and reads its
CUDA device report and free/total VRAM **once**. Neither command makes
network calls or changes drivers, clocks, performance settings, weights,
model cache or local files.

Preflight reason codes are specific: unreviewed data, missing optional
training dependencies, CUDA unprobed/unavailable, insufficient *advisory*
free VRAM, or `OPERATOR_TRAINING_EXPERIMENT_CANDIDATE`. Even the last
one is **not** a hardware-fit, working QLoRA, model-quality, or licensing
certification. Free VRAM threshold is a conservative scheduling hint
(6 GiB), not a claim that a 4B QLoRA job fits. Training remains a
**separate explicit** `python -m scripts.phi_model_train ... --execute`
action, guarded by review flags.

A Windows 11 / RTX 5070 user should start with an isolated Python
environment and select a compatible PyTorch CUDA wheel from
https://pytorch.org/get-started/locally/. The optional model-training
libraries are external to PhiStream's core dependencies. CUDA availability,
quantized kernel compatibility, LoRA trainer signatures and peak memory
must all be verified on that machine. WSL2 may be preferable if native
Windows package combinations prove incompatible; do not assume one
platform works from CPU CI.

## 2. Generate blinded base-versus-adapter review files

After executing the already-established held-out generation commands
twice with exactly the same `model` and `dataset`:

```powershell
python -m phistream.forge_review_pack \
  --dataset C:\path\to\reviewed.jsonl \
  --base C:\private\base-holdout.json \
  --adapter C:\private\adapter-holdout.json \
  --pack C:\private\review-a-b.json \
  --key C:\private\answer-key.json
```

The `--pack` file contains each held-out question, two **randomly**
labeled A/B answers, no automatic score, and a manual review rubric:
factual correctness, uncertainty, authority boundaries, useful action,
and regression risk. The **separate** `--key` maps A/B labels to
base/adapter only after review; keep it away from the reviewer.

Both sources must match the exact SHA-256 of the chosen dataset, the
declared base model, held-out IDs, topics and prompts. The command
rejects malformed receipts, tampered prompts, duplicate IDs, and
claims of production approval. It never invents a score or chooses a
winner. These files contain model outputs and **must be treated as
private local evaluation records**, not uploaded to CI or public GitHub.

## 3. The actual acceptance gate

The reviewer must grade each anonymized pair **before** reading the
answer key, record regressions and uncertainty, then compare results
against an independent larger corpus. Audit the selected model's
license and the original dataset's rights. Quantify
training reproducibility and on-machine GPU memory and inference
performance.

Only after that do we consider a separate, explicit adapter merge
or GGUF conversion experiment and PhiServe field validation.
A well-formatted QLoRA adapter is not an approved PhiOS model, and
neither green CI nor reduced refusal frequency proves improvement.

See [R0.1 seed and training procedure](PHI_MODEL_FORGE_R01.md) and
[PhiServe LoRA intake](https://github.com/MichaelWave369/PhiServe/blob/main/docs/PHI_MODEL_FORGE_INTAKE_R01.md).
