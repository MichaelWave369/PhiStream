# Φ Model Forge · R0.1 (not yet a fine-tuned model)

**Status: scaffolding and review-required fixtures only.** This is a separate
PhiStream research lane. PhiStream's existing inference path does **not**
load Qwen3 or PEFT adapters; PhiServe only registers GGUF and does not natively
serve LoRA adapter directories. No automatic production integration is claimed.

## Objective

Create a **PhiOS-Operator** adapter that understands how to respond about
PhiOS, PhiStream, PhiServe, PhiMadaPrime, GitHub CI, XR and local inference.
Teach *stable behaviors* (provenance, uncertainty, authority boundaries,
error isolation, memory/VRAM profiling, no invented checks). Keep *changing
facts* (repo branches, open PRs, incidents, documentation, versions) in
retrieval. Fine-tuning cannot replace an authenticated tool or permission system.

The seed includes 22 project-authored training fixtures and 6 held-out
evaluation prompts. They are **unreviewed** examples, too small to justify
a useful production model. No copied chat logs, proprietary README dumps,
secrets, tokens, identity data, medical data or model weights are included.
The text is designated CC0-1.0; upstream model licensing remains separate.

## Stage A: CPU-only dataset review

Run from PhiStream's repository root:

```sh
python -m phistream.model_forge --dataset model_forge/seed_examples.jsonl
python -m unittest tests.test_model_forge -v
```

The review tool reports SHA-256, example counts, topic coverage, held-out
split status and `REVIEW_REQUIRED`. It never trains, uploads, executes
commands from examples, or downloads models. For a **separate local copy**
of the dataset, inspect each example, check ownership and source permission,
remove unsuitable samples, then explicitly change its
`review_status` to `human-approved`. The default committed seed
must remain unreviewed. The training runner refuses unapproved examples.

## Stage B: GPU preflight and pilot training (operator only)

Use a fresh Python environment, preferably Linux or WSL2 CUDA on Windows.
Install a compatible CUDA PyTorch wheel using the official selector before
adding the optional dependencies. Example:

```sh
python -m venv .venv-forge
source .venv-forge/bin/activate
# Follow https://pytorch.org/get-started/locally/ for your CUDA build.
python -m pip install "trl[peft]" transformers datasets accelerate bitsandbytes
python -m pip install -e . --no-deps
python -m scripts.phi_model_train --dataset /path/to/reviewed.jsonl
```

That last command only reviews data. No model is loaded by default. For
the first mechanics check, prefer a **small locally cached open-weight
checkpoint** with `--model ... --allow-model-override`; the intended
4B candidate is `Qwen/Qwen3-4B-Instruct-2507` (Apache-2.0). A 12 GiB
RTX 5070 is a research candidate for NF4/QLoRA with batch size 1,
gradient accumulation 8, sequence length 512; this is **not a memory-fit
guarantee**, especially with ComfyUI/Ollama sharing the GPU.

Only once the operator has reviewed examples, checked the actual GPU,
accepted licensing and chosen a fresh directory outside the checkout:

```sh
python -m scripts.phi_model_train \
  --dataset /path/to/reviewed.jsonl \
  --output "$HOME/phi-model-forge/adapter-r01" \
  --steps 16 --max-length 512 --execute
```

Weights must already be cached locally; use `--allow-download` only
when the operator explicitly wants a Hugging Face model fetch.
No model upload occurs. An adapter directory and locally written
`phi-training-receipt.json` are created only after a successful run.
The receipt is self-reported, not a performance or quality certificate.

## Stage C: paired held-out outputs

Use the same base revision and evaluation data, one model per GPU session:

```sh
python -m scripts.phi_model_eval --mode base \
  --dataset /path/to/reviewed.jsonl \
  --output "$HOME/phi-model-forge/base-holdout.json"
python -m scripts.phi_model_eval --mode adapter \
  --adapter "$HOME/phi-model-forge/adapter-r01" \
  --dataset /path/to/reviewed.jsonl \
  --output "$HOME/phi-model-forge/adapter-holdout.json"
```

Outputs require **manual blinded review** for correctness, useful coding
advice, tool-action honesty, hallucinations, stable behavior and false
refusals. Repeat on a much larger independent corpus before promotion.
Test on the physical RTX and measure latency, memory, accuracy and
regressions. No score or superior model claim is fabricated.

## Stage D: PhiServe handoff (not implemented)

A PEFT adapter is not a PhiServe GGUF model. Future work may merge the
adapter into a licensed checkpoint (only by explicit approval), convert
to a supported GGUF, independently validate real inference and
register a unique alias through existing PhiServe commands. Preserve
the base model, the adapter, conversion provenance and rollback. Do not
enable an inference endpoint by default. PhiOS, Mobile and XR should
access a **governed local PhiServe endpoint**, not run separate gigantic
models in their browser UI.

## Legal and operational boundaries

- Project code is MIT; fixture data is CC0-1.0; Qwen3-4B-Instruct-2507 is Apache-2.0.
- Heretic is AGPL-3.0 and is **not** bundled or imported. Refusal ablation is
  a different experimental technique, not equivalent to supervised fine-tuning.
- Public data must be licensed; never train on tokens, private conversation
  histories or customer files without explicit source consent.
- No benchmark, CPU test, or green CI proves a successful RTX training run.
- An instruction model may still hallucinate; permissions remain enforced
  by PhiOS runtime, never solely by the model's generated text.
