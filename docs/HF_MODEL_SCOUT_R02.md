# Φ Model Scout R0.2: opt-in Hugging Face model metadata inspection

**Status: review only. Nothing downloads model weights, loads checkpoints, trains adapters, executes inference, registers a model, or changes Vessie's governed routing.**

This is the next research rung after [Model Atlas R0.1](HF_NINE_MODEL_ATLAS_R01.md), housed in PhiStream to avoid another standalone repo.

## What happens, and when

Default invocation never accesses the network:

    python -m phistream.hf_model_scout

An operator can explicitly select a model in the existing review-only catalog and opt in to a single Hugging Face Hub **model metadata** request:

    python -m pip install huggingface_hub
    python -m phistream.hf_model_scout --inspect --allow-network-metadata --model Qwen/Qwen3-4B-Instruct-2507

An optional exact commit pin protects against changes between metadata observations:

    python -m phistream.hf_model_scout --inspect --allow-network-metadata --model Qwen/Qwen3-4B-Instruct-2507 --revision THE_ACTUAL_40_CHARACTER_HF_COMMIT_SHA

Replace the placeholder with a real recorded model SHA from an earlier metadata receipt. The scanner never creates or approves pins automatically.

The explicit request uses the official Hugging Face Python `HfApi.model_info(..., files_metadata=True, token=False)` interface with a 12-second timeout, a fixed public endpoint, and no credential discovery. It does **not** invoke snapshot_download, hf_hub_download, from_pretrained, or any executable model code. Network metadata requests and associated external service logs can still reveal which public model you inspected.

## What it records

- Exact selected model ID and observed repository commit SHA.
- Reported author, model-card license and whether it matches the catalog declaration.
- Reported access gates (private, gated, disabled).
- Architecture names reported in model metadata when available.
- Count of repository files, number of safetensors and potentially pickle-based weight files.
- Sum of known repository-file bytes; full total is explicitly UNKNOWN if any per-file size is missing.
- Human review / license or access hold status, with explicit NO for weights authenticated, GPU fit proven, adapter compatibility proven, answer quality verified, execution authority and routing activation.

The file total includes **all listed repository files and potential alternative weight formats**, so it must NOT be mistaken for the size of one desired quantized checkpoint, the cost of one filtered download, required VRAM, or a training memory forecast.

The tool reports only bounded summary fields. No filenames, model-card body text, prompts, API keys or user files are emitted. Metadata is upstream-provided and is not independently authenticated.

## Security and policy boundaries

- The only valid model IDs are already present in the PhiStream Model Atlas candidate catalog; max 8 unique IDs per invocation.
- Exact repo ID and resolved 40-character commit SHA must agree; catalog revision pins and explicit pins are enforced.
- Missing or conflicting license/access metadata is not an approval; human review remains mandatory.
- Symlinks, malformed or oversized local catalog JSON and missing candidate IDs fail closed using existing R0.1 validation.
- No user token is passed. Hub's model card and security scan are evidence inputs, not permission grants.
- PhiStream does not run new routing, train weights, authorize downloads, or connect to GA108/Crane Fly here.
- A successful test or metadata scan does not imply that the model will run on the RTX 5070 or fit the machine.

## How this fits the actual Phi369 stack

**Vessie GA108:** chooses candidate specialist perspectives within existing human-governed policy.

**PhiStream Model Atlas:** proposes available model families and compares research metadata, never a production route.

**Phi Model Forge:** later runs explicit, reviewed dataset QLoRA training on physical hardware.

**PhiServe or Ollama:** distinct inference providers to be invoked only after separate approval.

**ΦPocketRotor:** observes native MoE token-level expert selections; this is not GA108 profile selection or LoRA fusion.

## Acceptance

    python -m unittest tests.test_hf_model_scout -v

CI validates only synthetic data and a mocked Hub API client. The external Hugging Face request is never performed in automated tests. No CI job should silently download 4B+ weights.

## Next evidence gate (R0.3)

After operator-reviewed metadata receipts and license checks, select the existing 4B Forge baseline. Run Forge's curation, token-audit, CUDA preflight, one explicitly approved QLoRA pilot, receipt integrity verification, and genuinely blinded held-out comparison. Maintain originals and rollback. Never enable production use because an adapter file exists.

Official Hugging Face API documentation:

- https://huggingface.co/docs/huggingface_hub/main/package_reference/hf_api
- https://huggingface.co/docs/huggingface_hub/main/guides/download
- https://huggingface.co/docs/hub/model-cards

**Ledger Above Ego: metadata receipts are not trained weights, authenticated truth, benchmark evidence or authority.**
