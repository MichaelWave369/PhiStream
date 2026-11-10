# Φ Model Forge R0.3 — exact public model revision, no-training handoff

**Status: OFFLINE, PUBLIC-METADATA REVIEW ONLY. No weights downloaded, local model loaded, CUDA training run, trained Nine Mind adapter, production activation, or checkpoint authenticity attestation.**

This is a bridge between PhiStream's already merged Model Scout R0.2 and Model Forge R0.5. It adds an **operator-reviewable exact Hugging Face repository commit**, not an automatic readiness approval.

## Publicly observed source

The public `https://huggingface.co/api/models/Qwen/Qwen3-4B-Instruct-2507` response inspected on **2026-11-09** reported:

- Repository: `Qwen/Qwen3-4B-Instruct-2507`
- SHA: `cdbee75f17c01a7cc42f958dc650907174af0554`
- Model-card license: `apache-2.0`
- Architecture: `Qwen3ForCausalLM`
- Owner/author label: `Qwen`
- Text generation model, 4.0B parameters per model card.

Public Hub metadata may change or be mistaken. This is **not** a SHA-256 digest of downloaded tensors, independent license counsel review, a GPU run, or a verified training result.

The revision is recorded in [forge_baseline_r03.json](../model_atlas/forge_baseline_r03.json). The original [Model Atlas catalog](../model_atlas/candidates.json) remains candidate-only, and no global alias or model qualification is silently pinned or promoted.

## Safe offline command

From a checkout containing this branch/merged code:

    python -m phistream.forge_baseline
    python -m unittest tests.test_forge_baseline -v

This checks the pinned revision, catalog model/license/architecture consistency, and **existing seed dataset review**. It performs no network request, checkpoint load, tokenizer load, PyTorch import, memory probe, or training.

The bundled Forge 22+6 seed is deliberately unapproved and only suitable for exercising the pipeline. Expect `DATASET_CURATION_AND_HUMAN_APPROVAL_REQUIRED` until the user reviews examples individually.

## Optional real Model Scout receipt comparison

On an operator-controlled development environment with the optional `huggingface_hub` dependency:

    python -m phistream.hf_model_scout --inspect --allow-network-metadata --model Qwen/Qwen3-4B-Instruct-2507 --revision cdbee75f17c01a7cc42f958dc650907174af0554

This contacts Hugging Face for metadata only and prints a JSON receipt. No model weights are downloaded. If saved privately into a file, compare that exact R0.2 receipt:

    python -m phistream.forge_baseline --scout-receipt /path/to/private-model-scout-receipt.json

A matching observation remains `MATCHED_DECLARED_METADATA_NOT_ATTESTED`, never a license/weights/training approval. A mismatching revision, architecture, observed license, source status, unauthorized-model claim, or operational-action claim is refused. The receipt is not independently signed, so operators must still verify its provenance.

## Before any actual local Forge pilot

The existing physical-machine checks remain required. The sequence is:

1. Review the model license, use conditions, and pinned source. Check whether the required local Hugging Face files are **already cached**. Do not silently download several gigabytes or implicitly enable executable remote model code.
2. Run the existing `phi_forge_windows_pilot.ps1 -Stage check` and `-Stage field-report`. Curate each private seed example individually with `-Stage curate`. Do not auto-approve the seed.
3. Separately test CUDA with `-Stage cuda` (physical RTX 5070 only), then the *actual cached tokenizer* with `-Stage audit-tokens -Revision cdbee75f17c01a7cc42f958dc650907174af0554 -MaxLength 512`.
4. Only after explicit local approval, model checkpoint availability, reviewed training/eval split, GPU preflight, and observed free VRAM should the operator consider the existing optional QLoRA train stage. The pin reviewer intentionally does not perform that approval or run it.
5. Preserve original model/adapter revisions and SHAs, then compare blinded held-out base/adapter outputs using the existing R0.5 receipts. Quality is **not** established from a green CI job or 16-step mechanics experiment.

Do not run nine adapters simultaneously on a 12 GiB GPU. Start with one baseline adapter to prove the training/evaluation mechanics, then measure sequential specialist switching and only later controlled weight fusion.

## Three independent evidence classes

- **Hugging Face model repo SHA:** commit identity of the repository tree.
- **Safetensors SHA-256:** separately computed byte hash of actual downloaded checkpoint files. Not provided or validated by this R0.3 manifest.
- **LoRA adapter SHA-256:** independently generated and verified against the selected training dataset, base model, revision, and evaluation receipts by the existing Forge R0.5 lane.

Never substitute one for another.

## Integration boundary

Model Forge and the Atlas remain research in PhiStream. SuperPhiVessel GA108/P1-C/Crane Fly hold specialist selection, model eligibility and independent executor authorization. PhiServe GGUF registration and Vessie integration remain separate and require approved qualification and explicit operator action.

**Ledger Above Ego: a pinned model is not yet a trained mind.**
