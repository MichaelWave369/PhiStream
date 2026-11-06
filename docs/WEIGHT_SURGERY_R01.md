# ΦStream Weight Surgery R0.1

Status: experimental offline checkpoint byte-editing laboratory. Local synthetic tests only; no claim of model-quality improvement, autonomous optimizer, or pretrained weight-editing success.

## Scope

This research lane is independent from the existing PhiStream inference engine, Model Forge QLoRA, and PocketRotor work. It supports **single-file safetensors** with **F32, F16 and BF16** tensor scalars, using only the Python standard library. Sharded checkpoints, GGUF/Ollama files, pickle, unsupported tensor formats, live model inference, activation analysis, model downloads, GPU execution, automatic agent decisions and deployment are deliberately out of scope.

The operator selects a checkpoint, tensor, flattened index and replacement scalar. The program never edits the original. It produces a SHA-256-bound plan, new patched checkpoint and explicit receipt; verification and restoration are independent commands.

## Commands

In an existing PhiStream checkout and Python environment:

    python -m phistream.weight_surgery --help
    python -m phistream.weight_surgery inspect "D:\Models\small\model.safetensors"
    python -m phistream.weight_surgery inspect "D:\Models\small\model.safetensors" --tensor model.layers.0.mlp.down_proj.weight --index 0

Draft a single deliberately chosen scalar change. The example numerical value is **not** a recommendation for any particular model:

    python -m phistream.weight_surgery draft "D:\Models\small\model.safetensors" --tensor model.layers.0.mlp.down_proj.weight --index 0 --new-value 0.0001 --output "D:\Trials\ws-001-plan.json"

After inspecting and explicitly approving the plan:

    python -m phistream.weight_surgery apply "D:\Models\small\model.safetensors" --plan "D:\Trials\ws-001-plan.json" --output "D:\Trials\ws-001-patched.safetensors" --receipt "D:\Trials\ws-001-receipt.json"
    python -m phistream.weight_surgery verify "D:\Trials\ws-001-patched.safetensors" --receipt "D:\Trials\ws-001-receipt.json"
    python -m phistream.weight_surgery revert "D:\Trials\ws-001-patched.safetensors" --receipt "D:\Trials\ws-001-receipt.json" --output "D:\Trials\ws-001-restored.safetensors"

The optional installed command entry point is phi-weight-surgery. Keep inputs and output under an explicitly private, excluded local experiment directory. Never commit downloaded models, derived weight files or private receipts.

## Contract and controls

- Plan schema: PHISTREAM_WEIGHT_SURGERY_PLAN_R01. Includes baseline SHA-256, checkpoint byte size and a list of precise tensor indices and expected/replacement little-endian hex bytes. The draft command produces one edit; an operator may manually add distinct reviewed entries up to 256 edits. Application revalidates each entry and refuses duplicate indices.
- Receipt schema: PHISTREAM_WEIGHT_SURGERY_RECEIPT_R01. Includes plan, plan SHA-256, input and output hashes and a LOCAL_UNVERIFIED authority label.
- Apply refuses changed source bytes, unexpected tensor bytes, malformed or overlapping tensor spans, unsupported dtypes, non-finite scalar values, no-op edits, and existing output/receipt files. Source-to-temporary copy is checked against the baseline hash before mutation. Output is published as a new file without clobbering existing paths.
- Verify checks the patched-file hash and every altered tensor entry. Revert reconstructs the original bytes in a different new file, and refuses success unless its full SHA-256 equals the baseline checkpoint SHA-256.
- These receipts are **not signed attestations**. They verify local byte-level consistency, not model identity, execution independence, good reasoning, quality, absence of training-data leakage or accuracy.
- Streaming disk copies avoid loading an entire checkpoint into RAM but require disk space for a complete second checkpoint. The operator is responsible for model licenses and safe checkpoint handling. This is not an appropriate substitute for a separate backup.
- No automatic model downloads, network requests, telemetry or model execution. No modification of PhiServe, model weights on GitHub, or other repositories.

## Verification

    python -m unittest discover -s tests -p test_weight_surgery.py -v

The tests construct tiny local synthetic safetensors fixtures and exercise F32/F16/BF16 edits, exact byte restoration, multi-edit plans, tampered receipts, wrong checkpoint hashes, malformed offsets, forbidden non-finite values, duplicate targets, existing-file protection and CLI inspection. GitHub Actions' existing unittest discovery will pick them up.

## R0.2 research gate (not part of R0.1)

Before claiming a model improvement, pin a trusted checkpoint revision and a separate compatible inference runtime. Freeze a private holdout and identical decoding settings, then compare: unmodified base, direct-edit candidate, decoding-only or prompt fix, and a QLoRA-trained baseline. Require paired seeds and repeat runs, regression thresholds, detailed per-item scores and latency/VRAM measurements. Evaluate a concrete failure (such as repetition) rather than optimizing a public benchmark alone.

PhiStream's native token-ID inference adapter currently covers a narrow Llama-family layout, **not Qwen3.5**. An eventual Qwen reproduction must use a separate pinned, verified reference runtime. A future PhiTriBrain proposal agent, NBGPocketMesh specialist router and GMED validation gate may consume plans and receipts **as untrusted data**, not independent evidence. No cross-repo execution has been authorized.
