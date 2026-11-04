# Φ Model Forge R0.3: local Windows GPU pilot

**Status: local operator procedure only.** No training, CUDA benchmark or model quality result is established by CPU CI.

## Purpose

R0.3 turns the previous research scaffolding into explicitly staged commands for an isolated Windows/RTX experiment. It never auto-approves data, trains, downloads or uploads by default.

## Prepare the machine once

From a PhiStream Git checkout, create a separate Python 3.11 virtual environment. Install a CUDA-enabled PyTorch build chosen from https://pytorch.org/get-started/locally/ and then the optional Hugging Face training dependencies. Do not replace the global PyTorch used by Ollama, ComfyUI or other local tools.

```powershell
py -3.11 -m venv .venv-forge
.\.venv-forge\Scripts\Activate.ps1
# Install official matching PyTorch CUDA wheel first.
python -m pip install "trl[peft]" transformers datasets accelerate bitsandbytes safetensors
python -m pip install -e . --no-deps
```

## One stage at a time

```powershell
.\scripts\phi_forge_windows_pilot.ps1                        # offline dataset/package preflight
.\scripts\phi_forge_windows_pilot.ps1 -Stage cuda           # explicit CUDA observation
.\scripts\phi_forge_windows_pilot.ps1 -Stage curate         # review each example: A/E/Q
.\scripts\phi_forge_windows_pilot.ps1 -Stage train -ApproveTraining
# To authorize fetching a model if not cached:
.\scripts\phi_forge_windows_pilot.ps1 -Stage train -ApproveTraining -AllowModelDownload
.\scripts\phi_forge_windows_pilot.ps1 -Stage eval-base
.\scripts\phi_forge_windows_pilot.ps1 -Stage eval-adapter
.\scripts\phi_forge_windows_pilot.ps1 -Stage blind-review
```

The curator writes only to a **new** private JSONL path outside the checkout, and only after the operator approves or excludes every example. Q aborts without writing. If both splits are not represented, curation fails. The committed dataset stays unapproved. The PowerShell training stage needs -ApproveTraining, and weights are not downloaded without -AllowModelDownload. Existing outputs are not overwritten.

Default pilot settings are 8 steps, a 256-token maximum sequence length, and a separate QLoRA adapter output path in the user's home directory. The 22+6 initial examples only exercise the training mechanics. They cannot establish a useful specialized assistant. A 12 GiB GPU is a candidate, not a fit guarantee: use the actual preflight, monitor VRAM and abort if incompatible.

An actual adapter must be evaluated against the exact same base model, prompts and dataset hash, with human blind scoring before examining the private answer key. Changing model versions between base and adapter runs invalidates a comparison. No model is shipped or registered in PhiServe by this rung.

See the existing R0.1 training runner, R0.2 local preflight and PhiServe adapter-intake documentation. Source code remains MIT; the fixture is CC0-1.0 and the model has its own upstream license.
