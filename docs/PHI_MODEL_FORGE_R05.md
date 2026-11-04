# Φ Model Forge R0.5 · operator field evidence and adapter A/B binding

**Status: code verified on CPU synthetic files only.** No trained ΦCore
checkpoint, CUDA fit proof, hardware benchmark, or model-quality claim.

## Why

R0.4 enforces token budgets, but no user-supplied **physical training run**
has been inspected yet. R0.5 adds one field-report stage that summarizes
what genuinely exists on the operator's machine, and forces the adapter
evaluation to prove it points at exactly the locally trained adapter file.

## On your Windows machine

In a local checked-out PhiStream repository, inside your isolated Forge
Python environment:

```powershell
git pull
.\scripts\phi_forge_windows_pilot.ps1 -Stage field-report
# If it says WAITING_FOR_CURATED_DATA:
.\scripts\phi_forge_windows_pilot.ps1 -Stage curate
# If training dependencies/CUDA are uncertain:
.\scripts\phi_forge_windows_pilot.ps1 -Stage cuda
# Ensure the dataset is reviewed and tokenized:
.\scripts\phi_forge_windows_pilot.ps1 -Stage audit-tokens -MaxLength 512
# Then ONLY if deliberately approving local QLoRA training:
.\scripts\phi_forge_windows_pilot.ps1 -Stage train -ApproveTraining -MaxLength 512
# Then collect a status-only receipt:
.\scripts\phi_forge_windows_pilot.ps1 -Stage field-report
```

Qwen3-4B may require additional memory on a 12 GiB GPU. The CLI
will refuse if CUDA is unavailable or the model is uncached without
explicit `-AllowModelDownload`; Python/package compatibility and
VRAM fit remain unproven until tested on the physical machine.
If the 22+6 curated seed is all you have, train only for **pipeline
mechanics**, not as a quality model.

## Result stages (no automatic progression)

- `WAITING_FOR_CURATED_DATA`: no reviewed local dataset at the supplied path.
- `DATASET_REVIEW_REQUIRED`: the data is not individually approved.
- `WAITING_FOR_LOCAL_TRAINING`: no adapter file and receipt yet.
- `INCOMPLETE_ADAPTER_ARTIFACT`: partial output, investigate before retrying.
- `WAITING_FOR_HELD_OUT_EVAL`: adapter exists and matches its local receipt;
  run explicit base/adapter evaluations.
- `HUMAN_BLIND_REVIEW_REQUIRED`: both matched evaluations exist; use the
  existing blind-review stage and manually grade before opening the answer key.

The field report reads bounded local metadata, a bounded LoRA safetensors
header, and hashes the whole adapter file. It compares dataset and
adapter hashes, training-step counts, tokenizer budget, base-model name,
declared model revision, and eval receipts. **It does not authenticate
publisher provenance or attest execution occurred.** A maliciously
fabricated locally consistent receipt can pass. This is an integrity
check of operator-supplied files, not independent hardware witnessing.

Adapter-mode generation now checks these same bindings **before**
loading the 4-bit base model onto a GPU. The evaluation output carries
the verified adapter SHA256. A base-model result has `adapter_sha256:null`.
The field report rejects mix-and-match results from another dataset,
base model, revision, or different adapter file.

The report intentionally contains no prompts, generated responses,
local file paths, usernames, access tokens, or daemon endpoints.
It may include public model names, dataset/adapter fingerprints and
training-step counts. The output is suitable to share for debugging
after the operator checks it. It does **not** modify PhiServe or grant
production authorization.

## Next genuinely useful gate

Run the pilot on real hardware; preserve the private original logs
for diagnostics, and share only the field-report JSON. Resolve actual
dependency/kernel/OOM problems observed on that machine before adding
more layers to the training pipeline.
