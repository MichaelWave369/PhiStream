# Φ Model Forge R0.4 · token-budget safety gate for first QLoRA experiment

**Status:** code and CPU fixture tests, not physical GPU training or validated Qwen fine-tuning.

## Problem

Our Windows R0.3 pilot deliberately defaults to `--max-length 256` to reduce
VRAM pressure. Hugging Face TRL's `SFTTrainer` truncates data longer than
`max_length`. An accidental truncation can remove part or all of the assistant
answer, making a successful training run misleading.

R0.4 audits the actual tokenizer's chat-template token counts **before**
loading any base-model weights. The R0.1 training runner now also invokes
the same gate immediately after loading the tokenizer, before
`AutoModelForCausalLM.from_pretrained`.

## Windows operator stages

In a previously installed, isolated PhiStream Python environment:

```powershell
# First review the starter examples individually (A/E/Q), if not yet done.
.\scripts\phi_forge_windows_pilot.ps1 -Stage curate

# Read-only after loading a LOCAL cached tokenizer, with no CUDA or model weights.
.\scripts\phi_forge_windows_pilot.ps1 -Stage audit-tokens -MaxLength 256

# Only if you approve a tokenizer download, still with no model weights:
.\scripts\phi_forge_windows_pilot.ps1 -Stage audit-tokens -AllowModelDownload -MaxLength 256

# If any example overflows, choose 512 or 1024 and audit again.
.\scripts\phi_forge_windows_pilot.ps1 -Stage audit-tokens -MaxLength 512

# Training is separate, requires explicit approval, and rechecks token budget.
.\scripts\phi_forge_windows_pilot.ps1 -Stage train -ApproveTraining -MaxLength 512
```

Inspect the JSON report's `overflow_example_ids`,
`template_mismatch_ids` and `missing_completion_ids` if the audit
blocks. No automatic truncation override exists. Preserve the train/eval
split; evaluation prompts do not participate in supervised training.

The CLI uses `local_files_only=True` by default, `trust_remote_code=False`,
and never instantiates a GPU model. An explicit `-AllowModelDownload`
allows tokenizer retrieval only during the audit stage. It does not
run the model. Tokenizer results are **user-supplied/local observations**,
not independent model-source authentication.

## Training receipt changes

A successful, separately authorized training run would include the
token-budget summary, `base_commit_reported` and
`tokenizer_commit_reported` when the libraries expose them. Those
are provenance hints, not verified publisher identities. The existing
adapter SHA256, original dataset SHA256, declared revision, and
explicit `production_authorized: false` remain unchanged.

The entire pilot is research-only. The 22 training fixtures are too
small for specialist quality claims. Actual QLoRA compatibility, model
fit on the 12 GB GPU, runtime versions, and base-versus-adapter quality
remain to be checked locally. Model routing into PhiServe is not
activated by this stage.

Reference: [Hugging Face TRL SFT Trainer](https://huggingface.co/docs/trl/sft_trainer)
documents prompt-completion loss masking and sequence truncation.
