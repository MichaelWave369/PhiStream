# Φ Nine Mind R0.5: Offline Two-/Three-Arm Blind Comparison

**STATUS: REVIEW-ONLY HARNESS. This PR does NOT run models, train LoRAs, call Hugging Face, modify PhiServe or Vessie, score answers, or certify specialist improvements.**

We now have original Nine Mind R0.4 fixtures and the existing Model Forge R0.5 two-arm A/B review. This research module extends the blinded comparison to a third possible arm, **without forking the existing trainer or evaluation receipt format**.

## Why three arms?

The central question is whether learning specialist method weights adds value beyond merely asking the unchanged model to use that method.

| Arm | Actual implementation | What it tests |
| --- | --- | --- |
| Base | Existing Model Forge `scripts.phi_model_eval --mode base` | Unmodified model and existing Forge system prompt |
| Prompted | Operator-supplied, separately labeled local output receipt | The same base model prompted to use a specific Nine Mind method, with its recipe SHA |
| Adapter | Existing Model Forge `scripts.phi_model_eval --mode adapter`, **only after a separately approved real training run** | Whether a same-base, same-revision LoRA actually improves upon both alternatives |

No prompted-mode GPU runner is added in this PR. That arm's JSON is operator-supplied, explicitly `OPERATOR_ASSERTED_NOT_ATTESTED`, and cannot be accepted as independent execution proof. Do not synthesize fabricated example answers and label them as actual model observations.

Start with **two arms** while no adapter exists; add a third only once training and adapter-integrity checks have genuinely passed. The input data is read locally only.

## Record format

Reuse Model Forge's `phi.model-forge.holdout-output.v1` receipt format, with shared:

- `model` and **exact pinned 40-character `model_revision`**;
- `dataset_sha256`, all identical held-out prompt IDs, topics, exact prompt text and generated outputs;
- `review_status: HUMAN_REVIEW_REQUIRED`;
- `quality_proven: false`, `deployment_authorized: false`, `network_upload: false`.

The base receipt must have `mode: "base"` and `adapter_sha256: null`. The optional adapter receipt must have `mode: "adapter"` and declare a 64-character lowercase `adapter_sha256`. Note that the new bench checks *declared* hashes only; the existing Forge R0.5 local adapter validator is the proper byte-level check upstream.

The prompted receipt has the same standard fields but `mode: "prompted"`, `adapter_sha256: null`, and the additional fields:

```json
{
  "profile_id": "tesla",
  "recipe_sha256": "64-lowercase-hex-digest-of-actual-method-system-prompt",
  "source_class": "OPERATOR_ASSERTED_NOT_ATTESTED",
  "method_prompt_applied": true
}
```

The actual prompted run must have been performed separately against the **identical model revision and dataset**. This harness does not attest that the claimed prompt was really applied or the actual generator was used. The recipe digest is descriptive unless backed by separately retained private material.

## Use (offline, in a real PhiStream checkout)

First generate actual output receipts on your own machine with the independently governed Forge runner and an operator-selected, locally evaluated prompt method. Preserve them outside the public repo. Create the prompted receipt only after the actual approved local run.

Then call:

```powershell
python -m phistream.nine_mind_bench `
  --dataset "C:\Private\reviewed-texas.jsonl" `
  --profile tesla `
  --base "C:\Private\baseline.json" `
  --prompted "C:\Private\method-prompted.json" `
  --pack "C:\Private\blind-review.json" `
  --key "C:\Private\private-answer-key.json"
```

After a verified actual adapter run, repeat with `--adapter "C:\Private\adapter-holdout.json"` to generate an A/B/C pack.

Run `python -m unittest tests.test_nine_mind_bench -v` for **CPU-only, mocked-receipt contract checks**.

The tool produces:
- a **private blinded review pack** with A/B labels or A/B/C labels randomized per question, task text and answers, manual grading rubric and blank scores;
- a **separate private answer key** with actual arm identities, model/revision, self-declared receipt-content hashes, profile, recipe SHA and optional adapter SHA;
- no automatic winner, measured quality improvement, trained-model status, or execution authorization.

Both destinations must be separate new paths **outside the checkout**, inside existing directories; files are created with owner-only permissions on POSIX. On Windows, validate effective NTFS ACLs and use a private user directory.

The R0.4 public seed is only for **protocol mechanics smoke tests**. It has one held-out row per profile, and the examples are already in a public repository. Treat them as disclosed, unreviewed and far too small for a real model-quality claim. A future genuine comparison requires an operator-reviewed, privately held, independently generated dataset with many scenario families, no training leakage, and controlled evaluation runs.

## Grading protocol

The human reviewer should grade each answer for correctness, relevance, uncertainty, practical usefulness, originality and regressions **before opening the key**. For scientific/math tasks, use independent reference calculations or tests; for visual/music concepts, use human domain review and eventual outputs from actual creative tools. Record failures and uncertainty, not only wins. Separate response quality from runtime VRAM/speed measures, which this tool does not observe.

Equal answer labels are not guaranteed to conceal model identity: outputs can reveal a distinctive style. The blind process reduces bias but cannot claim perfect blinding.

## Required boundaries

- No inference, Internet, model installation, model downloads, training, LoRA merging, native MoE expert selection, GA108 routing, or production promotion.
- Input/output may include private prompts or model text. Do not upload bench outputs or answer keys to GitHub, CI, or hosted LLMs.
- User-supplied receipts and their source hashes are *integrity correlators*, not independent machine/execution attestation.
- All comparisons remain human review, not scientific validation. A biased or fabricated prompted receipt would produce a meaningless comparison.
- Any training must still use existing [Model Forge R0.5](PHI_MODEL_FORGE_R05.md) safety and provenance checks; this bench never changes training authority.
- Any deployment must still pass Vessie's existing P1-C/GA108/Crane Fly eligibility and final executor authorization gates.

**Ledger Above Ego:** Test the specialist *against an ordinary prompt* before claiming the weights learned anything.
