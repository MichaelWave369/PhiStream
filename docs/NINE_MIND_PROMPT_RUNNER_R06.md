# Φ Nine Mind R0.6: Cache-Only Specialist Prompt Runner

**Status:** source-level research implementation with CPU-only synthetic tests. No local RTX 5070 inference has been performed by this pull request. No weights downloaded, models trained, adapters merged, scores fabricated, or Vessie routes changed.

## Why

[The R0.5 blind bench](NINE_MIND_BLIND_BENCH_R05.md) compares an ordinary base model, the **same** base using a specialist-method prompt, and eventually a trained adapter. Before R0.6 the middle arm needed an external operator-supplied JSON receipt. R0.6 provides a narrow, explicitly approved generator that can produce that missing receipt from an already-cached local model on the user's own machine.

This module is **not** a new trainer, inference service, model download manager, GA108 router, autonomous model chooser or proof that an experimental model is better.

## Nine methodological prompt recipes

The runner includes compact, original method instructions matching the nine R0.4 profiles:

- Tesla: electrical assumptions, units, loss budgets and verification
- da Vinci: mechanics, geometry and prototype iteration
- Feynman: clear definitions, uncertainty and falsifiable experiments
- Escher: geometric transformations, tessellation and finite recursion
- Dalí: original surreal symbolism without unsupported psychology
- Picasso: multiple perspectives and abstraction constraints
- Bach: original motifs, imitation and independent lines
- Beethoven: original motivic development, expressive arc and contrast
- Vivaldi: original rhythms, recurring refrains and movement contrast

Each is appended to the **existing** PhiOS Model Forge system instruction, so the base safety/authorization language remains. Each full system recipe is SHA-256 fingerprinted; the method does not claim to simulate the real historical individuals.

## CPU-only default

From a local PhiStream checkout containing R0.6:

```powershell
python -m phistream.nine_mind_prompt_run --profile tesla --dataset "C:\Private\reviewed-tesla.jsonl"
python -m unittest tests.test_nine_mind_prompt_run -v
```

Default mode reads a bounded local Forge JSONL file, checks specialist/topic compatibility, verifies the model catalog and R0.3 pinned baseline, counts eval rows, and prints a private-data-free **plan**. It never imports torch, loads a tokenizer/model, initializes CUDA, accesses the network, or writes an evaluation receipt.

The source model is fixed to `Qwen/Qwen3-4B-Instruct-2507` at the public Hub commit `cdbee75f17c01a7cc42f958dc650907174af0554`. This identifies a declared repo revision, **not** independently authenticated safetensors bytes or proven CUDA fit.

## Deliberately approved physical-machine run

Before attempting real inference:

1. Prepare a local Forge dataset whose **every row is individually human-approved** and whose held-out questions are independent of training. R0.4's public 27-example exercise collection is only unreviewed mechanics data and **will not pass the execution gate**.
2. Review license/use, exact cached model revision, optional PyTorch/CUDA/Transformers/bitsandbytes compatibility, and VRAM availability on the actual machine. There is no silent install or checkpoint download. A 12 GiB GPU is not a memory-fit guarantee.
3. Run only when the operator explicitly agrees to GPU inference and approves creating a private output:

```powershell
python -m phistream.nine_mind_prompt_run `
  --profile tesla `
  --dataset "C:\Private\reviewed-tesla.jsonl" `
  --max-new-tokens 128 `
  --execute --approve-local-inference `
  --output "C:\Private\tesla-method-holdout.json"
```

The generator requires **two opt-in switches**, loads the exact cached pinned model in NF4 4-bit quantization to `cuda:0` only, formats system/user messages using the actual tokenizer chat template, generates deterministically with `do_sample=False`, bounds context to 1024 tokens without truncation, and rejects empty/overlong outputs. No external inference API, CPU fallback, automatic model download, or `trust_remote_code` is used.

The resulting JSON receipt uses the **existing** Model Forge holdout schema (`phi.model-forge.holdout-output.v1`), with:
- `mode: prompted`, exact model and revision, dataset SHA-256 and matched question IDs/text;
- `profile_id`, `recipe_sha256`, `method_prompt_applied: true`, `source_class: OPERATOR_ASSERTED_NOT_ATTESTED`;
- `adapter_sha256: null`, `review_status: HUMAN_REVIEW_REQUIRED`;
- `quality_proven: false`, `deployment_authorized: false`, `network_upload: false`.

The file is **exclusive-create** in an already-existing private directory **outside the repository**; it never overwrites a record, silently publishes a prompt, prints model responses to stdout, or adds a token/secret to GitHub. On Windows, verify effective NTFS ACLs; POSIX creates owner-only files by default. Output can contain private prompts and model responses. Protect it accordingly.

The `method_prompt_applied` field describes the operation this local code attempted and is **not an independent attestation** that a particular GPU executed a specific checkpoint. Model-card identity remains different from actual checkpoint byte verification.

## Compare against the ordinary base

Use the existing Forge `scripts.phi_model_eval --mode base` with the **same** locally cached model, exact pinned revision, approved dataset, and identical decoding token limit. Supply its base receipt alongside the R0.6 prompted receipt to the existing R0.5 blind bench:

```powershell
python -m phistream.nine_mind_bench `
  --dataset "C:\Private\reviewed-tesla.jsonl" --profile tesla `
  --base "C:\Private\base-holdout.json" `
  --prompted "C:\Private\tesla-method-holdout.json" `
  --pack "C:\Private\two-arm-blind-review.json" `
  --key "C:\Private\two-arm-private-key.json"
```

This creates A/B questions in a privately graded pack and keeps the answer key separate. It cannot prove quality on its own. Compare prompt and adapter arms **only after a genuinely approved QLoRA run**, then independently grade larger held-out tasks.

## Known limitations

- Specialist selection is **operator-specified**, not a new GA108 learned router.
- A topic check cannot prove that a user-supplied `style` dataset actually belongs to one of the six artistic/musical profiles. The operator must verify semantic fit and a distinct holdout.
- Existing `scripts.phi_model_eval` handles the base/adapter arms separately; R0.6 only fills the prompted arm, never forges a base receipt or model run.
- CPU tests use injected synthetic responses and **cannot demonstrate real checkpoint load, Windows bitsandbytes compatibility, memory fit, quantization parity, model quality, or performance**.
- Holding a full 4B quantized model on `cuda:0` may fail for insufficient CUDA/VRAM. This is a valid field result and should not trigger a silent remote/CPU fallback.
- A successful local result is an operator-reported self-attested output, not execution attestation, safety certification, or production authorization.

**Ledger above ego. We compare behavior before we talk about smarter weights.**
