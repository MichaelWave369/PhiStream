# Φ Nine Mind R0.4 — Original Method Exercises and Held-Out Seed Corpus

**Status: ORIGINAL, UNREVIEWED, RESEARCH-ONLY FIXTURES. No model has been trained, evaluated or promoted by this PR.**

Nine Mind uses nine research *methods* inspired by historical contributions; it does **not** reproduce nine human minds, copy their protected artworks, or impersonate their viewpoints. This R0.4 adds a small, transparent and auditable corpus for a future one-adapter-per-profile experiment within the **existing** Phi Model Forge. No new training backend, Vessie router, GGUF converter, or model endpoint is added.

## All nine methods

| Council | Profile | Operational method |
| --- | --- | --- |
| Invention | Tesla | Electrical assumptions, energy accounting and measured performance |
| Invention | da Vinci | Physical mechanisms, geometric sketches and iterative prototyping |
| Invention | Feynman | Falsifiable hypotheses, measurement error and controlled experiments |
| Art | Escher | Original geometric constraints, tessellation and finite recursion |
| Art | Dalí | Original surreal symbolism and interpretive ambiguity |
| Art | Picasso | Deliberate multiview representation and visual abstraction |
| Music | Bach | Original motifs, interval transformations and independent voices |
| Music | Beethoven | Motivic variation, expressive arc and musical development |
| Music | Vivaldi | Movement contrast, recurring refrain and thematic motion |

The names are conceptual labels, **not evidence of historical persons' real thought processes**. Modern claims about science, geometry, and music must still be checked independently.

## What's included

- `nine_mind/method_seed_r04.jsonl`: 27 original, CC0-designated project-authored fixtures; **two train prompts and one held-out eval prompt per specialist**.
- Each row contains a profile, council, split, unique scenario family, reference response and a two-part rubric for **future human grading**. Evaluation references are not copied into the training split.
- `phistream/nine_mind_seed.py`: offline bounded seed validator and exclusive-create exporter to existing Forge's `phi.model-forge.example.v1` JSONL schema. Export chooses exactly **one specialist** and maps its source, license, topic and prompt/assistant examples to Forge format.
- `tests/test_nine_mind_seed.py`: contract tests for counts, provenance, rights claims, held-out isolation, malformed input, council mapping, unapproved export and overwrite/refusal.

Every seed row is deliberately `seed-unreviewed`. Exports **preserve that state** so Forge training remains blocked until the operator individually curates examples and verifies rights, accuracy, adequacy, privacy and purpose.

## Commands (offline / safe by default)

From a checked-out PhiStream repository after this branch merges:

    python -m phistream.nine_mind_seed
    python -m unittest tests.test_nine_mind_seed -v

To export **one** specialist for private operator review:

    python -m phistream.nine_mind_seed --export-profile tesla --output /path/outside/repository/tesla-r04-unreviewed.jsonl

The destination's parent must already exist and the file must **not** exist. Paths under the checkout are rejected to avoid accidental public corpus publication. No automatic post-export action, auto-review, training, GPU initialization or network call happens.

An export is **not a ready-to-train dataset**. It is a candidate handoff to the established `phi-forge-curate` process; training remains behind existing approvals, real tokenizer-budget checks, physical CUDA verification and the explicit train-stage approval flag. This R0.4 doesn't add a shortcut.

## What R0.4 deliberately does not claim

- Two training rows per profile and a single held-out example are far too small to demonstrate specialist capability or generalization. They are **pipeline mechanics fixtures only**.
- Reference responses and short rubrics are human-review aids, not ground-truth score certificates or permission for automatic LLM-as-judge grading.
- There are no LoRA weights, neural training, Hugging Face model downloads, adaptation or model merges.
- No real source has been ingested from a historical archive; original instructions describe methods rather than copying texts, drawings or songs.
- The nine profile IDs are **not automatically GA108 profile IDs**, and no Vessie runtime, BrainC authority or PocketRotor MoE expert-selection logic changes.

## R0.5: actual research design

1. Expand **rights-cleared original examples** per profile and prepare independent, leakage-audited held-out test families. Keep changing knowledge and contemporary project facts in retrieval, not static training weights.
2. Run one approved Qwen3-4B Forge mechanics pilot only after the physical RTX operator has completed curation, tokenizer audit, CUDA compatibility, disk and RAM review.
3. Record immutable base revision, dataset SHA, adapter SHA, real VRAM and throughput, and blind base-vs-adapter answers.
4. Compare specialist results against the same untouched base and against an ordinary role-prompt baseline. Evaluate correctness, relevance, groundedness, reproducibility, and creative utility with human judgments, not only training loss.
5. Later test **sequential specialist synthesis** and explicit LoRA switching. Only attempt compatible weight-level fusion when separate LoRAs are actually trained on the **same exact** foundation and tokenizer revision.

Keep the canonical Phi Model Forge's approval/evidence gates and Vessie's separate final Executor Authorization. `phi-weight-surgery` is not LoRA training.

**Ledger Above Ego: Nine inspirations are not nine trained models.**
