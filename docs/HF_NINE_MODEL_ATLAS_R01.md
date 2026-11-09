# Φ Model Atlas / Nine Mind R0.1: Hugging Face and adapter research

**Status: REVIEW ONLY. No downloaded weights, trained adapters, real inference, merges, GPU benchmarks, Vessie routing changes or execution permissions.**

PhiStream hosts a local research catalog; SuperPhiVessel retains GA108 identity and governed routing. This is not a new competing router or a replacement for Model Forge R0.5, PhiServe GGUF serving, or the experimental PhiStream Llama streaming engine.

## Topology

    Human intent and scoped permissions
      -> Vessie GA108 identity shortlist, P1-C eligibility and approved model pool
      -> PhiStream Model Atlas advisory catalog / eventual adapter evidence
      -> ONE Vessie BrainC/Crane Fly governed route
      -> mandatory downstream executor authorization
      -> explicitly selected, qualified model runtime
      -> independent verification and receipts

GA108 profiles are stable specialist identities, **not 108 models to load**. Its documented router usually selects one specialist, shortlists up to eight, and requires a separate approved budget for councils of at most four. The Nine Mind profiles are a research grouping (3 invention, 3 visual art, 3 music), not historical brain reconstructions, not GA108 profile IDs, and not nine autonomous executors. A future bridge must map them to real stable GA108 identifiers after review.

## Hugging Face candidate shortlist

| Model | Official license | Proposed use | Current status |
| --- | --- | --- | --- |
| [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | Apache-2.0 | Existing Forge QLoRA pilot baseline | Untrained/unqualified here |
| [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | Apache-2.0 | Multimodal challenger | Architecture review required |
| [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | Apache-2.0 | Larger challenger | GPU fit and training unknown |
| [Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) | MIT | Compact text reasoning challenger | Different architecture, review required |

Hugging Face model cards are not proof of installed weights, model quality, true GPU occupancy, or compatibility. Qwen3.5 is not a drop-in replacement for the existing Qwen3 causal-LM Forge code. In the catalog, revisions are deliberately unpinned and qualification UNVERIFIED.

Before any model download, use explicit metadata review through Hugging Face model_info, pin a specific commit, verify the rights, and inspect a snapshot_download dry run. A model's weight size is not a complete VRAM bound; account for activations, KV cache, workspace, optimizer, runtime, concurrent GPU applications, and context length.

## Four distinct weight operations

1. **Quantization:** compress base checkpoint weights for storage/inference or QLoRA; not learned expertise.
2. **QLoRA:** freeze one selected base and fine-tune an adapter on authorized, reviewed examples. Reuse Phi Model Forge R0.1–R0.5 gates. The 22+6 dataset only tests pipeline mechanics, not specialist quality.
3. **Adapter selection and fusion:** switch LoRA adapters from a common base, or test PEFT add_weighted_adapter (linear, TIES, DARE, etc.). LoRAs from different bases or tokenizer revisions must not be naively fused. Compatible shapes alone do not prove capability or a quality gain. Preserve untouched originals and an independent base/adapter test.
4. **PhiStream weight surgery:** existing reversible, SHA-bound individual safetensors scalar byte edits. This is NOT training, quantization, or LoRA fusion, and must remain isolated.

PhiServe handles registered GGUF models and does not currently serve PEFT adapter directories natively. Any conversion, merged export, registration, or deployment requires a separate permissioned experiment and rollback record.

## Existing R0.1 offline behavior

    python -m phistream.model_atlas
    python -m unittest tests.test_model_atlas -v
    python -m phistream.model_atlas --mix-plan /path/to/operator-proposal.json

The catalog validator checks bounded, strictly shaped local JSON for four Hugging Face candidates and nine research profiles. The optional mix reviewer accepts 1–4 declared adapter records using a single pinned 40-character base revision and matching declared tokenizer hash, target modules, LoRA rank and SHA-256 fingerprints. It requires normalized **advisory routing percentages** totaling 1.0; these are NOT necessarily the correct coefficients for a numerical PEFT merge.

All valid proposed mixes return HELD_UNVERIFIED_REVIEW_ONLY. The tool **does not examine actual model or adapter bytes**, verify declared SHA fingerprints, load tensors, download models, merge weights, execute prompts, access Vessie memory, or activate routes. Symlinks, oversized JSON, duplicate keys, invalid profiles, inconsistent base declarations, and incompatible declared metadata fail closed.

## Next evidence gates

**R0.2: Read-only Hugging Face model-card inspection.** Implement a separately opt-in metadata-only client with exact publisher, license, commit SHA, modality, family, and file-size receipts. Never auto-download or auto-approve a candidate; avoid arbitrary remote code.

**R0.3: User-machine QLoRA mechanics pilot.** One model per RTX 5070 12 GiB GPU session: curated rights-checked dataset -> real tokenizer-budget audit -> explicit CUDA preflight -> human-approved training -> adapter SHA-bound receipts -> blinded base/adapter evaluation. No CI or metadata receipt substitutes for physical measurements.

**R0.4: Nine method datasets.** Train the nine perspectives as *methods*: electrical hypotheses, mechanics, experiment design, geometry, symbolic composition, visual perspectives, counterpoint, thematic development, and rhythmic motion. Use original or explicitly licensed training examples, with separate retrieved source works and genuinely held-out problems. Do not claim to recreate historical people.

**R0.5: Benchmark three strategies.** Compare the base, a selected single adapter, separate specialist responses with synthesis, and compatible weight-level fusion. Measure correctness, originality/usefulness, failures, hallucinations, time-to-first-token, throughput, peak VRAM, and reversibility. Never call a successful merge API operation a proven smarter model.

**R0.6: Integrate under governed Vessie routing.** Add a read-only adapter evidence/status bridge first. Do not fork GA108/P1-C/BrainC/Crane Fly, move authority into a browser, widen permissions, or automatically spend money. Preserve operator approvals, review, rollback, and downstream executor authorization.

## Primary references

- [PhiStream Model Forge R0.1](PHI_MODEL_FORGE_R01.md) and [R0.5](PHI_MODEL_FORGE_R05.md)
- [Vessie routing rebase](https://github.com/MichaelWave369/SuperPhiVessel/blob/main/docs/routing/ROUTING_REBASE_V1.md) and [gateway contract](https://github.com/MichaelWave369/SuperPhiVessel/blob/main/docs/routing/GATEWAY_CONTRACT_V1.md)
- [Hugging Face PEFT](https://huggingface.co/docs/transformers/peft), [merging](https://huggingface.co/docs/peft/main/developer_guides/model_merging), [model_info API](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api), [download dry run](https://huggingface.co/docs/huggingface_hub/main/guides/download), and [model card guidance](https://huggingface.co/docs/hub/model-cards)

**Ledger above ego:** metadata, green unit tests, trained adapters, verified quality, runtime qualification, and execution authority are six different states.


## Reuse ΦPocketRotor before inventing another MoE router

The repository already contains a separate **ΦPocketRotor** research line. This is important because there are **three different routing layers**, which must not be conflated:

- **Vessie GA108**: task-level specialist perspectives and approval-bound cognitive orchestration. One canonical governed task route remains in SuperPhiVessel.
- **PhiStream Model Atlas / Model Forge**: proposed model selection, text-specialist LoRA switching, held-out evaluation and eventual compatible-adapter fusion, subject to real artifact verification and operator approval.
- **ΦPocketRotor**: **read-only observation of a MoE model's own token-level native expert gate**, not a policy router and not a replacement for training LoRA adapters.

Existing source-backed evidence and gates:

- [PocketRotor E3](POCKETROTOR_E3_QWEN35.md) hooks an already-loaded Hugging Face Qwen3.5 MoE router and observes native selected expert IDs (256 routed experts, top-8); it does not load or modify the model.
- [PocketRotor E4](POCKETROTOR_E4_NATIVE_CPU.md) exercises the actual Hugging Face Transformers MoE classes with **tiny synthetic CPU weights**. It cannot demonstrate genuine pretrained checkpoint behavior or RTX performance.
- [PocketRotor E5](POCKETROTOR_E5_PRETRAINED.md) specifies an **explicit manual-only** pretrained IBM Granite MoE test, using a pinned `ibm-granite/granite-3.0-1b-a400m-instruct` checkpoint and its separately checked sha256. The manual workflow is NOT triggered by pushes/PRs/merges. Unless actual run receipts are present, its completion and outputs remain unverified.

These complementary layers suggest a better long-term design: the Atlas proposes *which methods and model artifacts to test*; Vessie authorizes the *task and allowed provider*; native MoE routing remains the trained model's **own internal computational mechanism**. No naming of historical profiles as MoE expert IDs, and no user-defined adapter mixture should be mislabeled as native per-token gating.

The first optional real pretrained routing experiment can use E5's smaller Granite checkpoint, but it is **separate from** the Qwen3-4B Forge QLoRA pilot and must remain individually operator-approved. Neither experiment proves a Nine Mind improvement without held-out outcomes.
