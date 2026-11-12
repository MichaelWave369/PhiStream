# Φ Neural Tiny Pilot R1 (PhiStream)

**Status: tiny CPU code tested locally; operator RTX 5070 execution is not verified.**

This deliberately separate pilot trains an **original miniature causal Transformer**, using `torch.nn.MultiheadAttention` and a deterministic 256-token UTF-8 byte protocol. The Karpathy repositories informed the learning path, but none of their code or weights are copied or imported. It is an educational model and experiment plumbing, **not a production LLM**.

## Scope

- `phistream/neural_protocol.py`: fixed holdout evaluator, byte tokenizer contract, and independent evaluator fingerprint bound to exact code bytes **and context length**. Every study fixes these; changing one requires a new baseline.
- `phistream/neural_tiny.py`: opt-in and size-limited training; `preflight` is read-only and does not initialize CUDA; `train` requires explicit approval and defaults to CPU. Uses an operator-supplied 200 byte to 1 MiB UTF-8 local file.
- `examples/demo-phi-bytes.txt`: original synthetic repetitive lines for a mechanical pilot only. It is intentionally **not** an intelligence-quality benchmark.
- `tests/test_neural_tiny.py`: offline CPU and integrity tests included in normal `unittest discover`.

The byte-level Transformer is trained using random windows **only from the first 80%** of the supplied bytes. The last 20% is never used for gradient updates; the evaluator measures fixed, evenly spaced causal prediction windows from that held-out partition. Its `val_bpb` is mean cross-entropy divided by ln 2 (bits per byte). Repetitive text has strong train/holdout similarity and is just a smoke fixture.

## Operator preflight (Windows PowerShell)

From an up-to-date local PhiStream checkout using an environment with CUDA-ready PyTorch or CPU PyTorch:

```powershell
python -m phistream.neural_tiny preflight --data .\examples\demo-phi-bytes.txt --context 16
```

This hashes the complete local dataset and evaluator and reports tokenizer protocol, split sizes, and `training_started:false`. It makes no model, GPU, or network call.

## Explicit tiny training, CPU first

```powershell
python -m phistream.neural_tiny train `
  --data .\examples\demo-phi-bytes.txt `
  --out-dir "$HOME\phi-neural-runs\baseline" `
  --approve-training --device cpu --steps 12 --seconds 120 `
  --context 16 --dim 32 --heads 2 --layers 1 --batch 4 `
  --study-id phi-tiny-pilot --run-id baseline
```

For a first **explicit CUDA** attempt on RTX 5070, select `--device cuda` and a fresh `--out-dir "$HOME\phi-neural-runs\cuda-baseline"` and `--run-id cuda-baseline`. **Do not compare CPU and CUDA runs as identical hardware.** Defaults keep the neural network small; the advertised 10,240 MiB limit is a **post-run cap on PyTorch tensor allocations**, not an enforced process-wide VRAM quota. CUDA setup/runtime/driver memory can exceed the PyTorch counter. Runtime stopping is checked between optimizer steps, not a hard process timeout.

A successful training run writes **only** three files in a new external directory:

- `plan.json`: `PHI_NEURAL_PLAN_R1`, with `engine=phi_tiny`, actual data/evaluator/tokenizer digests and a locally observed but non-authenticated hardware descriptor.
- `metrics.log`: `val_bpb`, `training_seconds`, `total_seconds`, `peak_vram_mb`, `num_steps`, `num_params_M` in the same strict-line form accepted by PhiRepoNerve's Neural Lab.
- `run-meta.json`: baseline versus final held-out loss, config, code SHA, and explicitly unverified status. `peak_vram_mb:0` means CPU/no CUDA metric, *not* a measured 0 MiB process footprint.

No raw dataset text, generated text, model weights, usernames, tokens, or local source paths are exported in these three outputs. No checkpoint is saved; this is a **reproducible training-mechanics pilot**, not a finished trained product. The input text remains on the operator's machine. The code refuses to put output inside a Git checkout or overwrite prior runs.

## Handoff to PhiRepoNerve Neural Bridge

After the paired [PhiRepoNerve phi_tiny intake change](https://github.com/MichaelWave369/PhiRepoNerve), a RepoNerve checkout can record and verify this experiment:

```powershell
# Run from a PhiRepoNerve checkout, with paths adjusted to your local checkouts:
python -m phireponerve.neural_lab record `
  --plan "$HOME\phi-neural-runs\baseline\plan.json" `
  --log "$HOME\phi-neural-runs\baseline\metrics.log" `
  --code "..\PhiStream\phistream\neural_tiny.py" `
  --out "$HOME\phi-neural-runs\baseline\verified-receipt.json"
python -m phireponerve.neural_lab verify `
  --receipt "$HOME\phi-neural-runs\baseline\verified-receipt.json" `
  --log "$HOME\phi-neural-runs\baseline\metrics.log" `
  --code "..\PhiStream\phistream\neural_tiny.py"
```

The next stage is a distinct candidate run with the *same* dataset/evaluator/tokenizer, hardware, time cap and optimizer step count. Use RepoNerve's `compare` and `wiki` only after both receipts are recorded and verified.

## Claim limits and safety

- We validated tiny training on CPU in the isolated development environment. **We have not used or inspected your actual RTX 5070.** Do not claim GPU speed, fit, or a production model.
- A SHA digest binds bytes, not execution authenticity; operator-declared metrics can be forged. A truly independent replay still needs an isolated runner and signed provenance.
- `val_bpb` is a teaching metric on arbitrary input bytes, not downstream scientific validity or model usefulness. No auto-PR, auto-merge, autonomous training edit, code execution from a knowledge pack, or model promotion is installed.
- Dataset permissions, confidentiality, and suitability are the operator's responsibility; use only approved local text.

**Next rung:** reviewed candidate design, independent replay with the fixed evaluator, bounded agent permissions, provenance audit, and manual promotion.
