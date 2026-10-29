# SmolLM2-135M · independent RTX 5060 pretrained receipt audit (October 29, 2026)

**Status: LIMITED THIRD-PARTY RECEIPT CONSISTENCY PASS; NOT A CERTIFIED SPEEDUP.**

Hunter's Claude reported that the pretrained SmolLM2-135M-Instruct
four-case SHA-bound matrix and three frozen paired A/B gates passed on
the RTX 5060 under WSL2, after the first approved real-checkpoint run.
Source: 99,936-byte user-supplied archive
`20261029-smollm2-5060-replication-claude.zip`:

```text
SHA-256 5fde6fa6d4eff69ee67f942bdc310b0e3c8e1e5111baaa2bda438e4c41c4008d
```

The archive contains the receipt JSONs, frozen workload manifest,
preflight, independent HF reference traces, memory log, runner scripts
and narrative report, **but no checkpoint weights**. Attribution:
Claude at Hunter's Xi Research Table; PhiStream commit
`e205ad22490ab475848c89c48316526e45e9735f`. The report says
the user and machine owner granted execution/download approval; this
record does not authenticate consent independently.

## Offline independent consistency check (no model or GPU required)

Do **not** upload the original private workstation ZIP into the public
repository. On a machine holding a legitimate copy, run:

```bash
python -m scripts.smollm2_replication_audit \
  --archive ./20261029-smollm2-5060-replication-claude.zip \
  --expected-sha256 5fde6fa6d4eff69ee67f942bdc310b0e3c8e1e5111baaa2bda438e4c41c4008d \
  --contended-case spread_24 \
  --output ./smollm2-5060-independent-review.json
```

The script only uses Python's standard library, parses bounded ZIP
contents in memory and writes a **new, exclusive** JSON receipt. It
rehashes the entire supplied ZIP; validates the frozen prompt cases and
preflight SHA-256 declarations; verifies the manifest, reference,
matrix-review, candidate and paired-run SHA links; checks each step's
greedy/token and sampled-logit parity flags; independently recomputes
four-pair relative wall-clock and first-logit gains; and confirms that
scheduler production promotion remains explicitly off. It never
downloads, loads or executes neural weights and never contacts a GPU.
The report identifies `spread_24` as externally **reported contended**
and makes no claim to have measured GPU interference independently.

## What can be independently checked from the supplied files

- Checkpoint fingerprints *declared by receipts* match the earlier
  5070 `config.json` and `model.safetensors` digests. The original
  model bytes are not available to the auditor and the model publisher
  is not authenticated.
- Four frozen input cases, all 128/128 greedy tokens and 1,147/1,147
  sampled logit probes match the user-declared HF reference. The
  reference has 69 distinct greedy IDs across cases and four distinct
  five-ID prefixes, not just repetitive token 32. This is **sampled
  logit** parity, not full-vocabulary equality.
- The HF reference receipt claims Transformers 5.22.0, torch
  2.11.0+cu128, eager FP32, a separate venv and no PhiStream import.
  These are external environment declarations, not auditor-attested
  process isolation.
- All three candidate receipts claim successful real-model execution.
  Each A/B receipt contains two reference-matching warmups and
  eight reference-matching scored methods, in four balanced pairs,
  with seed 369 and one warmup per method.

| Frozen case | 5060 matched tokens | Sampled logits | Paired faster counts | 5060 descriptive median gain | 5070 previously reported gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| `short_1` | 32/32 | 288/288 | Control only | N/A | N/A |
| `mixed_7` | 32/32 | 285/285 | 4/4 | 14.50% | 15.53% |
| `repeat_12` | 32/32 | 286/286 | 4/4 | 25.32% | 24.77% |
| `spread_24` | 32/32 | 288/288 | 4/4 | 38.64%, **CONTENDED** | 39.68% |

The historical 5070 figures are from the [three-case A/B census]
(THREE_CASE_AB_CENSUS.md), not rechecked here against the original
5070 raw receipts. Apparent agreement between percentages is
**hypothesis-supporting, not a cross-device performance claim**.
Neither independent timing nor statistical significance is
established. Four paired samples and a single session per machine
cannot certify speedups or scheduler readiness.

## Important hardware and experimental caveats

Hunter's Claude reported that `qwen3:8b` reloaded unexpectedly from
other services and had to be unloaded again before continuing. Another
reload reportedly began at 16:35:02 and overlapped the final
`spread_24` paired A/B execution. The `spread_24` wall-clock data
must therefore be labeled *shared GPU / contaminated*; however,
matching token IDs and logits remain useful correctness evidence.

On this WSL2 host the torch CUDA free-memory query and
`nvidia-smi` used-memory reading disagreed. The reported hard
`nvidia-smi` 2,200 MiB minimum before execution groups, and
`--capacity-mib 1600 --reserve-mib 512`, are **guard choices,
not independently verified safe memory fit**. The attached memory log
documents the failed 1,377 MiB initial preflight window and subsequent
6,600+ MiB free windows. Guard outcomes and recovery are reported
by the external runner; checks between cases cannot prevent a model
reload **during** an active trial.

**Proposed next research step:** repeat the paired 5060 and 5070
experiments on another day with logged per-trial load and GPU
exclusivity, report every trial and the distribution, and preserve
the original SINGLE default. Never silently remove the contended
outlier, rewrite the past ledger, or promote a production scheduler
from these results.

**Ledger Above Ego.**
