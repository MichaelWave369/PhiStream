# Φ CLAVIS Φ369 R0.1 — Cognitive Key Route Planner

**State: research-only, additive, disabled for production inference.** CLAVIS is an
offline categorical route **proposal** for PhiStream. It does not invoke LLMs,
select the actual runtime model, load or mix adapters, download weights, train,
deploy, call tools, modify GA108/Vessie's router, or persist chat content.

The design takes visual inspiration from an old symbolic key engraving.
Historical symbolism is an artistic metaphor, not evidence of hidden machine
learning engineering. CLAVIS also does **not** impersonate Tesla, da Vinci,
Feynman, Escher, Dalí, Picasso, Bach, Beethoven or Vivaldi. The names are labels
for already documented *modern methodological profiles* in Nine Mind.

## Current interfaces

Run a pure planning example (only supplied categorical labels are output):

```sh
python -m phistream.clavis --mode engineering --need electrical --need physics --need prototype
# equivalent after installing the repo:
phi-clavis --mode creative --need recursion --need symbolism --need perspective
```

Optional `--max-specialists 1..4`, `--token-budget 128..32768`, and repeated
`--allow-profile` restrict an operator's proposed plan. No raw prompts, source
documents, credentials or user history are required or accepted by the CLI.

From Python:

```python
from phistream.clavis import plan, WorkingMemory
proposal = plan("engineering", ["electrical", "physics", "prototype"])

memory = WorkingMemory()
observation_id = memory.record(
    "observation", "Prototype current measurement: 0.8 A",
    source="local measurement note", confidence=0.8,
)
evidence_id = memory.record(
    "evidence", "Independent instrument log confirms 0.8 A",
    source="instrument-log-id", confidence=0.9,
)
decision_id = memory.record(
    "decision", "Review current budget before changing the circuit",
    source="operator-review", confidence=0.7, evidence_refs=(evidence_id,),
)
safe_metadata_only = memory.snapshot()  # never exports raw text or source detail
```

## The seven chambers

1. Perception: classify a declared task.
2. Memory: retrieve approved, relevant context (future adapter; not implemented).
3. Reason: structured explanations and testable claims.
4. Imagination: alternative possibilities, not asserted facts.
5. Synthesis: combine specialist perspectives.
6. Verification: required in **every** mode; no bypass.
7. Reflection: review provenance and what the evidence actually says.

R0.1 returns names of required functions, not seven independent agents.
There is no hidden reasoning loop, inter-agent message bus, or automatic model
inference here.

## Nine Mind collaboration rather than nine new weights

CLAVIS uses the existing Nine Mind R0.6 method identities and the R0.1 Model
Atlas catalog as its reference. Static, transparent capability overlaps
prioritize up to four specialist methods. Greedy incremental coverage gives
each selected profile a listed contribution. Unknown or unsupported requested
capabilities are rejected; uncovered needs yield
`HELD_UNCOVERED_CAPABILITIES`.

The score is **not** model quality. R0.1 does not measure latency, token
throughput, GPU occupancy, tool risks, accuracy or calibration. Every plan's
evidence object explicitly says so. Creating 7 x 9 possible function/profile
combinations does not instantiate 63 models.

## The 16 working registers

Four bounded slots each:

- observation (`o1..o4`);
- hypothesis (`h1..h4`);
- evidence (`e1..e4`);
- decision (`d1..d4`).

The memory is an in-process object only: it does not write to disk, log,
network, or automatically copy data from Vessie. Each entry has a caller-
supplied provenance string and declared 0..1 confidence **not calibrated**
against outcomes. Decisions require references to existing evidence slots.
When a bank fills, write attempts fail until a new memory instance is created.
The default `snapshot()` exports only slot IDs, unverified numeric confidence,
and evidence references, never raw contents or source strings. The caller
remains responsible for protecting process memory.

## Governance boundary and integration road map

R0.1 is deliberately NOT wired into `phistream.cli` inference commands,
PocketRotor's native expert selection, Vessie, SuperPhiVessel, PhiOS, or
GA108's specialist identities/authorization. It never claims those systems
are already integrated.

1. **R0.1 (this change):** versioned plan schema, deterministic coverage,
   read-only CLI, bounded in-process registers, CPU unit tests.
2. **R0.2 candidate:** integrate only through an explicitly approved upstream
   adapter which validates schema, reviewed identity mapping and privacy.
3. **R0.3 candidate:** use real blind held-out benchmark receipts from Nine
   Mind R0.5/R0.6 to estimate quality, latency and risk by profile. Include
   uncertainty and abstain when no reliable measurements exist.
4. **R0.4 candidate:** guarded inference orchestration, budget enforcement and
   evidence-aware verification. Require independent unit, local hardware,
   security, performance and regression gates before any production switch.

**Authority stays outside the proposal:** all plans declare
`operator_approval_required=true`, `verification_required=true`,
`inference_authorized=false`, `tool_execution_authorized=false`,
`adapter_activation_authorized=false` and `deployment_authorized=false`.
Even a fully covered proposal is NOT permission to execute.

## Testing

```sh
python -m unittest tests.test_clavis -v
python -m phistream.clavis --help
python -m phistream.clavis --mode creative --need music --need rhythm
```

All tests use CPU-only metadata and no weights or network. The CI suite
continues to run the existing inference/parity tests unchanged.

**Ledger above ego.**
