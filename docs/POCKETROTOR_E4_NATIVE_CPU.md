# ΦPocketRotor E4 · Real Transformers Classes, Tiny CPU Weights

**Pilot status: Native Hugging Face code integration, synthetic weights.**
This is a concrete step beyond E3's tiny *fake class* fixtures:
E4 instantiates **the actual** Hugging Face Qwen3.5 MoE sparse block,
native top-k router and native expert implementation from pinned
`transformers==5.21.0`, using tiny, randomly initialized CPU tensors.

## What E4 is designed to prove

Can the merged E3 hook attach to genuine installed Qwen3.5 classes,
observe native top-k expert indices, verify those same IDs are consumed
by native experts, and preserve exactly the same native outputs as
the uninstrumented computation?

The smoke test uses a two-layer toy wrapper and genuine native sparse
blocks with 256 experts / top-k 8 but **tiny** hidden size 16 and
expert width 8. For 16 deterministic synthetic activation inputs,
it compares all hooked outputs to unhooked outputs at exact
`atol=0, rtol=0`, verifies 32 layer-by-token gate-to-expert events,
then exports each layer via the existing E2-R bridge and replays E2
cache simulations. Route records are labeled `kind=synthetic`,
because **the inputs and weights are invented**, even though the
classes and operations really come from Hugging Face.

## Online GitHub Actions instead of a local 35B installation

This self-contained, bounded job lives at:

`.github/workflows/pocketrotor-e4-native-cpu.yml`

It runs on a pull request changing the native integration files, or
can be explicitly started through **Actions → PhiPocketRotor E4 Native
Qwen CPU Integration → Run workflow** once on the default branch.
It installs CPU-only PyTorch and **pinned Transformers 5.21.0**
through pip in a fresh GitHub-hosted runner. It never calls
`from_pretrained()`, a tokenizer or an inference provider.
During test execution `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`, disabling Hub retrieval. A 14-day small
JSON artifact contains run metadata only, not user prompts, router
events from a real checkpoint or model weights.

The core execution command is:

~~~sh
python -m scripts.pocketrotor_e4_native_smoke --output /tmp/phipocketrotor-e4-native-cpu.json
~~~

Without `--output`, JSON goes to stdout. With it, the result file
is created exclusively; overwrites are refused. The default
`PhiStream CI` remains lightweight and does not import Transformers.

The workflow is **not** a security sandbox for arbitrary pip packages;
the pinned library dependency is still an externally provided package.
The runner has no secrets and only read repository permission.
For higher-security reproduction, pin hashes for all dependencies
and use an offline artifact/cache verified against those hashes.

## Required successful result (limited scope)

`REAL_LIBRARY_NATIVE_BLOCK_CPU_HOOK_PARITY_SYNTHETIC_WEIGHTS`

Its evidence says:
- the pinned *library implementation* was executed on CPU;
- a native 256-expert / top-8 router generated exactly the events
  consumed by its native expert module;
- the E3 observer did not perturb the outputs in this small test;
- E2-R and E2 accepted those structurally genuine native route IDs;
- no pretrained model, GPU, checkpoint, prompt, or real activation
  corpus was involved.

It does **not** prove that E3 hooks work during real 35B model generation,
under quantization, with optimized kernels, or in Ollama/GGUF.
It does **not** measure speed or support a production inference claim.
An external MoE backend can still choose a different optimized path
that bypasses the Python hooks.

## Next evidence gate

After this pinned native-code smoke passes, a separate operator-controlled
small real *pretrained* model runtime or already-available compatible
local Qwen checkpoint should supply actual routed expert IDs, with
versioned provenance and frozen held-out workloads. A genuine 35B
full-weight Hugging Face model may far exceed 12 GB GPU memory.
Only once those traces exist should the bounded E2 policies be compared
on actual expert selections and then benchmarked for physical transfers.

**Ledger Above Ego.**
