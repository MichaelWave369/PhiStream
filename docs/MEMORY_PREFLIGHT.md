# Gate C candidate: GPU memory policy preflight

PhiStream now computes a **metadata-only working-set estimate** for one
compatible, local Llama safetensors checkpoint. It does NOT execute a model,
download files, query a GPU by default, or promise the model fits in memory.

The purpose is to reject clearly oversized configurations **before** reading
model tensors or starting inference, and to document why further local
measurements are needed.

## Offline planning on any computer

From the repository root in a Python environment with PhiStream installed:

```powershell
python -m phistream.cli plan-memory "C:\models\local-llama" --capacity-mib 12288 --reserve-mib 1024 --context-tokens 512 --device scenario --output "benchmarks\local\memory-plan.json"
```

This is explicitly a **user-specified scenario**. The 12288 MiB is a
hypothetical policy input, not measured RTX 5070 free VRAM. Because a
checkpoint can already be loaded in another process, *total* VRAM is not the
same as *currently free* VRAM.

The plan reports:
- Binary on-disk checkpoint bytes vs the active decoder's FP32 layer bytes.
- The largest single layer of FP32 weights staged simultaneously.
- Expected persistent FP32 KV cache for a **specified context length**.
- An estimated attention workspace, streaming head chunk and logits buffer.
- User budget, reserve, effective budget and policy refusal reasons.
- An explicit `no_fit_guarantee: true` on all results.

**Status meanings:** `BLOCKED_BY_POLICY` means this selected policy
prohibits the run because its estimated working set exceeds the effective
budget, or a tensor exceeds the configured read cap.
`REVIEW_REQUIRED` means that specific policy check did not block execution.
It does **not** mean "GPU compatible", "safe", or "can run 70B".

A smaller context can reduce KV memory estimates, but actual compute and
allocator overhead remain unmeasured. The estimator does not include CUDA
context, allocation fragmentation, kernel workspaces, Python object overhead,
all temporary host buffers or transfer latency, and cannot guarantee an OOM
will not occur.

## Optional CUDA free-memory cap

On a local machine with a compatible CUDA-enabled PyTorch installation:

```powershell
python -m phistream.cli doctor --device cuda
python -m phistream.cli plan-memory "C:\models\local-llama" --capacity-mib 12288 --reserve-mib 1024 --context-tokens 512 --device cuda --strict
```

With `--device cuda`, the model does **not** run: PhiStream queries current
CUDA free memory and reduces the effective policy budget to the lower of
requested capacity and observed free bytes. CUDA absence is an explicit error,
not a CPU fallback. Querying free VRAM is not hardware benchmark evidence.

`--strict` produces exit code 2 for a `BLOCKED_BY_POLICY` report, but
the JSON report is still emitted. Without `--strict`, planning does not
fail the CLI merely because the policy blocks.

## Optional execution guard

The existing `run-llama-ids` command can opt into preflight before loading
the first execution weight:

```powershell
python -m phistream.cli run-llama-ids "C:\models\local-llama" --prompt "1,2,3" --tokens 8 --device cuda --capacity-mib 12288 --reserve-mib 1024 --output "benchmarks\local\llama-guarded.json"
```

If the metadata-derived policy blocks, it raises a memory policy error **prior
to the decoder's weight loading**. A nonblocked plan is placed in the resulting
execution receipt, labeled as derived from checkpoint metadata. If no
`--capacity-mib` is specified, existing execution behavior is unchanged and
the optional receipt field is null.

The guard is a cooperative preflight policy, **not** a hard kernel allocator
or protection against CUDA OOM. PhiStream still lacks real-model parity,
performance validation, adaptive hot-layer residency, and GPU profiling.

All receipts remain local by default, and local paths in `benchmarks/local/`
are gitignored. Model weights and source identities are not published.

## Scientific boundary and next step

The GitHub CI tests exercise policy arithmetic on deliberately tiny untrained
safetensors fixtures and validate refusing unsafe scenarios. They do **not**
measure a live NVIDIA GPU. Before claiming any throughput, capacity, or
advantage over AirLLM, test the same real licensed model, same context, same
source revision, same hardware, and an independently verified reference.
