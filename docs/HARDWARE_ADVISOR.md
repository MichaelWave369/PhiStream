# ΦStream Hardware-Aware Advisor (PR #8, experimental)

Two **read-only, opt-in** CLI commands help evaluate available host RAM, CUDA VRAM
and drive capacity before deciding how to stage local Llama model layers.
This milestone is **not** an adaptive GPU scheduler and does not pin any
layers in the real-checkpoint Llama executor. It collects hardware counters
and proposes an explicitly unapproved policy for research.

## Run the snapshot now on Windows 11

From your existing `C:\Users\micro\Downloads\PhiStream` repository:

```powershell
git pull --ff-only
py -m phistream.cli hardware-snapshot --gpu cuda --drive "C:\" --drive "D:\" --output "benchmarks\local\hardware-rtx5070.json"
```

The output records OS **available physical RAM** (Win32 GlobalMemoryStatusEx),
CUDA **free** and total memory (torch.cuda.mem_get_info), OS/logical CPU count,
and free/total storage on **only** the paths you explicitly select.
It does **not** measure SSD speed, GPU performance, PCIe throughput or real
layer-transfer times. With no `--gpu cuda`, it does not initialize/query CUDA.
Requests for CUDA fail if PyTorch cannot access CUDA; there is no CPU fallback.
The exact output path must be new. Nothing is automatically uploaded or shared.

### Note about storage labels

The Windows `C:` and `D:` labels and drive names come from *your* inventory.
The snapshot checks volume capacity at an explicit path, but does not prove
drive speed or that it is a physically separate NVMe namespace. We leave
storage-performance profiling for a separately reviewed, controlled experiment.

## Recommend residency for a local safetensors model

Once you have a **compatible, licensed local Llama safetensors checkpoint**:

```powershell
py -m phistream.cli recommend-residency "D:\PhiStream\Models\your-local-llama" --context-tokens 512 --gpu-reserve-mib 1024 --ram-reserve-mib 4096 --drive "D:\" --output "benchmarks\local\residency-advice.json"
```

Do not run that second command against Ollama/GGUF files: the current model
inspector does not support GGUF or arbitrary transformer architectures.
Only use a local checkpoint you are authorized to use.

The advisor combines the latest sampled RAM/VRAM with the existing metadata
model preflight: FP32 execution layers, estimated KV, output-head chunks and
workspace. It keeps a GPU reserve (default **1 GiB**) and RAM reserve
(default **4 GiB**), and computes a **candidate** static pinned-layer set
for future implementation. The actual Llama backend does NOT implement it.

Statuses and boundaries:

- `REVIEW_REQUIRED`: the *specified* estimate did not exceed the policy.
  This does not prove the model will fit or perform well.
- `BLOCKED_BY_POLICY`: GPU estimated working set, tensor read limit or
  host layer staging failed its selected memory budget.
- `execution_approved: false` and `auto_placement_enabled: false` are always
  recorded. No execution, auto routing or model modification takes place.
- `--strict` exits with code 2 on a blocked recommendation; otherwise the
  receipt is emitted for review even when blocked.
- Hardware readings can change immediately due to Chrome, Windows, Ollama
  and other apps. All recommended placement is provisional.

## Provenance and privacy

`PHISTREAM_HARDWARE_SNAPSHOT_V1` uses
`LOCAL_HARDWARE_COUNTERS_NOT_A_BENCHMARK`. The advisor uses
`PHISTREAM_RESIDENCY_ADVICE_V1` and
`LIVE_MEMORY_COUNTERS_PLUS_METADATA_ESTIMATE_NOT_BENCHMARKED`.

Potential personal filesystem paths are preserved in local receipts only
when you explicitly selected drives. Review JSON before sharing. Receipts
under `benchmarks/local/` are gitignored. No telemetry service is present.

## Remaining development

The actual GPU/RAM/SSD **adaptive** cache requires measured transfer costs,
allocator-pressure observation, engineering a persistent layer-cache manager,
explicit execution approval, replay/parity verification and out-of-memory
fallbacks. The GitHub CPU tests cover simulated counters and checkpoint
metadata, *not* the RTX 5070. We will only close those gates with real
machine receipts and reproducible comparisons.
