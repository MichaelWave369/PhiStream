# Local device validation (Gate A candidate)

ΦStream's GPU validation is **opt-in and local-only**. The automated CI can test
the CPU code, but it **cannot claim an RTX 5070 measurement**. A machine with a
compatible CUDA-enabled PyTorch install must supply its own test result.

## Windows 11 and RTX 5070 / RTX 3060

Start in the checked-out PhiStream repository root with Python 3.10+,
NumPy and an **official CUDA-enabled PyTorch build compatible with your driver**.
Check the PyTorch installer instructions appropriate for your CUDA setup. The
script deliberately **does not install anything**, fetch weights or upload files.

```powershell
python -m pip install -e . --no-deps
.\scripts\validate-rtx.ps1 -Device cuda
```

PowerShell may require you to run the script from a terminal where local
script execution is permitted by your own policy. Do not lower machine-wide
security settings just to run PhiStream.

If you need to isolate a CPU-only test:

```powershell
.\scripts\validate-rtx.ps1 -Device cpu
```

The command **fails** if CUDA is requested but unavailable, rather than
silently running on CPU. For manual use without a PowerShell script:

```sh
python -m phistream.cli doctor --device cuda
python -m phistream.cli init-fixture artifacts/gpu-fixture
python -m phistream.cli verify artifacts/gpu-fixture
python -m phistream.cli validate-device artifacts/gpu-fixture --device cuda --tokens 8 --pin-mib 0.125 --output benchmarks/local/my-local-validation.json
```

`init-fixture` refuses overwriting an existing fixture. Use that step only
once, or choose a fresh path. `validate-device --output` also refuses to
overwrite a prior receipt.

## Receipt contract

The optional local JSON output uses
`schema: "PHISTREAM_DEVICE_VALIDATION_V1"`, with:

- Hardware inventory (GPU name, CUDA capability, available VRAM *before* tests).
- Explicit `device: cuda` or `device: cpu`. No implicit CPU fallback.
- Verified SHA256 tensor manifest for a **deterministic, untrained fixture**.
- Stream/resident 4-token logit parity and 8 greedy-token agreement.
- Three mode runs: stream, pinned-stream, resident; pinned budget separately
  stated in MiB. They are executed sequentially and are not a statistical study.
- Locally measured time, toy tokens/sec, weight-load counts and PyTorch peak
  *allocated* VRAM where available.

The v0.1 timing is generation only, **excluding engine initialization and
full-checkpoint verification**. The allocated-VRAM peak excludes CUDA context,
other processes, allocator reservation and OS-reported total usage. Disk/PCIe
traffic is not directly measured. Mode order may influence caching. These
numbers are **not** benchmarks of 8B/70B/405B checkpoints.

## Live dashboard receipt viewer

Visit https://michaelwave369.github.io/PhiStream/ and open the
**Local CUDA receipt viewer** under recorded benchmarks. Select your own
`benchmarks/local/phistream-cuda-*.json` file. Its data is parsed in your
browser; there is no server API, implicit publishing or telemetry. The viewer
labels files **user-provided**, not independently verified. CPU results and
invalid receipt schemas will not be displayed as CUDA measurements.

## Current state and release boundary

Until an actual RTX 5070/3060 user run is completed, the site must continue
to display **GPU PENDING** by default. The new receipt is local and not
automatically placed under the public `benchmarks/` folder. Publishing it
requires a separate deliberate, reviewable decision. Even a successful
synthetic GPU fixture does not complete real-model Gate B or performance Gate C.
