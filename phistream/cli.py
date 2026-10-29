"""CLI emits locally reproducible JSON benchmark receipts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import gc
import platform
import shutil
import statistics
import subprocess
import time

import torch

from .engine import PhiStreamEngine
from .format import ModelConfig, ModelStore, build_fixture


def _layer_parameter_bytes(store: ModelStore) -> list[int]:
    from .engine import BLOCK_NAMES
    return [sum(store.entries[f"layer.{i}.{n}"]["nbytes"] for n in BLOCK_NAMES)
            for i in range(store.config.n_layers)]


def _pin_report(store: ModelStore, mode: str, pin_budget_mib: float,
                pinned_count: int) -> dict:
    """Explain pin budgets; refuse to disguise zero-layer pinning as an optimization."""
    sizes = _layer_parameter_bytes(store)
    budget_bytes = int(pin_budget_mib * 1024 * 1024)
    smallest = min(sizes) if sizes else None
    warning = None
    if mode == "stream" and budget_bytes > 0 and pinned_count == 0 and smallest:
        warning = (f"PIN_BUDGET_PINS_ZERO_LAYERS: budget {budget_bytes} bytes "
                   f"({pin_budget_mib} MiB) < smallest layer {smallest} bytes; "
                   f"use --pin-mib >= {smallest / (1024 * 1024):.6f} to pin one layer. "
                   "This run is equivalent to plain streaming.")
    return {"pin_budget_bytes": budget_bytes,
            "layer_parameter_bytes_min": smallest,
            "layer_parameter_bytes_max": max(sizes) if sizes else None,
            "min_pin_mib_for_one_layer": round(smallest / (1024 * 1024), 6) if smallest else None,
            "pin_warning": warning}


def _gpu_telemetry(device: torch.device) -> dict | None:
    """Torch plus optional nvidia-smi device readings, untrusted if they disagree.

    The external query is best effort, no shell, fixed command/fields, and its
    output is metadata only. On WSL2 torch and nvidia-smi may have different
    accounting; neither reading is silently substituted for the other.
    """
    if device.type != "cuda":
        return None
    idx = device.index if device.index is not None else torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(idx)
    out = {"gpu_free_bytes": int(free), "gpu_total_bytes": int(total),
           "nvidia_smi": None}
    exe = shutil.which("nvidia-smi")
    if exe:
        fields = ("memory.used", "memory.total", "utilization.gpu", "temperature.gpu",
                  "clocks.sm", "clocks.mem", "power.draw", "pstate")
        try:
            row = subprocess.run(
                [exe, f"--id={idx}", "--query-gpu=" + ",".join(fields),
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=True).stdout.strip()
            values = [v.strip() for v in row.split(",")]
            if len(values) == len(fields):
                out["nvidia_smi"] = dict(zip(fields, values))
        except (OSError, subprocess.SubprocessError):
            pass
    return out


def benchmark(store: ModelStore, mode: str, device: str,
              prompt: list[int], n_tokens: int, pin_budget_mib: float,
              warmups: int = 0) -> dict:
    """Generation-only toy timing; each optional warmup uses a fresh engine.

    Unscored passes prime CPU/GPU runtime and OS caches, but must never
    contribute their loads, positions or elapsed seconds to the receipt.
    Default zero preserves historic results; use --warmups 1 or 2 when
    collecting new comparable warm-condition data.
    """
    if type(warmups) is not int or not 0 <= warmups <= 5:
        raise ValueError("warmups must be an integer from 0 through 5")
    if type(n_tokens) is not int or n_tokens < 1:
        raise ValueError("tokens must be a positive integer")
    if not isinstance(prompt, list) or not prompt or any(
        type(t) is not int for t in prompt
    ):
        raise ValueError("prompt must be nonempty integer token ID list")
    telemetry_device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
    gpu_before = _gpu_telemetry(telemetry_device)
    warm_tokens: list[list[int]] = []
    for _ in range(warmups):
        warming = PhiStreamEngine(
            store, mode=mode, device=device, pin_budget_mib=pin_budget_mib)
        warm_tokens.append(warming.generate(prompt, n_tokens))
        if warming.device.type == "cuda":
            torch.cuda.synchronize(warming.device)
        del warming
        gc.collect()  # outside timed region and before measured engine
    engine = PhiStreamEngine(store, mode=mode, device=device, pin_budget_mib=pin_budget_mib)
    if engine.device.type == "cuda":
        torch.cuda.synchronize(engine.device)
    start = time.perf_counter()
    tokens = engine.generate(prompt, n_tokens)
    if engine.device.type == "cuda":
        torch.cuda.synchronize(engine.device)
    elapsed = time.perf_counter() - start
    if any(ids != tokens for ids in warm_tokens):
        raise ValueError("Warmup and measured greedy token IDs differ")
    c = engine.counters
    gpu_after = _gpu_telemetry(engine.device)
    pin = _pin_report(store, mode, pin_budget_mib, len(engine.pinned))
    return {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": "MEASURED_ON_THIS_MACHINE_SYNTHETIC_UNTRAINED",
        "mode": mode, "device": str(engine.device),
        "warmup_passes_unscored": warmups,
        "timing_scope": "ONE_FRESH_ENGINE_GENERATION_ONLY_AFTER_UNSCORED_WARMUPS",
        "warmup_generation_parity_checked": bool(warmups),
        "torch_version": torch.__version__, "platform": platform.platform(),
        "gpu_name": torch.cuda.get_device_name(engine.device) if engine.device.type == "cuda" else None,
        "config": vars(store.config), "model_weight_bytes": store.bytes_total,
        "pin_budget_mib": pin_budget_mib, "pinned_layer_count": len(engine.pinned),
        "pinned_layer_parameter_bytes": engine.pinned_parameter_bytes,
        **pin,
        "gpu_telemetry_before": gpu_before, "gpu_telemetry_after": gpu_after,
        "prompt_token_ids": prompt, "generated_token_ids": tokens,
        "total_seconds": round(elapsed, 6),
        "generated_tokens_per_second_end_to_end": round(n_tokens / elapsed, 6) if elapsed else None,
        "weight_load_events": c.layer_loads,
        "weight_bytes_loaded_total": c.loaded_parameter_bytes,
        "weight_load_seconds": round(c.load_seconds, 6),
        "peak_cuda_allocated_bytes": c.peak_cuda_allocated_bytes,
        "interpretation": "Toy parity/performance fixture, not a usable language model or large-model benchmark",
    }


def benchmark_repeated(store: ModelStore, mode: str, device: str,
                       prompt: list[int], n_tokens: int, pin_budget_mib: float,
                       warmups: int = 0, repeats: int = 5) -> dict:
    """N separate measured engines in one process, warmups once before trial 1.

    Report all trials and spread, rather than claiming noisy medians are stable.
    The original non-repeated bench format is unchanged when --repeats 1.
    """
    if type(repeats) is not int or not 2 <= repeats <= 50:
        raise ValueError("repeats must be an integer from 2 through 50")
    trials = [benchmark(store, mode, device, prompt, n_tokens, pin_budget_mib,
                        warmups=warmups if i == 0 else 0) for i in range(repeats)]
    if any(t["generated_token_ids"] != trials[0]["generated_token_ids"] for t in trials):
        raise ValueError("Measured trials produced different greedy token IDs")

    def spread(key: str) -> dict:
        xs = [t[key] for t in trials]
        mean = statistics.fmean(xs)
        sd = statistics.stdev(xs)
        return {"median": round(statistics.median(xs), 6), "min": min(xs),
                "max": max(xs), "mean": round(mean, 6), "stdev": round(sd, 6),
                "cv": round(sd / mean, 6) if mean else None, "n": len(xs)}

    return {
        "schema": "phistream.bench.repeated.v1",
        "provenance": "MEASURED_ON_THIS_MACHINE_SYNTHETIC_UNTRAINED",
        "mode": mode, "device": trials[0]["device"], "repeats": repeats,
        "warmup_passes_unscored_before_first_trial": warmups,
        "timing_scope": "N_FRESH_ENGINE_GENERATIONS_ONE_PROCESS_WARMUPS_BEFORE_FIRST",
        "pin_warning": trials[0]["pin_warning"],
        "summary": {"generated_tokens_per_second_end_to_end":
                        spread("generated_tokens_per_second_end_to_end"),
                    "total_seconds": spread("total_seconds"),
                    "weight_load_seconds": spread("weight_load_seconds")},
        "trials": trials,
        "interpretation": "Report trial spread and external GPU activity; synthetic throughput only",
    }


def main() -> None:
    ap = argparse.ArgumentParser(prog="phistream", description="PhiStream original layer-streaming experiment")
    sp = ap.add_subparsers(dest="command", required=True)
    demo = sp.add_parser("init-fixture", help="Create deterministic untrained experimental model")
    demo.add_argument("directory")
    demo.add_argument("--seed", type=int, default=369)
    validate = sp.add_parser("verify", help="Check every tensor's SHA256")
    validate.add_argument("directory")
    bench = sp.add_parser("bench", help="Produce JSON measurement receipt")
    bench.add_argument("directory")
    bench.add_argument("--mode", choices=["stream", "resident"], default="stream")
    bench.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    bench.add_argument("--prompt", default="1,2,3")
    bench.add_argument("--tokens", type=int, default=4)
    bench.add_argument("--pin-mib", type=float, default=0.0)
    bench.add_argument("--warmups", type=int, default=0,
                       help="0-5 fresh-engine unscored warmup passes (default 0 for historical comparability)")
    bench.add_argument("--repeats", type=int, default=1,
                       help="1 = historic receipt; 2-50 = new fresh-engine trials with spread (use >=5)")
    parity = sp.add_parser("parity", help="Verify streamed and resident logits on this model")
    parity.add_argument("directory")
    parity.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    doctor = sp.add_parser("doctor", help="Inspect CPU or CUDA readiness (no downloads)")
    doctor.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    local = sp.add_parser("validate-device", help="Verify checksums, parity and CPU/CUDA toy-fixture receipts")
    local.add_argument("directory")
    local.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    local.add_argument("--tokens", type=int, default=8)
    local.add_argument("--pin-mib", type=float, default=0.125)
    local.add_argument("--output", help="Optional local receipt file; refuses to overwrite")
    inspect = sp.add_parser(
        "inspect-checkpoint",
        help="Inspect local safetensors header/shards and optional Llama shape layout; no execution",
    )
    inspect.add_argument("location", help="Explicit .safetensors file or local checkpoint directory")
    inspect.add_argument("--llama-layout", action="store_true", help="Check canonical Llama tensor names and shapes")
    inspect.add_argument("--hash", action="store_true", help="Stream SHA256 of every tensor (can be very slow)")
    inspect.add_argument("--probe-tensor", metavar="NAME", help="Materialize only one bounded CPU tensor")
    inspect.add_argument("--max-mib", type=int, default=64, help="Maximum tensor probe bytes in MiB (default 64)")
    run = sp.add_parser("run-llama-ids", help="Execute local canonical Llama token IDs; no tokenizer or reference claim")
    run.add_argument("location", help="Explicit local safetensors checkpoint directory")
    run.add_argument("--prompt", required=True, help="Comma-separated token IDs from a separately licensed tokenizer")
    run.add_argument("--tokens", type=int, default=4)
    run.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    run.add_argument("--max-tensor-mib", type=int, default=512)
    run.add_argument("--head-rows", type=int, default=128)
    run.add_argument("--output", help="Write receipt to new local JSON file; refuses overwrite")
    fingerprint = sp.add_parser("fingerprint-checkpoint",
        help="SHA256 all local safetensors shards and config; no uploads")
    fingerprint.add_argument("location")
    fingerprint.add_argument("--output", help="Write new local JSON manifest; refuse overwrite")
    compare = sp.add_parser("compare-reference",
        help="Compare local Llama logits/greedy IDs against a user-provided independent trace")
    compare.add_argument("location")
    compare.add_argument("--trace", required=True, help="Local JSON reference trace")
    compare.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    compare.add_argument("--max-tensor-mib", type=int, default=512)
    compare.add_argument("--head-rows", type=int, default=128)
    compare.add_argument("--output", help="Save new local JSON comparison receipt; refuse overwrite")
    mem = sp.add_parser("plan-memory",
        help="Offline Llama metadata memory policy estimate; never a GPU fit guarantee")
    mem.add_argument("location", help="Explicit local compatible Llama checkpoint")
    mem.add_argument("--capacity-mib", type=float, required=True,
                     help="User-specified memory budget in MiB (scenario or GPU cap)")
    mem.add_argument("--reserve-mib", type=float, default=512.0,
                     help="Additional safety reserve in MiB (default 512)")
    mem.add_argument("--context-tokens", type=int, required=True)
    mem.add_argument("--device", choices=["scenario", "cuda"], default="scenario",
                     help="Scenario never queries GPU; cuda caps by actual free memory")
    mem.add_argument("--max-tensor-mib", type=int, default=512)
    mem.add_argument("--head-rows", type=int, default=128)
    mem.add_argument("--output", help="Write a new local JSON plan; refuse overwrite")
    mem.add_argument("--strict", action="store_true",
                     help="Exit nonzero if policy blocks the requested plan")
    run.add_argument("--capacity-mib", type=float,
                     help="Opt-in memory policy guard before execution (not a fit guarantee)")
    run.add_argument("--reserve-mib", type=float, default=512.0,
                     help="Safety reserve, used only with --capacity-mib")
    hardware = sp.add_parser("hardware-snapshot",
        help="Read-only local RAM, optional CUDA and selected drive free-space counters")
    hardware.add_argument("--gpu", choices=["none", "cuda"], default="none")
    hardware.add_argument("--drive", action="append", default=[],
                          help="Explicit path to measure free/total bytes; repeat up to 8 times")
    hardware.add_argument("--output", help="Write a new local JSON receipt, never overwrite")
    advisor = sp.add_parser("recommend-residency",
        help="Review-only model-layer residency advice from live RAM/CUDA counters")
    advisor.add_argument("location", help="Explicit local compatible Llama safetensors checkpoint")
    advisor.add_argument("--context-tokens", type=int, required=True)
    advisor.add_argument("--gpu-reserve-mib", type=float, default=1024.0)
    advisor.add_argument("--ram-reserve-mib", type=float, default=4096.0)
    advisor.add_argument("--max-tensor-mib", type=int, default=512)
    advisor.add_argument("--head-rows", type=int, default=128)
    advisor.add_argument("--drive", action="append", default=[],
                         help="Optional explicit storage directory for capacity report")
    advisor.add_argument("--output", help="Write new local JSON recommendation, never overwrite")
    advisor.add_argument("--strict", action="store_true",
                         help="Exit 2 for a policy-blocked advisory (never permits execution)")
    transfer = sp.add_parser("transfer-lab",
        help="Bounded scratch SSD/host/CUDA transfer timing; no model files used")
    transfer.add_argument("--workspace", required=True,
                          help="Existing local directory that permits disposable scratch fixture")
    transfer.add_argument("--gpu", choices=["none", "cuda"], default="none")
    transfer.add_argument("--size-mib", type=int, default=16, help="1-256 MiB scratch fixture")
    transfer.add_argument("--repeats", type=int, default=5, help="3-15 repetitions")
    transfer.add_argument("--output", help="Save a new local JSON receipt; refuse overwrite")
    pipeline = sp.add_parser("pipeline-lab",
        help="Paired synthetic file-to-RAM-to-CUDA wall clock (no model files)")
    pipeline.add_argument("--workspace", required=True,
                          help="Existing local directory for exclusive disposable scratch file")
    pipeline.add_argument("--gpu", choices=["none", "cuda"], default="none")
    pipeline.add_argument("--size-mib", type=int, default=16, help="1-128 MiB")
    pipeline.add_argument("--repeats", type=int, default=8, help="4-20 paired trials")
    pipeline.add_argument("--warmups", type=int, default=2, help="1-5 warmup trials per method")
    pipeline.add_argument("--output", help="Write new local JSON receipt; refuse overwrite")
    reuse = sp.add_parser("reuse-lab",
        help="Compare synthetic pageable/new-pinned/reusable-pinned GPU stages across bounded sizes")
    reuse.add_argument("--workspace", required=True, help="Existing scratch-only directory")
    reuse.add_argument("--gpu", choices=["none", "cuda"], default="none")
    reuse.add_argument("--sizes-mib", default="4,16,64",
                       help="1–3 unique sizes, 1–128 MiB each and <=192 MiB total")
    reuse.add_argument("--rounds", type=int, default=6, choices=[6, 12])
    reuse.add_argument("--warmups", type=int, default=2, help="1–4 per method")
    reuse.add_argument("--output", help="Create a new local JSON receipt, refuse overwrite")
    direct = sp.add_parser("direct-read-lab",
        help="Opt-in synthetic direct file.readinto into preallocated host CPU/CUDA tensors")
    direct.add_argument("--workspace", required=True, help="Existing local scratch directory")
    direct.add_argument("--gpu", choices=["none", "cuda"], default="none")
    direct.add_argument("--sizes-mib", default="4,16,64", help="1–3 bounded MiB sizes")
    direct.add_argument("--rounds", type=int, default=6, choices=[6, 12])
    direct.add_argument("--warmups", type=int, default=2, help="1–4 per method")
    direct.add_argument("--output", help="Create local receipt file, never overwrite")
    chunk = sp.add_parser("chunk-lab",
        help="Synthetic pinned single-vs-double buffered CUDA chunk pipeline")
    chunk.add_argument("--workspace", required=True, help="Existing local scratch-only directory")
    chunk.add_argument("--gpu", choices=["none", "cuda"], default="none")
    chunk.add_argument("--chunk-mib", type=int, default=4, help="1,2,4,8 or 16 MiB")
    chunk.add_argument("--chunks", type=int, default=16, help="4–32; fixture <=256 MiB")
    chunk.add_argument("--repeats", type=int, default=6, help="4,6,8 balanced trial rounds")
    chunk.add_argument("--warmups", type=int, default=1, help="1–3 warmups per method")
    chunk.add_argument("--output", help="New local JSON receipt; never overwrite")
    overlap = sp.add_parser("overlap-lab",
        help="Local synthetic host timeline / persistent worker experiment, no model files")
    overlap.add_argument("--workspace", required=True, help="Existing local scratch directory")
    overlap.add_argument("--gpu", choices=["none", "cuda"], default="none")
    overlap.add_argument("--chunk-mib", type=int, default=8, help="1/2/4/8/16 MiB")
    overlap.add_argument("--chunks", type=int, default=8, help="4–32, total <=256 MiB")
    overlap.add_argument("--rounds", type=int, choices=[6, 12], default=6)
    overlap.add_argument("--warmups", type=int, default=1, help="1–3 per method")
    overlap.add_argument("--output", help="Create new JSON receipt, never overwrite")
    guarded = sp.add_parser("guarded-scheduler",
        help="Offline receipt gate and explicit scratch-only CUDA scheduler prototype")
    guarded.add_argument("--receipt-glob", required=True,
                         help="Local nonrecursive glob, preferably overlap-8m8-repeat-*.json")
    guarded.add_argument("--run-synthetic", action="store_true",
                         help="Run bounded CUDA scratch test; otherwise REVIEW ONLY")
    guarded.add_argument("--workspace",
                         help="Existing scratch-only folder, required with --run-synthetic")
    guarded.add_argument("--request-double", action="store_true",
                         help="Request synthetic double; never changes production policy")
    guarded.add_argument("--approve-synthetic-double", action="store_true",
                         help="Explicit operator consent for synthetic CUDA double path")
    guarded.add_argument("--randomize-control-order", action="store_true",
                         help="Only with explicitly approved synthetic double run; 3/3 balanced shuffled control order")
    guarded.add_argument("--output", help="Create new local JSON receipt, never overwrite")
    qualify = sp.add_parser("qualify-scheduler",
        help="Read-only, local qualification of ALL matching guarded synthetic receipts")
    qualify.add_argument("--receipt-glob", required=True,
                         help="Local nonrecursive glob, e.g. guarded-double-*.json")
    qualify.add_argument("--output", help="Exclusive new local JSON report")
    bridge = sp.add_parser("real-model-gate",
        help="Read-only real Llama safetensors preflight; separately approved >=32-step reference parity")
    bridge.add_argument("location", help="Explicit local canonical Llama safetensors directory/file")
    bridge.add_argument("--capacity-mib", type=float, required=True,
                        help="Memory scenario/policy in MiB; no hard fit guarantee")
    bridge.add_argument("--reserve-mib", type=float, default=1024)
    bridge.add_argument("--context-tokens", type=int, default=40,
                        help="Preflight context if no reference trace supplied")
    bridge.add_argument("--device", choices=["scenario", "cpu", "cuda"],
                        default="scenario")
    bridge.add_argument("--max-tensor-mib", type=int, default=512)
    bridge.add_argument("--head-rows", type=int, default=128)
    bridge.add_argument("--source", help="User-declared original model source (not verified)")
    bridge.add_argument("--revision", help="User-declared exact checkpoint revision")
    bridge.add_argument("--license-id", help="User-declared license identifier (not verified)")
    bridge.add_argument("--trace", help="Local user-declared independent reference JSON")
    bridge.add_argument("--hash-checkpoint", action="store_true",
                        help="Explicitly SHA256 entire checkpoint, can take time")
    bridge.add_argument("--execute-parity", action="store_true",
                        help="Request real checkpoint execution only against >=32-step reference")
    bridge.add_argument("--approve-local-execution", action="store_true",
                        help="Separate operator consent required to execute a real model")
    bridge.add_argument("--output", help="Create a new local JSON receipt, never overwrite")
    args = ap.parse_args()
    if args.command == "init-fixture":
        build_fixture(args.directory, ModelConfig(), args.seed)
        print(json.dumps({"status": "created", "path": args.directory,
                          "provenance": "SIMULATED_UNTRAINED_FIXTURE"}, indent=2))
        return
    if args.command == "doctor":
        from .diagnostics import device_information
        print(json.dumps(device_information(args.device), indent=2))
        return
    if args.command == "inspect-checkpoint":
        from .safeweights import LocalCheckpoint
        reader = LocalCheckpoint(args.location)
        receipt = reader.inspect(llama_layout=args.llama_layout, hash_all=args.hash)
        if args.probe_tensor:
            if args.max_mib < 0 or args.max_mib > 4096:
                raise ValueError("--max-mib must be between 0 and 4096")
            t = reader.tensor(args.probe_tensor, max_bytes=args.max_mib * 1024 * 1024)
            receipt["probe"] = {
                "name": args.probe_tensor, "shape": list(t.shape),
                "dtype": str(t.dtype), "device": str(t.device),
                "nbytes": t.numel() * t.element_size(),
                "note": "One tensor materialized on CPU; not inference or model compatibility proof",
            }
        print(json.dumps(receipt, indent=2, allow_nan=False))
        if receipt["status"] == "LAYOUT_MISMATCH":
            raise SystemExit(1)
        return
    if args.command == "run-llama-ids":
        from pathlib import Path
        from .safeweights import LocalCheckpoint
        from .llama import execute_local_tokens
        raw_ids = args.prompt.split(",")
        if not raw_ids or len(raw_ids) > 2048 or any(not x.strip().isdigit() for x in raw_ids):
            raise ValueError("--prompt must be 1-2048 comma-separated nonnegative decimal token IDs")
        prompt = [int(x.strip()) for x in raw_ids]
        receipt = execute_local_tokens(
            LocalCheckpoint(args.location), prompt, args.tokens,
            device=args.device, max_tensor_mib=args.max_tensor_mib,
            head_rows=args.head_rows,
            capacity_mib=args.capacity_mib, reserve_mib=args.reserve_mib,
        )
        payload = json.dumps(receipt, indent=2, allow_nan=False) + "\n"
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as output:
                output.write(payload)
        print(payload, end="")
        return
    if args.command in ("fingerprint-checkpoint", "compare-reference"):
        from pathlib import Path
        from .reference import checkpoint_fingerprint, compare_reference, read_reference_trace
        from .safeweights import LocalCheckpoint
        checkpoint = LocalCheckpoint(args.location)
        if args.command == "fingerprint-checkpoint":
            result = {
                "schema": "PHISTREAM_CHECKPOINT_FINGERPRINT_V1",
                "provenance": "LOCAL_SHA256_COMPUTED_NOT_A_SIGNATURE",
                "checkpoint_fingerprint": checkpoint_fingerprint(checkpoint),
                "note": "Full local bytes hashed, not a proof of model license or identity.",
            }
        else:
            result = compare_reference(
                checkpoint, read_reference_trace(args.trace), device=args.device,
                max_tensor_mib=args.max_tensor_mib, head_rows=args.head_rows,
            )
        payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            dest = Path(args.output)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("x", encoding="utf-8") as out:
                out.write(payload)
        print(payload, end="")
        if args.command == "compare-reference" and result["status"] == "MISMATCH":
            raise SystemExit(1)
        return
    if args.command == "plan-memory":
        from pathlib import Path
        from .safeweights import LocalCheckpoint
        from .memory import memory_preflight
        observed = None
        if args.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
            observed = int(torch.cuda.mem_get_info()[0])
        report = memory_preflight(
            LocalCheckpoint(args.location), context_tokens=args.context_tokens,
            capacity_mib=args.capacity_mib, reserve_mib=args.reserve_mib,
            max_tensor_mib=args.max_tensor_mib, head_rows=args.head_rows,
            observed_free_bytes=observed,
        )
        payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("x", encoding="utf-8") as out:
                out.write(payload)
        print(payload, end="")
        if args.strict and report["status"] == "BLOCKED_BY_POLICY":
            raise SystemExit(2)
        return
    if args.command in ("hardware-snapshot", "recommend-residency"):
        from pathlib import Path
        from .hardware import hardware_snapshot
        if args.command == "hardware-snapshot":
            receipt = hardware_snapshot(gpu=args.gpu, drives=tuple(args.drive))
        else:
            from .safeweights import LocalCheckpoint
            from .advisor import recommend_residency
            observation = hardware_snapshot(gpu="cuda", drives=tuple(args.drive))
            receipt = recommend_residency(
                LocalCheckpoint(args.location),
                observation, context_tokens=args.context_tokens,
                gpu_reserve_mib=args.gpu_reserve_mib,
                ram_reserve_mib=args.ram_reserve_mib,
                max_tensor_mib=args.max_tensor_mib,
                head_rows=args.head_rows,
            )
        content = json.dumps(receipt, indent=2, allow_nan=False) + "\n"
        if args.output:
            target = Path(args.output)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x", encoding="utf-8") as out:
                out.write(content)
        print(content, end="")
        if (args.command == "recommend-residency" and args.strict
                and receipt["status"] == "BLOCKED_BY_POLICY"):
            raise SystemExit(2)
        return
    if args.command == "transfer-lab":
        from pathlib import Path
        from .transfer import run_transfer_lab
        receipt = run_transfer_lab(
            args.workspace, gpu=args.gpu,
            size_mib=args.size_mib, repeats=args.repeats,
        )
        payload = json.dumps(receipt, indent=2, allow_nan=False) + "\n"
        if args.output:
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("x", encoding="utf-8") as out:
                out.write(payload)
        print(payload, end="")
        return
    if args.command == "pipeline-lab":
        from pathlib import Path
        from .pipeline import run_pipeline_lab
        result = run_pipeline_lab(
            args.workspace, gpu=args.gpu, size_mib=args.size_mib,
            repeats=args.repeats, warmups=args.warmups,
        )
        payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as target:
                target.write(payload)
        print(payload, end="")
        return
    if args.command == "reuse-lab":
        from pathlib import Path
        from .reuse import run_reuse_lab
        result = run_reuse_lab(
            args.workspace, gpu=args.gpu,
            sizes_mib=args.sizes_mib,
            rounds=args.rounds, warmups=args.warmups,
        )
        payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            dest = Path(args.output)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("x", encoding="utf-8") as fh:
                fh.write(payload)
        print(payload, end="")
        return
    if args.command == "direct-read-lab":
        from pathlib import Path
        from .direct import run_direct_lab
        result = run_direct_lab(
            args.workspace, gpu=args.gpu, sizes_mib=args.sizes_mib,
            rounds=args.rounds, warmups=args.warmups,
        )
        payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            dest = Path(args.output)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("x", encoding="utf-8") as target:
                target.write(payload)
        print(payload, end="")
        return
    if args.command == "chunk-lab":
        from pathlib import Path
        from .chunklab import run_chunk_lab
        dest = Path(args.output) if args.output else None
        # Fail early on known collisions, while open("x") below still
        # preserves exclusive no-overwrite semantics against races.
        if dest is not None and dest.exists():
            raise FileExistsError(f"Receipt already exists: {dest}")
        result = run_chunk_lab(
            args.workspace, gpu=args.gpu, chunk_mib=args.chunk_mib,
            chunks=args.chunks, repeats=args.repeats, warmups=args.warmups,
        )
        payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if dest is not None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("x", encoding="utf-8") as output:
                output.write(payload)
        print(payload, end="")
        return
    if args.command == "overlap-lab":
        from pathlib import Path
        from .overlap import run_overlap_lab
        dest = Path(args.output) if args.output else None
        if dest is not None and dest.exists():
            raise FileExistsError(f"Receipt already exists: {dest}")
        report = run_overlap_lab(
            args.workspace, gpu=args.gpu, chunk_mib=args.chunk_mib,
            chunks=args.chunks, rounds=args.rounds, warmups=args.warmups,
        )
        payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if dest is not None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("x", encoding="utf-8") as handle:
                handle.write(payload)
        print(payload, end="")
        return
    if args.command == "guarded-scheduler":
        from pathlib import Path
        from .guarded import find_receipts, review_receipts, run_synthetic_guarded
        if args.randomize_control_order and not (args.run_synthetic
                and args.request_double and args.approve_synthetic_double):
            ap.error("--randomize-control-order requires --run-synthetic, --request-double and --approve-synthetic-double")
        if args.request_double and not args.run_synthetic:
            ap.error("--request-double requires --run-synthetic")
        if args.approve_synthetic_double and not args.request_double:
            ap.error("--approve-synthetic-double requires --request-double")
        if args.run_synthetic and not args.workspace:
            ap.error("--run-synthetic requires an existing --workspace")
        destination = Path(args.output) if args.output else None
        if destination is not None and destination.exists():
            raise FileExistsError(f"Receipt already exists: {destination}")
        evidence = review_receipts(find_receipts(args.receipt_glob))
        if args.run_synthetic:
            result = run_synthetic_guarded(
                args.workspace, evidence,
                request_double=args.request_double,
                approve_synthetic_double=args.approve_synthetic_double,
                randomize_control_order=args.randomize_control_order,
            )
            result["gate_summary"] = evidence
        else:
            result = evidence
        payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if destination is not None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("x", encoding="utf-8") as output:
                output.write(payload)
        print(payload, end="")
        return
    if args.command == "qualify-scheduler":
        from pathlib import Path
        from .qualification import find_guarded_receipts, qualify_guarded
        dest = Path(args.output) if args.output else None
        if dest is not None and dest.exists():
            raise FileExistsError(f"Receipt already exists: {dest}")
        report = qualify_guarded(find_guarded_receipts(args.receipt_glob))
        payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if dest is not None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("x", encoding="utf-8") as handle:
                handle.write(payload)
        print(payload, end="")
        return
    if args.command == "real-model-gate":
        from pathlib import Path
        from .realbridge import evaluate_real_checkpoint
        dest = Path(args.output) if args.output else None
        if dest is not None and dest.exists():
            raise FileExistsError(f"Receipt already exists: {dest}")
        report = evaluate_real_checkpoint(
            args.location, capacity_mib=args.capacity_mib,
            reserve_mib=args.reserve_mib, context_tokens=args.context_tokens,
            device=args.device, max_tensor_mib=args.max_tensor_mib,
            head_rows=args.head_rows, source=args.source,
            revision=args.revision, license_id=args.license_id,
            trace_path=args.trace, hash_checkpoint=args.hash_checkpoint,
            execute_parity=args.execute_parity,
            approve_local_execution=args.approve_local_execution,
        )
        payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if dest is not None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("x", encoding="utf-8") as output:
                output.write(payload)
        print(payload, end="")
        if report["status"] in ("PARITY_MISMATCH", "HELD_MEMORY_POLICY"):
            raise SystemExit(2)
        return
    store = ModelStore(args.directory)
    if args.command == "validate-device":
        from .diagnostics import validate_device
        from pathlib import Path
        receipt = validate_device(store, args.device, pin_budget_mib=args.pin_mib, tokens=args.tokens)
        data = json.dumps(receipt, indent=2, allow_nan=False) + "\n"
        if args.output:
            target = Path(args.output)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x", encoding="utf-8") as output:
                output.write(data)
        print(data, end="")
        if receipt["status"] != "PASS":
            raise SystemExit(1)
        return
    if args.command == "verify":
        store.verify()
        print(json.dumps({"status": "verified", "tensor_count": len(store.entries)}, indent=2))
    elif args.command == "bench":
        prompt = [int(n.strip()) for n in args.prompt.split(",")]
        if args.repeats == 1:
            receipt = benchmark(store, args.mode, args.device, prompt, args.tokens, args.pin_mib, args.warmups)
        else:
            receipt = benchmark_repeated(store, args.mode, args.device, prompt, args.tokens,
                                         args.pin_mib, args.warmups, args.repeats)
        if receipt["pin_warning"]:
            import sys
            print("WARNING: " + receipt["pin_warning"], file=sys.stderr)
        print(json.dumps(receipt, indent=2))
    elif args.command == "parity":
        a = PhiStreamEngine(store, mode="stream", device=args.device)
        b = PhiStreamEngine(store, mode="resident", device=args.device)
        max_delta = 0.0
        for t in [1, 2, 3, 4]:
            l, r = a.step(t), b.step(t)
            max_delta = max(max_delta, (l-r).abs().max().item())
        passed = max_delta <= 1e-5
        print(json.dumps({"pass": passed, "max_abs_logit_delta": max_delta,
                          "context_tokens_checked": 4,
                          "provenance": "MEASURED_PARITY_SYNTHETIC_UNTRAINED"}, indent=2))
        if not passed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
