"""CLI emits locally reproducible JSON benchmark receipts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import platform
import time

import torch

from .engine import PhiStreamEngine
from .format import ModelConfig, ModelStore, build_fixture


def benchmark(store: ModelStore, mode: str, device: str,
              prompt: list[int], n_tokens: int, pin_budget_mib: float) -> dict:
    engine = PhiStreamEngine(store, mode=mode, device=device, pin_budget_mib=pin_budget_mib)
    start = time.perf_counter()
    tokens = engine.generate(prompt, n_tokens)
    elapsed = time.perf_counter() - start
    c = engine.counters
    return {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": "MEASURED_ON_THIS_MACHINE_SYNTHETIC_UNTRAINED",
        "mode": mode, "device": str(engine.device),
        "torch_version": torch.__version__, "platform": platform.platform(),
        "gpu_name": torch.cuda.get_device_name(engine.device) if engine.device.type == "cuda" else None,
        "config": vars(store.config), "model_weight_bytes": store.bytes_total,
        "pin_budget_mib": pin_budget_mib, "pinned_layer_count": len(engine.pinned),
        "pinned_layer_parameter_bytes": engine.pinned_parameter_bytes,
        "prompt_token_ids": prompt, "generated_token_ids": tokens,
        "total_seconds": round(elapsed, 6),
        "generated_tokens_per_second_end_to_end": round(n_tokens / elapsed, 6) if elapsed else None,
        "weight_load_events": c.layer_loads,
        "weight_bytes_loaded_total": c.loaded_parameter_bytes,
        "weight_load_seconds": round(c.load_seconds, 6),
        "peak_cuda_allocated_bytes": c.peak_cuda_allocated_bytes,
        "interpretation": "Toy parity/performance fixture, not a usable language model or large-model benchmark",
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
        print(json.dumps(benchmark(store, args.mode, args.device, prompt, args.tokens, args.pin_mib), indent=2))
    elif args.command == "parity":
        a = PhiStreamEngine(store, mode="stream", device=args.device)
        b = PhiStreamEngine(store, mode="resident", device=args.device)
        import torch
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
