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
