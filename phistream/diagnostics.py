"""Device-scoped synthetic validation with explicit, auditable GPU/CPU receipts.

No downloaded models, remote benchmarking, or implicit CPU fallback. The
existing toy fixture is untrained and these results cannot support LLM claims.
"""
from __future__ import annotations

from datetime import datetime, timezone
import gc
import math
import platform

import torch

from .cli import benchmark
from .engine import PhiStreamEngine
from .format import ModelStore


SCHEMA = "PHISTREAM_DEVICE_VALIDATION_V1"
PROVENANCE = "MEASURED_ON_THIS_MACHINE_SYNTHETIC_UNTRAINED"
ATOL = 3e-5
RTOL = 3e-5


def device_information(device: str) -> dict:
    """Report actual device discovery. CUDA requests are strict and fail closed."""
    if device not in ("cpu", "cuda"):
        raise ValueError("device must explicitly be cpu or cuda")
    cuda_ready = bool(torch.cuda.is_available())
    if device == "cuda" and not cuda_ready:
        raise RuntimeError(
            "CUDA requested but torch.cuda.is_available() is false. "
            "Install a compatible official CUDA-enabled PyTorch build and driver; "
            "no silent CPU fallback is permitted."
        )
    info = {
        "device": device,
        "device_requested": device,
        "cuda_available": cuda_ready,
        "torch_version": str(torch.__version__),
        "torch_cuda_build": str(torch.version.cuda) if torch.version.cuda else None,
        "platform": platform.platform(),
        "gpu_name": None,
        "compute_capability": None,
        "gpu_total_bytes": None,
        "gpu_free_bytes_before": None,
    }
    if device == "cuda":
        idx = torch.cuda.current_device()
        prop = torch.cuda.get_device_properties(idx)
        free, total = torch.cuda.mem_get_info(idx)
        info.update({
            "gpu_name": prop.name,
            "compute_capability": f"{prop.major}.{prop.minor}",
            "gpu_total_bytes": int(total),
            "gpu_free_bytes_before": int(free),
        })
    return info


def validate_device(
    store: ModelStore,
    device: str,
    *,
    pin_budget_mib: float = 0.125,
    tokens: int = 8,
    parity_ids: tuple[int, ...] = (1, 2, 3, 4),
) -> dict:
    """Run checksums, exact-token parity and isolated-mode synthetic benchmarks.

    Benchmarks use generation-only timing from the v0.1 benchmark function,
    excluding engine startup, disk checksum verification and model conversion.
    The outputs describe only this tiny fixture on the selected device.
    """
    if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 1:
        raise ValueError("tokens must be a positive integer")
    if not math.isfinite(pin_budget_mib) or pin_budget_mib < 0:
        raise ValueError("pin_budget_mib must be nonnegative and finite")
    if not parity_ids or any(not isinstance(n, int) or n < 0 or n >= store.config.vocab_size for n in parity_ids):
        raise ValueError("invalid parity token IDs")
    prompt = [1, 2, 3]
    if max(parity_ids) >= store.config.vocab_size or max(prompt) >= store.config.vocab_size:
        raise ValueError("fixture vocabulary too small")
    if max(len(parity_ids), len(prompt) + tokens - 1) > store.config.max_context:
        raise ValueError("fixture context is too short for requested validation")
    hardware = device_information(device)
    # Check that the checkpoint is the declared synthetic seed, not just an
    # arbitrary weight file with matching shape and a freshly signed manifest.
    if getattr(store, 'manifest_provenance', None) != "SIMULATED_UNTRAINED_FIXTURE":
        raise ValueError("device-validation accepts synthetic fixtures only")
    store.verify()
    resident = PhiStreamEngine(store, mode="resident", device=device)
    stream = PhiStreamEngine(store, mode="stream", device=device)
    delta = 0.0
    passing = True
    for token in parity_ids:
        left, right = stream.step(token), resident.step(token)
        if not torch.isfinite(left).all().item() or not torch.isfinite(right).all().item():
            passing = False
            delta = float("inf")
            break
        difference = float((left - right).abs().max().item())
        delta = max(delta, difference)
        passing = passing and bool(torch.allclose(left, right, atol=ATOL, rtol=RTOL))
    del resident, stream
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    runs = {}
    if passing:
        for mode, budget in (("stream", 0.0), ("pinned", pin_budget_mib), ("resident", 0.0)):
            if mode == "pinned":
                result = benchmark(store, "stream", device, prompt, tokens, budget)
            else:
                result = benchmark(store, mode, device, prompt, tokens, budget)
            runs[mode] = result
            if device == "cuda":
                torch.cuda.empty_cache()
        # A mismatch in greedy token IDs is a validation failure, not a caveat.
        passing = (
            runs["stream"]["generated_token_ids"] == runs["resident"]["generated_token_ids"]
            and runs["pinned"]["generated_token_ids"] == runs["resident"]["generated_token_ids"]
        )
    return {
        "schema": SCHEMA,
        "provenance": PROVENANCE,
        "status": "PASS" if passing else "FAIL",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware,
        "fixture": {
            "type": "SYNTHETIC_UNTRAINED",
            "format": "PHISTREAM_V1_F32_LE",
            "verified_sha256": True,
            "weight_bytes": store.bytes_total,
            "tensor_count": len(store.entries),
            "layers": store.config.n_layers,
        },
        "parity": {
            "passed": passing,
            "max_abs_logit_delta": delta if math.isfinite(delta) else None,
            "atol": ATOL, "rtol": RTOL,
            "token_ids_checked": list(parity_ids),
        },
        "runs": runs,
        "caveats": [
            "Untrained toy transformer; no language-model or real-model support claim.",
            "Throughput excludes engine startup and checkpoint verification.",
            "CUDA peak tracks PyTorch allocated tensors, not driver context, "
            "OS GPU usage, allocator reservation or PCIe/disk traffic.",
            "Comparisons are sequential and cache state may vary; not a fair competitor benchmark.",
            "Validation receipt originates on the user's machine; upload or publish only by explicit choice.",
        ],
    }
