"""Read-only local readiness gate for the optional PhiOS QLoRA training lane.

Never downloads weights, loads models, installs packages or changes GPU clocks.
CUDA device initialization occurs only if the operator uses --probe-cuda.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any, Callable

from .model_forge import DEFAULT_DATA, review

SCHEMA = "phi.model-forge.preflight.v1"
DEPENDENCIES = ("torch", "transformers", "trl", "peft", "datasets",
                "accelerate", "bitsandbytes", "safetensors", "huggingface-hub")
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
# Advisory scheduling gate, not a VRAM requirement or memory-fit guarantee.
FREE_MEMORY_ADVISORY_MIB = 6144


def installed_versions(lookup: Callable[[str], str] = importlib.metadata.version) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in DEPENDENCIES:
        try:
            versions[name] = lookup(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def cuda_observation() -> dict[str, Any]:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"status": "unavailable", "reason": "CUDA unavailable"}
        count = torch.cuda.device_count()
        if count < 1:
            return {"status": "unavailable", "reason": "No CUDA adapters"}
        free, total = torch.cuda.mem_get_info(0)
        return {
            "status": "observed",
            "reported_device": str(torch.cuda.get_device_name(0))[:128],
            "cuda_device_count": min(count, 32),
            "free_mib": int(free // (1024 * 1024)),
            "total_mib": int(total // (1024 * 1024)),
            "bf16_reported": bool(torch.cuda.is_bf16_supported()),
        }
    except (OSError, RuntimeError, ImportError, AssertionError):
        return {"status": "unavailable", "reason": "CUDA initialization failed"}


def inspect(
    dataset: Path = DEFAULT_DATA,
    *,
    probe_cuda: bool = False,
    versions: dict[str, str | None] | None = None,
    gpu_probe: Callable[[], dict[str, Any]] = cuda_observation,
) -> dict[str, Any]:
    dataset_report = review(dataset)
    dependencies = installed_versions() if versions is None else versions
    if set(dependencies) != set(DEPENDENCIES):
        raise ValueError("Dependency inventory does not match expected packages")
    gpu = gpu_probe() if probe_cuda else {"status": "not_probed"}
    missing = [name for name in DEPENDENCIES if not dependencies[name]]
    if not dataset_report["human_review_complete"]:
        reason = "DATASET_NOT_HUMAN_APPROVED"
    elif missing:
        reason = "DEPENDENCIES_MISSING"
    elif not probe_cuda:
        reason = "CUDA_NOT_PROBED"
    elif gpu.get("status") != "observed":
        reason = "CUDA_UNAVAILABLE"
    elif not isinstance(gpu.get("free_mib"), int) or gpu["free_mib"] < FREE_MEMORY_ADVISORY_MIB:
        reason = "INSUFFICIENT_FREE_VRAM_ADVISORY"
    else:
        reason = "OPERATOR_TRAINING_EXPERIMENT_CANDIDATE"
    return {
        "schema": SCHEMA,
        "model_candidate": MODEL,
        "python": ".".join(str(n) for n in sys.version_info[:3]),
        "platform": platform.system(),
        "dataset_sha256": dataset_report["dataset_sha256"],
        "dataset_approval": dataset_report["status"],
        "train_examples": dataset_report["train_examples"],
        "eval_examples": dataset_report["eval_examples"],
        "installed_versions": dependencies,
        "missing_packages": missing,
        "gpu": gpu,
        "free_vram_advisory_mib": FREE_MEMORY_ADVISORY_MIB,
        "status": "REVIEW_REQUIRED",
        "reason_code": reason,
        "training_started": False,
        "model_loaded": False,
        "weights_downloaded": False,
        "adapter_created": False,
        "quality_improvement_verified": False,
        "fit_guarantee": False,
        "deployment_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phi Model Forge: local dataset/package preflight, no training")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--probe-cuda", action="store_true",
                        help="Opt in to local PyTorch CUDA initialization and VRAM snapshot")
    args = parser.parse_args()
    try:
        report = inspect(args.dataset, probe_cuda=args.probe_cuda)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Preflight held: {exc}\n")


if __name__ == "__main__":
    main()
