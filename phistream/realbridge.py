"""Phase-1 governed bridge from locally held Llama safetensors to reference parity.

No discovery, downloads, tokenizer, checkpoint edits, production scheduler,
automatic prefetch, or claims of reference independence. Metadata review is
the default. CUDA/model execution requires separate explicit consent.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
import platform
import time

from .llama import LlamaSettings
from .memory import memory_preflight
from .reference import (
    checkpoint_fingerprint, compare_reference, read_reference_trace,
    validate_reference_trace,
)
from .safeweights import LocalCheckpoint

SCHEMA = "PHISTREAM_REAL_MODEL_BRIDGE_V1"
PROVENANCE = "LOCAL_USER_DECLARED_CHECKPOINT_AND_REFERENCE_UNAUTHENTICATED"
MAX_NAME = 256


def _declaration(value: str | None, field: str) -> str | None:
    if value is None:
        return None
    if (not isinstance(value, str) or not 1 <= len(value.strip()) <= MAX_NAME
            or "\x00" in value or "\n" in value or "\r" in value):
        raise ValueError(f"{field} must be a 1–256 character single-line declaration")
    return value.strip()


def evaluate_real_checkpoint(
    location: str | Path, *, capacity_mib: float, reserve_mib: float = 1024.0,
    context_tokens: int = 40, device: str = "scenario",
    max_tensor_mib: int = 512, head_rows: int = 128,
    source: str | None = None, revision: str | None = None,
    license_id: str | None = None, trace_path: str | Path | None = None,
    hash_checkpoint: bool = False, execute_parity: bool = False,
    approve_local_execution: bool = False,
) -> dict:
    """Preflight always; opt-in executes only prevalidated >=32 reference steps.

    A reported matching user-provided trace never authenticates independence.
    The conservative single-buffer model executor is never reconfigured.
    """
    if device not in ("scenario", "cpu", "cuda"):
        raise ValueError("device must be scenario, cpu or cuda")
    if (type(hash_checkpoint) is not bool or type(execute_parity) is not bool
            or type(approve_local_execution) is not bool):
        raise ValueError("Execution/hash controls must be explicit boolean values")
    if approve_local_execution and not execute_parity:
        raise ValueError("Approval without --execute-parity is invalid")
    if execute_parity and (not approve_local_execution or not trace_path or device == "scenario"):
        raise ValueError("Execution requires --approve-local-execution, --trace and cpu/cuda device")
    if type(context_tokens) is not int or not 1 <= context_tokens <= 8192:
        raise ValueError("context-tokens must be from 1 to 8192")
    source = _declaration(source, "source")
    revision = _declaration(revision, "revision")
    license_id = _declaration(license_id, "license-id")
    source_ready = all((source, revision, license_id))
    if execute_parity and not source_ready:
        raise ValueError("Execution requires declared --source, --revision and --license-id")
    if trace_path is not None and not isinstance(trace_path, (str, Path)):
        raise ValueError("Reference path must be a local filename")
    # Refuse symlink-selected paths, including the final directory or file.
    chosen = Path(location)
    if chosen.is_symlink():
        raise ValueError("Symlink checkpoint path is not allowed")
    ckpt = LocalCheckpoint(chosen)
    # Validate execution-specific architecture restrictions, not only shapes.
    settings = LlamaSettings.from_checkpoint(ckpt)

    trace = None
    validated = None
    if trace_path is not None:
        trace = read_reference_trace(trace_path)
        validated = validate_reference_trace(trace, ckpt)
    if execute_parity and len(validated["expected_ids"]) < 32:
        raise ValueError("Real-model parity needs at least 32 independent-reference token steps")
    required_context = (
        len(validated["prompt"]) + len(validated["expected_ids"]) - 1
        if validated is not None else context_tokens
    )
    if required_context > settings.max_context:
        raise ValueError("Prompt and reference token count exceeds model max context")
    # Unlike synthetic chunk benchmarks, this is a metadata-based policy
    # and cannot guarantee a hard RAM/VRAM ceiling.
    observed_free = None
    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA unavailable; never falling back to CPU")
        observed_free = int(torch.cuda.mem_get_info()[0])
    plan = memory_preflight(
        ckpt, context_tokens=required_context, capacity_mib=capacity_mib,
        reserve_mib=reserve_mib, max_tensor_mib=max_tensor_mib,
        head_rows=head_rows, observed_free_bytes=observed_free,
    )
    blocked = plan["status"] == "BLOCKED_BY_POLICY"
    report = {
        "schema": SCHEMA, "provenance": PROVENANCE,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "device_requested": device,
        "status": "HELD_MEMORY_POLICY" if blocked else "PREFLIGHT_REVIEW_REQUIRED",
        "scope": "EXPLICIT_LOCAL_CANONICAL_LLAMA_SAFETENSORS_ONLY",
        "checkpoint": {
            "format": "SAFETENSORS", "model_type": "llama",
            "tensor_count": len(ckpt.entries),
            "weight_file_count": len(ckpt.files),
            "weight_bytes": sum(t.nbytes for t in ckpt.entries.values()),
            "dtype_set": sorted({t.dtype for t in ckpt.entries.values()}),
            "layers": settings.layers,
            # Do not publish selected absolute model path by default.
            "path_withheld": True,
        },
        "user_asserted_source": {
            "source": source, "revision": revision,
            "license_id": license_id, "declared": bool(source_ready),
            "verified_by_phistream": False,
        },
        "reference_trace": {
            "supplied": trace is not None,
            "steps": len(validated["expected_ids"]) if validated else None,
            "declared_runtime": validated["origin"]["runtime"] if validated else None,
            "asserted_independent": (
                validated["origin"]["asserted_independent"] if validated else None
            ),
            "independence_authenticated": False,
        },
        "memory_preflight": plan,
        "checkpoint_fingerprint": None,
        "execution": None,
        "real_model_run_authorized": execute_parity and approve_local_execution,
        "scheduler_policy_applied": False, "double_buffer_enabled": False,
        "model_execution": False,
        "limits": [
            "Local user-provided model only; no automatic discovery or downloads.",
            "User-declared license, source, revision, and reference independence are NOT authenticated.",
            "Memory plan estimates FP32 per-layer execution and is NOT a guaranteed VRAM or RAM cap.",
            "No tokenizer, EOS, GGUF, quantized architecture, custom RoPE scaling, or production model service.",
            "Reference checks compare teacher-forced greedy IDs and sparse sampled logits.",
            "Real model speedup vs Ollama/AirLLM and GPU/SSD physical overlap have NOT been demonstrated.",
            "Single-buffer streaming only; synthetic scheduler qualification cannot change this executor.",
        ],
    }

    # Optional full SHA256 check is local read-only but may be expensive.
    if hash_checkpoint and not blocked:
        report["checkpoint_fingerprint"] = checkpoint_fingerprint(ckpt)
    if not execute_parity:
        if not blocked:
            report["status"] = (
                "PREFLIGHT_TRACE_REVIEW_REQUIRED" if validated is not None
                else "PREFLIGHT_REVIEW_REQUIRED"
            )
        return report
    if blocked:
        report["status"] = "HELD_MEMORY_POLICY"
        return report

    # Existing comparator checks every shard SHA256 against the reference
    # trace before decoding. Do not execute a nonmatching model.
    start = time.perf_counter()
    result = compare_reference(
        ckpt, trace, device=device, max_tensor_mib=max_tensor_mib,
        head_rows=head_rows,
    )
    elapsed = time.perf_counter() - start
    report["checkpoint_fingerprint"] = result["checkpoint_fingerprint"]
    report["execution"] = result
    report["comparison_command_wall_seconds_including_full_hash"] = round(elapsed, 6)
    report["model_execution"] = True
    if result["status"] == "MATCH_USER_REFERENCE_UNVERIFIED":
        report["status"] = "PARITY_MATCH_USER_REFERENCE_UNAUTHENTICATED"
    elif result["status"] == "MISMATCH":
        report["status"] = "PARITY_MISMATCH"
    else:
        report["status"] = "HELD_REFERENCE_LENGTH"
    return report
