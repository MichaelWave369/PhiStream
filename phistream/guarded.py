"""Receipt-gated, operator-approved *synthetic-only* PhiStream scheduler.

This module neither opens real model checkpoints nor changes any production
scheduler. Imported local evidence is unauthenticated and never promotes
itself to runtime approval. Default single; optional explicit CUDA experiment.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import glob
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import statistics
import tempfile
import time

from .chunklab import _CudaBackend, _fixture, _check_parameters
from .hardware import physical_memory
from .overlap import _trial, _order, recommend
from .reuse import _duration
from .transfer import MIB

SCHEMA = "PHISTREAM_GUARDED_SCHEDULER_V1"
PROVENANCE = "LOCAL_UNAUTHENTICATED_SYNTHETIC_RECEIPT_REVIEW"
RECEIPT_SCHEMA = "PHISTREAM_OVERLAP_LAB_V1"
RECEIPT_PROVENANCE = "MEASURED_LOCAL_SYNTHETIC_HOST_TIMELINE_NO_MODEL"
SHAPE = (8, 8)  # 8 MiB x 8 chunks: only tested evidence cohort
MIN_INDEPENDENT_RUNS = 3
MIN_ROUNDS = 12
MAX_INPUT_FILES = 12
MAX_JSON_BYTES = 512 * 1024
MIN_FREE_VRAM_MARGIN = 512 * MIB


def _finite_positive(value) -> bool:
    return type(value) in (float, int) and math.isfinite(value) and value > 0


def _utc(raw: str) -> datetime:
    if not isinstance(raw, str):
        raise ValueError("Timestamp missing or not a string")
    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timestamp lacks timezone offset")
    return parsed.astimezone(timezone.utc)


def _inspect_receipt(obj: dict) -> dict:
    if not isinstance(obj, dict):
        raise ValueError("JSON must be an object")
    if (obj.get("schema") != RECEIPT_SCHEMA or
        obj.get("provenance") != RECEIPT_PROVENANCE or
        obj.get("status") != "PASS_CUDA_OVERLAP_LAB" or
        obj.get("gpu_requested") != "cuda" or
        obj.get("model_execution") is not False or
        obj.get("automatic_layer_placement") is not False):
        raise ValueError("Wrong schema/provenance, device or execution scope")
    if (type(obj.get("chunk_mib")) is not int or
        type(obj.get("chunks")) is not int or
        (obj["chunk_mib"], obj["chunks"]) != SHAPE):
        raise ValueError("Only the independently tested 8 MiB x 8 shape is eligible")
    design = obj.get("design") or {}
    if (type(design.get("paired_rounds")) is not int or
        design["paired_rounds"] != MIN_ROUNDS or
        design.get("host_buffers_reused_only_after_cuda_completion_event") is not True or
        design.get("same_preallocated_host_device_buffer_budget") is not True or
        design.get("scratch_cache_controlled") is not False or
        design.get("gpu_compute_overlap_tested") is not False or
        design.get("gpu_clock_aligned_with_cpu") is not False):
        raise ValueError("Insufficient or nonmatching experiment controls")
    if design.get("balanced_orders") != _order(MIN_ROUNDS):
        raise ValueError("Missing exact 12-round balanced three-way permutation schedule")
    fixture = obj.get("fixture") or {}
    validation = obj.get("validation") or {}
    if (fixture.get("synthetic") is not True or
        fixture.get("scratch_deleted") is not True or
        type(fixture.get("total_bytes")) is not int or
        fixture["total_bytes"] != 64 * MIB or
        not isinstance(fixture.get("sha256"), str) or
        len(fixture["sha256"]) != 64 or
        validation.get("full_sha256_host_and_gpu_audit") is not True or
        validation.get("audit_is_outside_timing") is not True or
        type(validation.get("audit_chunks")) is not int or
        validation["audit_chunks"] != SHAPE[1]):
        raise ValueError("Incomplete or mismatched scratch/GPU replay integrity receipt")
    hw = obj.get("hardware") or {}
    if not all(isinstance(hw.get(key), str) and hw[key] for key in
               ("gpu_name", "torch_version", "cuda_build")):
        raise ValueError("Missing CUDA/GPU identity")
    if not _finite_positive(hw.get("free_vram_before_bytes")):
        raise ValueError("Missing positive observed free VRAM")
    platform_name = obj.get("platform")
    if not isinstance(platform_name, str) or not platform_name:
        raise ValueError("Missing measured platform")
    started = _utc(obj.get("started_utc"))
    ended = _utc(obj.get("completed_utc"))
    if not started < ended:
        raise ValueError("Invalid receipt interval")
    a = ((obj.get("methods") or {}).get("single") or {}).get("end_to_end") or {}
    b = ((obj.get("methods") or {}).get("double_reuse_worker") or {}).get("end_to_end") or {}
    def samples(series):
        values = series.get("samples_seconds")
        if (not isinstance(values, list) or len(values) != MIN_ROUNDS or
            any(not _finite_positive(x) for x in values)):
            raise ValueError("Incomplete or invalid paired elapsed samples")
        median_reported = series.get("median_seconds")
        if (not _finite_positive(median_reported) or
            abs(statistics.median(values) - median_reported) > 0.000002):
            raise ValueError("Receipt median does not match raw samples")
        return values
    single = samples(a)
    double = samples(b)
    original_policy = obj.get("recommendation") or {}
    recalculated = recommend(single, double, gpu=True)
    for field in ("decision", "default", "double_wins", "minimum_wins",
                  "policy_applied"):
        if original_policy.get(field) != recalculated[field]:
            raise ValueError("Published recommendation disagrees with measured samples")
    if (not _finite_positive(original_policy.get("minimum_relative_gain")) or
        abs(original_policy["minimum_relative_gain"] - 0.05) > 1e-9 or
        not isinstance(original_policy.get("median_paired_relative_gain_double"),
                       (int, float)) or
        not math.isfinite(original_policy["median_paired_relative_gain_double"]) or
        abs(original_policy["median_paired_relative_gain_double"] -
            recalculated["median_paired_relative_gain_double"]) > 0.00001):
        raise ValueError("Receipt reports different policy thresholds or arithmetic")
    return {
        "started": started,
        "ended": ended,
        "key": (platform_name, hw["gpu_name"], hw["torch_version"],
                hw["cuda_build"]),
        "observed_free": hw["free_vram_before_bytes"],
        "decision": recalculated["decision"],
        "wins": recalculated["double_wins"],
        "gain": recalculated["median_paired_relative_gain_double"],
        "single": single, "double": double,
    }


def review_receipts(receipt_files: list[str | Path]) -> dict:
    """Read only, no GPU calls. Gate is deliberately stricter than PR #16."""
    if not isinstance(receipt_files, list) or not 1 <= len(receipt_files) <= MAX_INPUT_FILES:
        raise ValueError("Provide 1–12 receipt paths")
    results, rejected, keys_seen = [], [], set()
    seen_hashes = set()
    for path_value in receipt_files:
        path = Path(path_value)
        name = str(path)
        try:
            if not path.is_file() or path.is_symlink():
                raise ValueError("Missing, non-file or symlink receipt")
            if path.stat().st_size > MAX_JSON_BYTES:
                raise ValueError("Receipt exceeds maximum local JSON size")
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if digest in seen_hashes:
                raise ValueError("Duplicate receipt content")
            seen_hashes.add(digest)
            observation = _inspect_receipt(json.loads(raw))
            start_key = observation["started"].isoformat()
            if start_key in keys_seen:
                raise ValueError("Duplicate experiment start time")
            keys_seen.add(start_key)
            results.append(observation)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError,
                TypeError, OverflowError) as exc:
            rejected.append({"name":path.name, "reason":str(exc)[:200]})
    guards = []
    if rejected:
        guards.append("INVALID_OR_DUPLICATE_RECEIPTS")
    if len(results) < MIN_INDEPENDENT_RUNS:
        guards.append("INSUFFICIENT_VALID_RUNS")
    unique_hw = {r["key"] for r in results}
    if len(unique_hw) != 1:
        guards.append("INCONSISTENT_HARDWARE_OR_SOFTWARE")
    # Reject concurrent captures; separate invocations are not necessarily
    # statistically independent, but overlapping timestamps are clearly not.
    intervals = sorted((r["started"], r["ended"]) for r in results)
    if any(b[0] <= a[1] for a, b in zip(intervals, intervals[1:])):
        guards.append("OVERLAPPING_EXPERIMENT_PERIODS")
    if any(r["decision"] != "DOUBLE_CANDIDATE" for r in results):
        guards.append("NOT_ALL_RUNS_PASS_LOCAL_GATE")
    run_count = len(results)
    max_wins = sum(r["wins"] for r in results)
    all_samples = [x for r in results for x in zip(r["single"], r["double"])]
    gains = [(s-d)/s for s,d in all_samples]
    paired_gain = statistics.median(gains) if gains else None
    consistency = max_wins / len(all_samples) if all_samples else 0
    if paired_gain is None or paired_gain < 0.05 or consistency < 0.75:
        guards.append("AGGREGATE_EVIDENCE_BELOW_THRESHOLD")
    allowed = len(guards) == 0
    identity = results[0]["key"] if len(unique_hw) == 1 and results else None
    return {
        "schema": SCHEMA, "provenance": PROVENANCE,
        "status": "ELIGIBLE_FOR_SYNTHETIC_OPT_IN" if allowed else "HELD_SINGLE_DEFAULT",
        "unverified_external_inputs": True,
        "validated_receipt_count": run_count,
        "rejected_receipts": rejected,
        "conditions": {"required_runs":MIN_INDEPENDENT_RUNS,
                       "rounds_per_run":MIN_ROUNDS,
                       "chunk_mib":SHAPE[0], "chunks":SHAPE[1],
                       "per_run_min_paired_gain":0.05,
                       "per_run_min_win_share":0.75},
        "aggregate": {
            "paired_trials":len(all_samples), "double_wins":max_wins,
            "double_win_share":round(consistency,6),
            "median_paired_gain_double":(round(paired_gain,6)
                                          if paired_gain is not None else None),
        },
        "hardware_claim": ({
            "platform": identity[0], "gpu_name":identity[1],
            "torch_version":identity[2], "cuda_build":identity[3],
            "minimum_observed_free_vram_bytes":min(r["observed_free"] for r in results),
        } if identity else None),
        "guards":guards, "suggested": "DOUBLE_CANDIDATE" if allowed else "SINGLE",
        "default":"SINGLE", "production_policy_applied":False,
        "model_execution":False, "automatic_layer_placement":False,
        "interpretation":"Receipts are user-supplied and unauthenticated. Passing this gate permits only an explicitly approved synthetic scratch experiment, not production scheduling.",
    }


def find_receipts(pattern: str) -> list[str]:
    """Glob is local only. No recursive traversal and no network lookup."""
    if not isinstance(pattern, str) or not pattern or len(pattern) > 1024 or "**" in pattern:
        raise ValueError("Specify one nonrecursive local receipt-glob")
    names = sorted(glob.glob(pattern))
    if not names:
        raise ValueError("Receipt pattern matched no local files")
    if len(names) > MAX_INPUT_FILES:
        raise ValueError("Too many matching receipt files")
    return names


def _local_cuda_check(evidence: dict, chunk_bytes: int) -> tuple[bool, str]:
    """Return approval-safety advisory; do not claim authenticated devices."""
    from .chunklab import _CudaBackend  # no CUDA calls on review-only path
    import torch
    claim = evidence.get("hardware_claim") or {}
    if not torch.cuda.is_available():
        return False, "CUDA_NOT_AVAILABLE"
    name = torch.cuda.get_device_name()
    if (name != claim.get("gpu_name") or platform.platform() != claim.get("platform")
        or str(torch.__version__) != claim.get("torch_version")
        or str(torch.version.cuda) != claim.get("cuda_build")):
        return False, "HARDWARE_OR_SOFTWARE_CHANGED"
    free, _ = torch.cuda.mem_get_info()
    needed = 4 * chunk_bytes + 256 * MIB
    if free < needed:
        return False, "CUDA_CAPACITY_BELOW_SAFE_MINIMUM"
    floor = claim.get("minimum_observed_free_vram_bytes",0)
    if free + MIN_FREE_VRAM_MARGIN < floor:
        return False, "AVAILABLE_VRAM_SHRANK"
    return True, "MATCHED_UNAUTHENTICATED_RECEIPT_COHORT"


def run_synthetic_guarded(workspace: str | Path, evidence: dict, *,
                          request_double: bool = False,
                          approve_synthetic_double: bool = False,
                          rounds: int = 6, warmups: int = 1) -> dict:
    """Only temporary owned bytes. Single is guaranteed default/held fallback."""
    if type(request_double) is not bool or type(approve_synthetic_double) is not bool:
        raise ValueError("Explicit boolean approvals required")
    _check_parameters(SHAPE[0], SHAPE[1], 6, warmups)
    if type(rounds) is not int or rounds != 6:
        raise ValueError("Prototype uses exactly six synthetic observation rounds")
    root = Path(workspace)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Existing nonsymlink scratch workspace required")
    chunk_bytes = SHAPE[0]*MIB
    if shutil.disk_usage(root).free < 3*64*MIB + 128*MIB:
        raise RuntimeError("Insufficient scratch space; no fallback run possible")
    if physical_memory()["available_bytes"] < 6*chunk_bytes + 256*MIB:
        raise RuntimeError("Insufficient system RAM; refusing execution")
    eligible = evidence.get("status") == "ELIGIBLE_FOR_SYNTHETIC_OPT_IN"
    cuda_ok, cuda_reason = _local_cuda_check(evidence, chunk_bytes)
    if not cuda_ok and cuda_reason == "CUDA_NOT_AVAILABLE":
        raise RuntimeError("CUDA unavailable; no CPU fallback or fake performance")
    use_double = bool(eligible and request_double and approve_synthetic_double and cuda_ok)
    chosen = "double_reuse_worker" if use_double else "single"
    fallback_reasons = []
    if request_double and not eligible: fallback_reasons.append("EVIDENCE_HELD")
    if request_double and not approve_synthetic_double: fallback_reasons.append("OPERATOR_NOT_APPROVED")
    if request_double and not cuda_ok: fallback_reasons.append(cuda_reason)
    # A safety mismatch can block any run if the CUDA resource minimum itself
    # cannot support even the single-buffer scratch allocation.
    if cuda_reason == "CUDA_CAPACITY_BELOW_SAFE_MINIMUM":
        raise RuntimeError("CUDA capacity below guarded scratch minimum")

    backend = _CudaBackend(chunk_bytes)
    values = []
    control = []
    # Keep everything in private owned temp directory; no checkpoint paths.
    began = datetime.now(timezone.utc).isoformat()
    with tempfile.TemporaryDirectory(prefix="phistream-guarded-", dir=root) as tmp:
        filename = Path(tmp)/"own-synthetic-chunks.bin"
        digest, digests, _ = _fixture(filename, chunk_bytes, SHAPE[1])
        with ThreadPoolExecutor(max_workers=1) as worker:
            worker.submit(lambda:None).result()
            for _ in range(warmups):
                _trial(filename, chunk_bytes, SHAPE[1], chosen, backend, worker)
                if use_double:
                    _trial(filename, chunk_bytes, SHAPE[1],
                           "double_reuse_worker", backend, worker,
                           serialize_worker=True)
            for i in range(rounds):
                # Approved double is measured alongside a same-worker
                # deliberately serialized negative control. Alternate order.
                if use_double and i%2==0:
                    control.append(_trial(filename, chunk_bytes, SHAPE[1],
                                   "double_reuse_worker", backend, worker,
                                   serialize_worker=True)["elapsed_seconds"])
                values.append(_trial(filename, chunk_bytes, SHAPE[1], chosen,
                                     backend, worker)["elapsed_seconds"])
                if use_double and i%2==1:
                    control.append(_trial(filename, chunk_bytes, SHAPE[1],
                                   "double_reuse_worker", backend, worker,
                                   serialize_worker=True)["elapsed_seconds"])
        audit = backend.replay_audit(filename, digests, chunk_bytes)
        with filename.open("rb") as stream:
            if hashlib.sha256(stream.read()).hexdigest()!=digest:
                raise RuntimeError("Guarded synthetic fixture digest changed")
    return {
        "schema":"PHISTREAM_GUARDED_SYNTHETIC_RUN_V1",
        "provenance":"LOCALLY_EXECUTED_SYNTHETIC_GPU_SCRATCH_NO_MODEL",
        "status":"PASS_GUARDED_CUDA_SYNTHETIC",
        "started_utc":began,
        "completed_utc":datetime.now(timezone.utc).isoformat(),
        "requested_double":request_double,
        "explicit_operator_approval":approve_synthetic_double,
        "selected_mode":chosen, "fallback_reasons":fallback_reasons,
        "original_gate_status":evidence.get("status"),
        "live_hardware_check":cuda_reason,
        "chunk_mib":SHAPE[0], "chunks":SHAPE[1],
        "selected_end_to_end":_duration(values),
        "serialized_same_worker_negative_control":_duration(control) if control else None,
        "control_is_deliberately_nonoverlapping":True if control else False,
        "integrity":audit, "temporary_scratch_deleted":True,
        "model_execution":False, "production_policy_applied":False,
        "automatic_layer_placement":False,
        "caveats":[
            "User-provided receipts are unauthenticated and may be altered.",
            "The only applied mode is a bounded local scratch experiment, not production.",
            "Clock domains are not calibrated; the serialized control cannot prove physical GPU/read overlap.",
            "OS page cache and ambient system processes remain uncontrolled.",
            "All receipts and payloads remain local; no Ollama/Vessie files or settings accessed.",
        ],
    }
