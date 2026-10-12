"""Offline, fail-closed comparison of PhiStream against user-supplied reference traces.

Never launches, downloads, or imports a reference model. A trace's origin is
user-declared, NOT cryptographically authenticated independent evidence.
Compares exact greedy token IDs and optionally specified logits coordinates
under teacher forcing on an identical checkpoint fingerprint.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import time
from typing import Any

import torch

from .llama import StreamingLlama
from .safeweights import LocalCheckpoint, _no_duplicate_fields

TRACE_SCHEMA = "PHISTREAM_REFERENCE_TRACE_V1"
RESULT_SCHEMA = "PHISTREAM_REFERENCE_COMPARISON_V1"
MAX_TRACE_BYTES = 8 * 1024 * 1024
MAX_STEPS = 128
MAX_PROBES_PER_STEP = 128
CHUNK_BYTES = 4 * 1024 * 1024
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
ID_RE = re.compile(r"^(0|[1-9][0-9]*)$")


def _read_trace(path: str | Path) -> dict:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ValueError("Reference trace must be a local regular JSON file (no symlinks)")
    if target.stat().st_size > MAX_TRACE_BYTES:
        raise ValueError("Reference trace exceeds the 8 MiB size limit")
    try:
        with target.open("r", encoding="utf-8") as fh:
            trace = json.load(fh, object_pairs_hook=_no_duplicate_fields,
                              parse_constant=lambda x: (_ for _ in ()).throw(
                                  ValueError(f"Non-finite JSON value {x}")))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError("Invalid reference JSON") from exc
    if not isinstance(trace, dict):
        raise ValueError("Reference trace root must be an object")
    return trace


def _hash_file(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Missing or symlinked checkpoint file: {path.name}")
    before = path.stat()
    digest = sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    after = path.stat()
    if (before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ino != after.st_ino):
        raise ValueError(f"Checkpoint file changed during hashing: {path.name}")
    return digest.hexdigest()


def checkpoint_fingerprint(checkpoint: LocalCheckpoint) -> dict:
    """Complete SHA256 of local weights + config, not a model/license signature.

    Deliberately reads every byte. Huge checkpoints can take a long time.
    Fingerprints can be published independently without publishing weights.
    """
    names = ["config.json", *sorted(checkpoint.files)]
    result = {}
    for name in names:
        result[name] = _hash_file(checkpoint.base / name)
    return {"algorithm": "sha256", "files": result}


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def validate_reference_trace(trace: dict, ckpt: LocalCheckpoint) -> dict:
    """Validate entire input before any expensive inference or side effects."""
    if trace.get("schema") != TRACE_SCHEMA:
        raise ValueError("Unsupported reference trace schema")
    origin = trace.get("reference")
    if not isinstance(origin, dict):
        raise ValueError("Reference identification is required")
    for field in ("runtime", "version", "execution_device"):
        value = origin.get(field)
        if not isinstance(value, str) or not 1 <= len(value.strip()) <= 150:
            raise ValueError(f"Reference {field} must be a nonempty string")
    if origin.get("asserted_independent") is not True:
        raise ValueError("Reference must explicitly declare an independent implementation")
    manifest = trace.get("checkpoint_fingerprint")
    if not isinstance(manifest, dict) or manifest.get("algorithm") != "sha256":
        raise ValueError("Trace requires a complete SHA256 checkpoint fingerprint")
    expected = manifest.get("files")
    names = {"config.json", *ckpt.files.keys()}
    if not isinstance(expected, dict) or set(expected) != names:
        raise ValueError("Reference fingerprint must list exactly the config and all weight shards")
    if any(not isinstance(v, str) or not DIGEST_RE.fullmatch(v) for v in expected.values()):
        raise ValueError("Expected SHA256 digests must be lowercase 64-character hex")
    prompt = trace.get("prompt_token_ids")
    expected_ids = trace.get("expected_generated_token_ids")
    probes = trace.get("logit_probes")
    if (not isinstance(prompt, list) or not 1 <= len(prompt) <= 2048
            or any(type(t) is not int or t < 0 or t >= ckpt.config.get("vocab_size", -1)
                   for t in prompt)):
        raise ValueError("Invalid prompt token IDs")
    if (not isinstance(expected_ids, list) or not 1 <= len(expected_ids) <= MAX_STEPS
            or any(type(t) is not int or t < 0 or t >= ckpt.config["vocab_size"]
                   for t in expected_ids)):
        raise ValueError("Invalid generated token IDs")
    if len(prompt) + len(expected_ids) - 1 > ckpt.config.get("max_position_embeddings", 2048):
        raise ValueError("Reference prompt and generation exceed model context")
    if not isinstance(probes, list) or len(probes) != len(expected_ids):
        raise ValueError("One logit probe map per generated token is required")
    cleaned = []
    for step, item in enumerate(probes):
        if not isinstance(item, dict) or not 1 <= len(item) <= MAX_PROBES_PER_STEP:
            raise ValueError("Each step needs 1-128 named logit probes")
        parsed = {}
        for key, value in item.items():
            if not isinstance(key, str) or not ID_RE.fullmatch(key):
                raise ValueError("Logit probe positions must be canonical decimal token IDs")
            idx = int(key)
            if idx >= ckpt.config["vocab_size"]:
                raise ValueError("Logit probe position outside vocabulary")
            parsed[idx] = _finite_number(value, f"logit probe in step {step}")
        cleaned.append(parsed)
    atol = _finite_number(trace.get("atol", 1e-4), "atol")
    rtol = _finite_number(trace.get("rtol", 1e-4), "rtol")
    if not 0 <= atol <= 1 or not 0 <= rtol <= 1:
        raise ValueError("Tolerances must be nonnegative and at most 1")
    return {"origin": origin, "prompt": prompt, "expected_ids": expected_ids,
            "probes": cleaned, "atol": atol, "rtol": rtol, "fingerprint": expected}


def compare_reference(
    checkpoint: LocalCheckpoint, trace: dict, *,
    device: str = "cpu", max_tensor_mib: int = 512, head_rows: int = 128,
) -> dict:
    """Teacher-forced reference trace evaluation; full checkpoint bound by hashes.

    No false 'verified independent' status. The trace could be fabricated or
    exported by the same PhiStream engine; provenance remains unverified.
    """
    validated = validate_reference_trace(trace, checkpoint)
    actual_fp = checkpoint_fingerprint(checkpoint)
    expected_fp = validated["fingerprint"]
    # Before invoking the decoder, require identical complete file hashes.
    if actual_fp["files"] != expected_fp:
        raise ValueError("Checkpoint fingerprint mismatch: do not compare different weights")
    execution_start = time.perf_counter()
    engine = StreamingLlama(checkpoint, device=device, max_tensor_mib=max_tensor_mib,
                            head_rows=head_rows)
    prompt = validated["prompt"]
    reference_ids = validated["expected_ids"]
    for token in prompt[:-1]:
        engine.step(token)
    supplied_token = prompt[-1]
    comparisons = []
    matched_tokens = 0
    matched_probes = 0
    max_delta = 0.0
    worst_step = None
    first_logits_seconds = None
    per_step_seconds = []
    peak_cuda_allocated_bytes = None
    for i, expected in enumerate(reference_ids):
        step_start = time.perf_counter()
        logits = engine.step(supplied_token)
        # The executor already synchronizes each load; explicitly synchronize
        # at the end of CUDA step to bound the compute duration as well.
        if engine.device.type == "cuda":
            torch.cuda.synchronize(engine.device)
        finished = time.perf_counter()
        per_step_seconds.append(finished - step_start)
        if i == 0:
            first_logits_seconds = finished - execution_start
        if engine.last_peak_cuda_allocated is not None:
            peak_cuda_allocated_bytes = max(
                peak_cuda_allocated_bytes or 0, int(engine.last_peak_cuda_allocated)
            )
        greedy = int(torch.argmax(logits).item())
        token_match = greedy == expected
        if token_match:
            matched_tokens += 1
        step_ok = True
        for token_id, target in validated["probes"][i].items():
            found = float(logits[token_id].item())
            err = abs(found - target)
            if not math.isfinite(found):
                step_ok = False
                max_delta = float("inf")
                worst_step = i
                continue
            if err > max_delta:
                max_delta, worst_step = err, i
            if err <= validated["atol"] + validated["rtol"] * abs(target):
                matched_probes += 1
            else:
                step_ok = False
        comparisons.append({
            "step": i, "expected_next_id": expected, "actual_next_id": greedy,
            "token_match": token_match, "probe_match": step_ok,
        })
        # Teacher forcing preserves matching positions after an earlier mismatch.
        supplied_token = expected
    execution_total_seconds = time.perf_counter() - execution_start
    all_tokens = matched_tokens == len(reference_ids)
    total_probes = sum(len(v) for v in validated["probes"])
    all_probes = matched_probes == total_probes
    complete_length = len(reference_ids) >= 32
    match = all_tokens and all_probes
    status = ("MISMATCH" if not match else
              "MATCH_USER_REFERENCE_UNVERIFIED" if complete_length else
              "MATCH_TOO_SHORT_USER_REFERENCE_UNVERIFIED")
    return {
        "schema": RESULT_SCHEMA,
        "status": status,
        "provenance": "USER_SUPPLIED_REFERENCE_NOT_INDEPENDENTLY_AUTHENTICATED",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "reference": validated["origin"],
        "checkpoint_fingerprint": actual_fp,
        "device": str(engine.device),
        "gpu_name": torch.cuda.get_device_name(engine.device) if engine.device.type == "cuda" else None,
        "steps": len(reference_ids),
        "required_steps_for_gate_candidate": 32,
        "exact_greedy_tokens_matched": matched_tokens,
        "total_logit_probes": total_probes,
        "logit_probes_matched": matched_probes,
        "max_abs_sampled_logit_error": max_delta if math.isfinite(max_delta) else None,
        "worst_probe_step": worst_step,
        "atol": validated["atol"], "rtol": validated["rtol"],
        "comparisons": comparisons,
        "performance": {
            "measurement_scope": "DECODER_ONLY_EXCLUDES_CHECKPOINT_SHA_AND_TRACE_VALIDATION",
            "total_with_engine_setup_seconds": round(execution_total_seconds, 6),
            "time_to_first_logits_including_prompt_seconds": round(first_logits_seconds, 6),
            "subsequent_step_seconds": [round(x, 6) for x in per_step_seconds[1:]],
            "subsequent_decode_tokens_per_second": (
                round((len(per_step_seconds)-1) / sum(per_step_seconds[1:]), 6)
                if len(per_step_seconds) > 1 and sum(per_step_seconds[1:]) > 0
                else None
            ),
            "max_observed_cuda_allocated_bytes_per_step": peak_cuda_allocated_bytes,
            "weight_load_count": engine.weight_load_count,
            "weight_bytes_staged_total": engine.weight_bytes_staged,
            "weight_load_seconds_including_transfers": round(engine.weight_load_seconds, 6),
            "prefill_definition": "includes initialization, prompt processing and first-token logits; not prefill-only throughput",
            "no_reference_speedup_claim": True,
        },
        "logit_coverage": "SPARSE_COORDINATES_ONLY",
        "independent_reference_authenticated": False,
        "verification_claim": (
            "Comparison against a USER-DECLARED independent reference trace; "
            "runtime and reference provenance are NOT independently verified."
        ),
    }


def read_reference_trace(path: str | Path) -> dict:
    return _read_trace(path)
