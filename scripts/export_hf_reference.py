#!/usr/bin/env python3
"""Offline HF Transformers reference ORACLE for PhiStream validation only.

Separate executable. ZERO phistream imports, ZERO tokenizer/network/model downloads,
ZERO checkpoint writes, ZERO real-model scheduler integration. This script uses
installed third-party Transformers as an external comparison runtime, not as
PhiStream's implementation or a production dependency.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

# Set before even considering importing Hugging Face code.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

SCHEMA = "PHISTREAM_REFERENCE_TRACE_V1"
PREFLIGHT = "PHISTREAM_REAL_MODEL_BRIDGE_V1"
HEX = re.compile(r"^[0-9a-f]{64}$")
MAX_JSON = 1024 * 1024
BLOCK = 4 * 1024 * 1024
STEPS = 32


def unique_fields(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("Duplicate JSON field: " + name)
        result[name] = value
    return result


def read_json(path, *, max_bytes=MAX_JSON):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > max_bytes:
        raise ValueError("Missing/symlinked/oversized JSON: " + path.name)
    data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_fields,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    if not isinstance(data, dict):
        raise ValueError("Expected JSON object")
    return data


def file_digest(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Not a normal local file: " + path.name)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(BLOCK)
            if not chunk:
                break
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size, after.st_mtime_ns, after.st_ino
    ):
        raise ValueError("Checkpoint changed during hashing: " + path.name)
    return digest.hexdigest()


def check_checkpoint(model_dir, receipt):
    """Independent file hashing, no calls to PhiStream or HF."""
    root = Path(model_dir)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Explicit existing non-symlink local checkpoint directory required")
    if (receipt.get("schema") != PREFLIGHT
            or receipt.get("status") != "PREFLIGHT_REVIEW_REQUIRED"
            or receipt.get("model_execution") is not False
            or receipt.get("double_buffer_enabled") is not False
            or receipt.get("scheduler_policy_applied") is not False
            or (receipt.get("memory_preflight") or {}).get("status") != "REVIEW_REQUIRED"
            or (receipt.get("checkpoint") or {}).get("format") != "SAFETENSORS"
            or (receipt.get("checkpoint") or {}).get("model_type") != "llama"):
        raise ValueError("Need a successful prior non-execution Llama fingerprint preflight")
    declaration = receipt.get("user_asserted_source") or {}
    if (declaration.get("declared") is not True
            or declaration.get("verified_by_phistream") is not False
            or any(not isinstance(declaration.get(field), str)
                   or not declaration[field].strip()
                   for field in ("source", "revision", "license_id"))):
        raise ValueError("Declared pinned source, revision, license required")
    fp = receipt.get("checkpoint_fingerprint") or {}
    claimed = fp.get("files")
    if fp.get("algorithm") != "sha256" or not isinstance(claimed, dict) or (
        set(claimed) != {"config.json", "model.safetensors"}
    ):
        raise ValueError("Phase-1 exporter needs complete config + single-shard fingerprint")
    for name, digest in claimed.items():
        if not isinstance(digest, str) or not HEX.fullmatch(digest):
            raise ValueError("Invalid SHA256 in preflight manifest")
    current = {name: file_digest(root / name) for name in sorted(claimed)}
    if current != claimed:
        raise ValueError("Checkpoint SHA256 mismatch. Refusing reference model execution.")
    cfg = read_json(root / "config.json")
    if (cfg.get("model_type") != "llama" or
            cfg.get("architectures") not in (None, ["LlamaForCausalLM"]) or
            type(cfg.get("vocab_size")) is not int or not 8 <= cfg["vocab_size"] <= 1_000_000 or
            type(cfg.get("max_position_embeddings", 2048)) is not int):
        raise ValueError("Invalid or unsupported Llama config")
    return cfg, current, declaration


def parse_prompt(value, cfg):
    parts = value.split(",")
    if not 1 <= len(parts) <= 64 or any(not x.isdecimal() for x in parts):
        raise ValueError("Provide 1–64 comma-separated unsigned token IDs")
    ids = [int(x) for x in parts]
    if any(i >= cfg["vocab_size"] for i in ids):
        raise ValueError("Prompt token outside checkpoint vocabulary")
    if len(ids) + STEPS - 1 > cfg.get("max_position_embeddings", 2048):
        raise ValueError("Prompt and reference length exceed checkpoint context")
    return ids


def reference_forward(root, ids, vocab, device):
    """Independent HF full-sequence, no cache. Never imports PhiStream."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. No CPU fallback.")
        free_bytes, _ = torch.cuda.mem_get_info()
        weight_bytes = (root / "model.safetensors").stat().st_size
        if free_bytes < (1024 * 1024 * 1024 + 4 * weight_bytes):
            raise RuntimeError("CUDA memory too low for guarded reference trial")
    model = AutoModelForCausalLM.from_pretrained(
        str(root),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()
    model.to(device)
    selected = ids.copy()
    generated = []
    probes = []
    min_margin = math.inf
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for index in range(STEPS):
                seq = torch.tensor([selected], dtype=torch.long, device=device)
                output = model(input_ids=seq, use_cache=False)
                logits = output.logits[0, -1, :].detach().to(
                    device="cpu", dtype=torch.float32
                )
                if logits.numel() != vocab or not bool(torch.isfinite(logits).all()):
                    raise RuntimeError("Reference logits are nonfinite or have wrong vocabulary")
                highest = torch.topk(logits, k=2)
                next_id = int(highest.indices[0].item())
                runner_up = int(highest.indices[1].item())
                margin = float((highest.values[0] - highest.values[1]).item())
                min_margin = min(margin, min_margin)
                probe_ids = sorted({0, 1, 3, 7, vocab // 4, vocab // 2,
                                    vocab - 1, next_id, runner_up})
                probes.append({str(i): float(logits[i].item()) for i in probe_ids})
                generated.append(next_id)
                # Deliberately ignore tokenizer and EOS. PhiStream's comparator
                # teacher-forces this exact generated sequence at 32 positions.
                selected.append(next_id)
        if device == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - started
    finally:
        del model
    return generated, probes, {
        "runtime": "Hugging Face Transformers AutoModelForCausalLM",
        "version": str(transformers.__version__),
        "execution_device": device,
        "asserted_independent": True,
        "torch_version": str(torch.__version__),
        "algorithm": "FULL_SEQUENCE_GREEDY_NO_CACHE",
        "attention_implementation": "eager",
        "weight_compute_dtype": "float32",
    }, {"reference_generation_seconds": round(seconds, 6),
        "minimum_top2_logit_margin": round(min_margin, 8),
        "continuation_ignores_eos": True,
        "tokenizer_used": False}


def export_trace(model_dir, prior_receipt, *, device, prompt,
                 output, execute_reference=False, approved=False):
    if type(execute_reference) is not bool or type(approved) is not bool:
        raise ValueError("Require explicit boolean consent")
    if approved and not execute_reference:
        raise ValueError("Cannot approve without requesting reference execution")
    if execute_reference and (not approved or device not in ("cpu", "cuda")):
        raise ValueError("Reference execution needs --run-reference, --approve-local-execution, --device cpu|cuda")
    if device not in ("none", "cpu", "cuda"):
        raise ValueError("Unsupported device")
    destination = Path(output) if output is not None else None
    if execute_reference and destination is None:
        raise ValueError("Reference execution needs explicit --output JSON path")
    if destination is not None and (
        destination.exists() or destination.suffix.lower() != ".json"
    ):
        raise ValueError("Output must be a new local .json file (no overwrite)")
    root = Path(model_dir)
    report = read_json(prior_receipt)
    cfg, checksums, declaration = check_checkpoint(root, report)
    ids = parse_prompt(prompt, cfg)
    if not execute_reference:
        return {
            "status": "HASH_VERIFIED_REFERENCE_NOT_EXECUTED",
            "reference_trace_written": False,
            "checkpoint_sha256": {"algorithm": "sha256", "files": checksums},
            "planned_steps": STEPS, "device": "none",
            "source_license_are_user_asserted": True, "local_only": True,
        }
    # A second read after the reference protects against concurrent accidental edits.
    generated, probes, origin, measurements = reference_forward(
        root, ids, cfg["vocab_size"], device
    )
    after = {name: file_digest(root / name) for name in sorted(checksums)}
    if after != checksums:
        raise ValueError("Checkpoint changed during reference execution; refusing receipt")
    if len(generated) != STEPS or len(probes) != STEPS:
        raise ValueError("Reference must produce exactly 32 steps")
    trace = {
        "schema": SCHEMA,
        "reference": origin,
        "checkpoint_fingerprint": {"algorithm": "sha256", "files": checksums},
        "prompt_token_ids": ids,
        "expected_generated_token_ids": generated,
        "logit_probes": probes,
        "atol": 0.0001, "rtol": 0.0001,
        "experiment": {
            "status": "REFERENCE_OBSERVED_NOT_AUTHENTICATED",
            "observed_utc": datetime.now(timezone.utc).isoformat(),
            "source_revision_license_user_asserted": declaration,
            "checkpoint_bytes_rehashed_after_execution": True,
            "no_model_download": True,
            "model_files_modified": False,
            "no_tokenizer_or_eos": True,
            "phistream_decoder_not_imported": True,
            "no_phi_scheduler_activation": True,
            **measurements,
        },
    }
    payload = json.dumps(trace, indent=2, allow_nan=False) + "\n"
    if len(payload.encode("utf-8")) > 8 * 1024 * 1024:
        raise ValueError("Reference trace exceeds PhiStream 8 MiB input limit")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    return {
        "status": "LOCAL_HF_REFERENCE_TRACE_CREATED_NOT_YET_COMPARED",
        "trace_path": str(destination),
        "steps": STEPS, "probe_maps": len(probes), "device": device,
        "model_weights_unchanged": True,
        "source_license_are_user_asserted": True,
        "reference_independence_authenticated": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Independent HF offline reference exporter (no PhiStream decoder imports)"
    )
    parser.add_argument("--checkpoint", required=True, help="Existing local safetensors folder")
    parser.add_argument("--preflight-receipt", required=True,
                        help="Existing local real-model-gate --hash-checkpoint JSON")
    parser.add_argument("--prompt", default="1,4,3",
                        help="1–64 literal token IDs, no tokenizer, default 1,4,3")
    parser.add_argument("--device", choices=("none", "cpu", "cuda"), default="none")
    parser.add_argument("--run-reference", action="store_true",
                        help="Actually load local model in separate HF runtime")
    parser.add_argument("--approve-local-execution", action="store_true",
                        help="Explicit operator approval for local model execution")
    parser.add_argument("--output", help="New local 32-step reference JSON; never overwrite")
    args = parser.parse_args(argv)
    result = export_trace(args.checkpoint, args.preflight_receipt,
                          device=args.device, prompt=args.prompt,
                          output=args.output,
                          execute_reference=args.run_reference,
                          approved=args.approve_local_execution)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
