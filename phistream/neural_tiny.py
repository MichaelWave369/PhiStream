"""Small, explicit, operator-approved byte-level Transformer training experiment.

No auto-downloads, network calls, imports of Karpathy projects, checkpoints,
external shell commands, autonomous edits, or model promotion. CPU by default.
Records are declared local measurements, not hardware attestations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from . import neural_protocol as protocol

SCHEMA = "PHI_TINY_TRAIN_R1"
PLAN_SCHEMA = "PHI_NEURAL_PLAN_R1"
MAX_DATASET = 1024 * 1024
NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}\Z")


class ExperimentError(ValueError):
    """Unsafe input, disallowed training, or an incomplete experiment."""


def checked_dataset(path: Path, context: int) -> bytes:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ExperimentError("Dataset must be a regular file, not a symlink")
    size = path.stat().st_size
    if not 200 <= size <= MAX_DATASET:
        raise ExperimentError("Dataset size must be 200 bytes to 1 MiB")
    data = path.read_bytes()
    try:
        decoded = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ExperimentError("Dataset must be valid UTF-8") from exc
    if "\x00" in decoded:
        raise ExperimentError("NUL bytes are not accepted")
    train, val = protocol.split_bytes(data)
    if min(len(train), len(val)) < context + 2:
        raise ExperimentError("Training and holdout partitions must exceed context size")
    return data


def preflight(data_path: Path, context: int = 16) -> dict:
    data = checked_dataset(data_path, context)
    train, val = protocol.split_bytes(data)
    return {
        "schema": "PHI_TINY_PREFLIGHT_R1", "status": "READY_FOR_EXPLICIT_OPERATOR_TRAINING",
        "dataset_sha256": hashlib.sha256(data).hexdigest(),
        "dataset_bytes": len(data), "train_bytes": len(train), "holdout_bytes": len(val),
        "tokenizer_sha256": protocol.tokenizer_digest(),
        "evaluator_sha256": protocol.evaluator_digest(context),
        "context": context, "holdout_windows": len(protocol.windows(val, context)),
        "training_started": False, "gpu_probed": False, "downloaded": False,
        "data_text_exported": False, "quality_improvement_proven": False,
    }


@dataclass(frozen=True)
class Config:
    context: int = 16
    dim: int = 32
    heads: int = 2
    layers: int = 1
    batch: int = 4
    steps: int = 12
    lr: float = 0.003
    seconds: float = 120.0
    vram_mb: float = 10240.0
    seed: int = 369
    device: str = "cpu"
    study_id: str = "phi-tiny-v1"
    run_id: str = "baseline"

    def check(self):
        if not (4 <= self.context <= 128 and 8 <= self.dim <= 256
                and 1 <= self.heads <= 8 and self.dim % self.heads == 0
                and 1 <= self.layers <= 4 and 1 <= self.batch <= 16
                and 1 <= self.steps <= 1000 and 1 <= self.seed < 2**31
                and 0 < self.lr <= 0.03 and 1 <= self.seconds <= 3600
                and 256 <= self.vram_mb <= 24576 and self.device in ("cpu", "cuda")):
            raise ExperimentError("Training config outside bounded tiny-model policy")
        if any(not math.isfinite(x) for x in (self.lr, self.seconds, self.vram_mb)):
            raise ExperimentError("Nonfinite training configuration")
        if not NAME.fullmatch(self.study_id) or not NAME.fullmatch(self.run_id):
            raise ExperimentError("Study and run IDs must be simple, short names")
        return self


def make_model(config: Config):
    """Build an original tiny causal decoder; reference projects are not imported."""
    import torch
    from torch import nn

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(config.dim)
            self.attention = nn.MultiheadAttention(config.dim, config.heads, batch_first=True)
            self.norm2 = nn.LayerNorm(config.dim)
            self.ffn = nn.Sequential(nn.Linear(config.dim, 4 * config.dim),
                                     nn.GELU(), nn.Linear(4 * config.dim, config.dim))

        def forward(self, x, causal_mask):
            z = self.norm1(x)
            x = x + self.attention(z, z, z, attn_mask=causal_mask, need_weights=False)[0]
            return x + self.ffn(self.norm2(x))

    class TinyDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.token = nn.Embedding(256, config.dim)
            self.position = nn.Embedding(config.context, config.dim)
            self.blocks = nn.ModuleList(Block() for _ in range(config.layers))
            self.norm = nn.LayerNorm(config.dim)
            self.head = nn.Linear(config.dim, 256, bias=False)

        def forward(self, ids):
            if ids.ndim != 2 or ids.size(1) > config.context:
                raise ValueError("Token input length exceeds context")
            positions = torch.arange(ids.shape[1], device=ids.device)
            x = self.token(ids) + self.position(positions)
            mask = torch.ones(ids.shape[1], ids.shape[1], device=ids.device, dtype=torch.bool).triu(1)
            for block in self.blocks:
                x = block(x, mask)
            return self.head(self.norm(x))

    return TinyDecoder()


def _hardware_id(device: str, torch_module) -> str:
    if device == "cuda":
        info = str(torch_module.cuda.get_device_name(0)) + "/" + str(torch_module.cuda.get_device_capability(0))
    else:
        info = platform.machine() + "/" + platform.processor() + "/" + platform.system()
    # Stable *declared* device profile, not a unique or authenticated machine ID.
    return device + "-" + hashlib.sha256((info + "/torch-" + torch_module.__version__).encode()).hexdigest()[:16]


def _new_folder(path: Path):
    path = Path(path).expanduser()
    if path.is_symlink() or path.exists():
        raise ExperimentError("Output directory must not exist; past runs are immutable")
    # Avoid inadvertently writing receipts and private context in source control.
    resolved = path.resolve()
    if any((ancestor / ".git").exists() for ancestor in resolved.parents):
        raise ExperimentError("Output must be outside every Git checkout")
    return resolved


def _write_exclusive(path: Path, content: str):
    with path.open("x", encoding="utf-8") as file:
        file.write(content)


def run(data_path: Path, out_dir: Path, config: Config, *, approved: bool = False):
    config.check()
    if not approved:
        raise ExperimentError("Training requires explicit --approve-training")
    out_dir = _new_folder(out_dir)
    data = checked_dataset(data_path, config.context)
    train_bytes, validation_bytes = protocol.split_bytes(data)
    import torch
    import torch.nn.functional as F

    if config.device == "cuda" and not torch.cuda.is_available():
        raise ExperimentError("CUDA unavailable; no CPU fallback without operator approval")
    torch.manual_seed(config.seed)
    if config.device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.reset_peak_memory_stats()
    device = torch.device(config.device)
    model = make_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.0)
    n_params = sum(x.numel() for x in model.parameters())
    initial_bpb = protocol.bits_per_byte(model, validation_bytes, context=config.context, device=device)
    model.train()
    rng = torch.Generator(device="cpu").manual_seed(config.seed)
    # Every training window uses the training partition only; the held-out bytes
    # are reserved strictly for the locked evaluator.
    train_tensor = torch.tensor(list(train_bytes), dtype=torch.long)
    start = time.monotonic()
    for _ in range(config.steps):
        if time.monotonic() - start >= config.seconds:
            raise ExperimentError("Stopped at training time cap; no completed receipt written")
        upper = len(train_bytes) - config.context - 1
        indices = torch.randint(0, upper + 1, (config.batch,), generator=rng)
        offsets = torch.arange(config.context)
        x = train_tensor[indices[:, None] + offsets].to(device)
        y = train_tensor[indices[:, None] + offsets + 1].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        if not torch.isfinite(loss):
            raise ExperimentError("Nonfinite training loss; experiment abandoned")
        loss.backward()
        optimizer.step()
    if config.device == "cuda":
        torch.cuda.synchronize()
    training_seconds = time.monotonic() - start
    if training_seconds > config.seconds:
        raise ExperimentError("Training exceeded time cap; no completed receipt written")
    result_bpb = protocol.bits_per_byte(model, validation_bytes, context=config.context, device=device)
    peak_vram_mb = (torch.cuda.max_memory_allocated(0) / 1024**2) if config.device == "cuda" else 0.0
    if config.device == "cuda" and peak_vram_mb > config.vram_mb:
        raise ExperimentError("CUDA allocation exceeded declared limit; no completed receipt written")
    # Metric budget excludes setup and initial/final evaluation, matching the
    # reported training_seconds. total_seconds includes the final evaluation.
    total_seconds = time.monotonic() - start
    plan = {
        "schema": PLAN_SCHEMA, "study_id": config.study_id, "run_id": config.run_id,
        "engine": "phi_tiny", "metric": "val_bpb", "direction": "min",
        "dataset_sha256": hashlib.sha256(data).hexdigest(),
        "evaluator_sha256": protocol.evaluator_digest(config.context),
        "tokenizer_sha256": protocol.tokenizer_digest(),
        "hardware_id": _hardware_id(config.device, torch),
        "training_budget_seconds": config.seconds,
        "max_peak_vram_mb": config.vram_mb,
        "source_ref": "PhiStream:phistream/neural_tiny.py",
    }
    log = "\n".join([
        f"val_bpb: {result_bpb:.9f}",
        f"training_seconds: {training_seconds:.6f}",
        f"total_seconds: {total_seconds:.6f}",
        f"peak_vram_mb: {peak_vram_mb:.6f}",
        f"num_steps: {config.steps}",
        f"num_params_M: {n_params / 1_000_000:.9f}",
    ]) + "\n"
    metadata = {
        "schema": SCHEMA, "status": "LOCAL_TRAINING_COMPLETED_UNVERIFIED",
        "plan": plan, "initial_val_bpb": initial_bpb,
        "final_val_bpb": result_bpb, "delta_val_bpb": initial_bpb - result_bpb,
        "steps": config.steps, "parameters": n_params,
        "config": {field: getattr(config, field) for field in config.__dataclass_fields__},
        "torch_version": str(torch.__version__), "python_version": sys.version.split()[0],
        "train_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "evaluator_is_fixed": True, "training_proven_authentic": False,
        "quality_improvement_proven": False, "production_authorized": False,
        "checkpoint_saved": False, "network_used": False,
        "vram_metric_scope": "torch.cuda.max_memory_allocated; 0 means CPU/not_applicable",
        "time_limit_scope": "best_effort_step_boundary; does not hard-kill hung GPU kernels",
    }
    out_dir.mkdir(parents=True)
    _write_exclusive(out_dir / "plan.json", json.dumps(plan, indent=2, sort_keys=True) + "\n")
    _write_exclusive(out_dir / "metrics.log", log)
    _write_exclusive(out_dir / "run-meta.json", json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description="Explicit, offline tiny-Transformer training pilot")
    actions = parser.add_subparsers(dest="command", required=True)
    check = actions.add_parser("preflight", help="Read data metadata without loading model or GPU")
    check.add_argument("--data", type=Path, required=True)
    check.add_argument("--context", type=int, default=16)
    train = actions.add_parser("train", help="Run bounded local training; CPU default")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--out-dir", type=Path, required=True)
    train.add_argument("--approve-training", action="store_true")
    train.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    train.add_argument("--context", type=int, default=16)
    train.add_argument("--dim", type=int, default=32)
    train.add_argument("--heads", type=int, default=2)
    train.add_argument("--layers", type=int, default=1)
    train.add_argument("--batch", type=int, default=4)
    train.add_argument("--steps", type=int, default=12)
    train.add_argument("--lr", type=float, default=0.003)
    train.add_argument("--seconds", type=float, default=120)
    train.add_argument("--vram-mb", type=float, default=10240)
    train.add_argument("--seed", type=int, default=369)
    train.add_argument("--study-id", default="phi-tiny-v1")
    train.add_argument("--run-id", default="baseline")
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            report = preflight(args.data, args.context)
        else:
            config = Config(**{field: getattr(args, field) for field in Config.__dataclass_fields__})
            report = run(args.data, args.out_dir, config, approved=args.approve_training)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (ExperimentError, OSError, ValueError) as exc:
        print("phi-tiny held: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())