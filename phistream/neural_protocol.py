"""Frozen, independent byte-tokenizer + held-out score contract for Φ tiny experiments.

Training architectures may change, but this evaluator must remain unchanged within
one comparison study. An implementation edit changes its SHA-256 and invalidates
cross-version comparison until new baselines are run.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

BYTE_TOKENIZER_SPEC = (
    "phi-byte-v1; UTF-8 byte tokens 0..255; no BOS/EOS; "
    "contiguous last-20-percent holdout; shifted causal windows; "
    "fixed 8 evenly-spaced evaluation windows; log2 bits-per-byte"
)


def tokenizer_digest() -> str:
    return hashlib.sha256(BYTE_TOKENIZER_SPEC.encode("utf-8")).hexdigest()


def evaluator_digest(context: int) -> str:
    """Bind the exact evaluator code *and* its context-sized holdout protocol."""
    if type(context) is not int or not 4 <= context <= 128:
        raise ValueError("Invalid evaluation context")
    content = Path(__file__).read_bytes()
    return hashlib.sha256(content + f"\ncontext={context};windows=8".encode("ascii")).hexdigest()


def split_bytes(data: bytes) -> tuple[bytes, bytes]:
    if not data:
        raise ValueError("Dataset is empty")
    boundary = (len(data) * 4) // 5
    return data[:boundary], data[boundary:]


def windows(data: bytes, context: int, count: int = 8):
    """Deterministic, disjoint-from-training evaluation positions."""
    if not (isinstance(context, int) and 4 <= context <= 128):
        raise ValueError("Invalid evaluation context")
    if not (isinstance(count, int) and 1 <= count <= 32):
        raise ValueError("Invalid evaluation window count")
    maximum_start = len(data) - context - 1
    if maximum_start < 0:
        raise ValueError("Evaluation holdout is shorter than context + 1")
    count = min(count, maximum_start + 1)
    positions = [0] if count == 1 else [i * maximum_start // (count - 1) for i in range(count)]
    return [(data[i:i + context], data[i + 1:i + context + 1]) for i in positions]


def bits_per_byte(model, val_data: bytes, *, context: int, device) -> float:
    """Mean held-out token cross-entropy (nats) converted to bits/byte.

    Inputs are raw UTF-8 bytes and the output vocabulary is exactly 256 bytes.
    The scorer never shuffles, samples training windows, or changes loss weights.
    """
    import torch
    import torch.nn.functional as F

    was_training = model.training
    model.eval()
    total_loss, total_tokens = 0.0, 0
    try:
        with torch.inference_mode():
            for xs, ys in windows(val_data, context):
                x = torch.tensor([list(xs)], dtype=torch.long, device=device)
                y = torch.tensor([list(ys)], dtype=torch.long, device=device)
                logits = model(x)
                if tuple(logits.shape) != (1, context, 256):
                    raise ValueError("Model output shape is incompatible with fixed byte evaluator")
                total_loss += float(F.cross_entropy(
                    logits.reshape(-1, 256), y.reshape(-1), reduction="sum"
                ).item())
                total_tokens += y.numel()
        result = total_loss / total_tokens / math.log(2)
        if not math.isfinite(result) or result <= 0:
            raise ValueError("Nonfinite or nonpositive evaluation loss")
        return result
    finally:
        model.train(was_training)