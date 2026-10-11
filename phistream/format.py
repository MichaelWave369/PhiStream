"""Zero-pickle, offline, memory-mapped tensor storage for our experimental format."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping
import json
import math
import os

import numpy as np


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 128
    dim: int = 64
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 2
    ffn_dim: int = 128
    max_context: int = 128
    rope_theta: float = 10000.0
    rms_eps: float = 1e-5

    def validate(self) -> None:
        fields = (self.vocab_size, self.dim, self.n_layers, self.n_heads,
                  self.n_kv_heads, self.ffn_dim, self.max_context)
        if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in fields):
            raise ValueError("All dimension and count fields must be positive integers")
        if self.dim % self.n_heads or self.n_heads % self.n_kv_heads:
            raise ValueError("dim must divide n_heads; n_heads must divide n_kv_heads")
        if (self.dim // self.n_heads) % 2:
            raise ValueError("RoPE requires an even head_dim")
        if not (math.isfinite(self.rope_theta) and self.rope_theta > 0
                and math.isfinite(self.rms_eps) and self.rms_eps > 0):
            raise ValueError("Invalid rope_theta or rms_eps")

    def shapes(self) -> dict[str, tuple[int, ...]]:
        self.validate()
        d, head_dim, kv_dim = self.dim, self.dim // self.n_heads, self.n_kv_heads * (self.dim // self.n_heads)
        shapes = {"embed": (self.vocab_size, d), "final_norm": (d,), "lm_head": (self.vocab_size, d)}
        for i in range(self.n_layers):
            p = f"layer.{i}."
            shapes.update({
                p + "attn_norm": (d,), p + "ffn_norm": (d,),
                p + "wq": (d, d), p + "wk": (kv_dim, d), p + "wv": (kv_dim, d),
                p + "wo": (d, d), p + "gate": (self.ffn_dim, d),
                p + "up": (self.ffn_dim, d), p + "down": (d, self.ffn_dim),
            })
        return shapes


class ModelStore:
    """Manifest-validated, contiguous float32 file; maps bytes without eager read.

    Only local, fixed-name files are used. Tensor checksum verification is explicit.
    This is our own format, not a checkpoint importer or safetensors clone.
    """
    def __init__(self, directory: str | Path):
        directory = Path(directory)
        with (directory / "manifest.json").open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        if manifest.get("format") != "PHISTREAM_V1_F32_LE":
            raise ValueError("Unsupported PhiStream format")
        self.config = ModelConfig(**manifest["config"])
        expected = self.config.shapes()
        raw_tensors = manifest["tensors"]
        if set(raw_tensors) != set(expected):
            raise ValueError("Manifest tensor set does not match architecture")
        blob = directory / "weights.bin"
        blob_size = blob.stat().st_size
        spans: list[tuple[int,int]] = []
        for name, shape in expected.items():
            entry = raw_tensors[name]
            offset, nbytes = entry["offset"], entry["nbytes"]
            if (entry["dtype"] != "<f4" or entry["shape"] != list(shape)
                or not isinstance(offset, int) or not isinstance(nbytes, int)
                or offset < 0 or nbytes != math.prod(shape) * 4
                or offset + nbytes > blob_size):
                raise ValueError(f"Bad tensor description: {name}")
            spans.append((offset, offset+nbytes))
        spans.sort()
        if any(b[0] < a[1] for a,b in zip(spans,spans[1:])):
            raise ValueError("Tensor regions overlap")
        self.entries = raw_tensors
        self.path = blob
        self.bytes_total = sum(entry["nbytes"] for entry in raw_tensors.values())
        # COW view is writeable to torch but edits never persist to checkpoint.
        self.mapping = np.memmap(blob, dtype=np.uint8, mode="c", shape=(blob_size,))

    def array(self, name: str) -> np.ndarray:
        entry = self.entries[name]
        return np.ndarray(tuple(entry["shape"]), dtype="<f4", buffer=self.mapping,
                          offset=entry["offset"])

    def verify(self) -> None:
        for name, entry in self.entries.items():
            digest = sha256()
            start, length = entry["offset"], entry["nbytes"]
            # Chunked to avoid materializing large tensors in Python.
            for pos in range(start, start+length, 8*1024*1024):
                end = min(start + length, pos + 8*1024*1024)
                digest.update(self.mapping[pos:end].tobytes())
            if digest.hexdigest() != entry["sha256"]:
                raise ValueError(f"Checksum mismatch: {name}")


def _fixture_weights(config: ModelConfig, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = {}
    for name, shape in config.shapes().items():
        if name.endswith("_norm"):
            array = np.ones(shape, dtype="<f4")
        else:
            array = (rng.standard_normal(shape).astype("<f4") * 0.045).astype("<f4")
        out[name] = array
    return out


def build_fixture(directory: str | Path, config: ModelConfig | None = None,
                  seed: int = 369) -> None:
    """Create a deterministic untrained model for correctness tests, NOT a chat LLM."""
    config = config or ModelConfig()
    config.validate()
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    if (target / "manifest.json").exists() or (target / "weights.bin").exists():
        raise FileExistsError("Model files already exist; refusing to overwrite")
    weights = _fixture_weights(config, seed)
    entries: dict[str, dict] = {}
    with (target / "weights.bin").open("xb") as fh:
        for name, tensor in weights.items():
            # Align to page boundaries so the OS can manage mapped pages efficiently.
            padding = (-fh.tell()) % 4096
            if padding:
                fh.write(b"\0" * padding)
            offset = fh.tell()
            raw = tensor.tobytes(order="C")
            fh.write(raw)
            entries[name] = {"dtype": "<f4", "shape": list(tensor.shape),
                             "offset": offset, "nbytes": len(raw),
                             "sha256": sha256(raw).hexdigest()}
    manifest = {"format": "PHISTREAM_V1_F32_LE", "config": asdict(config),
                "seed": seed, "provenance": "SIMULATED_UNTRAINED_FIXTURE",
                "tensors": entries}
    tmp = target / "manifest.json.tmp"
    with tmp.open("x", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, target / "manifest.json")
