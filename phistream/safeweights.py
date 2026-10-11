"""PhiStream's independent, read-only local safetensors checkpoint inspector.

Bounded JSON headers, exact binary offsets, explicit model paths, and no
pickle, deserialization hooks, internet fetches or checkpoint mutations.
Inspection is NOT compatible generation or benchmark evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import struct
import sys
from typing import Any

MAX_HEADER = 16 * 1024 * 1024
MAX_INDEX = 8 * 1024 * 1024
MAX_TENSORS = 200_000
MAX_SHARDS = 512
BYTES_PER_ELEMENT = {
    "BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "I32": 4,
    "I64": 8, "F16": 2, "BF16": 2, "F32": 4, "F64": 8,
    "F8_E4M3": 1, "F8_E5M2": 1,
}
TORCH_TYPES = {
    "BOOL": "bool", "U8": "uint8", "I8": "int8",
    "I16": "int16", "I32": "int32", "I64": "int64",
    "F16": "float16", "BF16": "bfloat16", "F32": "float32",
    "F64": "float64",
}
VALID_LLM_DTYPES = {"BF16", "F16", "F32"}


def _no_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict:
    obj = {}
    for key, val in pairs:
        if key in obj:
            raise ValueError(f"Duplicate JSON key: {key}")
        obj[key] = val
    return obj


def _json_object(data: bytes, label: str) -> dict:
    try:
        parsed = json.loads(data.decode("utf-8"), object_pairs_hook=_no_duplicate_fields,
                            parse_constant=lambda val: (_ for _ in ()).throw(ValueError(
                                f"Nonfinite JSON number: {val}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON in {label}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object")
    return parsed


def _safe_basename(value: str) -> str:
    if (not isinstance(value, str) or not value.endswith(".safetensors")
            or "/" in value or "\\" in value or value in (".", "..")
            or value.startswith(".")):
        raise ValueError(f"Invalid shard basename: {value!r}")
    return value


def _load_json(path: Path, max_bytes: int) -> dict:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Missing or symlinked JSON: {path.name}")
    if path.stat().st_size > max_bytes:
        raise ValueError(f"JSON file too large: {path.name}")
    return _json_object(path.read_bytes(), path.name)


@dataclass(frozen=True)
class TensorEntry:
    name: str
    dtype: str
    shape: tuple[int, ...]
    file: str
    start: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start


class SafeTensorFile:
    """Parse a local .safetensors file without loading its weights."""

    def __init__(self, path: Path):
        if path.is_symlink() or not path.is_file() or path.suffix != ".safetensors":
            raise ValueError("Expected a regular, local .safetensors file (no symlinks)")
        self.path = path
        size = path.stat().st_size
        if size < 10:
            raise ValueError("Truncated safetensors file")
        with path.open("rb") as fh:
            header_len = struct.unpack("<Q", fh.read(8))[0]
            if header_len < 2 or header_len > MAX_HEADER or header_len > size - 8:
                raise ValueError("Invalid safetensors header length")
            header = _json_object(fh.read(header_len), path.name)
        if len(header) > MAX_TENSORS + 1:
            raise ValueError("Too many tensor entries")
        metadata = header.pop("__metadata__", {})
        if not isinstance(metadata, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items()
        ):
            raise ValueError("safetensors __metadata__ must map strings to strings")
        data_start = 8 + header_len
        payload_len = size - data_start
        entries: dict[str, TensorEntry] = {}
        ranges = []
        for name, info in header.items():
            if not isinstance(name, str) or not name or not isinstance(info, dict):
                raise ValueError("Malformed tensor entry")
            dtype, shape, offsets = info.get("dtype"), info.get("shape"), info.get("data_offsets")
            if not isinstance(dtype, str) or dtype not in BYTES_PER_ELEMENT:
                raise ValueError(f"Unsupported tensor dtype: {dtype!r}")
            if (not isinstance(shape, list) or len(shape) > 16 or any(
                type(dim) is not int or dim < 0 for dim in shape
            )):
                raise ValueError(f"Invalid shape: {name}")
            if (not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(n) is not int or n < 0 for n in offsets)):
                raise ValueError(f"Invalid tensor offsets: {name}")
            lo, hi = offsets
            expected = math.prod(shape) * BYTES_PER_ELEMENT[dtype]
            if lo > hi or hi > payload_len or hi - lo != expected:
                raise ValueError(f"Invalid tensor bounds / dtype size: {name}")
            entries[name] = TensorEntry(name, dtype, tuple(shape), path.name,
                                        data_start + lo, data_start + hi)
            ranges.append((lo, hi))
        # The binary payload must be fully described, without holes or overlaps.
        # Zero-sized tensors may share the same byte boundary.
        cursor = 0
        for lo, hi in sorted(ranges):
            if lo != cursor:
                raise ValueError("Overlapping or unindexed tensor payload bytes")
            cursor = hi
        if cursor != payload_len:
            raise ValueError("Unindexed trailing payload bytes")
        self.metadata = metadata
        self.entries = entries
        self.payload_bytes = payload_len
        self.file_size = size

    def read_tensor(self, name: str, *, max_bytes: int = 256 * 1024 * 1024):
        """Materialize ONE bounded CPU tensor with owned writable storage.

        Explicitly supported for basic numeric dtypes. It does not dispatch any
        execution or import model classes. Large tensors require an opted-in cap.
        """
        if name not in self.entries:
            raise KeyError(name)
        t = self.entries[name]
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("Invalid maximum tensor byte budget")
        if t.nbytes > max_bytes:
            raise ValueError(f"Tensor exceeds {max_bytes} byte read budget")
        if t.dtype not in TORCH_TYPES:
            raise ValueError(f"Tensor {name} has no supported PyTorch CPU type")
        if sys.byteorder != "little":
            raise RuntimeError("This reader currently requires a little-endian CPU")
        import torch  # lazy; metadata inspection itself needs no GPU libraries
        with self.path.open("rb") as fh:
            fh.seek(t.start)
            payload = fh.read(t.nbytes)
        if len(payload) != t.nbytes:
            raise ValueError("Checkpoint changed while reading a tensor")
        dtype = getattr(torch, TORCH_TYPES[t.dtype])
        if t.nbytes == 0:
            return torch.empty(t.shape, dtype=dtype, device="cpu")
        # Own bytes in a bytearray so no non-writable numpy alias reaches torch.
        result = torch.frombuffer(bytearray(payload), dtype=dtype)
        return result.reshape(t.shape)

    def read_axis0_rows(self, name: str, start: int, end: int, *,
                        max_bytes: int = 256 * 1024 * 1024):
        """Read only requested contiguous rows of a local 2D weight matrix.

        This allows embedding lookup and an LM-head projection to stream through
        large vocabulary matrices without retaining the entire matrix in RAM.
        """
        if name not in self.entries:
            raise KeyError(name)
        entry = self.entries[name]
        if len(entry.shape) != 2 or entry.dtype not in TORCH_TYPES:
            raise ValueError("Only supported 2D numeric tensors allow row reads")
        rows, columns = entry.shape
        if (type(start) is not int or type(end) is not int or not 0 <= start <= end <= rows
                or type(max_bytes) is not int or max_bytes < 0):
            raise ValueError("Invalid row slice or byte budget")
        length = (end - start) * columns * BYTES_PER_ELEMENT[entry.dtype]
        if length > max_bytes:
            raise ValueError("Requested rows exceed configured tensor read budget")
        if sys.byteorder != "little":
            raise RuntimeError("Row reader requires little-endian CPU")
        import torch
        dtype = getattr(torch, TORCH_TYPES[entry.dtype])
        if length == 0:
            return torch.empty((end - start, columns), dtype=dtype)
        with self.path.open("rb") as fh:
            fh.seek(entry.start + start * columns * BYTES_PER_ELEMENT[entry.dtype])
            data = fh.read(length)
        if len(data) != length:
            raise ValueError("Checkpoint changed during row read")
        return torch.frombuffer(bytearray(data), dtype=dtype).reshape(end - start, columns)

    def fingerprint(self, name: str, *, chunk_bytes: int = 4 * 1024 * 1024) -> str:
        """Optional full-tensor hash, streamed in bounded chunks."""
        t = self.entries[name]
        result = sha256()
        with self.path.open("rb") as fh:
            fh.seek(t.start)
            remaining = t.nbytes
            while remaining:
                data = fh.read(min(chunk_bytes, remaining))
                if not data:
                    raise ValueError("Truncated tensor during hash verification")
                result.update(data)
                remaining -= len(data)
        return result.hexdigest()


class LocalCheckpoint:
    """Resolve one explicit single/sharded safetensors checkpoint, locally."""

    def __init__(self, location: str | Path):
        source = Path(location)
        self.base = source if source.is_dir() else source.parent
        if self.base.is_symlink():
            raise ValueError("Checkpoint directory symlinks are not permitted")
        if not self.base.is_dir():
            raise ValueError("Checkpoint directory does not exist")
        self.config = None
        if (self.base / "config.json").exists():
            self.config = _load_json(self.base / "config.json", MAX_INDEX)
        index = self.base / "model.safetensors.index.json"
        if source.is_dir() and index.exists():
            index_data = _load_json(index, MAX_INDEX)
            mapping = index_data.get("weight_map")
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError("Index must include a non-empty weight_map")
            if len(mapping) > MAX_TENSORS:
                raise ValueError("Too many indexed tensors")
            if any(not isinstance(k, str) or not k or not isinstance(v, str)
                   for k, v in mapping.items()):
                raise ValueError("Invalid index weight map")
            names = sorted({_safe_basename(v) for v in mapping.values()})
        else:
            file = source / "model.safetensors" if source.is_dir() else source
            if file.suffix != ".safetensors":
                raise ValueError("Choose a local safetensors file or indexed checkpoint directory")
            names = [_safe_basename(file.name)]
            mapping = None
        if len(names) > MAX_SHARDS:
            raise ValueError("Too many weight files")
        self.files = {}
        self.entries = {}
        for name in names:
            path = self.base / name
            part = SafeTensorFile(path)
            self.files[name] = part
            for tensor, entry in part.entries.items():
                if tensor in self.entries:
                    raise ValueError(f"Duplicate tensor between shards: {tensor}")
                self.entries[tensor] = entry
        if mapping is not None:
            if set(mapping) != set(self.entries):
                raise ValueError("Index tensor map does not match shard contents")
            for tensor, name in mapping.items():
                if self.entries[tensor].file != name:
                    raise ValueError(f"Index maps tensor to wrong shard: {tensor}")
        if len(self.entries) > MAX_TENSORS:
            raise ValueError("Too many tensors")
        self.payload_bytes = sum(part.payload_bytes for part in self.files.values())

    def tensor(self, name: str, *, max_bytes: int = 256 * 1024 * 1024):
        return self.files[self.entries[name].file].read_tensor(name, max_bytes=max_bytes)

    def inspect(self, *, llama_layout: bool = False, hash_all: bool = False) -> dict:
        result = {
            "schema": "PHISTREAM_CHECKPOINT_INSPECTION_V1",
            "provenance": "LOCAL_METADATA_INSPECTED_NOT_BENCHMARKED",
            "status": "STRUCTURE_VALID",
            "format": "SAFETENSORS",
            "file_count": len(self.files),
            "tensor_count": len(self.entries),
            "weight_bytes": sum(t.nbytes for t in self.entries.values()),
            "shard_names": sorted(self.files),
            "dtypes": sorted({t.dtype for t in self.entries.values()}),
            "config_model_type": self.config.get("model_type") if self.config else None,
            "llama_layout": None,
            "sha256": None,
            "warnings": [
                "Local file/header validation only. Not a compatibility, correctness, performance or GPU inference proof.",
                "Licenses and tokenizer are not inspected. No checkpoint weights are redistributed.",
            ],
        }
        if llama_layout:
            result["llama_layout"] = self.validate_llama_layout()
            if not result["llama_layout"]["passed"]:
                result["status"] = "LAYOUT_MISMATCH"
        if hash_all:
            result["sha256"] = {
                name: self.files[t.file].fingerprint(name)
                for name, t in sorted(self.entries.items())
            }
        return result

    def validate_llama_layout(self) -> dict:
        """Shape-only preflight for the canonical untied/tied Llama decoder.

        Explicitly excludes quantized, MoE, tensor-parallel, custom kernels,
        bias-heavy or variant architecture support.
        """
        c = self.config
        if not isinstance(c, dict) or c.get("model_type") != "llama":
            return {"passed": False, "reason": "Requires local config.json with model_type llama"}
        keys = ("vocab_size", "hidden_size", "num_hidden_layers",
                "num_attention_heads", "intermediate_size")
        if any(type(c.get(k)) is not int or c[k] < 1 for k in keys):
            return {"passed": False, "reason": "Invalid or absent required Llama dimensions"}
        d, nh, nl, ff, vocab = (c["hidden_size"], c["num_attention_heads"],
                                 c["num_hidden_layers"], c["intermediate_size"], c["vocab_size"])
        kv = c.get("num_key_value_heads", nh)
        if (type(kv) is not int or kv < 1 or d % nh or nh % kv
                or (d // nh) % 2 or nl > 10000):
            return {"passed": False, "reason": "Invalid Llama grouped-query / RoPE dimensions"}
        if type(c.get("tie_word_embeddings", False)) is not bool:
            return {"passed": False, "reason": "Invalid embedding tie flag"}
        kvdim = kv * (d // nh)
        expected = {"model.embed_tokens.weight": (vocab, d), "model.norm.weight": (d,)}
        if not c.get("tie_word_embeddings", False):
            expected["lm_head.weight"] = (vocab, d)
        for i in range(nl):
            p = f"model.layers.{i}."
            expected.update({
                p + "input_layernorm.weight": (d,),
                p + "post_attention_layernorm.weight": (d,),
                p + "self_attn.q_proj.weight": (d, d),
                p + "self_attn.k_proj.weight": (kvdim, d),
                p + "self_attn.v_proj.weight": (kvdim, d),
                p + "self_attn.o_proj.weight": (d, d),
                p + "mlp.gate_proj.weight": (ff, d),
                p + "mlp.up_proj.weight": (ff, d),
                p + "mlp.down_proj.weight": (d, ff),
            })
        missing = [name for name in expected if name not in self.entries]
        mismatched = [name for name, shape in expected.items()
                      if name in self.entries and (self.entries[name].shape != shape
                      or self.entries[name].dtype not in VALID_LLM_DTYPES)]
        extras = len(set(self.entries) - set(expected))
        return {
            "passed": not missing and not mismatched,
            "reason": "Canonical projection shapes checked only; no inference compatibility established",
            "layers": nl,
            "expected_core_tensors": len(expected),
            "extra_tensors": extras,
            "missing_count": len(missing),
            "mismatch_count": len(mismatched),
            "sample_missing": missing[:8],
            "sample_mismatched": mismatched[:8],
        }
