"""PhiStream R0.1: offline, reversible, SHA-bound safetensors scalar edits.

This is a byte-patching research utility, NOT an automatic capability optimizer.
No inference, training, model downloads, or edits to the source checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile

PLAN_SCHEMA = "PHISTREAM_WEIGHT_SURGERY_PLAN_R01"
RECEIPT_SCHEMA = "PHISTREAM_WEIGHT_SURGERY_RECEIPT_R01"
MAX_HEADER = 8 * 1024 * 1024
MAX_DOCUMENT = 1024 * 1024
MAX_EDITS = 256
DTYPE_BYTES = {"F32": 4, "F16": 2, "BF16": 2}


class SurgeryError(ValueError):
    """Explicitly rejected checkpoint, plan, or output."""


def _fail(message: str) -> None:
    raise SurgeryError(message)


def _no_duplicate_keys(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            _fail("duplicate JSON key: " + key)
        obj[key] = value
    return obj


def _load_json(path: Path):
    if path.stat().st_size > MAX_DOCUMENT:
        _fail("JSON document exceeds maximum size")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(obj) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")


def _exclusive_json(path: Path, obj) -> None:
    data = _json_bytes(obj)
    # Exclusive creation, including no following existing symlinks.
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _real_number(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(name + " must be a nonnegative integer")
    return value


def _tensor_table(path: Path):
    length = path.stat().st_size
    if length < 10:
        _fail("truncated safetensors checkpoint")
    with path.open("rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        if header_size < 2 or header_size > MAX_HEADER or header_size > length - 8:
            _fail("invalid or oversized safetensors header")
        header = json.loads(f.read(header_size).decode("utf-8"), object_pairs_hook=_no_duplicate_keys)
    if not isinstance(header, dict) or not header:
        _fail("invalid safetensors header object")
    data_size = length - 8 - header_size
    spans = []
    result = {}
    for name, entry in header.items():
        if name == "__metadata__":
            if not isinstance(entry, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in entry.items()):
                _fail("invalid safetensors metadata")
            continue
        if not isinstance(entry, dict) or set(entry) != {"dtype", "shape", "data_offsets"}:
            _fail("invalid tensor descriptor")
        dtype = entry["dtype"]
        if dtype not in DTYPE_BYTES:
            _fail("R0.1 supports only F32, F16 and BF16 tensors")
        shape, offsets = entry["shape"], entry["data_offsets"]
        if not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
            _fail("invalid tensor shape or offsets")
        count = 1
        for dim in shape:
            _real_number(dim, "tensor dimension")
            count *= dim
            if count > 2**63:
                _fail("tensor element count overflow")
        start, end = (_real_number(i, "tensor offset") for i in offsets)
        if end < start or end > data_size or end - start != count * DTYPE_BYTES[dtype]:
            _fail("tensor offsets/shape disagree with checkpoint size")
        spans.append((start, end))
        result[name] = {"dtype": dtype, "shape": shape, "count": count, "offset": 8 + header_size + start}
    if not result:
        _fail("checkpoint contains no tensors")
    previous = 0
    for start, end in sorted(spans):
        if start < previous:
            _fail("overlapping tensor spans")
        previous = end
    return result


def _scalar(table, tensor: str, index: int):
    if tensor not in table:
        _fail("tensor does not exist: " + tensor)
    info = table[tensor]
    _real_number(index, "flat index")
    if index >= info["count"]:
        _fail("tensor index out of bounds")
    return info["offset"] + index * DTYPE_BYTES[info["dtype"]], info["dtype"]


def _decode_scalar(raw: bytes, dtype: str) -> float:
    if dtype == "F32":
        return struct.unpack("<f", raw)[0]
    if dtype == "F16":
        return struct.unpack("<e", raw)[0]
    return struct.unpack("<f", b"\0\0" + raw)[0]


def _encode_scalar(value: float, dtype: str) -> bytes:
    if not math.isfinite(value):
        _fail("non-finite replacement value")
    try:
        if dtype == "F32":
            raw = struct.pack("<f", value)
        elif dtype == "F16":
            raw = struct.pack("<e", value)
        else:
            bits = struct.unpack("<I", struct.pack("<f", value))[0]
            bits += 0x7FFF + ((bits >> 16) & 1)  # BF16 ties-to-even
            raw = struct.pack("<H", (bits >> 16) & 0xFFFF)
    except (OverflowError, struct.error):
        _fail("replacement cannot be represented in tensor dtype")
    if not math.isfinite(_decode_scalar(raw, dtype)):
        _fail("replacement rounds to non-finite value")
    return raw


def _validate_plan(plan, checkpoint: Path, table, actual_sha: str):
    if not isinstance(plan, dict) or set(plan) != {"schema", "checkpoint_sha256", "checkpoint_size", "edits"}:
        _fail("invalid plan structure")
    if plan["schema"] != PLAN_SCHEMA:
        _fail("unsupported plan schema")
    if plan["checkpoint_sha256"] != actual_sha or plan["checkpoint_size"] != checkpoint.stat().st_size:
        _fail("baseline checkpoint SHA-256 or size mismatch")
    edits = plan["edits"]
    if not isinstance(edits, list) or not 1 <= len(edits) <= MAX_EDITS:
        _fail("edit count outside R0.1 limit")
    patches = []
    seen = set()
    with checkpoint.open("rb") as f:
        for e in edits:
            if not isinstance(e, dict) or set(e) != {"tensor", "index", "expected_hex", "new_hex"}:
                _fail("invalid edit entry")
            if not isinstance(e["tensor"], str):
                _fail("tensor name must be string")
            offset, dtype = _scalar(table, e["tensor"], e["index"])
            key = (e["tensor"], e["index"])
            if key in seen:
                _fail("duplicate tensor/index edit")
            seen.add(key)
            size = DTYPE_BYTES[dtype]
            for field in ("expected_hex", "new_hex"):
                v = e[field]
                if not isinstance(v, str) or len(v) != size * 2 or any(c not in "0123456789abcdef" for c in v):
                    _fail("raw weight bytes must be lowercase, exact-length hex")
                if not math.isfinite(_decode_scalar(bytes.fromhex(v), dtype)):
                    _fail("cannot edit non-finite scalar")
            if e["expected_hex"] == e["new_hex"]:
                _fail("no-op weight edits are not allowed")
            f.seek(offset)
            if f.read(size) != bytes.fromhex(e["expected_hex"]):
                _fail("expected scalar bytes do not match baseline")
            patches.append((offset, bytes.fromhex(e["new_hex"])))
    return patches


def _write_patched(src: Path, dest: Path, patches, expected_sha=None) -> str:
    if src.resolve() == dest.resolve() or os.path.lexists(dest):
        _fail("destination must be a new distinct file")
    if not dest.parent.is_dir():
        _fail("destination parent directory does not exist")
    fd, tmpname = tempfile.mkstemp(prefix=".weight-surgery-", suffix=".tmp", dir=str(dest.parent))
    tmp = Path(tmpname)
    try:
        with src.open("rb") as source, os.fdopen(fd, "w+b") as target:
            shutil.copyfileobj(source, target, 8 * 1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        if expected_sha is not None and _sha256(tmp) != expected_sha:
            _fail("source changed during checkpoint copy")
        with tmp.open("r+b") as target:
            for offset, newbytes in patches:
                target.seek(offset)
                target.write(newbytes)
            target.flush()
            os.fsync(target.fileno())
        result_sha = _sha256(tmp)
        os.link(tmp, dest)  # exclusive atomic publication; no clobbering
        return result_sha
    finally:
        tmp.unlink(missing_ok=True)


def draft(checkpoint: Path, tensor: str, index: int, value: float, output: Path):
    table = _tensor_table(checkpoint)
    offset, dtype = _scalar(table, tensor, index)
    with checkpoint.open("rb") as f:
        f.seek(offset)
        original = f.read(DTYPE_BYTES[dtype])
    new = _encode_scalar(value, dtype)
    plan = {"schema": PLAN_SCHEMA, "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_size": checkpoint.stat().st_size,
            "edits": [{"tensor": tensor, "index": index,
                       "expected_hex": original.hex(), "new_hex": new.hex()}]}
    _validate_plan(plan, checkpoint, table, plan["checkpoint_sha256"])
    _exclusive_json(output, plan)
    return plan


def apply(checkpoint: Path, plan_path: Path, output: Path, receipt_path: Path):
    if output.resolve() == receipt_path.resolve() or os.path.lexists(receipt_path):
        _fail("receipt must be a distinct new file")
    table = _tensor_table(checkpoint)
    actual_sha = _sha256(checkpoint)
    plan = _load_json(plan_path)
    patches = _validate_plan(plan, checkpoint, table, actual_sha)
    patched_sha = _write_patched(checkpoint, output, patches, expected_sha=actual_sha)
    receipt = {"schema": RECEIPT_SCHEMA, "plan": plan,
               "plan_sha256": hashlib.sha256(_json_bytes(plan)).hexdigest(),
               "baseline_sha256": actual_sha, "patched_sha256": patched_sha,
               "edit_count": len(patches), "authority": "LOCAL_UNVERIFIED"}
    try:
        _exclusive_json(receipt_path, receipt)
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return receipt


def _read_receipt(path: Path):
    receipt = _load_json(path)
    if not isinstance(receipt, dict) or set(receipt) != {"schema", "plan", "plan_sha256", "baseline_sha256", "patched_sha256", "edit_count", "authority"}:
        _fail("malformed receipt")
    if receipt["schema"] != RECEIPT_SCHEMA or receipt["authority"] != "LOCAL_UNVERIFIED":
        _fail("unsupported receipt")
    plan = receipt["plan"]
    if not isinstance(plan, dict) or receipt["plan_sha256"] != hashlib.sha256(_json_bytes(plan)).hexdigest():
        _fail("plan digest mismatch")
    if receipt["baseline_sha256"] != plan.get("checkpoint_sha256") or receipt["edit_count"] != len(plan.get("edits", [])):
        _fail("receipt metadata mismatch")
    return receipt


def verify(patched: Path, receipt_path: Path):
    receipt = _read_receipt(receipt_path)
    actual_sha = _sha256(patched)
    if actual_sha != receipt["patched_sha256"] or patched.stat().st_size != receipt["plan"]["checkpoint_size"]:
        _fail("patched checkpoint hash or size mismatch")
    table = _tensor_table(patched)
    inverse = {**receipt["plan"], "checkpoint_sha256": actual_sha,
               "edits": [{**e, "expected_hex": e["new_hex"], "new_hex": e["expected_hex"]}
                         for e in receipt["plan"]["edits"]]}
    patches = _validate_plan(inverse, patched, table, actual_sha)
    return receipt, patches


def revert(patched: Path, receipt_path: Path, output: Path):
    receipt, patches = verify(patched, receipt_path)
    result_sha = _write_patched(patched, output, patches, expected_sha=receipt["patched_sha256"])
    if result_sha != receipt["baseline_sha256"]:
        output.unlink(missing_ok=True)
        _fail("restored checkpoint differs from original baseline")
    return {"status": "RESTORED_BYTE_IDENTICAL", "sha256": result_sha, "authority": "LOCAL_UNVERIFIED"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    inspect = subs.add_parser("inspect", help="offline metadata and optional scalar inspection")
    inspect.add_argument("checkpoint", type=Path)
    inspect.add_argument("--tensor")
    inspect.add_argument("--index", type=int, default=0)
    dr = subs.add_parser("draft", help="draft one SHA-bound scalar edit; never changes weights")
    dr.add_argument("checkpoint", type=Path)
    dr.add_argument("--tensor", required=True)
    dr.add_argument("--index", type=int, required=True)
    dr.add_argument("--new-value", required=True, type=float)
    dr.add_argument("--output", required=True, type=Path)
    ap = subs.add_parser("apply", help="write a new patched checkpoint plus receipt")
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--plan", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--receipt", required=True, type=Path)
    ve = subs.add_parser("verify", help="verify exact patched checkpoint and receipt")
    ve.add_argument("checkpoint", type=Path)
    ve.add_argument("--receipt", required=True, type=Path)
    re = subs.add_parser("revert", help="reconstruct original checkpoint to new file")
    re.add_argument("checkpoint", type=Path)
    re.add_argument("--receipt", required=True, type=Path)
    re.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            table = _tensor_table(args.checkpoint)
            result = {"sha256": _sha256(args.checkpoint), "tensor_count": len(table),
                      "tensors": {k: {key: v for key, v in entry.items() if key != "offset"} for k, entry in table.items()}}
            if args.tensor is not None:
                offset, dtype = _scalar(table, args.tensor, args.index)
                with args.checkpoint.open("rb") as f:
                    f.seek(offset)
                    raw = f.read(DTYPE_BYTES[dtype])
                result["scalar"] = {"tensor": args.tensor, "index": args.index,
                                    "raw_hex": raw.hex(), "value": _decode_scalar(raw, dtype)}
        elif args.command == "draft":
            result = draft(args.checkpoint, args.tensor, args.index, args.new_value, args.output)
        elif args.command == "apply":
            result = apply(args.checkpoint, args.plan, args.output, args.receipt)
        elif args.command == "verify":
            receipt, _ = verify(args.checkpoint, args.receipt)
            result = {"status": "VERIFIED_BYTES_NOT_QUALITY", "sha256": receipt["patched_sha256"], "edit_count": receipt["edit_count"]}
        else:
            result = revert(args.checkpoint, args.receipt, args.output)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (SurgeryError, OSError, ValueError, UnicodeError) as exc:
        print("weight surgery rejected: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
