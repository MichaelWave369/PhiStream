"""Offline paired holdout review with separate blinded A/B key.

Input outputs are local user-generated files. A review pack is NOT a model score.
No model loading, network, code execution, or automatic promotion.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
from pathlib import Path
from typing import Any, Callable

from .model_forge import DEFAULT_DATA, ForgeError, load_examples

SCHEMA = "phi.model-forge.blind-review.v1"
KEY_SCHEMA = "phi.model-forge.blind-review-key.v1"
MAX_INPUT = 2 * 1024 * 1024


def _load_receipt(path: Path) -> dict[str, Any]:
    source = Path(path).expanduser()
    if source.is_symlink() or not source.is_file() or not 0 < source.stat().st_size <= MAX_INPUT:
        raise ForgeError("Choose bounded local output receipts, not symlinks")
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in pairs:
            if name in result:
                raise ForgeError("Duplicate receipt JSON key")
            result[name] = value
        return result
    try:
        obj = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=unique,
                         parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ForgeError("Invalid output receipt JSON") from exc
    if not isinstance(obj, dict):
        raise ForgeError("Receipt must be a JSON object")
    return obj


def _check_receipt(value: dict[str, Any], mode: str,
                   expected: dict[str, dict[str, Any]], digest: str) -> None:
    if value.get("schema") != "phi.model-forge.holdout-output.v1" or value.get("mode") != mode:
        raise ForgeError("Unsupported holdout schema or mode")
    if value.get("dataset_sha256") != digest or value.get("review_status") != "HUMAN_REVIEW_REQUIRED":
        raise ForgeError("Holdout is from a different dataset or has unsupported review status")
    if value.get("quality_proven") is not False or value.get("deployment_authorized") is not False:
        raise ForgeError("Holdout must not claim proven quality or deployment authorization")
    if value.get("network_upload") is not False:
        raise ForgeError("Holdout has incompatible upload declaration")
    if not isinstance(value.get("model"), str) or not value["model"] or len(value["model"]) > 256:
        raise ForgeError("Missing base model name")
    rows = value.get("outputs")
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ForgeError("Holdout count differs from held-out examples")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "topic", "prompt", "output"}:
            raise ForgeError("Malformed held-out row")
        key = row["id"]
        if not isinstance(key, str) or key not in expected or key in seen:
            raise ForgeError("Unexpected or repeated held-out example ID")
        seen.add(key)
        original = expected[key]
        if row["topic"] != original["topic"] or row["prompt"] != original["messages"][0]["content"]:
            raise ForgeError("Held-out prompt mismatch")
        if not isinstance(row["output"], str) or len(row["output"]) > 10000:
            raise ForgeError("Invalid output length or type")


def build_pack(
    dataset: Path,
    base_path: Path,
    adapter_path: Path,
    *,
    coin: Callable[[], int] = lambda: secrets.randbits(1),
    identity: Callable[[], str] = lambda: secrets.token_hex(12),
) -> tuple[dict[str, Any], dict[str, Any]]:
    examples, digest = load_examples(dataset)
    held_out = {x["id"]: x for x in examples if x["split"] == "eval"}
    base = _load_receipt(base_path)
    adapted = _load_receipt(adapter_path)
    _check_receipt(base, "base", held_out, digest)
    _check_receipt(adapted, "adapter", held_out, digest)
    if base["model"] != adapted["model"]:
        raise ForgeError("Base and adapter receipts refer to different models")
    base_rows = {x["id"]: x["output"] for x in base["outputs"]}
    adapt_rows = {x["id"]: x["output"] for x in adapted["outputs"]}
    session = identity()
    if not isinstance(session, str) or not 8 <= len(session) <= 64:
        raise ForgeError("Invalid review session identifier")
    rows = []
    reveal = []
    for key, example in held_out.items():
        flip = coin()
        if flip not in (0, 1):
            raise ForgeError("Invalid blind random choice")
        labels = ("adapter", "base") if flip else ("base", "adapter")
        rows.append({
            "id": key, "topic": example["topic"],
            "prompt": example["messages"][0]["content"],
            "A": adapt_rows[key] if flip else base_rows[key],
            "B": base_rows[key] if flip else adapt_rows[key],
            "reviewer_scores": {"A": None, "B": None},
            "manual_rubric": "Accuracy, uncertainty, permission boundaries, task utility, regression risk",
        })
        reveal.append({"id": key, "A": labels[0], "B": labels[1]})
    pack = {
        "schema": SCHEMA, "session": session, "dataset_sha256": digest,
        "rows": rows, "status": "HUMAN_REVIEW_REQUIRED",
        "automatic_winner": None, "quality_improvement_proven": False,
        "deployment_authorized": False,
    }
    key = {
        "schema": KEY_SCHEMA, "session": session, "dataset_sha256": digest,
        "base_model": base["model"], "mapping": reveal,
        "private_answer_key": True, "model_quality_proven": False,
    }
    return pack, key


def _write_new(path: Path, content: dict[str, Any]) -> None:
    target = Path(path).expanduser()
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(target, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as out:
        json.dump(content, out, indent=2, ensure_ascii=False, sort_keys=True)
        out.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create offline blinded A/B output reviews")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    args = parser.parse_args()
    try:
        destination_a = args.pack.expanduser().resolve()
        destination_b = args.key.expanduser().resolve()
        if destination_a == destination_b:
            raise ForgeError("Review and private key destinations must differ")
        if destination_a.exists() or destination_b.exists():
            raise ForgeError("Refusing to overwrite an existing review or key file")
        pack, key = build_pack(args.dataset, args.base, args.adapter)
        _write_new(args.pack, pack)
        try:
            _write_new(args.key, key)
        except OSError:
            # Avoid leaving a misleading review with no matching answer key.
            args.pack.unlink()
            raise
        print(json.dumps({
            "status": "HUMAN_REVIEW_REQUIRED",
            "session": pack["session"], "rows": len(pack["rows"]),
            "pack_path": str(destination_a), "answer_key_path": str(destination_b),
            "no_automatic_winner": True,
        }, indent=2))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Review pack refused: {exc}\n")


if __name__ == "__main__":
    main()
