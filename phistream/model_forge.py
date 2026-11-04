"""Pure-Python schema gate for an optional, operator-run PhiOS model adapter lab.

No model loading, network access, code execution, or training happens here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

SCHEMA = "phi.model-forge.example.v1"
RECEIPT = "phi.model-forge.dataset-review.v1"
EXAMPLE_KEYS = {"schema", "id", "split", "topic", "source", "license", "review_status", "messages"}
ALLOWED_SPLITS = {"train", "eval"}
ALLOWED_TOPICS = {
    "authority", "hardware", "performance", "models", "security", "source",
    "xr", "deployment", "style",
}
MAX_BYTES = 2 * 1024 * 1024
DEFAULT_DATA = Path(__file__).resolve().parents[1] / "model_forge" / "seed_examples.jsonl"
_ID = re.compile(r"phi-[0-9]{3,6}\Z")


class ForgeError(ValueError):
    """Malformed or unreviewed model training candidate."""


def load_examples(path: Path) -> tuple[list[dict[str, Any]], str]:
    path = Path(path).expanduser()
    if path.is_symlink() or not path.is_file():
        raise ForgeError("Choose a regular existing local JSONL file, not a symlink")
    size = path.stat().st_size
    if size == 0 or size > MAX_BYTES:
        raise ForgeError("Dataset must be 1 byte to 2 MiB")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ForgeError("Dataset is not UTF-8") from exc
    examples: list[dict[str, Any]] = []
    ids: set[str] = set()
    prompts: set[str] = set()
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise ForgeError(f"Blank example at line {lineno}")
        if len(line) > 12000:
            raise ForgeError(f"Example too large at line {lineno}")
        try:
            entry = json.loads(line, parse_constant=lambda _x: (_ for _ in ()).throw(ValueError("nonfinite number")))
        except (ValueError, json.JSONDecodeError) as exc:
            raise ForgeError(f"Invalid JSON at line {lineno}") from exc
        if not isinstance(entry, dict) or set(entry) != EXAMPLE_KEYS or entry.get("schema") != SCHEMA:
            raise ForgeError(f"Unexpected schema or fields at line {lineno}")
        eid = entry["id"]
        if not isinstance(eid, str) or not _ID.fullmatch(eid) or eid in ids:
            raise ForgeError(f"Invalid or duplicate example id at line {lineno}")
        ids.add(eid)
        if entry["split"] not in ALLOWED_SPLITS or entry["topic"] not in ALLOWED_TOPICS:
            raise ForgeError(f"Unsupported split or topic at line {lineno}")
        if entry["source"] != "project-authored-fixture" or entry["license"] != "CC0-1.0":
            raise ForgeError(f"Example lacks reviewed source/rights provenance at line {lineno}")
        if entry["review_status"] not in ("seed-unreviewed", "human-approved"):
            raise ForgeError(f"Invalid review status at line {lineno}")
        messages = entry["messages"]
        if not isinstance(messages, list) or len(messages) != 2:
            raise ForgeError(f"Expected one user and one assistant message at line {lineno}")
        for message, role, limit in zip(messages, ("user", "assistant"), (2000, 4000)):
            if (not isinstance(message, dict) or set(message) != {"role", "content"}
                    or message["role"] != role or not isinstance(message["content"], str)
                    or not message["content"].strip() or len(message["content"]) > limit):
                raise ForgeError(f"Invalid message structure at line {lineno}")
        key = " ".join(messages[0]["content"].casefold().split())
        if key in prompts:
            raise ForgeError(f"Duplicate prompt/leakage across dataset splits at line {lineno}")
        prompts.add(key)
        examples.append(entry)
    if len(examples) > 5000:
        raise ForgeError("Candidate dataset contains over 5000 examples")
    if not any(x["split"] == "train" for x in examples) or not any(x["split"] == "eval" for x in examples):
        raise ForgeError("Need nonempty disjoint train and eval sets")
    return examples, hashlib.sha256(raw).hexdigest()


def review(path: Path = DEFAULT_DATA) -> dict[str, Any]:
    examples, digest = load_examples(path)
    train = [x for x in examples if x["split"] == "train"]
    evaluation = [x for x in examples if x["split"] == "eval"]
    reviewed = all(x["review_status"] == "human-approved" for x in examples)
    return {
        "schema": RECEIPT,
        "dataset_sha256": digest,
        "train_examples": len(train),
        "eval_examples": len(evaluation),
        "topics": sorted({x["topic"] for x in examples}),
        "human_review_complete": reviewed,
        "status": "REVIEW_REQUIRED" if not reviewed else "APPROVED_DATASET_ONLY",
        "training_run": False,
        "gpu_measured": False,
        "model_quality_proven": False,
        "production_ready": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Φ Model Forge: offline dataset review, no training")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    try:
        print(json.dumps(review(args.dataset), indent=2, sort_keys=True))
    except (ForgeError, OSError) as exc:
        parser.exit(2, f"Model Forge dataset held: {exc}\n")


if __name__ == "__main__":
    main()
