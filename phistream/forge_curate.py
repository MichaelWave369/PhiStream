"""Human-in-loop local approval of each PhiOS Model Forge example.

No bulk approve, automatic training, model loading, network activity or uploads.
The original dataset stays unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Any

from .model_forge import DEFAULT_DATA, ForgeError, load_examples, review

def curate(source: Path, output: Path, choice: Callable[[dict[str, Any]], str]) -> dict:
    source = Path(source).expanduser()
    output = Path(output).expanduser()
    examples, original_sha = load_examples(source)
    target = output.resolve()
    repo = Path(__file__).resolve().parents[1]
    if output.exists() or output.is_symlink():
        raise ForgeError("Curated dataset already exists; never overwrite")
    if target == source.resolve() or target == repo or repo in target.parents:
        raise ForgeError("Private dataset output must be outside the Git checkout")
    accepted = []
    excluded = 0
    for example in examples:
        decision = choice(example)
        if decision == "quit":
            raise ForgeError("Review cancelled; nothing was written")
        if decision == "exclude":
            excluded += 1
            continue
        if decision != "approve":
            raise ForgeError("Every example needs an explicit approve/exclude decision")
        accepted.append({**example, "review_status": "human-approved"})
    train = sum(x["split"] == "train" for x in accepted)
    holdout = sum(x["split"] == "eval" for x in accepted)
    if train == 0 or holdout == 0:
        raise ForgeError("Curated examples must retain train and held-out cases")
    raw = "".join(json.dumps(e, ensure_ascii=False, separators=(",", ":")) + "\n" for e in accepted).encode("utf-8")
    if len(raw) > 2 * 1024 * 1024:
        raise ForgeError("Curated dataset too large")
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        result = review(output)
        if not result["human_review_complete"]:
            raise ForgeError("Produced dataset did not pass schema review")
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return {
        "schema": "phi.model-forge.curated-dataset.v1",
        "status": "APPROVED_DATASET_ONLY",
        "original_sha256": original_sha,
        "curated_sha256": hashlib.sha256(raw).hexdigest(),
        "approved_train": train,
        "approved_eval": holdout,
        "excluded": excluded,
        "output_path": str(output.resolve()),
        "training_performed": False,
        "quality_proven": False,
        "production_authorized": False,
    }

def operator_choice(example: dict[str, Any]) -> str:
    print("\n" + "=" * 65)
    print(f"{example['id']} | {example['split']} | {example['topic']}")
    print(f"Source: {example['source']} | Rights: {example['license']}")
    print("USER:\n" + example["messages"][0]["content"])
    print("ASSISTANT:\n" + example["messages"][1]["content"])
    while True:
        try:
            answer = input("[A]pprove / [E]xclude / [Q]uit, no save: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            return "quit"
        if answer in ("a", "approve"):
            return "approve"
        if answer in ("e", "exclude"):
            return "exclude"
        if answer in ("q", "quit"):
            return "quit"
        print("Only A, E or Q. No automatic approvals.")

def main() -> None:
    parser = argparse.ArgumentParser(description="Review every local ΦCore training example")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(curate(args.dataset, args.output, operator_choice), indent=2, sort_keys=True))
    except (ForgeError, OSError) as exc:
        parser.exit(2, f"Model Forge review refused: {exc}\n")

if __name__ == "__main__":
    main()
