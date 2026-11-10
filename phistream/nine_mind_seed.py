"""Nine Mind R0.4: offline reviewed-method seed validator and safe Forge export.

Original project-authored fixtures only; NOT a trained or evaluated model.
Exports unapproved per-specialist examples to existing Model Forge schema,
which independently refuses training until each sample is human-approved.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re

from .model_atlas import CATALOG_PATH, read_json, validate_catalog
from .model_forge import ForgeError, load_examples

SCHEMA = "phistream.nine-mind.method-example.v0.4"
REPORT_SCHEMA = "phistream.nine-mind.seed-review.v0.4"
SEED = Path(__file__).resolve().parents[1] / "nine_mind" / "method_seed_r04.jsonl"
REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 256 * 1024
MAX_LINE = 12000
RECORD_KEYS = {"schema", "id", "profile_id", "council", "split", "task_family",
               "source", "license", "review_status", "prompt",
               "reference_response", "rubric"}
EXACT_SPLITS = {"train": 2, "eval": 1}
PROFILES = 9
ROWS = 27
_FAMILY = re.compile(r"[a-z][a-z0-9-]{3,60}\Z")
_ID = re.compile(r"nine-[0-9]{3}\Z")
TOPIC = {
    "tesla": "hardware", "da-vinci": "hardware", "feynman": "performance",
    "escher": "style", "dali": "style", "picasso": "style",
    "bach": "style", "beethoven": "style", "vivaldi": "style",
}


class SeedError(ValueError):
    """Nine Mind seed data does not meet the offline research contract."""


def _unique_json(pairs):
    doc = {}
    for k, v in pairs:
        if k in doc:
            raise SeedError("duplicate JSON object key")
        doc[k] = v
    return doc


def _load_seed(path: Path):
    path = Path(path).expanduser()
    if path.is_symlink() or not path.is_file():
        raise SeedError("select a local regular JSONL seed, not a symlink")
    if not 0 < path.stat().st_size <= MAX_BYTES:
        raise SeedError("seed file must be between 1 byte and 256 KiB")
    data = path.read_bytes()
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise SeedError("seed must be UTF-8") from exc
    examples = []
    for i, line in enumerate(lines, 1):
        if not line.strip() or len(line) > MAX_LINE:
            raise SeedError(f"blank or oversized row at {i}")
        try:
            row = json.loads(line, object_pairs_hook=_unique_json,
                             parse_constant=lambda _: (_ for _ in ()).throw(SeedError("nonfinite JSON")))
        except (ValueError, TypeError) as exc:
            raise SeedError(f"invalid row {i}") from exc
        if not isinstance(row, dict) or set(row) != RECORD_KEYS or row["schema"] != SCHEMA:
            raise SeedError(f"unexpected row schema at {i}")
        examples.append(row)
    return examples, hashlib.sha256(data).hexdigest()


def validate(path: Path = SEED, *, catalog_path: Path = CATALOG_PATH):
    catalog = read_json(catalog_path)
    validate_catalog(catalog)
    profiles = {item["id"]: item["council"] for item in catalog["profiles"]}
    if set(profiles) != set(TOPIC):
        raise SeedError("specialist identities disagree with Model Atlas")
    records, digest = _load_seed(path)
    if len(records) != ROWS:
        raise SeedError("R0.4 requires exactly 27 seed rows")
    counts = Counter()
    ids, families, prompts = set(), set(), set()
    for row in records:
        identity, profile, split = row["id"], row["profile_id"], row["split"]
        if not isinstance(identity, str) or not _ID.fullmatch(identity) or identity in ids:
            raise SeedError("invalid or duplicate Nine Mind record ID")
        ids.add(identity)
        if not isinstance(profile, str) or profile not in profiles or row["council"] != profiles[profile]:
            raise SeedError("unrecognized or mismatched specialist council")
        if split not in EXACT_SPLITS:
            raise SeedError("invalid split")
        counts[(profile, split)] += 1
        family = row["task_family"]
        if not isinstance(family, str) or not _FAMILY.fullmatch(family) or family in families:
            raise SeedError("duplicate or invalid scenario family, possible train/eval leakage")
        families.add(family)
        if (row["source"], row["license"], row["review_status"]) != (
            "project-authored-fixture", "CC0-1.0", "seed-unreviewed"
        ):
            raise SeedError("only unreviewed original CC0 project fixtures allowed")
        prompt, response = row["prompt"], row["reference_response"]
        if (not isinstance(prompt, str) or not 20 <= len(prompt) <= 2000
                or not isinstance(response, str) or not 20 <= len(response) <= 4000):
            raise SeedError("invalid prompt or response length")
        normalized = " ".join(prompt.casefold().split())
        if normalized in prompts:
            raise SeedError("duplicate prompt, possible split leakage")
        prompts.add(normalized)
        rubric = row["rubric"]
        if (not isinstance(rubric, list) or not 2 <= len(rubric) <= 4
                or any(not isinstance(item, str) or not 10 <= len(item) <= 180 for item in rubric)
                or len(set(rubric)) != len(rubric)):
            raise SeedError("invalid evaluation rubric")
    if any(counts[(profile, split)] != number
           for profile in profiles for split, number in EXACT_SPLITS.items()):
        raise SeedError("each specialist must have two training and one independent held-out example")
    return {
        "schema": REPORT_SCHEMA, "status": "UNREVIEWED_SEED_FIXTURES_ONLY",
        "seed_sha256": digest, "specialist_count": PROFILES,
        "train_count": PROFILES * EXACT_SPLITS["train"],
        "eval_count": PROFILES * EXACT_SPLITS["eval"],
        "councils": {name: sum(1 for c in profiles.values() if c == name)
                     for name in ("invention", "art", "music")},
        "source_class": "ORIGINAL_CC0_PROJECT_FIXTURES_NOT_HISTORICAL_TEXT",
        "model_loaded": False, "training_started": False,
        "adapter_created": False, "model_quality_proven": False,
        "approval_complete": False, "deployment_authorized": False,
    }


def forge_rows(profile: str, path: Path = SEED):
    """Translate ONE specialist to existing Forge data format with approvals locked off."""
    validate(path)
    if profile not in TOPIC:
        raise SeedError("unknown specialist")
    records, _ = _load_seed(path)
    selected = [x for x in records if x["profile_id"] == profile]
    rows = []
    for x in selected:
        ident = int(x["id"].split("-")[-1])
        rows.append({
            "schema": "phi.model-forge.example.v1",
            "id": f"phi-{2000 + ident}",
            "split": x["split"],
            "topic": TOPIC[profile],
            "source": "project-authored-fixture",
            "license": "CC0-1.0",
            "review_status": "seed-unreviewed",
            "messages": [
                {"role": "user", "content": x["prompt"]},
                {"role": "assistant", "content": x["reference_response"]},
            ],
        })
    if len(rows) != 3:
        raise SeedError("expected exactly three specialist rows")
    return rows


def export(profile: str, output: Path, *, seed: Path = SEED):
    rows = forge_rows(profile, seed)
    destination = Path(output).expanduser()
    parent = destination.parent.resolve(strict=True)
    full = (parent / destination.name).resolve(strict=False)
    if full == REPO_ROOT or REPO_ROOT in full.parents:
        raise SeedError("output must stay outside the Git checkout to avoid public accidental disclosure")
    if destination.is_symlink() or destination.exists():
        raise SeedError("output already exists; never overwrite operator data")
    if not parent.is_dir():
        raise SeedError("output parent must be an existing directory")
    payload = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)
    # Exclusive creation. A preexisting symlink/file is not opened for writing.
    try:
        with full.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
        try:
            validated, _ = load_examples(full)
            if len(validated) != 3 or any(x["review_status"] != "seed-unreviewed" for x in validated):
                raise SeedError("export failed existing Forge contract")
        except BaseException:
            full.unlink(missing_ok=True)
            raise
    except FileExistsError as exc:
        raise SeedError("output already exists") from exc
    return {
        "schema": "phistream.nine-mind.forge-export.v0.4",
        "status": "EXPORTED_UNREVIEWED_FIXTURES_REQUIRES_INDIVIDUAL_HUMAN_REVIEW",
        "profile_id": profile, "examples": len(rows),
        "dataset_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "model_loaded": False, "training_started": False,
        "approval_complete": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, default=SEED)
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--export-profile", choices=sorted(TOPIC))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.export_profile:
            if not args.output or args.catalog != CATALOG_PATH:
                raise SeedError("export requires --output and canonical catalog")
            report = export(args.export_profile, args.output, seed=args.seed)
        else:
            if args.output:
                raise SeedError("--output is valid only with --export-profile")
            report = validate(args.seed, catalog_path=args.catalog)
        print(json.dumps(report, indent=2, sort_keys=True))
    except (SeedError, AtlasError, ForgeError, OSError, ValueError) as exc:
        parser.exit(2, "Nine Mind HELD: " + str(exc) + "\n")


if __name__ == "__main__":
    main()
