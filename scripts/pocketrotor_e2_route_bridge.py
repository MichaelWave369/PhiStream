"""PocketRotor E2-R: read-only native-router event capture to E2 replay bridge.

Standard library only. No models, transformers, CUDA, token text, downloads or
network. A runtime must supply its *own already selected* expert IDs; this
adapter does not guess or re-run its expert router. Source claims are not
independently authenticated.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

from scripts.pocketrotor_e2 import SCHEMA as E2_SCHEMA, _digest as e2_digest, validate as validate_e2

CAPTURE_SCHEMA = "PHISTREAM_POCKETROTOR_E2_NATIVE_ROUTE_CAPTURE_V1"
BRIDGE_SCHEMA = "PHISTREAM_POCKETROTOR_E2_ROUTE_BRIDGE_V1"
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_EVENTS = 16384
MAX_LAYERS = 128
NAME = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SOURCE_FIELDS = {
    "kind", "runtime", "model_id", "checkpoint_sha256",
    "route_semantics", "token_text_recorded",
}
CAPTURE_FIELDS = {"schema", "expert_count", "source", "events"}
EVENT_FIELDS = {"split", "layer", "step", "experts"}


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _no_duplicate(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        _require(key not in result, "Duplicate JSON key in route capture")
        result[key] = value
    return result


def _read(path: str | Path) -> dict:
    p = Path(path)
    _require(p.is_file() and not p.is_symlink()
             and 0 < p.stat().st_size <= MAX_FILE_BYTES,
             "Input must be a bounded regular local file, not a symlink")
    with p.open(encoding="utf-8") as stream:
        return json.load(
            stream, object_pairs_hook=_no_duplicate,
            parse_constant=lambda x: (_ for _ in ()).throw(
                ValueError("Nonfinite JSON value in route capture")),
        )


def _file_sha(path: str | Path) -> str:
    p = Path(path)
    _require(p.is_file() and not p.is_symlink(), "Input file missing or symlinked")
    before = p.stat()
    h = sha256()
    with p.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    after = p.stat()
    _require(
        (before.st_ino, before.st_mtime_ns, before.st_size)
        == (after.st_ino, after.st_mtime_ns, after.st_size),
        "Capture changed during hashing",
    )
    return h.hexdigest()


def _source(source: object) -> dict:
    _require(isinstance(source, dict) and set(source) == SOURCE_FIELDS,
             "Source requires exact bounded provenance fields")
    _require(source["kind"] in ("synthetic", "operator_asserted_native"),
             "Source kind must clearly distinguish synthetic from asserted native")
    for key in ("runtime", "model_id"):
        value = source[key]
        _require(isinstance(value, str) and 1 <= len(value) <= 120
                 and all(32 <= ord(c) <= 126 for c in value),
                 "Source runtime/model_id must be short printable ASCII")
    _require(isinstance(source["checkpoint_sha256"], str)
             and HEX64.fullmatch(source["checkpoint_sha256"]) is not None,
             "Checkpoint fingerprint must be lowercase SHA256")
    _require(source["route_semantics"] == "POST_NATIVE_ROUTER_SELECTED_EXPERT_IDS",
             "Only original post-router expert choices are accepted")
    _require(source["token_text_recorded"] is False,
             "Route capture must contain no token or prompt text")
    return source


def validate_capture(capture: dict) -> dict:
    """Reject missing layers, reordered time, overlap, unknown fields and gaps."""
    _require(isinstance(capture, dict) and set(capture) == CAPTURE_FIELDS
             and capture["schema"] == CAPTURE_SCHEMA,
             "Unknown capture schema or fields")
    source = _source(capture["source"])
    n = capture["expert_count"]
    _require(type(n) is int and 4 <= n <= 128, "Expert count must be 4..256")
    events = capture["events"]
    _require(isinstance(events, list) and 4 <= len(events) <= MAX_EVENTS,
             "Capture must have 4..16384 routed events")
    streams: dict[str, dict[int, list[list[int]]]] = {}
    phases: list[str] = []
    closed: set[str] = set()
    for event in events:
        _require(isinstance(event, dict) and set(event) == EVENT_FIELDS,
                 "Event has unexpected or missing fields")
        split, layer, step, experts = (
            event["split"], event["layer"], event["step"], event["experts"]
        )
        _require(isinstance(split, str) and NAME.fullmatch(split) is not None,
                 "Invalid split label")
        _require(type(layer) is int and 0 <= layer < MAX_LAYERS,
                 "Layer must be within 0..127")
        _require(type(step) is int and 0 <= step < 4096,
                 "Step must be a nonnegative contiguous per-layer index")
        _require(isinstance(experts, list) and 1 <= len(experts) <= 8
                 and all(type(x) is int and 0 <= x < n for x in experts)
                 and len(set(experts)) == len(experts),
                 "Selected experts must be distinct native IDs within range")
        if not phases or split != phases[-1]:
            _require(split not in closed, "A previously closed split was resumed")
            if phases:
                closed.add(phases[-1])
            if not phases:
                _require(split == "train", "Training must occur before holdouts")
            elif split == "train":
                raise ValueError("Training may not resume after evaluation")
            phases.append(split)
        layer_steps = streams.setdefault(split, {}).setdefault(layer, [])
        _require(step == len(layer_steps),
                 "Out-of-order, duplicate or missing layer/step event")
        layer_steps.append(list(experts))

    _require(phases[0] == "train" and 2 <= len(phases) <= 9,
             "Require training and 1..8 disjoint holdout splits")
    layers = sorted(streams["train"])
    _require(1 <= len(layers) <= MAX_LAYERS,
             "Capture must contain one or more routed layers")
    counts = {}
    for split in phases:
        mapping = streams[split]
        _require(sorted(mapping) == layers,
                 "Every split must contain the exact same routed layers")
        step_lengths = {len(mapping[layer]) for layer in layers}
        _require(len(step_lengths) == 1,
                 "All layers must have the same number of steps per split")
        count = next(iter(step_lengths))
        _require(2 <= count <= 4096,
                 "Each layer/split must contain 2..4096 steps")
        counts[split] = count
    return {
        "source": source,
        "expert_count": n,
        "layers": layers,
        "splits": phases,
        "counts": counts,
        "streams": streams,
        "declared_source_authenticity_verified": False,
    }


def convert_capture(capture: dict, *, layer: int, expert_bytes: int,
                    capacity_bytes: int, max_prefetch: int) -> tuple[dict, dict]:
    """Select one layer so E2 never incorrectly connects different layers."""
    checked = validate_capture(capture)
    _require(type(layer) is int and layer in checked["layers"],
             "Selected layer must be present in every split")
    result = {
        "schema": E2_SCHEMA,
        "expert_count": checked["expert_count"],
        "expert_bytes": expert_bytes,
        "capacity_bytes": capacity_bytes,
        "max_prefetch": max_prefetch,
        "train": checked["streams"]["train"][layer],
        "holdouts": {
            split: checked["streams"][split][layer]
            for split in checked["splits"][1:]
        },
    }
    validate_e2(result)
    manifest = {
        "schema": BRIDGE_SCHEMA,
        "status": "E2_REPLAY_TRACE_EXPORTED_SOURCE_UNAUTHENTICATED",
        "model_execution": False,
        "gpu_queried": False,
        "real_weight_transfer_executed": False,
        "source_authentication": "OPERATOR_ASSERTED_NOT_VERIFIED",
        "source_kind": checked["source"]["kind"],
        "source_runtime": checked["source"]["runtime"],
        "source_model_id": checked["source"]["model_id"],
        "checkpoint_sha256_declared_unverified":
            checked["source"]["checkpoint_sha256"],
        "routing_semantics": checked["source"]["route_semantics"],
        "layer": layer,
        "layer_count": len(checked["layers"]),
        "capture_training_steps": checked["counts"]["train"],
        "capture_holdout_steps": {
            key: checked["counts"][key] for key in checked["splits"][1:]
        },
        "expert_bytes_operator_assumed_not_measured": expert_bytes,
        "e2_trace_canonical_sha256": e2_digest(result),
        "representation": "PER_LAYER_POST_NATIVE_ROUTER_ID_TRACE",
        "weight_precision_verified": False,
        "independent_native_router_reference_verified": False,
        "synthetic_source": checked["source"]["kind"] == "synthetic",
        "production_promoted": False,
        "limitations": [
            "No model is instrumented or executed by this offline converter.",
            "Operator-provided source and SHA256 may be fabricated; provenance is not attested.",
            "Equal expert size and transfer budget are assumptions for E2 simulation.",
            "Layer-specific traces avoid false cross-layer transition correlations.",
            "No raw text, token IDs, prompts or per-token router logits are accepted.",
        ],
    }
    return result, manifest


class RouteEventRecorder:
    """Opt-in callback interface for a trusted runtime's native top-k stage.

    Integration MUST call record() after the runtime has selected its true
    expert IDs. This class never decides experts or calls a model itself.
    """
    def __init__(self, source: dict, expert_count: int):
        self.source = dict(_source(source))
        _require(type(expert_count) is int and 4 <= expert_count <= 128,
                 "Expert count must be 4..128")
        self.expert_count = expert_count
        self.events: list[dict] = []

    def record(self, *, split: str, layer: int, step: int,
               experts: list[int]) -> None:
        _require(len(self.events) < MAX_EVENTS, "Capture event cap reached")
        item = {
            "split": split, "layer": layer, "step": step,
            "experts": list(experts) if isinstance(experts, list) else experts,
        }
        # Validate the new event in its strict, complete shape while checking
        # temporal and per-layer completeness during final export.
        _require(isinstance(item["experts"], list)
                 and 1 <= len(item["experts"]) <= 8
                 and all(type(x) is int and 0 <= x < self.expert_count
                         for x in item["experts"])
                 and len(set(item["experts"])) == len(item["experts"]),
                 "Native expert IDs must be bounded distinct integers")
        _require(type(layer) is int and 0 <= layer < MAX_LAYERS
                 and type(step) is int and 0 <= step < 4096
                 and isinstance(split, str) and NAME.fullmatch(split) is not None,
                 "Malformed layer/step/split")
        self.events.append(item)

    def export(self) -> dict:
        capture = {
            "schema": CAPTURE_SCHEMA,
            "source": dict(self.source),
            "expert_count": self.expert_count,
            "events": list(self.events),
        }
        validate_capture(capture)
        return capture


def demo_capture() -> dict:
    """Clearly synthetic fixture; never attested as a real MoE trace."""
    source = {
        "kind": "synthetic",
        "runtime": "pocketrotor-e2-r-fixture",
        "model_id": "untrained-declared-fixture",
        "checkpoint_sha256": "0" * 64,
        "route_semantics": "POST_NATIVE_ROUTER_SELECTED_EXPERT_IDS",
        "token_text_recorded": False,
    }
    rec = RouteEventRecorder(source, 16)
    for split, count, offset in (
        ("train", 48, 0), ("same_family", 16, 3), ("shifted", 16, 7)
    ):
        for step in range(count):
            for layer in (1, 3):
                if split == "shifted":
                    group = (step * 3 + layer + offset) % 16
                else:
                    group = (step + layer + offset) % 16
                rec.record(
                    split=split, layer=layer, step=step,
                    experts=[group, (group + 3) % 16],
                )
    return rec.export()


def _new_file(target: Path, obj: dict) -> None:
    body = json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _require(target.parent.is_dir() and not target.is_symlink(),
             "Output parent missing or output is symlinked")
    with target.open("x", encoding="utf-8") as stream:
        stream.write(body)


def main() -> None:
    cli = argparse.ArgumentParser(description="PocketRotor E2-R offline native-router event bridge")
    sub = cli.add_subparsers(dest="action", required=True)
    demo = sub.add_parser("demo", help="Emit a clearly labeled synthetic capture")
    demo.add_argument("--output", required=True, help="NEW local capture file")
    inspect = sub.add_parser("inspect", help="Read capture metadata; no runtime execution")
    inspect.add_argument("--input", required=True)
    convert = sub.add_parser("convert", help="Export one selected layer to E2 replay format")
    convert.add_argument("--input", required=True)
    convert.add_argument("--layer", type=int, required=True)
    convert.add_argument("--expert-bytes", type=int, required=True)
    convert.add_argument("--capacity-bytes", type=int, required=True)
    convert.add_argument("--max-prefetch", type=int, default=1)
    convert.add_argument("--trace-output", required=True, help="NEW E2 trace JSON")
    convert.add_argument("--manifest-output", required=True, help="NEW provenance manifest JSON")
    args = cli.parse_args()
    if args.action == "demo":
        _new_file(Path(args.output), demo_capture())
    elif args.action == "inspect":
        checked = validate_capture(_read(args.input))
        print(json.dumps({
            "schema": BRIDGE_SCHEMA,
            "status": "METADATA_INSPECTED_NO_MODEL_EXECUTION",
            "source_capture_file_sha256": _file_sha(args.input),
            "source_kind": checked["source"]["kind"],
            "source_authentication": "OPERATOR_ASSERTED_NOT_VERIFIED",
            "expert_count": checked["expert_count"],
            "layers": checked["layers"], "splits": checked["splits"],
            "split_steps_per_layer": checked["counts"],
            "model_execution": False, "production_promoted": False,
        }, indent=2, sort_keys=True))
    else:
        source = Path(args.input)
        trace_out, manifest_out = Path(args.trace_output), Path(args.manifest_output)
        _require(trace_out.resolve() not in (
                     source.resolve(), manifest_out.resolve())
                 and manifest_out.resolve() != source.resolve(),
                 "Input, trace output and manifest output must be distinct")
        _require(not trace_out.exists() and not manifest_out.exists(),
                 "Refuse to overwrite any original trace or provenance manifest")
        _require(trace_out.parent.is_dir() and manifest_out.parent.is_dir(),
                 "All output parents must already exist")
        result, manifest = convert_capture(
            _read(source), layer=args.layer, expert_bytes=args.expert_bytes,
            capacity_bytes=args.capacity_bytes, max_prefetch=args.max_prefetch,
        )
        manifest["source_capture_file_sha256"] = _file_sha(source)
        # Outputs are intentionally exclusive-created. If the second write
        # fails, preserve the first partial artifact for forensic review.
        _new_file(trace_out, result)
        _new_file(manifest_out, manifest)


if __name__ == "__main__":
    main()
