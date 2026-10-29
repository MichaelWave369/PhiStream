"""PhiPocketRotor E2: trace-only nested expert residency / prefetch experiment.

Standard-library-only simulation. It does not execute a neural model, move
real tensors, or estimate physical bandwidth. Routing traces are read-only.
The predictor sees a frozen *training* prefix and the current routed experts,
never a future holdout token/label.
"""
from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from hashlib import sha256
import argparse
import json
from pathlib import Path
import random

SCHEMA = "PHISTREAM_POCKETROTOR_E2_TRACE_V1"
RESULT_SCHEMA = "PHISTREAM_POCKETROTOR_E2_RESULT_V1"
MODES = ("lru", "frequency", "flat-markov", "nested-bubble")
MAX_TRACE_BYTES = 2 * 1024 * 1024


def quad(index: int, depth: int) -> str:
    if type(index) is not int or type(depth) is not int or depth < 1 or not 0 <= index < 4**depth:
        raise ValueError("Expert outside finite quadtree")
    return "".join(str((index // (4**shift)) % 4) for shift in reversed(range(depth)))


def _digest(data) -> str:
    packed = json.dumps(data, separators=(",", ":"), sort_keys=True, allow_nan=False).encode()
    return sha256(packed).hexdigest()


def validate(trace: dict) -> dict:
    if not isinstance(trace, dict) or trace.get("schema") != SCHEMA:
        raise ValueError("Unsupported expert trace schema")
    expert_count = trace.get("expert_count")
    expert_bytes = trace.get("expert_bytes")
    capacity_bytes = trace.get("capacity_bytes")
    max_prefetch = trace.get("max_prefetch")
    if type(expert_count) is not int or not 4 <= expert_count <= 128:
        raise ValueError("Expert count must be 4..128")
    if type(expert_bytes) is not int or not 1 <= expert_bytes <= 16 * 1024 * 1024:
        raise ValueError("Expert byte size must be 1..16 MiB")
    if type(capacity_bytes) is not int or not 1 <= capacity_bytes <= 2 * 1024**3:
        raise ValueError("Cache capacity must be 1 byte..2 GiB")
    if type(max_prefetch) is not int or not 0 <= max_prefetch <= 4:
        raise ValueError("Prefetch slots must be 0..4")
    slots = capacity_bytes // expert_bytes
    if slots < 1:
        raise ValueError("Cache capacity cannot hold one expert")
    depth = 1
    while 4 ** depth < expert_count:
        depth += 1

    def steps(raw, label, *, maximum):
        if not isinstance(raw, list) or not 2 <= len(raw) <= maximum:
            raise ValueError(f"{label}: need 2..{maximum} steps")
        out = []
        for group in raw:
            if (not isinstance(group, list) or not 1 <= len(group) <= 8 or
                    any(type(v) is not int or not 0 <= v < expert_count for v in group) or
                    len(set(group)) != len(group) or len(group) > slots):
                raise ValueError(f"{label}: invalid or over-budget routed expert group")
            out.append(tuple(group))
        return out

    train = steps(trace.get("train"), "train", maximum=4096)
    holdouts = trace.get("holdouts")
    if not isinstance(holdouts, dict) or not 1 <= len(holdouts) <= 8:
        raise ValueError("Require 1..8 named independent holdout traces")
    accepted = {}
    for name, raw in sorted(holdouts.items()):
        if (not isinstance(name, str) or not 1 <= len(name) <= 32 or
                not name.replace("_", "").replace("-", "").isalnum()):
            raise ValueError("Invalid holdout identifier")
        accepted[name] = steps(raw, name, maximum=4096)
    if set(trace) != {"schema", "expert_count", "expert_bytes", "capacity_bytes",
                      "max_prefetch", "train", "holdouts"}:
        raise ValueError("Unknown or missing trace fields")
    return {
        "expert_count": expert_count, "expert_bytes": expert_bytes,
        "capacity_bytes": capacity_bytes, "max_prefetch": max_prefetch,
        "slots": slots, "depth": depth, "train": train, "holdouts": accepted,
    }


def learn(history: list[tuple[int, ...]], depth: int) -> dict:
    """Frozen frequency and conditional transition counts from training only."""
    global_counts = Counter()
    conditional = defaultdict(Counter)
    by_parent = defaultdict(Counter)
    addresses = {}
    for group in history:
        for expert in group:
            global_counts[expert] += 1
            addresses[expert] = quad(expert, depth)[:-1]
    for previous, following in zip(history, history[1:]):
        for source in previous:
            for target in following:
                conditional[source][target] += 1
        for parent in {quad(source, depth)[:-1] for source in previous}:
            for target in following:
                by_parent[parent][target] += 1
    return {"global": global_counts, "direct": conditional, "parents": by_parent,
            "depth": depth}


def predict(model: dict, previous: tuple[int, ...], mode: str, expert_count: int) -> list[int]:
    """Returns only a ranking; never routes or replaces the canonical demand."""
    if mode not in MODES or mode == "lru":
        return []
    parents = {quad(e, model["depth"])[:-1] for e in previous}
    scores = {}
    for expert in range(expert_count):
        global_score = model["global"][expert]
        if mode == "frequency":
            scores[expert] = global_score
        else:
            scores[expert] = sum(model["direct"][src][expert] for src in previous)
            scores[expert] += .01 * global_score
            if mode == "nested-bubble":
                scores[expert] += .75 * sum(model["parents"][p][expert] for p in parents)
    return sorted(range(expert_count), key=lambda expert: (-scores[expert], expert))


def replay(config: dict, model: dict, holdout: list[tuple[int, ...]], mode: str) -> dict:
    """Transfer accounting under one fixed finite cache. No simulated clocks."""
    if mode not in MODES:
        raise ValueError("Unknown cache mode")
    bytes_per = config["expert_bytes"]
    slots = config["slots"]
    active_cache: OrderedDict[int, None] = OrderedDict()
    pending_prefetch: set[int] = set()
    counters = Counter()

    def place(expert: int, *, speculated: bool) -> None:
        if expert in active_cache:
            active_cache.move_to_end(expert)
            return
        while len(active_cache) >= slots:
            evicted, _ = active_cache.popitem(last=False)
            counters["evictions"] += 1
            if evicted in pending_prefetch:
                pending_prefetch.remove(evicted)
                counters["wasted_prefetch_evictions"] += 1
        active_cache[expert] = None
        if speculated:
            pending_prefetch.add(expert)
            counters["prefetch_loads"] += 1
            counters["prefetch_bytes"] += bytes_per
        else:
            counters["demand_loads"] += 1
            counters["demand_bytes"] += bytes_per

    for index, required in enumerate(holdout):
        for expert in required:
            counters["routed_uses"] += 1
            if expert in active_cache:
                counters["demand_hits"] += 1
                if expert in pending_prefetch:
                    pending_prefetch.remove(expert)
                    counters["useful_prefetch"] += 1
                active_cache.move_to_end(expert)
            else:
                counters["demand_misses"] += 1
                place(expert, speculated=False)
        # Only the *current* route and frozen training counts are visible.
        # There is deliberately no prefetch after the final holdout event.
        if index + 1 < len(holdout) and mode != "lru":
            budget = config["max_prefetch"]
            for candidate in predict(model, required, mode, config["expert_count"]):
                if budget == 0:
                    break
                if candidate in active_cache:
                    continue
                place(candidate, speculated=True)
                budget -= 1
        counters["peak_resident_slots"] = max(
            counters["peak_resident_slots"], len(active_cache)
        )
    counters["unused_prefetch_at_end"] = len(pending_prefetch)
    return {
        "mode": mode, "status": "ROUTE_REPLAY_SIMULATED",
        "routed_expert_uses": counters["routed_uses"],
        "demand_hits": counters["demand_hits"],
        "demand_misses": counters["demand_misses"],
        "demand_bytes": counters["demand_bytes"],
        "prefetch_bytes": counters["prefetch_bytes"],
        "total_transfer_bytes": counters["demand_bytes"] + counters["prefetch_bytes"],
        "prefetch_loads": counters["prefetch_loads"],
        "useful_prefetch": counters["useful_prefetch"],
        "wasted_prefetch_evictions": counters["wasted_prefetch_evictions"],
        "unused_prefetch_at_end": counters["unused_prefetch_at_end"],
        "evictions": counters["evictions"],
        "peak_resident_bytes": counters["peak_resident_slots"] * bytes_per,
        "cache_capacity_bytes": config["capacity_bytes"],
        "all_original_expert_ids_preserved": True,
        "gpu_compute_executed": False,
        "hardware_time_measured": False,
    }


def analyze(trace: dict) -> dict:
    cfg = validate(trace)
    model = learn(cfg["train"], cfg["depth"])
    outcomes = {}
    for name, steps in cfg["holdouts"].items():
        results = {mode: replay(cfg, model, steps, mode) for mode in MODES}
        baseline = results["lru"]
        nested = results["nested-bubble"]
        markov = results["flat-markov"]
        outcomes[name] = {
            "trace_sha256": _digest([list(group) for group in steps]),
            "steps": len(steps),
            "results": results,
            "nested_demand_bytes_vs_lru": nested["demand_bytes"] - baseline["demand_bytes"],
            "nested_demand_bytes_vs_flat_markov": nested["demand_bytes"] - markov["demand_bytes"],
            "nested_total_bytes_vs_lru": nested["total_transfer_bytes"] - baseline["total_transfer_bytes"],
            "nested_cuts_demand_bytes_by_10pct": (
                baseline["demand_bytes"] > 0 and
                nested["demand_bytes"] <= .9 * baseline["demand_bytes"]
            ),
            "nested_within_125pct_total_transfer_of_lru": (
                nested["total_transfer_bytes"] <= 1.25 * baseline["total_transfer_bytes"]
            ),
        }
    return {
        "schema": RESULT_SCHEMA, "status": "SIMULATED_TRACE_ONLY_NO_HARDWARE_CLAIM",
        "source_trace_sha256": _digest(trace),
        "training_sha256": _digest([list(group) for group in cfg["train"]]),
        "training_steps": len(cfg["train"]),
        "expert_count": cfg["expert_count"], "expert_bytes": cfg["expert_bytes"],
        "cache_capacity_bytes": cfg["capacity_bytes"], "max_prefetch": cfg["max_prefetch"],
        "quadtree_depth": cfg["depth"],
        "address_scheme": "Finite base-4 NW/NE/SW/SE path, grouping by shared parent",
        "methods": list(MODES),
        "holdouts": outcomes,
        "simulated_expert_routing_only": True,
        "model_execution": False, "cuda_used": False,
        "real_weight_bytes_transferred": False,
        "performance_certified": False, "production_promoted": False,
        "limitations": [
            "Only recorded expert IDs are replayed; no real MoE model is executed.",
            "Demand bytes are a critical-path proxy, NOT measured GPU or NVMe stall time.",
            "Prefetch bytes are counted toward total bandwidth and finite residency.",
            "Prediction uses training transitions and current route only, never future holdout events.",
            "Zero actual compute overlap, hardware timing, quantization or GPU energy measurement.",
            "Nested grouping may underperform flat Markov or LRU, especially under distribution shift.",
        ],
    }


def demo_trace() -> dict:
    """Declared synthetic train/holdout generation, disjoint time sequences."""
    def sequence(n: int, *, seed: int, shift: bool) -> list[list[int]]:
        rng = random.Random(seed)
        cluster = seed % 4
        result = []
        for i in range(n):
            if shift:
                cluster = (cluster + (3 if i % 5 else 2)) % 4
            else:
                cluster = (cluster + (1 if i % 7 else 0)) % 4
            item = 4 * cluster + (i % 4)
            second = 4 * cluster + ((i + 1 + rng.randint(0, 1)) % 4)
            if second == item:
                second = 4 * cluster + ((i + 2) % 4)
            result.append([item, second])
        return result
    return {
        "schema": SCHEMA, "expert_count": 16, "expert_bytes": 4096,
        "capacity_bytes": 4 * 4096, "max_prefetch": 1,
        "train": sequence(128, seed=369, shift=False),
        "holdouts": {
            "same_family": sequence(64, seed=137, shift=False),
            "shifted": sequence(64, seed=991, shift=True),
        },
    }


def _read_trace(path: str | Path) -> dict:
    target = Path(path)
    if target.is_symlink() or not target.is_file() or not 0 < target.stat().st_size <= MAX_TRACE_BYTES:
        raise ValueError("Require a bounded regular local JSON file without symlinks")
    def no_duplicates(pairs):
        data = {}
        for key, value in pairs:
            if key in data:
                raise ValueError("Duplicate trace JSON key")
            data[key] = value
        return data
    with target.open(encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=no_duplicates,
                         parse_constant=lambda v: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def main() -> None:
    cli = argparse.ArgumentParser(description="PhiPocketRotor E2: trace-only local cache simulator")
    cli.add_argument("action", choices=("demo", "evaluate"))
    cli.add_argument("--trace", help="Required for evaluate: local structured expert trace")
    cli.add_argument("--output", help="Optional NEW, exclusive local JSON result")
    args = cli.parse_args()
    if args.action == "evaluate" and not args.trace:
        cli.error("evaluate requires --trace")
    if args.action == "demo" and args.trace:
        cli.error("demo does not accept --trace")
    result = analyze(demo_trace() if args.action == "demo" else _read_trace(args.trace))
    body = json.dumps(result, indent=2, allow_nan=False, sort_keys=True) + "\n"
    if args.output:
        target = Path(args.output)
        if target.is_symlink() or not target.parent.is_dir():
            raise ValueError("Output parent must exist; symlink refused")
        with target.open("x", encoding="utf-8") as stream:
            stream.write(body)
    else:
        print(body, end="")


if __name__ == "__main__":
    main()
