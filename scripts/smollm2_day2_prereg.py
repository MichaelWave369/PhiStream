"""Frozen, offline preregistration for PhiStream SmolLM2 day-two paired study.

This module generates a deterministic research plan. It NEVER contacts a
network, enumerates devices, downloads a model, runs pretrained weights,
starts monitoring, pauses services, or initiates a GPU experiment.

Day-two execution remains unimplemented/unauthorized here. The existing
paired runner and its independent parity/preflight gates remain canonical.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

SCHEMA = "PHISTREAM_SMOLLM2_DAY2_PREREGISTRATION_V1"
MASTER_SEED = 4277796034
DAY1_SEED = 369
PAIR_COUNT = 8
MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
FINGERPRINTS = {
    "config.json": "8eb740e8bbe4cff95ea7b4588d17a2432deb16e8075bc5828ff7ba9be94d982a",
    "model.safetensors": "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c",
}
FROZEN_PROMPTS = {
    "short_1": [1],
    "mixed_7": [9, 28, 7, 139, 302, 900, 4096],
    "repeat_12": [42, 42, 42, 42, 42, 42, 13, 13, 13, 13, 1, 1],
    "spread_24": [
        495, 871, 1604, 2361, 3141, 4004, 5127, 6203,
        7129, 8261, 9347, 10589, 11827, 13219, 14641,
        15883, 17257, 18631, 19937, 21247, 22621, 23917,
        25127, 26393,
    ],
}
AB_CASES = ("mixed_7", "repeat_12", "spread_24")
SEED_DERIVATION_PREFIX = "PhiStream:SmolLM2:day2:v1:"


def case_seed(case: str) -> int:
    if case not in AB_CASES:
        raise ValueError("Only frozen A/B cases have derived seeds")
    phrase = f"{SEED_DERIVATION_PREFIX}{MASTER_SEED}:{case}"
    return int.from_bytes(hashlib.sha256(phrase.encode("utf-8")).digest()[:4], "big")


def balanced_orders(case: str) -> list[list[str]]:
    """Use the exact design of paired_real_prefill_ab.schedule, at 8 pairs."""
    seed = case_seed(case)
    orders = [["SINGLE", "BATCHED"] for _ in range(PAIR_COUNT // 2)]
    orders += [["BATCHED", "SINGLE"] for _ in range(PAIR_COUNT // 2)]
    random.Random(seed).shuffle(orders)
    return orders


def manifest() -> dict:
    return {
        "schema": SCHEMA,
        "status": "PREREGISTRATION_ONLY_NO_GPU_EXECUTION_AUTHORIZED",
        "model": {"source": MODEL, "revision": REVISION,
                  "license_id": "Apache-2.0",
                  "declared_sha256": dict(FINGERPRINTS),
                  "source_verified_by_this_plan": False},
        "design": {
            "master_seed": MASTER_SEED, "seed_derivation":
            "uint32-big-endian(first_four_bytes(sha256(utf8('PhiStream:SmolLM2:day2:v1:{master_seed}:{case}'))))",
            "day_one_seed": DAY1_SEED,
            "cases_order": list(FROZEN_PROMPTS),
            "prompts": FROZEN_PROMPTS,
            "greedy_steps_per_case": 32, "paired_cases": list(AB_CASES),
            "pairs_per_case": PAIR_COUNT,
            "warmups_per_method": 1,
            "per_case": {
                case: {"seed": case_seed(case), "orders": balanced_orders(case)}
                for case in AB_CASES
            },
            "balanced_per_case": True,
            "run_order_frozen_before_outcomes": True,
        },
        "memory_policy": {
            "capacity_mib": 1600, "reserve_mib": 512,
            "external_pre_group_nvidia_smi_free_floor_mib": 2200,
            "policy_is_not_device_fit_attestation": True,
            "torch_and_nvidia_smi_conflicts_must_both_be_preserved": True,
            "preflight_and_sha_gates_required": True,
        },
        "contention_plan": {
            "poll_interval_seconds": 1,
            "requested_query": "nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits",
            "require_timestamped_trial_start_end": True,
            "require_monotonic_clock_and_utc_mapping": True,
            "sample_source_unavailable_or_gapped": "UNKNOWN_NOT_CLEAN",
            "unexpected_process_overlap": "CONTENDED_PRESERVE_ALL_TRIALS",
            "no_other_process_observed": "OBSERVED_CLEAR_NOT_EXCLUSIVITY_PROOF",
            "no_automatic_unload_or_kill": True,
            "stop_new_trials_if_memory_gate_fails": True,
            "all_trial_labels_preserved_without_selective_deletion": True,
            "monitor_or_trial_timestamp_binding_must_be_implemented_and_tested_before_execution": True,
        },
        "analysis": {
            "report_raw_trial_timing_and_invariant_weight_bytes": True,
            "report_every_pair_gain_and_median_and_spread": True,
            "include_all_trials_and_contended_stratum_separately": True,
            "no_exclusion_by_timing_outcome": True,
            "cross_machine_same_case_seed_and_orders": True,
            "no_statistical_significance_or_production_speedup_presumed": True,
            "single_default_remains_unchanged": True,
            "day_one_archive_and_contended_spread_24_preserved": True,
        },
        "authorization": {
            "model_download": "REQUIRES_SEPARATE_EXPLICIT_HUNTER_APPROVAL",
            "model_execution": "REQUIRES_SEPARATE_EXPLICIT_OPERATOR_APPROVAL",
            "service_pause_unload_restart": "REQUIRES_HUNTER_APPROVAL_AND_RESTORATION_CHECK",
            "this_manifest_authorizes_execution": False,
        },
    }


def canonical_bytes() -> bytes:
    return (json.dumps(manifest(), sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path,
                    help="Explicit new .json file; exclusive create, never overwrite")
    args = ap.parse_args()
    data = canonical_bytes()
    if args.output is not None:
        path = args.output
        if path.suffix != ".json" or path.exists() or path.is_symlink():
            ap.error("Output must be a new .json file, never overwrite")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
    print(json.dumps({"status": manifest()["status"],
                      "seed": MASTER_SEED,
                      "pairs_per_case": PAIR_COUNT,
                      "manifest_sha256": hashlib.sha256(data).hexdigest(),
                      "model_executed": False}, indent=2))


if __name__ == "__main__":
    main()
