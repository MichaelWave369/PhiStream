"""E2 trace-only laboratory tests; no model, network, or hardware GPU."""
from __future__ import annotations

import copy
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.pocketrotor_e2 import (
    MODES, SCHEMA, _read_trace, analyze, demo_trace, learn, predict, quad,
    replay, validate,
)


class PocketRotorE2Tests(unittest.TestCase):
    def setUp(self):
        self.trace = demo_trace()

    def test_addresses_are_finite_and_match_original_pocketrotor(self):
        from phistream.pocketrotor import _quad
        for depth in (1, 2, 3):
            for index in range(4**depth):
                self.assertEqual(quad(index, depth), _quad(index, depth))
        for args in ((-1, 2), (16, 2), (0, 0), (True, 2)):
            with self.assertRaises(ValueError):
                quad(*args)

    def test_fixture_has_disjoint_labeled_sequences_and_no_hardware_claims(self):
        result = analyze(self.trace)
        self.assertEqual(result, analyze(demo_trace()))
        self.assertEqual(result["status"], "SIMULATED_TRACE_ONLY_NO_HARDWARE_CLAIM")
        self.assertEqual(result["methods"], list(MODES))
        self.assertFalse(result["model_execution"])
        self.assertFalse(result["cuda_used"])
        self.assertFalse(result["performance_certified"])
        self.assertFalse(result["production_promoted"])
        self.assertEqual(set(result["holdouts"]), {"same_family", "shifted"})
        self.assertNotEqual(result["holdouts"]["same_family"]["trace_sha256"],
                            result["holdouts"]["shifted"]["trace_sha256"])

    def test_every_policy_replays_exact_route_under_bounded_memory(self):
        cfg = validate(self.trace)
        model = learn(cfg["train"], cfg["depth"])
        for name, holdout in cfg["holdouts"].items():
            for mode in MODES:
                with self.subTest(name=name, mode=mode):
                    r = replay(cfg, model, holdout, mode)
                    self.assertTrue(r["all_original_expert_ids_preserved"])
                    self.assertEqual(r["routed_expert_uses"],
                                     sum(len(group) for group in holdout))
                    self.assertEqual(r["demand_hits"] + r["demand_misses"],
                                     r["routed_expert_uses"])
                    self.assertEqual(r["total_transfer_bytes"],
                                     r["demand_bytes"] + r["prefetch_bytes"])
                    self.assertLessEqual(r["peak_resident_bytes"],
                                         cfg["capacity_bytes"])
                    self.assertFalse(r["hardware_time_measured"])
                    if mode == "lru":
                        self.assertEqual(r["prefetch_bytes"], 0)

    def test_zero_prefetch_produces_identical_lru_loads(self):
        trace = copy.deepcopy(self.trace)
        trace["max_prefetch"] = 0
        result = analyze(trace)
        for name in result["holdouts"]:
            arms = result["holdouts"][name]["results"]
            baseline = arms["lru"]
            for mode in MODES:
                self.assertEqual(arms[mode]["demand_bytes"],
                                 baseline["demand_bytes"])
                self.assertEqual(arms[mode]["total_transfer_bytes"],
                                 baseline["total_transfer_bytes"])
                self.assertEqual(arms[mode]["demand_hits"],
                                 baseline["demand_hits"])

    def test_model_cannot_read_hidden_future_holdout(self):
        cfg = validate(self.trace)
        model = learn(cfg["train"], cfg["depth"])
        current = (4, 5)
        old = predict(model, current, "nested-bubble", cfg["expert_count"])
        poisoned = copy.deepcopy(self.trace)
        poisoned["holdouts"]["same_family"] = [
            [i % cfg["expert_count"]] for i in reversed(range(64))
        ]
        other_cfg = validate(poisoned)
        model_again = learn(other_cfg["train"], other_cfg["depth"])
        self.assertEqual(old, predict(model_again, current, "nested-bubble",
                                      cfg["expert_count"]))
        self.assertEqual(set(old), set(range(cfg["expert_count"])))
        self.assertEqual(len(old), cfg["expert_count"])

    def test_predictive_cache_can_reduce_critical_demand_without_invented_speed(self):
        trace = {
            "schema": SCHEMA, "expert_count": 4, "expert_bytes": 10,
            "capacity_bytes": 10, "max_prefetch": 1,
            "train": [[0], [1]] * 16,
            "holdouts": {"alternating": [[0], [1]] * 16},
        }
        result = analyze(trace)
        arms = result["holdouts"]["alternating"]["results"]
        self.assertLess(arms["nested-bubble"]["demand_bytes"],
                        arms["lru"]["demand_bytes"])
        self.assertGreater(arms["nested-bubble"]["prefetch_bytes"], 0)
        self.assertFalse(result["real_weight_bytes_transferred"])
        self.assertFalse(result["performance_certified"])

    def test_adversarial_distribution_does_not_get_hidden(self):
        trace = {
            "schema": SCHEMA, "expert_count": 4, "expert_bytes": 10,
            "capacity_bytes": 10, "max_prefetch": 1,
            "train": [[0], [1]] * 16,
            "holdouts": {"shifted": [[0], [2]] * 16},
        }
        result = analyze(trace)
        arms = result["holdouts"]["shifted"]["results"]
        self.assertGreater(arms["nested-bubble"]["total_transfer_bytes"],
                           arms["lru"]["total_transfer_bytes"])
        self.assertEqual(result["holdouts"]["shifted"]["steps"], 32)

    def test_strict_budgets_duplicates_and_no_hidden_label(self):
        cases = [
            {"expert_count": True},
            {"expert_bytes": 0},
            {"capacity_bytes": 1},
            {"max_prefetch": 5},
            {"max_prefetch": True},
            {"train": [[0]]},
            {"train": [[0, 0], [1]]},
            {"train": [[True], [1]]},
            {"holdouts": {}},
            {"holdouts": {"bad/../name": [[0], [1]]}},
            {"holdouts": {"a": [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]}},
            {"label": "forbidden hidden answer"},
        ]
        for diff in cases:
            with self.subTest(diff=str(diff)[:40]):
                with self.assertRaises(ValueError):
                    validate({**self.trace, **diff})

    def test_malformed_files_and_exclusive_receipt(self):
        from scripts.pocketrotor_e2 import main
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = root / "trace.json"
            original.write_text(json.dumps(self.trace), encoding="utf-8")
            outcome = root / "results.json"
            with patch("sys.argv", ["pocketrotor_e2", "evaluate",
                                    "--trace", str(original),
                                    "--output", str(outcome)]):
                main()
            self.assertEqual(json.loads(outcome.read_text())["schema"],
                             "PHISTREAM_POCKETROTOR_E2_RESULT_V1")
            with patch("sys.argv", ["pocketrotor_e2", "evaluate",
                                    "--trace", str(original),
                                    "--output", str(outcome)]):
                with self.assertRaises(FileExistsError):
                    main()
            old_content = original.read_bytes()
            self.assertEqual(old_content, json.dumps(self.trace).encode())
            with patch("sys.argv", ["pocketrotor_e2", "demo"]):
                with redirect_stdout(StringIO()) as output:
                    main()
            self.assertEqual(json.loads(output.getvalue())["schema"],
                             "PHISTREAM_POCKETROTOR_E2_RESULT_V1")
            malformed = root / "duplicate.json"
            malformed.write_text('{"schema":"a","schema":"b"}')
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                _read_trace(malformed)
            invalid = root / "nonfinite.json"
            invalid.write_text('{"value":NaN}')
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                _read_trace(invalid)


if __name__ == "__main__":
    unittest.main()
