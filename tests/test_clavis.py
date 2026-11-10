"""CLAVIS R0.1: offline, deterministic, non-executing cognitive planning tests."""
from __future__ import annotations

import json
import unittest

from phistream.clavis import (
    CAPABILITIES, CHAMBERS, ClavisError, MODES, REGISTER_KINDS,
    SCHEMA, SPECIALISTS, WorkingMemory, plan,
)
from phistream.model_atlas import CATALOG_PATH, read_json


class ClavisPlanTests(unittest.TestCase):
    def test_profiles_match_existing_nine_mind_atlas(self):
        catalog = read_json(CATALOG_PATH)
        self.assertEqual(set(SPECIALISTS), {item["id"] for item in catalog["profiles"]})
        self.assertEqual(len(SPECIALISTS), 9)
        self.assertEqual(len(CHAMBERS), 7)
        self.assertEqual(len(REGISTER_KINDS) * 4, 16)

    def test_each_mode_keeps_verification_and_no_execution(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                result = plan(mode, ["engineering"])
                self.assertEqual(result["schema"], SCHEMA)
                self.assertIn("verification", result["chambers"])
                self.assertTrue(result["gate"]["verification_required"])
                self.assertTrue(result["gate"]["operator_approval_required"])
                self.assertFalse(result["gate"]["inference_authorized"])
                self.assertFalse(result["gate"]["tool_execution_authorized"])
                self.assertFalse(result["gate"]["deployment_authorized"])
                self.assertFalse(result["evidence"]["model_quality_measured"])
                self.assertEqual(len(result["memory_registers"]), 4)

    def test_engineering_routes_cover_multiple_needs(self):
        result = plan("engineering", ["electrical", "physics", "prototype"])
        self.assertEqual(result["status"], "PLANNING_ONLY_NO_EXECUTION")
        self.assertEqual(result["uncovered_capabilities"], [])
        self.assertEqual(
            {item["profile_id"] for item in result["specialists"]},
            {"tesla", "feynman", "da-vinci"},
        )
        self.assertEqual(result["max_specialists"], 3)
        self.assertEqual(result, plan("engineering", ["electrical", "physics", "prototype"]))

    def test_creative_and_music_councils_supported(self):
        art = plan("creative", ["recursion", "symbolism", "perspective"])
        self.assertEqual({p["profile_id"] for p in art["specialists"]},
                         {"escher", "dali", "picasso"})
        music = plan("creative", ["harmony", "expression", "rhythm"])
        self.assertEqual({p["profile_id"] for p in music["specialists"]},
                         {"bach", "beethoven", "vivaldi"})

    def test_uncovered_or_restricted_profiles_held(self):
        partial = plan("engineering", ["electrical", "physics"], max_specialists=1)
        self.assertEqual(partial["status"], "HELD_UNCOVERED_CAPABILITIES")
        self.assertTrue(partial["uncovered_capabilities"])
        restricted = plan("engineering", ["electrical"], allowed_profiles=["bach"])
        self.assertEqual(restricted["specialists"], [])
        self.assertEqual(restricted["uncovered_capabilities"], ["electrical"])
        self.assertFalse(restricted["gate"]["inference_authorized"])

    def test_invalid_inputs_fail_closed(self):
        invalid = [
            lambda: plan("unknown", ["physics"]),
            lambda: plan("engineering", ["quantum-magic"]),
            lambda: plan("engineering", []),
            lambda: plan("engineering", ["physics", "physics"]),
            lambda: plan("engineering", ["physics"], max_specialists=True),
            lambda: plan("engineering", ["physics"], token_budget=0),
            lambda: plan("engineering", ["physics"], allowed_profiles=[]),
            lambda: plan("engineering", ["physics"], allowed_profiles=["tesla", "tesla"]),
        ]
        for index, fn in enumerate(invalid):
            with self.subTest(index=index), self.assertRaises(ClavisError):
                fn()

    def test_catalog_capabilities_all_known(self):
        self.assertTrue(CAPABILITIES)
        self.assertTrue(all(SPECIALISTS[profile] <= CAPABILITIES for profile in SPECIALISTS))


class ClavisMemoryTests(unittest.TestCase):
    def test_sixteen_bounded_slots_and_sanitized_snapshot(self):
        mem = WorkingMemory()
        for kind in ("observation", "hypothesis", "evidence"):
            for number in range(4):
                self.assertEqual(
                    mem.record(kind, f"PRIVATE_{kind}_{number}", source="private/path",
                               confidence=0.25),
                    f"{kind[0]}{number + 1}",
                )
        for number in range(4):
            mem.record("decision", f"PRIVATE_decision_{number}",
                       source="private/path", confidence=0.9,
                       evidence_refs=(f"e{number + 1}",))
        snap = mem.snapshot()
        self.assertEqual(sum(len(x) for x in snap["registers"].values()), 16)
        as_json = json.dumps(snap)
        self.assertNotIn("PRIVATE_", as_json)
        self.assertNotIn("private/path", as_json)
        self.assertFalse(snap["raw_content_exported"])
        with self.assertRaisesRegex(ClavisError, "full"):
            mem.record("observation", "extra", source="source", confidence=0.8)

    def test_decision_without_evidence_is_denied(self):
        mem = WorkingMemory()
        with self.assertRaisesRegex(ClavisError, "require existing evidence"):
            mem.record("decision", "unsupported", source="claim", confidence=1.0)
        with self.assertRaisesRegex(ClavisError, "require existing evidence"):
            mem.record("decision", "invented", source="claim", confidence=1.0,
                       evidence_refs=("e1",))
        self.assertEqual(mem.record("evidence", "measured", source="receipt",
                                    confidence=0.8), "e1")
        self.assertEqual(mem.record("decision", "supported", source="audit",
                                    confidence=0.7, evidence_refs=("e1",)), "d1")
        self.assertEqual(mem.snapshot()["registers"]["decision"][0]["evidence_refs"], ["e1"])

    def test_invalid_confidence_and_nondecision_links_denied(self):
        mem = WorkingMemory()
        for confidence in (float("nan"), float("inf"), -0.1, 1.1, True):
            with self.subTest(confidence=confidence), self.assertRaises(ClavisError):
                mem.record("observation", "test", source="source", confidence=confidence)
        with self.assertRaisesRegex(ClavisError, "only decision"):
            mem.record("hypothesis", "claim", source="source", confidence=0.2,
                       evidence_refs=("e1",))
        with self.assertRaises(ClavisError):
            mem.record("evidence", "claim", source="", confidence=0.2)


if __name__ == "__main__":
    unittest.main()
