"""E2-R bridge tests: synthetic recorder only; no trained model or network."""
from __future__ import annotations

import copy
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.pocketrotor_e2 import analyze, _digest as e2_digest
from scripts.pocketrotor_e2_route_bridge import (
    CAPTURE_SCHEMA, BRIDGE_SCHEMA, RouteEventRecorder, _read, convert_capture,
    demo_capture, validate_capture,
)


class PocketRotorRouteBridgeTests(unittest.TestCase):
    def setUp(self):
        self.capture = demo_capture()

    def test_demo_is_explicit_synthetic_complete_and_no_prompt_text(self):
        c = self.capture
        self.assertEqual(c["schema"], CAPTURE_SCHEMA)
        self.assertEqual(c["source"]["kind"], "synthetic")
        self.assertFalse(c["source"]["token_text_recorded"])
        info = validate_capture(c)
        self.assertEqual(info["layers"], [1, 3])
        self.assertEqual(info["splits"], ["train", "same_family", "shifted"])
        self.assertEqual(info["counts"], {
            "train": 48, "same_family": 16, "shifted": 16,
        })
        self.assertFalse(info["declared_source_authenticity_verified"])
        self.assertEqual(len(c["events"]), 160)

    def test_conversion_is_per_layer_and_e2_analyzable(self):
        for layer in (1, 3):
            with self.subTest(layer=layer):
                converted, manifest = convert_capture(
                    self.capture, layer=layer, expert_bytes=4096,
                    capacity_bytes=4 * 4096, max_prefetch=1,
                )
                self.assertEqual(converted["schema"],
                                 "PHISTREAM_POCKETROTOR_E2_TRACE_V1")
                self.assertEqual(converted["train"][0],
                                 [event["experts"] for event in self.capture["events"]
                                  if event["split"] == "train"
                                  and event["layer"] == layer][0])
                self.assertEqual(manifest["layer"], layer)
                self.assertEqual(manifest["schema"], BRIDGE_SCHEMA)
                self.assertEqual(manifest["e2_trace_canonical_sha256"],
                                 e2_digest(converted))
                self.assertTrue(manifest["synthetic_source"])
                self.assertFalse(manifest["independent_native_router_reference_verified"])
                analysis = analyze(converted)
                self.assertEqual(analysis["status"],
                                 "SIMULATED_TRACE_ONLY_NO_HARDWARE_CLAIM")
                self.assertFalse(analysis["cuda_used"])
                self.assertEqual(analysis["source_trace_sha256"],
                                 manifest["e2_trace_canonical_sha256"])

    def test_source_claim_never_becomes_authenticated(self):
        modified = copy.deepcopy(self.capture)
        modified["source"]["kind"] = "operator_asserted_native"
        trace, manifest = convert_capture(
            modified, layer=1, expert_bytes=4096,
            capacity_bytes=16384, max_prefetch=1,
        )
        self.assertEqual(len(trace["train"]), 48)
        self.assertEqual(manifest["source_kind"], "operator_asserted_native")
        self.assertEqual(manifest["source_authentication"],
                         "OPERATOR_ASSERTED_NOT_VERIFIED")
        self.assertFalse(manifest["synthetic_source"])
        self.assertFalse(manifest["weight_precision_verified"])

    def test_recorder_exposes_callback_not_router_computation(self):
        recorder = RouteEventRecorder(copy.deepcopy(self.capture["source"]), 16)
        for event in self.capture["events"]:
            recorder.record(
                split=event["split"], layer=event["layer"],
                step=event["step"], experts=event["experts"],
            )
        self.assertEqual(recorder.export(), self.capture)
        with self.assertRaises(ValueError):
            recorder.record(split="train", layer=1, step=99, experts=[True, 2])
        with self.assertRaises(ValueError):
            recorder.record(split="train", layer=1, step=99, experts=[1, 1])
        with self.assertRaises(ValueError):
            recorder.record(split="train", layer=1, step=99, experts=[16])
        with self.assertRaises(ValueError):
            recorder.record(split="train", layer=-1, step=99, experts=[1])

    def test_missing_layer_and_missing_step_are_rejected(self):
        altered = copy.deepcopy(self.capture)
        altered["events"] = [
            e for e in altered["events"] if not
            (e["split"] == "same_family" and e["layer"] == 3)
        ]
        with self.assertRaisesRegex(ValueError, "same routed layers"):
            validate_capture(altered)
        altered = copy.deepcopy(self.capture)
        altered["events"] = [
            e for e in altered["events"] if not
            (e["split"] == "train" and e["layer"] == 1 and e["step"] == 5)
        ]
        with self.assertRaisesRegex(ValueError, "Out-of-order"):
            validate_capture(altered)
        altered = copy.deepcopy(self.capture)
        altered["events"].append(copy.deepcopy(altered["events"][0]))
        with self.assertRaisesRegex(ValueError, "closed split"):
            validate_capture(altered)

    def test_invalid_source_privacy_and_relabeling_rejected(self):
        cases = [
            lambda c: c["source"].update(token_text_recorded=True),
            lambda c: c["source"].update(route_semantics="RECOMPUTED_TOPK"),
            lambda c: c["source"].update(kind="verified_native"),
            lambda c: c["source"].update(checkpoint_sha256="not-a-hash"),
            lambda c: c["source"].update(prompt="leaked text"),
            lambda c: c.update(prompt="leaked text"),
            lambda c: c["events"][0].update(text="private prompt"),
            lambda c: c["events"][0].update(experts=[0, 0]),
            lambda c: c["events"][0].update(experts=[True, 2]),
            lambda c: c["events"][0].update(step=True),
            lambda c: c["events"][0].update(split="holdout"),
            lambda c: c["events"][0].update(layer=128),
        ]
        for i, mutate in enumerate(cases):
            with self.subTest(case=i):
                bad = copy.deepcopy(self.capture)
                mutate(bad)
                with self.assertRaises(ValueError):
                    validate_capture(bad)

    def test_conversion_cannot_invent_layers_or_exceed_e2_budget(self):
        with self.assertRaisesRegex(ValueError, "Selected layer"):
            convert_capture(self.capture, layer=0, expert_bytes=4096,
                            capacity_bytes=16384, max_prefetch=1)
        with self.assertRaisesRegex(ValueError, "Cache capacity"):
            convert_capture(self.capture, layer=1, expert_bytes=4096,
                            capacity_bytes=1, max_prefetch=1)
        with self.assertRaises(ValueError):
            convert_capture(self.capture, layer=1, expert_bytes=4096,
                            capacity_bytes=16384, max_prefetch=True)

    def test_cli_demo_inspect_convert_and_no_overwrite(self):
        from scripts.pocketrotor_e2_route_bridge import main
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            source = p / "route.json"
            output = p / "layer.json"
            manifest = p / "manifest.json"
            with patch("sys.argv", ["route_bridge", "demo",
                                    "--output", str(source)]):
                main()
            self.assertEqual(_read(source), demo_capture())
            with patch("sys.argv", ["route_bridge", "inspect", "--input", str(source)]):
                with redirect_stdout(StringIO()) as stream:
                    main()
            inspected = json.loads(stream.getvalue())
            self.assertFalse(inspected["model_execution"])
            self.assertEqual(inspected["layers"], [1, 3])
            args = ["route_bridge", "convert", "--input", str(source),
                    "--layer", "1", "--expert-bytes", "4096",
                    "--capacity-bytes", "16384", "--max-prefetch", "1",
                    "--trace-output", str(output),
                    "--manifest-output", str(manifest)]
            with patch("sys.argv", args):
                main()
            self.assertEqual(analyze(_read(output))["schema"],
                             "PHISTREAM_POCKETROTOR_E2_RESULT_V1")
            man = json.loads(manifest.read_text())
            self.assertEqual(man["source_capture_file_sha256"],
                             inspected["source_capture_file_sha256"])
            self.assertEqual(man["e2_trace_canonical_sha256"], e2_digest(_read(output)))
            self.assertEqual(man["status"],
                             "E2_REPLAY_TRACE_EXPORTED_SOURCE_UNAUTHENTICATED")
            with patch("sys.argv", args):
                with self.assertRaisesRegex(ValueError, "overwrite"):
                    main()
            with patch("sys.argv", args[:-1] + [str(source)]):
                with self.assertRaisesRegex(ValueError, "distinct"):
                    main()

    def test_bad_json_duplicate_nonfinite_and_unknown_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = root / "dup.json"
            d.write_text('{"schema":"one","schema":"two"}')
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                _read(d)
            d.write_text('{"foo":NaN}')
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                _read(d)


if __name__ == "__main__":
    unittest.main()
