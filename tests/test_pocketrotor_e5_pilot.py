"""E5 pilot guard tests. Pretrained download never occurs in standard CI."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.pocketrotor_e5_pilot import (
    MODEL_ID, MODEL_REVISION, SPLIT_STEPS, PUBLIC_TEXT, SCHEMA, _source, main,
)


class E5PilotGuardTests(unittest.TestCase):
    def test_public_input_splits_are_frozen_and_distinct(self):
        self.assertEqual(SPLIT_STEPS, {
            "train": 32, "same_family": 16, "shifted": 16})
        self.assertEqual(set(PUBLIC_TEXT), set(SPLIT_STEPS))
        self.assertEqual(len(set(PUBLIC_TEXT.values())), 3)
        self.assertTrue(all(len(s) > 100 for s in PUBLIC_TEXT.values()))
        self.assertEqual(MODEL_ID, "ibm-granite/granite-3.0-1b-a400m-instruct")
        self.assertEqual(len(MODEL_REVISION), 40)
        self.assertEqual(SCHEMA, "PHISTREAM_POCKETROTOR_E5_PILOT_RECEIPT_V1")

    def test_provenance_contract_excludes_token_text(self):
        source = _source(
            "operator_asserted_native", MODEL_ID + "@" + MODEL_REVISION,
            "a" * 64)
        self.assertFalse(source["token_text_recorded"])
        self.assertNotIn("prompt", source)
        self.assertNotIn("token_ids", source)
        self.assertEqual(source["checkpoint_sha256"], "a" * 64)

    def test_pretrained_requires_explicit_flag_before_imports_or_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report, capture = root / "r.json", root / "c.json"
            args = [
                "e5", "pretrained",
                "--model-dir", str(root),
                "--report-output", str(report),
                "--capture-output", str(capture),
            ]
            with patch("sys.argv", args):
                with self.assertRaises(SystemExit):
                    main()
            self.assertFalse(report.exists())
            self.assertFalse(capture.exists())
            # With explicit consent but absent model dir, no downloads.
            with patch("sys.argv", args + ["--authorize-download"]):
                # Existing root *is* a model dir; do not call main in this case.
                pass
            missing = root / "missing"
            with patch("sys.argv", args + [
                "--authorize-download", "--model-dir", str(missing)]):
                with self.assertRaisesRegex(ValueError, "must already exist"):
                    main()
            self.assertFalse(report.exists())
            self.assertFalse(capture.exists())

    def test_no_overwriting_prior_receipts_in_any_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b = root / "report.json", root / "capture.json"
            a.write_text(json.dumps({"prior": "preserve"}))
            for mode in ("native-smoke", "pretrained"):
                with self.subTest(mode=mode):
                    args = ["e5", mode, "--report-output", str(a),
                            "--capture-output", str(b)]
                    with patch("sys.argv", args):
                        with self.assertRaisesRegex(ValueError, "overwrite"):
                            main()
                    self.assertEqual(json.loads(a.read_text())["prior"], "preserve")
                    self.assertFalse(b.exists())

    def test_native_smoke_rejects_download_flag_and_model_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = ["e5", "native-smoke", "--authorize-download",
                    "--report-output", str(root / "r.json"),
                    "--capture-output", str(root / "c.json")]
            with patch("sys.argv", args):
                with self.assertRaises(SystemExit):
                    main()
            self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
