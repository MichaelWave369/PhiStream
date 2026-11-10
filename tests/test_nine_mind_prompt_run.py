"""Nine Mind R0.6: offline tests of the gated prompted generation contract.

Tests inject a tiny synthetic generator; they do not initialize CUDA, load
weights, download model files, or claim physical inference has occurred.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from phistream.model_forge import ForgeError, load_examples
from phistream.nine_mind_seed import SEED, forge_rows
from phistream.nine_mind_prompt_run import (
    METHODS, _private_destination, prepare, recipe, run, RUN_SCHEMA,
)
from phistream.nine_mind_bench import _validate_prompted, _check_adapter_provenance


class NineMindPromptRunTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.dataset = self.root / "reviewed.jsonl"
        source = forge_rows("tesla")
        # Synthetic approvals are used ONLY to unit-test branch gating.
        for item in source:
            item["review_status"] = "human-approved"
        self.dataset.write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in source),
            encoding="utf-8",
        )
        self.output = self.root / "prompted.json"

    def fake(self):
        tracker = Mock()

        def generator(model, revision, examples, system, max_new_tokens):
            tracker(model, revision, examples, system, max_new_tokens)
            return ["Synthetic response; NOT an executed model" for _ in examples]
        return generator, tracker

    def test_all_nine_method_recipes_are_distinct_original_instructions(self):
        self.assertEqual(len(METHODS), 9)
        receipts = {profile: hashlib.sha256(recipe(profile).encode()).hexdigest()
                    for profile in METHODS}
        self.assertEqual(len(set(receipts.values())), 9)
        self.assertTrue(all("not as a simulated historical person" in recipe(x)
                            for x in METHODS))

    def test_default_prepare_never_loads_models(self):
        result = prepare(self.dataset, "tesla")
        self.assertEqual(result["status"], "LOCAL_INFERENCE_ELIGIBLE_FOR_OPERATOR_REVIEW")
        self.assertEqual(result["held_out_count"], 1)
        self.assertTrue(result["dataset_all_rows_approved"])
        self.assertTrue(result["cached_weights_required"])
        self.assertFalse(result["network_access_authorized"])
        self.assertFalse(result["model_loaded"])
        self.assertFalse(result["inference_started"])
        self.assertEqual(len(result["model_revision"]), 40)

    def test_untouched_seed_is_refused_for_execution(self):
        unapproved = self.root / "unapproved.jsonl"
        unapproved.write_text("".join(json.dumps(r) + "\n" for r in forge_rows("tesla")),
                              encoding="utf-8")
        plan = prepare(unapproved, "tesla")
        self.assertEqual(plan["status"], "DATASET_HUMAN_REVIEW_REQUIRED")
        generator, tracker = self.fake()
        with self.assertRaisesRegex(ForgeError, "individual human approval"):
            run(unapproved, "tesla", self.output,
                approve_local_inference=True, generator=generator)
        tracker.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_approval_is_required_prior_to_generator_call(self):
        generator, tracker = self.fake()
        with self.assertRaisesRegex(ForgeError, "explicit"):
            run(self.dataset, "tesla", self.output, generator=generator)
        tracker.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_synthetic_generator_produces_compatible_self_reported_receipt(self):
        generator, tracker = self.fake()
        report = run(self.dataset, "tesla", self.output,
                     approve_local_inference=True, generator=generator,
                     max_new_tokens=128)
        tracker.assert_called_once()
        arguments = tracker.call_args.args
        self.assertIn("electrical or energy-system ideas", arguments[3])
        self.assertEqual(arguments[4], 128)
        self.assertEqual(len(arguments[2]), 1)
        rec = json.loads(self.output.read_text(encoding="utf-8"))
        examples, digest = load_examples(self.dataset)
        held = {x["id"]: x for x in examples if x["split"] == "eval"}
        _validate_prompted(rec, held, digest, "tesla")
        _check_adapter_provenance(rec, "prompted")
        self.assertEqual(rec["schema"], RUN_SCHEMA)
        self.assertEqual(rec["recipe_sha256"], report["recipe_sha256"])
        self.assertEqual(rec["model_revision"], report["model_revision"])
        self.assertEqual(rec["outputs"][0]["prompt"], examples[-1]["messages"][0]["content"])
        self.assertEqual(rec["source_class"], "OPERATOR_ASSERTED_NOT_ATTESTED")
        self.assertFalse(report["source_attested"])
        self.assertFalse(report["quality_proven"])
        self.assertFalse(report["production_authorized"])

    def test_generator_failure_never_writes_receipt(self):
        def fail(*_):
            raise ForgeError("simulated failure")
        with self.assertRaisesRegex(ForgeError, "simulated failure"):
            run(self.dataset, "tesla", self.output,
                approve_local_inference=True, generator=fail)
        self.assertFalse(self.output.exists())

    def test_invalid_generator_output_denied(self):
        for value in ([], [""], ["x" * 10001], [42], "non-list"):
            with self.subTest(value_type=type(value).__name__, value_len=len(value)):
                with self.assertRaisesRegex(ForgeError, "output"):
                    run(self.dataset, "tesla", self.output,
                        approve_local_inference=True,
                        generator=lambda *_: value)
                self.assertFalse(self.output.exists())

    def test_cannot_overwrite_output_even_with_approval(self):
        self.output.write_text("operator original", encoding="utf-8")
        generator, tracker = self.fake()
        with self.assertRaisesRegex(ForgeError, "overwrite"):
            run(self.dataset, "tesla", self.output,
                approve_local_inference=True, generator=generator)
        self.assertEqual(self.output.read_text(), "operator original")
        tracker.assert_not_called()

    def test_output_within_checkout_refused_before_generator(self):
        generator, tracker = self.fake()
        with self.assertRaisesRegex(ForgeError, "outside the Git checkout"):
            run(self.dataset, "tesla", SEED.parent / "private-output-should-never-exist.json",
                approve_local_inference=True, generator=generator)
        tracker.assert_not_called()

    def test_reject_mismatched_topic(self):
        source = json.loads(self.dataset.read_text(encoding="utf-8").splitlines()[0])
        rows = [json.loads(x) for x in self.dataset.read_text(encoding="utf-8").splitlines()]
        rows[0]["topic"] = "style"
        self.dataset.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
        with self.assertRaisesRegex(ForgeError, "topic is incompatible"):
            prepare(self.dataset, "tesla")

    def test_invalid_token_budgets_or_profile_denied(self):
        for n in (0, 257, -4, 1.2, True):
            with self.subTest(n=n), self.assertRaisesRegex(ForgeError, "max_new_tokens"):
                prepare(self.dataset, "tesla", max_new_tokens=n)
        with self.assertRaisesRegex(ForgeError, "unknown Nine Mind"):
            prepare(self.dataset, "bogus")

    def test_private_path_validation_exclusive_and_extension(self):
        with self.assertRaisesRegex(ForgeError, "json receipt"):
            _private_destination(self.root / "output.txt")
        with self.assertRaisesRegex(ForgeError, "parent directory"):
            _private_destination(self.root / "missing" / "receipt.json")
        self.assertEqual(_private_destination(self.output), self.output.resolve())

    def test_symlink_refused(self):
        link = self.root / "linked.json"
        try:
            link.symlink_to(self.dataset)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable")
        with self.assertRaisesRegex(ForgeError, "symlinked"):
            _private_destination(link)


if __name__ == "__main__":
    unittest.main()
