"""Offline protocol guardrails for the independently reviewed RTX 5060 lab.

These checks read documentation only; no checkpoint, GPU, network, or
pretrained execution is accessed.
"""
from pathlib import Path
import unittest


DOC = Path(__file__).resolve().parents[1] / "docs" / "SECOND_MACHINE_REPLICATION.md"


class SecondMachineProtocolDocTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.body = DOC.read_text(encoding="utf-8")

    def test_preflight_filename_is_discoverable_and_not_misspelled(self):
        self.assertIn("smollm2-cuda-preflight-", self.body)
        self.assertNotIn("smolllm2-cuda-preflight", self.body)

    def test_preflight_contract_is_explicit(self):
        for required in (
            "PREFLIGHT_REVIEW_REQUIRED", "REVIEW_REQUIRED",
            "double_buffer_enabled", "scheduler_policy_applied",
            "SAFETENSORS", "llama", "source, revision, and license_id",
            "config.json", "model.safetensors", "sha256",
            "not authenticate the publisher",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.body)

    def test_5070_memory_sample_has_early_5060_do_not_copy_warning(self):
        dangerous = self.body.index("--capacity-mib 8192")
        caution = self.body.index("STOP: RTX 5070 HISTORICAL COMMAND ONLY")
        self.assertLess(caution, dangerous)
        self.assertIn("Do not copy or execute", self.body[caution:dangerous])

    def test_real_weight_procedure_is_separately_authorized(self):
        self.assertIn("separate operator approval", self.body)
        self.assertIn("not a guarantee of fit", self.body)
        self.assertIn("explicitly authorized execution", self.body.lower())
        self.assertIn("no real SmolLM2 model run", self.body)

    def test_independent_bundle_binding_and_limitations(self):
        self.assertIn(
            "934ceba7119e8351d817ccb810980c0bf3ea2415891a55b06de62c08d5345de8",
            self.body)
        self.assertIn("internally consistent third-party artifacts", self.body)
        self.assertIn("not independently witnessed", self.body)
        self.assertIn("5,984,512", self.body)
        self.assertIn("4,652,800", self.body)
        self.assertIn("657,664", self.body)


if __name__ == "__main__":
    unittest.main()
