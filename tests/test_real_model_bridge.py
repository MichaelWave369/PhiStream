"""PR19 real-checkpoint bridge: local CPU-only tests, no pretrained models."""
from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from phistream.realbridge import SCHEMA, evaluate_real_checkpoint
from test_llama_execution import local_llama
from test_reference_gate import make_trace


class RealCheckpointBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = self.root / "locally_owned_untrained"
        self.model.mkdir()
        self.cfg, self.weights = local_llama(self.model)
        from phistream.safeweights import LocalCheckpoint
        self.trace = make_trace(self.model, self.cfg, self.weights)
        self.trace_path = self.root / "independent-synthetic-trace.json"
        self.trace_path.write_text(json.dumps(self.trace))
        self.kw = {"capacity_mib": 16, "reserve_mib": 1, "max_tensor_mib": 1,
                   "head_rows": 7}

    def tearDown(self):
        self.temp.cleanup()

    def test_default_metadata_only_never_decodes_or_hashes(self):
        with patch("phistream.realbridge.compare_reference",
                   side_effect=AssertionError("never execute")):
            with patch("phistream.realbridge.checkpoint_fingerprint",
                       side_effect=AssertionError("default should not hash")):
                receipt = evaluate_real_checkpoint(self.model, **self.kw)
        self.assertEqual(receipt["schema"], SCHEMA)
        self.assertEqual(receipt["status"], "PREFLIGHT_REVIEW_REQUIRED")
        self.assertFalse(receipt["model_execution"])
        self.assertFalse(receipt["double_buffer_enabled"])
        self.assertFalse(receipt["scheduler_policy_applied"])
        self.assertEqual(receipt["checkpoint"]["layers"],2)
        self.assertTrue(receipt["checkpoint"]["path_withheld"])
        self.assertIsNone(receipt["checkpoint_fingerprint"])
        self.assertEqual(receipt["memory_preflight"]["status"],"REVIEW_REQUIRED")

    def test_optional_fingerprint_and_trace_validation_are_read_only(self):
        with patch("phistream.realbridge.compare_reference",
                   side_effect=AssertionError("no decoder")):
            result=evaluate_real_checkpoint(self.model, **self.kw,
                        trace_path=self.trace_path, hash_checkpoint=True,
                        source="user supplied", revision="synthetic1",
                        license_id="TEST_NOT_A_REAL_LICENSE")
        self.assertEqual(result["status"],"PREFLIGHT_TRACE_REVIEW_REQUIRED")
        self.assertEqual(result["reference_trace"]["steps"],32)
        self.assertEqual(len(result["checkpoint_fingerprint"]["files"]),2)
        self.assertFalse(result["model_execution"])
        self.assertFalse(result["user_asserted_source"]["verified_by_phistream"])

    def test_approved_cpu_parity_32_steps_still_unverified_independence(self):
        result=evaluate_real_checkpoint(self.model, **self.kw,
                device="cpu", trace_path=self.trace_path,
                execute_parity=True, approve_local_execution=True,
                source="locally generated synthetic fixture",revision="fixture-v1",
                license_id="SYNTHETIC-TEST")
        self.assertEqual(result["status"],"PARITY_MATCH_USER_REFERENCE_UNAUTHENTICATED")
        self.assertTrue(result["model_execution"])
        self.assertFalse(result["scheduler_policy_applied"])
        self.assertFalse(result["double_buffer_enabled"])
        compare=result["execution"]
        self.assertEqual(compare["steps"],32)
        self.assertEqual(compare["exact_greedy_tokens_matched"],32)
        self.assertFalse(compare["independent_reference_authenticated"])
        self.assertEqual(compare["performance"]["measurement_scope"],
                          "DECODER_ONLY_EXCLUDES_CHECKPOINT_SHA_AND_TRACE_VALIDATION")
        self.assertGreater(compare["performance"]["time_to_first_logits_including_prompt_seconds"],0)
        self.assertEqual(len(compare["performance"]["subsequent_step_seconds"]),31)
        self.assertGreater(compare["performance"]["weight_bytes_staged_total"],0)
        self.assertIsNone(compare["performance"]["max_observed_cuda_allocated_bytes_per_step"])
        json.dumps(result,allow_nan=False)

    def test_missing_consent_source_trace_and_bad_length_block(self):
        base={"device":"cpu", "trace_path":self.trace_path,
              "source":"local source","revision":"r1","license_id":"DECLARED",
              "execute_parity":True, "approve_local_execution":True}
        for remove in ("approve_local_execution","source","revision","license_id",
                       "trace_path"):
            args=base.copy()
            if remove=="approve_local_execution":args[remove]=False
            else:args.pop(remove)
            with self.subTest(remove=remove):
                with patch("phistream.realbridge.compare_reference",
                           side_effect=AssertionError("must not execute")):
                    with self.assertRaisesRegex(ValueError,"requires|Execution"):
                        evaluate_real_checkpoint(self.model,**self.kw,**args)
        short=self.trace.copy()
        short["expected_generated_token_ids"]=self.trace["expected_generated_token_ids"][:2]
        short["logit_probes"]=self.trace["logit_probes"][:2]
        self.trace_path.write_text(json.dumps(short))
        with patch("phistream.realbridge.compare_reference",
                   side_effect=AssertionError("must not execute")):
            with self.assertRaisesRegex(ValueError,"at least 32"):
                evaluate_real_checkpoint(self.model,**self.kw,**base)

    def test_memory_policy_holds_without_decoding(self):
        with patch("phistream.realbridge.compare_reference",
                   side_effect=AssertionError("no execution when budget blocked")):
            result=evaluate_real_checkpoint(self.model,
                    capacity_mib=0.0001,reserve_mib=0,
                    device="cpu",trace_path=self.trace_path,
                    execute_parity=True,approve_local_execution=True,
                    source="local",revision="r1",license_id="SYNTH",
                    max_tensor_mib=1)
        self.assertEqual(result["status"],"HELD_MEMORY_POLICY")
        self.assertFalse(result["model_execution"])
        self.assertEqual(result["memory_preflight"]["status"],"BLOCKED_BY_POLICY")

    def test_fingerprint_change_denies_before_decode(self):
        trace=self.trace.copy()
        trace["checkpoint_fingerprint"]=dict(trace["checkpoint_fingerprint"])
        files=dict(trace["checkpoint_fingerprint"]["files"])
        files["config.json"]="a"*64
        trace["checkpoint_fingerprint"]["files"]=files
        self.trace_path.write_text(json.dumps(trace))
        with patch("phistream.reference.StreamingLlama",
                   side_effect=AssertionError("must not decode mismatched bytes")):
            with self.assertRaisesRegex(ValueError,"fingerprint mismatch"):
                evaluate_real_checkpoint(self.model,**self.kw,
                    device="cpu", trace_path=self.trace_path,
                    execute_parity=True,approve_local_execution=True,
                    source="local",revision="r1",license_id="SYNTH")

    def test_variant_rope_refused_before_memory_or_execution(self):
        cfg=json.loads((self.model/"config.json").read_text())
        cfg["rope_scaling"]={"type":"dynamic","factor":2}
        (self.model/"config.json").write_text(json.dumps(cfg))
        with patch("phistream.realbridge.compare_reference",
                   side_effect=AssertionError("must not decode")):
            with self.assertRaisesRegex(ValueError,"Unsupported Llama feature"):
                evaluate_real_checkpoint(self.model,**self.kw)

    def test_cli_report_exclusive_and_exit2_on_block(self):
        from phistream.cli import main
        output=self.root/"out"/"bridge.json"
        argv=["phistream","real-model-gate",str(self.model),
              "--capacity-mib","16","--reserve-mib","1",
              "--max-tensor-mib","1","--output",str(output)]
        with patch("sys.argv",argv),redirect_stdout(StringIO()):
            main()
        self.assertEqual(json.loads(output.read_text())["status"],
                         "PREFLIGHT_REVIEW_REQUIRED")
        with patch("sys.argv",argv),redirect_stdout(StringIO()):
            with self.assertRaises(FileExistsError):
                main()
        blocked=self.root/"out"/"blocked.json"
        argv=["phistream","real-model-gate",str(self.model),
              "--capacity-mib","0.0001","--reserve-mib","0",
              "--max-tensor-mib","1","--output",str(blocked)]
        with patch("sys.argv",argv),redirect_stdout(StringIO()):
            with self.assertRaises(SystemExit) as caught:
                main()
        self.assertEqual(caught.exception.code,2)
        self.assertEqual(json.loads(blocked.read_text())["status"],"HELD_MEMORY_POLICY")


if __name__=="__main__":
    unittest.main()
