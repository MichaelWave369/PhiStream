"""Four-case real-weight matrix contracts. All fixtures are synthetic mocks.

No pretrained checkpoint execution, Hugging Face import, CUDA, or source
provenance authentication in these tests.
"""
from __future__ import annotations
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.real_weight_matrix import (
    CASES, frozen_cases, main, plan, review, SCHEMA,
)
from scripts.export_hf_reference import file_digest


class RealMatrixTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = self.root / "user_model"
        self.model.mkdir()
        self.preflight = self.root / "preflight.json"
        self.preflight.write_text("{}", encoding="utf-8")
        self.evidence = self.root / "evidence"
        self.evidence.mkdir()
        self.fp = {"algorithm": "sha256",
                   "files": {"config.json": "a"*64, "model.safetensors": "b"*64}}
        self.decl = {
            "source":"EXPERIMENTAL_SYNTHETIC", "revision":"fake-v1",
            "license_id":"SYNTHETIC", "declared":True,
            "verified_by_phistream":False,
        }
        self.patch_pinned = patch(
            "scripts.real_weight_matrix.pinned_checkpoint",
            return_value=({"vocab_size":49152},self.fp,self.decl),
        )
        self.patch_pinned.start()
        self.addCleanup(self.patch_pinned.stop)
        self.manifest = self.root/"manifest.json"
        plan(self.model,self.preflight,self.manifest)

    def tearDown(self):
        self.temp.cleanup()

    def make_pair(self, *, diversity=True):
        for case_index,(name,prompt) in enumerate(CASES):
            ids=([(case_index*3 + i) % 50 for i in range(32)]
                 if diversity else [32]*32)
            probes=[{str(x):float(x) for x in range(9)} for _ in range(32)]
            origin={
                "runtime":"Hugging Face Transformers AutoModelForCausalLM",
                "version":"TEST_MOCK","algorithm":"FULL_SEQUENCE_GREEDY_NO_CACHE",
                "execution_device":"cuda","asserted_independent":True,
            }
            trace={
                "schema":"PHISTREAM_REFERENCE_TRACE_V1",
                "reference":origin, "checkpoint_fingerprint":self.fp,
                "prompt_token_ids":list(prompt),
                "expected_generated_token_ids":ids,
                "logit_probes":probes,
                "experiment":{
                    "status":"REFERENCE_OBSERVED_NOT_AUTHENTICATED",
                    "checkpoint_bytes_rehashed_after_execution":True,
                    "phistream_decoder_not_imported":True,
                    "source_revision_license_user_asserted":self.decl,
                },
            }
            trace_path=self.evidence/("reference_"+name+".json")
            trace_path.write_text(json.dumps(trace),encoding="utf-8")
            parity={
                "schema":"PHISTREAM_REAL_MODEL_BRIDGE_V1",
                "status":"PARITY_MATCH_USER_REFERENCE_UNAUTHENTICATED",
                "device_requested":"cuda","model_execution":True,
                "real_model_run_authorized":True,
                "double_buffer_enabled":False,"scheduler_policy_applied":False,
                "checkpoint_fingerprint":self.fp,"user_asserted_source":self.decl,
                "reference_trace":{"trace_sha256":file_digest(trace_path),"steps":32},
                "execution":{
                    "reference":origin,
                    "checkpoint_fingerprint":self.fp,
                    "device":"cuda","steps":32,
                    "status":"MATCH_USER_REFERENCE_UNVERIFIED",
                    "exact_greedy_tokens_matched":32,
                    "total_logit_probes":288,"logit_probes_matched":288,
                    "max_abs_sampled_logit_error":1e-5,
                    "comparisons":[
                        {"step":i, "expected_next_id":v,
                         "actual_next_id":v,"token_match":True,
                         "probe_match":True} for i,v in enumerate(ids)
                    ],
                    "performance":{
                        "total_with_engine_setup_seconds":11.0,
                        "weight_load_seconds_including_transfers":9.0,
                        "weight_bytes_staged_total":9000000000,
                    }
                },
            }
            (self.evidence/("parity_"+name+".json")).write_text(
                json.dumps(parity),encoding="utf-8")

    def evaluate(self):
        out=self.root/"review.json"
        if out.exists():out.unlink()
        return review(self.model,self.preflight,self.manifest,self.evidence,out)

    def test_plan_only_and_frozen_cases(self):
        doc=json.loads(self.manifest.read_text())
        self.assertEqual(doc["schema"],SCHEMA)
        self.assertEqual(doc["status"],"FROZEN_PLAN_NOT_EXECUTED")
        self.assertFalse(doc["local_execution_authorized"])
        self.assertEqual([len(x["prompt_token_ids"]) for x in doc["cases"]],
                         [1,7,12,24])
        self.assertEqual(doc["cases"],frozen_cases())
        self.assertEqual(doc["checkpoint_fingerprint"],self.fp)
        with self.assertRaisesRegex(ValueError,"never overwrite"):
            plan(self.model,self.preflight,self.manifest)

    def test_all_four_synthetic_mock_pairs_match_limited_scope(self):
        self.make_pair()
        r=self.evaluate()
        self.assertEqual(r["status"],"MATCHED_LIMITED_FOUR_CASES")
        self.assertEqual(r["passing_cases"],4)
        self.assertEqual(r["required_cases"],4)
        self.assertFalse(r["model_execution"])
        self.assertFalse(r["real_model_double_buffer_promoted"])
        self.assertEqual(r["production_readiness"],"NOT_ESTABLISHED")
        self.assertEqual(r["median_decoder_seconds_descriptive_only"],11.0)
        self.assertGreater(r["unique_reference_greedy_ids_across_cases"],4)

    def test_all_numeric_matches_but_low_diversity_wont_promote(self):
        self.make_pair(diversity=False)
        r=self.evaluate()
        self.assertEqual(r["status"],"MATCHED_LIMITED_LOW_OUTPUT_DIVERSITY")
        self.assertEqual(r["passing_cases"],4)
        self.assertEqual(r["unique_reference_greedy_ids_across_cases"],1)

    def test_missing_case_is_not_pass(self):
        self.make_pair()
        (self.evidence/"parity_spread_24.json").unlink()
        r=self.evaluate()
        self.assertEqual(r["status"],"INCOMPLETE_EVIDENCE")
        self.assertEqual(r["passing_cases"],3)
        self.assertEqual(r["missing_or_invalid"][0]["id"],"spread_24")

    def test_swap_trace_rejected_by_sha_binding(self):
        self.make_pair()
        a=self.evidence/"reference_mixed_7.json"
        b=self.evidence/"reference_short_1.json"
        a.write_bytes(b.read_bytes())
        r=self.evaluate()
        self.assertEqual(r["status"],"FAIL_TRUST_CONTRACT")
        self.assertTrue(any(e["id"]=="mixed_7" for e in r["missing_or_invalid"]))

    def test_reference_trace_mutation_rejected(self):
        self.make_pair()
        target=self.evidence/"reference_repeat_12.json"
        trace=json.loads(target.read_text())
        trace["logit_probes"][0]["0"]=1234.
        target.write_text(json.dumps(trace))
        r=self.evaluate()
        self.assertEqual(r["status"],"FAIL_TRUST_CONTRACT")

    def test_fake_match_report_does_not_override_failed_step(self):
        self.make_pair()
        target=self.evidence/"parity_short_1.json"
        data=json.loads(target.read_text())
        data["execution"]["comparisons"][5]["token_match"]=False
        target.write_text(json.dumps(data))
        r=self.evaluate()
        self.assertEqual(r["status"],"FAIL_PARITY")
        self.assertEqual(r["passing_cases"],3)

    def test_forged_executed_or_double_buffer_status_denied(self):
        self.make_pair()
        target=self.evidence/"parity_repeat_12.json"
        data=json.loads(target.read_text())
        data["double_buffer_enabled"]=True
        target.write_text(json.dumps(data))
        r=self.evaluate()
        self.assertEqual(r["status"],"FAIL_TRUST_CONTRACT")

    def test_frozen_manifest_rewriting_denied(self):
        self.make_pair()
        data=json.loads(self.manifest.read_text())
        data["cases"][0]["prompt_token_ids"]=[2]
        self.manifest.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,"modified frozen matrix"):
            self.evaluate()

    def test_cli_review_exit_code_two_on_missing(self):
        argv=[
            "review","--checkpoint",str(self.model),
            "--preflight-receipt",str(self.preflight),
            "--manifest",str(self.manifest),
            "--receipts-dir",str(self.evidence),
            "--output",str(self.root/"cli-report.json")
        ]
        with redirect_stdout(StringIO()):
            with self.assertRaises(SystemExit) as err:
                main(argv)
        self.assertEqual(err.exception.code,2)
        self.assertEqual(json.loads((self.root/"cli-report.json").read_text())["status"],
                         "INCOMPLETE_EVIDENCE")


if __name__=="__main__":
    unittest.main()
