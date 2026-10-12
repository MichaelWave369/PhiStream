"""Independent reference-exporter contracts; no Transformers import in CI.

Only tiny untrained scratch model fixtures and mocked Hugging Face forward.
This DOES NOT demonstrate pretrained-model parity or GPU execution.
"""
from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.export_hf_reference import (
    check_checkpoint, export_trace, file_digest, main,
    parse_prompt, read_json,
)
from test_llama_execution import local_llama


class ExternalReferenceExporterTests(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.root = Path(self.t.name)
        self.model = self.root / "model"
        self.model.mkdir()
        local_llama(self.model, seed=369)
        files = {
            name:file_digest(self.model/name)
            for name in ("config.json","model.safetensors")
        }
        self.r = {
            "schema":"PHISTREAM_REAL_MODEL_BRIDGE_V1",
            "status":"PREFLIGHT_REVIEW_REQUIRED",
            "model_execution":False,
            "double_buffer_enabled":False,
            "scheduler_policy_applied":False,
            "checkpoint":{"format":"SAFETENSORS","model_type":"llama"},
            "memory_preflight":{"status":"REVIEW_REQUIRED"},
            "checkpoint_fingerprint":{"algorithm":"sha256","files":files},
            "user_asserted_source":{
                "source":"synthetic CI fixture","revision":"v1","license_id":"TEST",
                "declared":True, "verified_by_phistream":False,
            },
        }
        self.receipt = self.root / "fingerprint.json"
        self.receipt.write_text(json.dumps(self.r), encoding="utf-8")

    def tearDown(self):
        self.t.cleanup()

    def test_hash_only_check_without_transformers_or_model_decode(self):
        with patch("scripts.export_hf_reference.reference_forward",
                   side_effect=AssertionError("no inference")):
            out = export_trace(
                self.model,self.receipt,device="none",prompt="1,4,3",
                output=None,execute_reference=False,approved=False)
        self.assertEqual(out["status"],"HASH_VERIFIED_REFERENCE_NOT_EXECUTED")
        self.assertFalse(out["reference_trace_written"])
        self.assertEqual(out["planned_steps"],32)
        self.assertEqual(len(out["checkpoint_sha256"]["files"]),2)

    def test_execution_requires_explicit_approval_and_output(self):
        with patch("scripts.export_hf_reference.reference_forward",
                   side_effect=AssertionError("must not decode")):
            with self.assertRaisesRegex(ValueError,"--approve-local-execution"):
                export_trace(self.model,self.receipt,device="cuda",
                             prompt="1,4,3",output=self.root/"t.json",
                             execute_reference=True,approved=False)
            with self.assertRaisesRegex(ValueError,"explicit --output"):
                export_trace(self.model,self.receipt,device="cpu",
                             prompt="1,4,3",output=None,
                             execute_reference=True,approved=True)
            with self.assertRaisesRegex(ValueError,"Cannot approve"):
                export_trace(self.model,self.receipt,device="none",
                             prompt="1,4,3",output=None,
                             execute_reference=False,approved=True)

    def test_refuse_stale_checksum_before_any_hf_model_load(self):
        bad=dict(self.r)
        bad["checkpoint_fingerprint"]={
            "algorithm":"sha256",
            "files":{"config.json":"0"*64,
                     "model.safetensors":self.r["checkpoint_fingerprint"]["files"]["model.safetensors"]},
        }
        self.receipt.write_text(json.dumps(bad))
        with patch("scripts.export_hf_reference.reference_forward",
                   side_effect=AssertionError("model must not load")):
            with self.assertRaisesRegex(ValueError,"SHA256 mismatch"):
                export_trace(self.model,self.receipt,device="cpu",
                             prompt="1,4,3",output=self.root/"t.json",
                             execute_reference=True,approved=True)
        self.assertFalse((self.root/"t.json").exists())

    def test_mocked_reference_serializes_expected_comparison_contract(self):
        expected=list(range(32))
        def independent_stub(root, ids, vocab, device):
            self.assertEqual(device,"cpu")
            self.assertEqual(ids,[1,4,3])
            self.assertEqual(vocab,32)
            return expected,[{"0":0.0,"1":0.1} for _ in range(32)],{
                "runtime":"MOCK_Fake_Reference_Only",
                "version":"test", "execution_device":"cpu",
                "asserted_independent":True,
            }, {"reference_generation_seconds":0.5}
        output=self.root/"oracle.json"
        with patch("scripts.export_hf_reference.reference_forward",side_effect=independent_stub):
            info=export_trace(self.model,self.receipt,device="cpu",
                             prompt="1,4,3",output=output,
                             execute_reference=True,approved=True)
        self.assertEqual(info["status"],"LOCAL_HF_REFERENCE_TRACE_CREATED_NOT_YET_COMPARED")
        trace=json.loads(output.read_text())
        self.assertEqual(trace["schema"],"PHISTREAM_REFERENCE_TRACE_V1")
        self.assertEqual(trace["expected_generated_token_ids"],expected)
        self.assertEqual(len(trace["logit_probes"]),32)
        self.assertEqual(trace["checkpoint_fingerprint"],self.r["checkpoint_fingerprint"])
        self.assertTrue(trace["experiment"]["checkpoint_bytes_rehashed_after_execution"])
        self.assertFalse(trace["experiment"]["no_tokenizer_or_eos"] is False)
        self.assertTrue(trace["experiment"]["phistream_decoder_not_imported"])
        self.assertFalse(any("weights" in k for k in trace if k != "checkpoint_fingerprint"))
        with patch("scripts.export_hf_reference.reference_forward",
                   side_effect=AssertionError("cannot overwrite / execute")):
            with self.assertRaisesRegex(ValueError,"no overwrite"):
                export_trace(self.model,self.receipt,device="cpu",
                             prompt="1,4,3",output=output,
                             execute_reference=True,approved=True)

    def test_changed_weights_after_mocked_reference_abort_write(self):
        def tamper(root, ids, vocab, device):
            with (root/"model.safetensors").open("ab") as out:
                out.write(b"x")
            return list(range(32)),[{"0":1.0} for _ in range(32)],{
                "runtime":"MOCK", "version":"v1", "execution_device":"cpu",
                "asserted_independent":True,
            },{}
        output=self.root/"corrupt.json"
        with patch("scripts.export_hf_reference.reference_forward",side_effect=tamper):
            with self.assertRaisesRegex(ValueError,"changed during reference execution"):
                export_trace(self.model,self.receipt,device="cpu",
                             prompt="1,4,3",output=output,
                             execute_reference=True,approved=True)
        self.assertFalse(output.exists())

    def test_refuse_bad_prompt_unsigned_shards_symlink_or_duplicate_json(self):
        cfg, digest, decl=check_checkpoint(self.model,self.r)
        for v in ("-1,2", "1,,3", "1000", ",".join(["1"]*65)):
            with self.subTest(v=v),self.assertRaises(ValueError):
                parse_prompt(v,cfg)
        bad=self.r.copy()
        bad["checkpoint_fingerprint"]={"algorithm":"sha256","files":{"model.safetensors":"a"*64}}
        with self.assertRaisesRegex(ValueError,"single-shard"):
            check_checkpoint(self.model,bad)
        duplicate=self.root/"bad.json"
        duplicate.write_text('{"a":1,"a":2}')
        with self.assertRaisesRegex(ValueError,"Duplicate JSON"):
            read_json(duplicate)
        (self.root/"symlink").symlink_to(self.model,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,"non-symlink"):
            check_checkpoint(self.root/"symlink",self.r)

    def test_cli_default_is_offline_hash_review_only(self):
        args=["--checkpoint",str(self.model),"--preflight-receipt",
              str(self.receipt)]
        with patch("scripts.export_hf_reference.reference_forward",
                   side_effect=AssertionError("no reference execution")):
            with redirect_stdout(StringIO()) as out:
                main(args)
        report=json.loads(out.getvalue())
        self.assertEqual(report["status"],"HASH_VERIFIED_REFERENCE_NOT_EXECUTED")


if __name__=="__main__":
    unittest.main()
