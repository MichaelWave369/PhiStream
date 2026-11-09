"""Offline tests for Φ Model Scout R0.2. No Hugging Face calls."""
from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from phistream.model_atlas import CATALOG_PATH, read_json
from phistream.hf_model_scout import (
    ScoutError, inspect_candidates, inspect_info, main,
)


def example_info(model_id="Qwen/Qwen3-4B-Instruct-2507"):
    return SimpleNamespace(
        id=model_id,
        sha="a" * 40,
        card_data=SimpleNamespace(license="apache-2.0"),
        author="Qwen",
        private=False, gated=False, disabled=False,
        config={"architectures": ["Qwen3ForCausalLM"]},
        siblings=[
            SimpleNamespace(rfilename="model-00001-of-00002.safetensors", size=1_000_000),
            SimpleNamespace(rfilename="model-00002-of-00002.safetensors", size=1_250_000),
            SimpleNamespace(rfilename="config.json", size=4000),
        ],
    )


class HfModelScoutTests(unittest.TestCase):
    def setUp(self):
        self.catalog = read_json(CATALOG_PATH)
        self.candidate = self.catalog["models"][0]

    def test_no_network_by_default(self):
        from unittest.mock import patch
        with patch("builtins.print") as output:
            self.assertEqual(main([]), 0)
            self.assertIn("OFFLINE_CATALOG_ONLY", output.call_args.args[0])
        with self.assertRaises(SystemExit) as ctx:
            main(["--inspect", "--model", self.candidate["id"]])
        self.assertEqual(ctx.exception.code, 2)

    def test_known_good_upstream_metadata_never_approves(self):
        observation = inspect_info(self.candidate, example_info())
        self.assertEqual(observation["selected_revision"], "a" * 40)
        self.assertEqual(observation["review_status"], "REVIEW_REQUIRED")
        self.assertEqual(observation["repository_bytes_reported"], 2_254_000)
        self.assertEqual(observation["safetensors_bytes_known_files_only"], 2_250_000)
        self.assertFalse(observation["model_weight_downloaded"])
        self.assertFalse(observation["activation_authorized"])
        self.assertFalse(observation["hardware_fit_verified"])

    def test_missing_file_size_never_pretends_complete(self):
        info = example_info()
        info.siblings[1].size = None
        observed = inspect_info(self.candidate, info)
        self.assertFalse(observed["file_sizes_complete"])
        self.assertIsNone(observed["repository_bytes_reported"])
        self.assertEqual(observed["repository_bytes_known_files_only"], 1_004_000)

    def test_license_mismatch_is_held(self):
        info = example_info()
        info.card_data.license = "unknown-custom-license"
        observed = inspect_info(self.candidate, info)
        self.assertFalse(observed["license_matches_catalog"])
        self.assertEqual(observed["review_status"], "HELD_LICENSE_CONFLICT")

    def test_unsafe_file_indicator_does_not_download(self):
        info = example_info()
        info.siblings.append(SimpleNamespace(rfilename="model/pytorch_model.bin", size=100))
        observed = inspect_info(self.candidate, info)
        self.assertEqual(observed["potential_pickle_weight_file_count"], 1)
        self.assertFalse(observed["weights_authenticated"])

    def test_revision_or_id_mismatch_fails_closed(self):
        with self.assertRaisesRegex(ScoutError, "identity"):
            inspect_info(self.candidate, example_info("other/Model"))
        with self.assertRaisesRegex(ScoutError, "explicitly requested"):
            inspect_info(self.candidate, example_info(), requested_revision="b" * 40)
        info = example_info()
        info.sha = "not-sha"
        with self.assertRaisesRegex(ScoutError, "commit SHA"):
            inspect_info(self.candidate, info)

    def test_catalog_selection_before_network(self):
        client = Mock()
        with self.assertRaisesRegex(ScoutError, "not in reviewed catalog"):
            inspect_candidates(self.catalog, ["random/unapproved"], client=client)
        client.model_info.assert_not_called()
        with self.assertRaisesRegex(ScoutError, "distinct"):
            inspect_candidates(self.catalog, [self.candidate["id"]] * 2, client=client)
        client.model_info.assert_not_called()

    def test_observed_model_info_is_bounded_and_has_no_token(self):
        client = Mock()
        client.model_info.return_value = example_info()
        result = inspect_candidates(self.catalog, [self.candidate["id"]], client=client)
        self.assertEqual(result["status"], "UPSTREAM_METADATA_ONLY_HUMAN_REVIEW_REQUIRED")
        self.assertFalse(result["weights_downloaded"])
        self.assertFalse(result["vessie_authority_granted"])
        client.model_info.assert_called_once_with(
            repo_id=self.candidate["id"], revision=None, timeout=12,
            files_metadata=True, token=False,
        )

    def test_reject_bad_file_path_and_sizes(self):
        info = example_info()
        info.siblings[0].rfilename = "../bad.safetensors"
        with self.assertRaisesRegex(ScoutError, "file name"):
            inspect_info(self.candidate, info)
        info = example_info()
        info.siblings[0].size = -1
        with self.assertRaisesRegex(ScoutError, "file size"):
            inspect_info(self.candidate, info)

    def test_access_restriction_held(self):
        info = example_info()
        info.gated = "auto"
        self.assertEqual(inspect_info(self.candidate, info)["review_status"], "HELD_ACCESS_REVIEW")


if __name__ == "__main__":
    unittest.main()
