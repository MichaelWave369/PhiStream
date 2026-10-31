"""Regression: run-llama-ids must expose the memory options it passes to execution."""

import io
import sys
import unittest
from unittest.mock import patch

from phistream.cli import main


class TokenIdsCliTest(unittest.TestCase):
    def test_capacity_and_reserve_are_defined_and_forwarded(self):
        args = ["phistream", "run-llama-ids", "unused-local-path", "--prompt", "1,2,3",
                "--tokens", "4", "--device", "cpu",
                "--capacity-mib", "4096", "--reserve-mib", "1024"]
        dummy = {"schema": "PHISTREAM_LLAMA_TOKEN_IDS_V1", "status": "EXECUTED_UNVERIFIED_REFERENCE"}
        with patch.object(sys, "argv", args), patch("phistream.safeweights.LocalCheckpoint") as checkpoint:
            with patch("phistream.llama.execute_local_tokens", return_value=dummy) as execute:
                with patch("sys.stdout", new_callable=io.StringIO):
                    main()
        execute.assert_called_once()
        positional, keyword = execute.call_args
        self.assertEqual(positional[1], [1, 2, 3])
        self.assertEqual(positional[2], 4)
        self.assertEqual(keyword["capacity_mib"], 4096.0)
        self.assertEqual(keyword["reserve_mib"], 1024.0)
        checkpoint.assert_called_once_with("unused-local-path")

    def test_omitted_memory_budget_preserves_ungated_baseline(self):
        args = ["phistream", "run-llama-ids", "unused-local-path", "--prompt", "1"]
        dummy = {"schema": "PHISTREAM_LLAMA_TOKEN_IDS_V1"}
        with patch.object(sys, "argv", args), patch("phistream.safeweights.LocalCheckpoint"):
            with patch("phistream.llama.execute_local_tokens", return_value=dummy) as execute:
                with patch("sys.stdout", new_callable=io.StringIO):
                    main()
        self.assertIsNone(execute.call_args.kwargs["capacity_mib"])
        self.assertEqual(execute.call_args.kwargs["reserve_mib"], 512.0)


if __name__ == "__main__":
    unittest.main()
