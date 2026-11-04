"""Phi Model Forge R0.4: test token budget before loading any GPU weights."""
from __future__ import annotations

import unittest
from pathlib import Path

from phistream.forge_token_audit import audit_examples, require_valid, SCHEMA
from phistream.model_forge import ForgeError, DEFAULT_DATA, load_examples


class TokenizerFixture:
    """Minimal deterministic prefix-compatible chat-token fixture (CPU only)."""
    def __init__(self, corrupt_prefix: bool = False):
        self.corrupt_prefix = corrupt_prefix

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize
        content = "".join(
            f"<{x['role']}>{x['content']}</{x['role']}>"
            for x in messages if x['role'] != 'assistant'
        )
        if messages[-1]['role'] == 'assistant':
            content += f"<assistant>{messages[-1]['content']}</assistant>"
        elif add_generation_prompt:
            content += "<assistant>"
        codes = list(content.encode("utf-8"))
        if self.corrupt_prefix and messages[-1]['role'] == 'assistant':
            codes[0] = (codes[0] + 1) % 256
        return codes


def item(split: str, ident: str, prompt: str, reply: str) -> dict:
    return {
        "id": ident, "split": split,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reply},
        ],
    }


class ForgeTokenAuditTests(unittest.TestCase):
    def test_short_realistic_chat_template_passes(self):
        rows = [item("train", "phi-001", "What is a PR?", "A change proposal."),
                item("eval", "phi-002", "Does green mean merged?", "No.")]
        report = audit_examples(rows, TokenizerFixture(), "Advisory only.", 256)
        self.assertEqual(report["schema"], SCHEMA)
        self.assertEqual(report["status"], "TOKEN_BUDGET_PASSED")
        self.assertEqual(report["train_count"], 1)
        self.assertEqual(report["held_out_count"], 1)
        self.assertFalse(report["model_weights_loaded"])
        require_valid(report)

    def test_overlength_blocks_not_silent_truncation(self):
        rows = [item("train", "phi-001", "long " * 100, "Short answer.")]
        report = audit_examples(rows, TokenizerFixture(), "Advisory only.", 128)
        self.assertEqual(report["overflow_example_ids"], ["phi-001"])
        self.assertEqual(report["status"], "BLOCKED_BY_TOKEN_BUDGET")
        with self.assertRaisesRegex(ForgeError, "blocked training"):
            require_valid(report)

    def test_mismatched_template_refused(self):
        rows = [item("train", "phi-001", "Hello", "World.")]
        report = audit_examples(rows, TokenizerFixture(corrupt_prefix=True), "System", 256)
        self.assertEqual(report["template_mismatch_ids"], ["phi-001"])
        with self.assertRaises(ForgeError):
            require_valid(report)

    def test_huge_length_and_no_train_fail(self):
        with self.assertRaisesRegex(ForgeError, "128 to 1024"):
            audit_examples([item("train", "phi-001", "a", "b")], TokenizerFixture(), "System", 9999)
        with self.assertRaisesRegex(ForgeError, "No training rows"):
            audit_examples([item("eval", "phi-002", "a", "b")], TokenizerFixture(), "System", 256)

    def test_seed_is_still_unreviewed_by_default(self):
        rows, _ = load_examples(DEFAULT_DATA)
        self.assertTrue(any(x["review_status"] == "seed-unreviewed" for x in rows))
        self.assertEqual(sum(x["split"] == "eval" for x in rows), 6)


if __name__ == "__main__":
    unittest.main()
