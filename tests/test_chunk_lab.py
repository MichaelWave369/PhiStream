"""Scratch-only CPU checks and fake completion events for PR15 scheduler."""
from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from phistream import chunklab
from phistream.chunklab import (
    SCHEMA, PROVENANCE, _check_parameters, _fixture, _read_at,
    _run_sequence, run_chunk_lab,
)


class ChunkLabTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_cpu_scheduler_receipt_and_owned_scratch_cleanup(self):
        existing = self.root / "existing-checkpoint.bin"
        existing.write_bytes(b"NO TOUCH")
        before = set(self.root.iterdir())
        result = run_chunk_lab(
            self.root, chunk_mib=1, chunks=4, repeats=4,
            warmups=1, gpu="none",
        )
        self.assertEqual(result["schema"], SCHEMA)
        self.assertEqual(result["provenance"], PROVENANCE)
        self.assertEqual(result["status"], "PASS_CPU_CHUNK_SCHEDULER")
        self.assertFalse(result["model_execution"])
        self.assertFalse(result["automatic_layer_placement"])
        self.assertTrue(result["fixture"]["scratch_deleted"])
        self.assertEqual(result["validation"]["audit_chunks"], 4)
        self.assertFalse(result["validation"]["full_sha256_host_and_gpu_audit"])
        self.assertEqual(result["design"]["paired_rounds"], 4)
        for name in ("single", "double"):
            self.assertEqual(len(result["methods"][name]["end_to_end"]["samples_seconds"]), 4)
        self.assertEqual(set(self.root.iterdir()), before)
        self.assertEqual(existing.read_bytes(), b"NO TOUCH")
        json.dumps(result, allow_nan=False)

    def test_fake_gpu_events_refuse_reuse_before_completion(self):
        # A fake deferred DMA read retains a reference to its input memory.
        # A premature readinto on that slot corrupts the expected snapshot,
        # which our fake completion handler immediately detects.
        path = self.root / "owned"
        digest, hashes, _ = _fixture(path, 4096, 6)
        self.assertEqual(len(set(hashes)), 6)
        buffers = [memoryview(bytearray(4096)) for _ in range(2)]
        tickets = []
        def submit(slot):
            ticket = {"slot": slot, "expected": bytes(buffers[slot]), "complete": False}
            tickets.append(ticket)
            return ticket
        def wait(ticket):
            self.assertFalse(ticket["complete"])
            self.assertEqual(
                bytes(buffers[ticket["slot"]]), ticket["expected"],
                "Overwrote host buffer while synthetic GPU transfer pending",
            )
            ticket["complete"] = True
        def query(ticket):
            return ticket["complete"]

        single = _run_sequence(path, 4096, 6, "single", buffers, submit, wait, query)
        self.assertEqual(len(single["tickets"]), 6)
        self.assertTrue(all(t["complete"] for t in single["tickets"]))
        double = _run_sequence(path, 4096, 6, "double", buffers, submit, wait, query)
        self.assertEqual(len(double["tickets"]), 6)
        self.assertTrue(all(t["complete"] for t in double["tickets"]))
        self.assertGreater(double["read_scheduled_while_previous_copy_pending"], 0)
        self.assertEqual(single["read_scheduled_while_previous_copy_pending"], 0)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_outstanding_fake_gpu_copy_waited_when_next_read_fails(self):
        path = self.root / "owned-for-failure"
        _fixture(path, 4096, 4)
        buffers = [memoryview(bytearray(4096)) for _ in range(2)]
        released = []
        class Pending:
            complete = False
        def submit(slot):
            return Pending()
        def wait(ticket):
            ticket.complete = True
            released.append(ticket)
        from phistream import chunklab
        original_read = chunklab._read_at
        counter = {"n": 0}
        def bad_read(*args):
            counter["n"] += 1
            if counter["n"] == 2:
                raise RuntimeError("Injected second read failure")
            return original_read(*args)
        with patch("phistream.chunklab._read_at", side_effect=bad_read):
            with self.assertRaisesRegex(RuntimeError, "Injected second read"):
                _run_sequence(path, 4096, 4, "double", buffers, submit, wait)
        self.assertGreaterEqual(len(released), 1)
        self.assertTrue(all(t.complete for t in released))

    def test_bad_limits_are_rejected_without_creating_scratch(self):
        cases = (
            {"chunk_mib": 0}, {"chunk_mib": 3}, {"chunk_mib": 32},
            {"chunk_mib": True}, {"chunks": 3}, {"chunks": 33},
            {"chunks": True}, {"chunks": 32, "chunk_mib": 16},
            {"repeats": 5}, {"repeats": True},
            {"warmups": 0}, {"warmups": 4}, {"warmups": True},
            {"gpu": "auto"},
        )
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                run_chunk_lab(self.root, **case)
        self.assertEqual(list(self.root.iterdir()), [])
        _check_parameters(1, 4, 4, 1)

    def test_low_disk_ram_and_cuda_absence_fail_before_scratch(self):
        from collections import namedtuple
        Disk = namedtuple("Disk", "total used free")
        with patch("phistream.chunklab.shutil.disk_usage",
                   return_value=Disk(999999999, 999999998, 1)):
            with self.assertRaisesRegex(RuntimeError, "scratch disk"):
                run_chunk_lab(self.root, chunk_mib=1, chunks=4, repeats=4)
        with patch("phistream.chunklab.physical_memory",
                   return_value={"available_bytes": 2}):
            with self.assertRaisesRegex(RuntimeError, "host RAM"):
                run_chunk_lab(self.root, chunk_mib=1, chunks=4, repeats=4)
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "no fallback"):
                run_chunk_lab(self.root, gpu="cuda", chunk_mib=1,
                              chunks=4, repeats=4)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_short_read_refused_and_no_unrelated_files_deleted(self):
        path = self.root / "short.dat"
        path.write_bytes(b"small")
        with self.assertRaisesRegex(RuntimeError, "Short read"):
            _read_at(path, 0, memoryview(bytearray(100)))
        self.assertEqual(path.read_bytes(), b"small")

    def test_corrupted_scratch_rejected_and_removed(self):
        original = chunklab._fixture
        def corrupted(path, count, chunks):
            digest, hashes, elapsed = original(path, count, chunks)
            with path.open("r+b") as f:
                f.write(b"BAD!")
            return digest, hashes, elapsed
        with patch("phistream.chunklab._fixture", side_effect=corrupted):
            with self.assertRaisesRegex(RuntimeError, "modified|integrity"):
                run_chunk_lab(self.root, chunk_mib=1, chunks=4,
                              repeats=4, warmups=1, gpu="none")
        self.assertEqual(list(self.root.iterdir()), [])

    def test_cli_exclusive_receipt_and_no_network(self):
        from phistream.cli import main
        target = self.root / "receipts" / "chunks.json"
        argv = ["phistream", "chunk-lab", "--workspace", str(self.root),
                "--gpu", "none", "--chunk-mib", "1", "--chunks", "4",
                "--repeats", "4", "--warmups", "1",
                "--output", str(target)]
        with patch("sys.argv", argv), redirect_stdout(StringIO()) as output:
            main()
        self.assertEqual(json.loads(output.getvalue())["status"],
                         "PASS_CPU_CHUNK_SCHEDULER")
        self.assertEqual(json.loads(target.read_text())["schema"], SCHEMA)
        with patch("sys.argv", argv), redirect_stdout(StringIO()):
            with self.assertRaises(FileExistsError):
                main()


if __name__ == "__main__":
    unittest.main()
