#!/usr/bin/env python3
"""Bounded-memory tokenizer comparison tests on tiny synthetic samples."""
from __future__ import annotations

import inspect
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "dataset_v2" / "pipeline"
CONFIG = ROOT / "dataset_v2" / "configs" / "dataset_config.json"
FIXTURE = ROOT / "dataset_v2" / "tests" / "fixtures" / "representative_persian_sample.txt"
TOKENIZER = ROOT / "dataset_v2" / "tokenizer" / "persian-bpe-32k.json"
sys.path.insert(0, str(PIPELINE))
import compare_tokenizers as CT  # noqa: E402

try:
    from tokenizers import Tokenizer  # type: ignore
    HAVE_TOKENIZERS = True
except ImportError:
    HAVE_TOKENIZERS = False


class TestCompareTokenizersMemory(unittest.TestCase):
    def setUp(self):
        self.tmp_obj = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmp_obj.name)

    def tearDown(self):
        self.tmp_obj.cleanup()

    def test_sample_iterator_is_streaming_and_capped(self):
        path = self.tmp / "sample.txt"
        path.write_text("12345\n67890\nabcde\n", encoding="utf-8")
        iterator = CT.iter_sample_texts(path, 10)
        self.assertTrue(inspect.isgenerator(iterator))
        docs, chars, byte_count = CT.sample_stats(path, 10)
        self.assertEqual((docs, chars, byte_count), (2, 10, 10))
        self.assertEqual(list(CT.iter_sample_texts(path, 10)), ["12345", "67890"])

    def test_40m_character_request_is_refused_as_unbounded(self):
        out = self.tmp / "comparison.json"
        p = subprocess.run([
            sys.executable, str(PIPELINE / "compare_tokenizers.py"),
            "--config", str(CONFIG), "--sample", str(FIXTURE),
            "--tokenizer-baseline", str(TOKENIZER),
            "--max-chars", "40000000", "--out", str(out),
        ], cwd=str(ROOT), capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("bounded-memory", p.stderr.lower())
        self.assertFalse(out.exists())

    @unittest.skipUnless(HAVE_TOKENIZERS, "tokenizers package is required")
    def test_evaluation_streams_into_scalar_aggregates(self):
        tok = Tokenizer.from_file(str(TOKENIZER))
        report = CT.evaluate(tok, FIXTURE, 1000, "baseline", 32000, [512], 0)
        self.assertGreater(report["evaluation_documents"], 0)
        self.assertGreater(report["token_count"], 0)
        self.assertIn("streaming documents", report["evaluation_memory_strategy"])
        self.assertIn("persian_words_measured", report["token_fragmentation"])
        self.assertIn("distinct_token_types_used", report["token_fragmentation"])

    @unittest.skipUnless(HAVE_TOKENIZERS, "tokenizers package is required")
    def test_bpe_training_accepts_bounded_iterator_length(self):
        path = self.tmp / "tiny.txt"
        path.write_text("زبان فارسی آزمایشی است.\nاین جمله دوم است.\n", encoding="utf-8")
        docs, chars, _bytes = CT.sample_stats(path, 1000)
        tok = CT.train_bpe(path, 1000, 100, docs)
        # ByteLevel.initial_alphabet() has 256 entries, so a requested 100
        # cannot be honoured; the important check is that the bounded
        # iterator/length call completes on tokenizers 0.23.3.
        self.assertGreaterEqual(tok.get_vocab_size(), 256)
        self.assertLess(tok.get_vocab_size(), 1000)
        self.assertGreater(chars, 0)


if __name__ == "__main__":
    unittest.main()
