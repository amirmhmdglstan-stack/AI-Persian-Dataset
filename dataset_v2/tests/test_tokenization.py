#!/usr/bin/env python3
import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dataset_v2" / "pipeline"))
from common import load_config  # noqa: E402

CFG = load_config(ROOT / "dataset_v2" / "configs" / "dataset_config.json")
TOK_PATH = ROOT / "dataset_v2" / "tokenizer" / "persian-bpe-32k.json"
SEP = "<|endoftext|>"

try:
    from tokenizers import Tokenizer  # type: ignore
    HAVE_TOKENIZERS = True
except ImportError:
    HAVE_TOKENIZERS = False


@unittest.skipUnless(HAVE_TOKENIZERS, "tokenizers package is required")
class TestBaselineTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = Tokenizer.from_file(str(TOK_PATH))

    def test_vocab_and_special_token(self):
        self.assertEqual(self.tok.get_vocab_size(), 32000)
        self.assertEqual(self.tok.token_to_id(SEP), 0)
        self.assertEqual(self.tok.encode(SEP, add_special_tokens=False).ids, [0])

    def test_persian_roundtrip_zwnj_and_latin(self):
        text = ("زبان فارسی و نیم\u200cفاصله و اعداد ۱۴۰۲ و واژه\u200cهای Python و API. "
                "این یک جمله\u200cی آزمایشی است؟ بله، درست است!")
        ids = self.tok.encode(text, add_special_tokens=False).ids
        self.assertEqual(self.tok.decode(ids), text)
        self.assertGreater(len(text) / max(1, len(ids)), 2.0)

    def test_baseline_json_matches_v1_baseline_record(self):
        baseline = json.loads((ROOT / "dataset_v2" / "sources" / "v1_baseline.json").read_text(
            encoding="utf-8"))
        expected = baseline["files"]["tokenizer/persian-bpe-32k.json"]["sha256"]
        self.assertEqual(hashlib.sha256(TOK_PATH.read_bytes()).hexdigest(), expected)

    def test_document_separator_not_needed_in_text(self):
        text = "این متن معمولی فارسی است."
        ids = self.tok.encode(text, add_special_tokens=False).ids
        self.assertNotIn(0, ids)  # <|endoftext|> is not inserted automatically


if __name__ == "__main__":
    unittest.main()
