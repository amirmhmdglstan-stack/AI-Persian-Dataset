#!/usr/bin/env python3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dataset_v2" / "pipeline"))
from common import load_config  # noqa: E402
from deduplicate import SQLiteCorpusIndex  # noqa: E402

CFG = load_config(ROOT / "dataset_v2" / "configs" / "dataset_config.json")
BASE = (
    "زبان فارسی یکی از زبان\u200cهای مهم منطقه است و در ایران و افغانستان و تاجیکستان رایج است. "
    "این زبان تاریخی طولانی دارد و آثار فراوانی در زمینه\u200cهای علمی و ادبی و تاریخی به آن نوشته شده است. "
    "پژوهشگران پردازش زبان طبیعی برای آموزش مدل\u200cهای زبانی به پیکره\u200cهای متنی متنوع نیاز دارند. "
    "کیفیت پیکره به حذف تکرار و حفظ ساختار جمله و رعایت نیم\u200cفاصله و نشانه\u200cهای نگارشی بستگی دارد. "
    "منبع هر سند باید مشخص باشد و حقوق متن اصلی پیش از آموزش بررسی شود."
)


class TestDeduplication(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "index.sqlite3"
        self.index = SQLiteCorpusIndex(self.path, CFG, reset=True)

    def tearDown(self):
        self.index.close()
        self.tmp.cleanup()

    def _add(self, doc_id, text):
        h = self.index.exact_hash(text)
        sig, n = self.index.make_signature(text, "test-seed")
        self.index.add_kept(
            doc_id=doc_id, source_id="smoke", profile="A", license_="CC0 test",
            domain="encyclopedia", shard="dummy.jsonl", offset=0, byte_len=1,
            chars=len(text), quality=0.8, exact_hash=h, signature=sig,
            meta={"source_id": "smoke"})
        return h, sig, n

    def test_exact_duplicate_detection(self):
        h, _, _ = self._add("doc1", BASE)
        self.assertEqual(self.index.exact_duplicate_of(h), "doc1")
        different = BASE.replace("زبان فارسی", "زبان پارسی") + " جمله\u200cای تازه افزوده شد."
        self.assertEqual(self.index.exact_duplicate_of(self.index.exact_hash(different)), None)

    def test_exact_identity_preserves_distinctions_and_separates_profiles(self):
        original = 'می\u200cرود، این ۱۲ است!'
        for variant in ('میرود، این ۱۲ است!', 'می\u200cرود این ۱۲ است!',
                        'می\u200cرود، این 12 است!', 'می\u200cرود،  این ۱۲ است!'):
            with self.subTest(variant=variant):
                self.assertNotEqual(self.index.exact_hash(original, 'A'),
                                    self.index.exact_hash(variant, 'A'))
        self.assertNotEqual(self.index.exact_hash(original, 'A'),
                            self.index.exact_hash(original, 'B'))
        self.assertEqual(self.index.exact_hash(original, 'A'),
                         self.index.exact_hash(original, 'A'))

    def test_near_duplicate_detection_and_cluster(self):
        _, _, n = self._add("doc1", BASE)
        self.assertGreaterEqual(n, 8)
        near_text = BASE.replace("کیفیت پیکره", "کیفیت این پیکره").replace("متنوع نیاز دارند", "گوناگون نیاز دارند")
        sig2, n2 = self.index.make_signature(near_text, "test-seed")
        dup, sim, candidates = self.index.near_duplicate_of(sig2, n2)
        self.assertEqual(dup, "doc1", f"similarity={sim}, candidates={candidates}")
        self.assertGreaterEqual(sim, CFG["dedup"]["similarity_threshold"])
        self.index.add_duplicate(
            doc_id="doc2", source_id="smoke", profile="A", license_="CC0 test",
            domain="encyclopedia", status="near_duplicate", cluster_id=dup,
            quality=0.8, excerpt=near_text[:100], paired_doc_id=dup,
            similarity=sim, meta={"source_id": "smoke"})
        row = self.index.conn.execute(
            "SELECT status, cluster_id FROM docs WHERE doc_id='doc2'").fetchone()
        self.assertEqual(row["status"], "near_duplicate")
        self.assertEqual(row["cluster_id"], "doc1")

    def test_signature_is_deterministic_and_short_text_exact_only(self):
        a, n1 = self.index.make_signature(BASE, "seed")
        b, n2 = self.index.make_signature(BASE, "seed")
        self.assertEqual(a, b)
        self.assertEqual(n1, n2)
        short_sig, short_n = self.index.make_signature("سلام دنیا", "seed")
        self.assertEqual(short_n, 0)
        self.assertEqual(self.index.near_duplicate_of(short_sig, short_n),
                         (None, 0.0, 0))

    def test_sqlite_index_is_disk_backed(self):
        self.assertTrue(self.path.exists())
        self.assertGreater(self.path.stat().st_size, 0)
        row = self.index.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='bands'").fetchone()
        self.assertIsNotNone(row)


if __name__ == "__main__":
    unittest.main()
