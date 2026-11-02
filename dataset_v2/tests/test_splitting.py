#!/usr/bin/env python3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dataset_v2" / "pipeline"))
from common import load_config  # noqa: E402
from deduplicate import SQLiteCorpusIndex  # noqa: E402
from split_dataset import assign_splits  # noqa: E402

CFG = load_config(ROOT / "dataset_v2" / "configs" / "dataset_config.json")
# The production config forces minimum representation only for large sources;
# this tiny unit test deliberately exercises that post-pass.
TEST_CFG = {**CFG, "min_clusters_for_forced_representation": 3}
TEXT = "این یک سند فارسی برای آزمایش تقسیم\u200cبندی است. " * 12


class TestSplitting(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.index = SQLiteCorpusIndex(Path(self.tmp.name) / "index.sqlite3", CFG, reset=True)
        for source in ("source_one", "source_two"):
            for i in range(6):
                doc_id = f"{source}-{i}"
                self.index.add_kept(
                    doc_id=doc_id, source_id=source, profile="A", license_="CC0 test",
                    domain="encyclopedia", shard="dummy.jsonl", offset=0, byte_len=1,
                    chars=1000 + i, quality=0.8,
                    exact_hash=self.index.exact_hash(TEXT + doc_id),
                    signature=self.index.make_signature(TEXT + doc_id, "seed")[0],
                    meta={"source_id": source})
        # Simulate a near-duplicate cluster with two kept members: both must land
        # in the same split.
        self.index.conn.execute(
            "UPDATE docs SET cluster_id='source_one-0' WHERE doc_id='source_one-1'")
        self.index.conn.execute("UPDATE docs SET selected=1 WHERE status='kept'")
        self.index.commit()

    def tearDown(self):
        self.index.close()
        self.tmp.cleanup()

    def test_deterministic_cluster_consistent_splits(self):
        report1 = assign_splits(self.index, TEST_CFG)
        first = {r["doc_id"]: r["split"] for r in self.index.conn.execute(
            "SELECT doc_id, split FROM docs WHERE status='kept' AND selected=1")}
        self.assertEqual(report1["cluster_leakage_count"], 0)
        self.assertEqual(report1["documents_assigned"], 12)
        self.assertEqual(first["source_one-0"], first["source_one-1"])
        self.assertTrue(all(v in ("train", "validation", "test") for v in first.values()))
        for source in ("source_one", "source_two"):
            splits = {v for k, v in first.items() if k.startswith(source)}
            self.assertEqual(splits, {"train", "validation", "test"}, (source, splits))
        report2 = assign_splits(self.index, TEST_CFG)
        second = {r["doc_id"]: r["split"] for r in self.index.conn.execute(
            "SELECT doc_id, split FROM docs WHERE status='kept' AND selected=1")}
        self.assertEqual(first, second)
        self.assertEqual(report2["cluster_leakage_count"], 0)


if __name__ == "__main__":
    unittest.main()
