#!/usr/bin/env python3
"""Full-run balancing cap tests on a tiny synthetic SQLite corpus."""
from __future__ import annotations

import sqlite3
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dataset_v2" / "pipeline"))
from build_corpus import apply_balance  # noqa: E402


class DummyLogger:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


class FakeIndex:
    def __init__(self, conn):
        self.conn = conn


def make_index(n_docs: int = 10, chars_per_doc: int = 1000) -> FakeIndex:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE docs(
        doc_id TEXT PRIMARY KEY, source_id TEXT, domain TEXT,
        chars INTEGER, status TEXT, selected INTEGER DEFAULT 0)""")
    for i in range(n_docs):
        conn.execute(
            "INSERT INTO docs(doc_id,source_id,domain,chars,status,selected) "
            "VALUES(?,?,?,?,?,0)",
            (f"doc-{i:03d}", "wikipedia", "encyclopedia", chars_per_doc, "kept"))
    conn.commit()
    return FakeIndex(conn)


def config(**overrides):
    cfg = {
        "target_char_budget": 480_000_000,
        "source_caps": {"wikipedia": 0.50},
        "domain_targets": {"encyclopedia": 0.45},
    }
    cfg.update(overrides)
    return cfg


class TestBalancing(unittest.TestCase):
    def selected_ids(self, index):
        return [r[0] for r in index.conn.execute(
            "SELECT doc_id FROM docs WHERE status='kept' AND selected=1 ORDER BY doc_id")]

    def test_full_run_applies_caps_below_480m_target(self):
        index = make_index()
        report = apply_balance(index, config(), DummyLogger(), scope="full_run")
        self.assertEqual(report["scope"], "full_run")
        self.assertTrue(report["caps_applied"])
        self.assertEqual(report["balance_scope_behavior"],
                         "full_run_caps_always_applied_even_when_available_chars_below_target")
        self.assertEqual(report["target_char_budget"], 480_000_000)
        self.assertEqual(report["effective_budget_characters"], 10_000)
        self.assertEqual(report["available_characters"], 10_000)
        self.assertEqual(report["selected_documents"], 4)
        self.assertEqual(report["selected_characters"], 4_000)
        self.assertEqual(report["unfilled_characters"], 6_000)
        self.assertIn("domain_cap", report["skipped_by_cap"])
        self.assertFalse(report["source_percentages_changed"])
        self.assertEqual(report["source_cap_shares"], {"wikipedia": 0.50})
        ids = self.selected_ids(index)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 4)

    def test_effective_budget_is_min_target_and_available(self):
        index = make_index()
        report = apply_balance(
            index, config(target_char_budget=5_000,
                          domain_targets={"encyclopedia": 1.0}),
            DummyLogger(), scope="full_run")
        self.assertEqual(report["effective_budget_characters"], 5_000)
        self.assertEqual(report["selected_characters"], 2_000)
        self.assertEqual(report["unfilled_characters"], 3_000)
        self.assertTrue(report["caps_applied"])
        self.assertIn("source_cap", report["skipped_by_cap"])

    def test_small_sample_may_retain_all_accepted_documents(self):
        index = make_index()
        report = apply_balance(index, config(), DummyLogger(), scope="small_sample")
        self.assertEqual(report["scope"], "small_sample")
        self.assertFalse(report["caps_applied"])
        self.assertEqual(report["balance_scope_behavior"],
                         "small_sample_caps_bypassed_for_synthetic_tests_only")
        self.assertEqual(report["selected_documents"], 10)
        self.assertEqual(report["selected_characters"], 10_000)
        self.assertEqual(report["unfilled_characters"], 0)
        self.assertEqual(report["skipped_by_cap"], {})
        self.assertEqual(len(self.selected_ids(index)), 10)

    def test_unknown_scope_is_rejected(self):
        index = make_index()
        with self.assertRaises(ValueError):
            apply_balance(index, config(), DummyLogger(), scope="production")


if __name__ == "__main__":
    unittest.main()
