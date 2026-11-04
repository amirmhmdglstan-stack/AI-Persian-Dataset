#!/usr/bin/env python3
"""Repository-relative path resolution and launcher path-safety tests (Linux)."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dataset_v2" / "pipeline"))
from common import (  # noqa: E402
    config_policy, config_repo_root, config_source_metadata, config_tokenizer,
    load_config, repo_root_from_here, resolve_config_path,
)

V2_WINDOWS_REPO = r"D:\GPT\dataset\repo_v2_review"
V1_WINDOWS_REPO = r"D:\GPT\dataset\repo"


class TestPathResolution(unittest.TestCase):
    def test_no_machine_specific_repo_root_in_config(self):
        cfg = load_config(ROOT / "dataset_v2" / "configs" / "dataset_config.json")
        self.assertNotIn("repo_root", cfg)
        self.assertEqual(cfg["config_version"], "2.1.0")

    def test_all_config_paths_resolve_to_actual_v2_checkout(self):
        cfg = load_config()
        self.assertEqual(repo_root_from_here().resolve(), ROOT.resolve())
        self.assertEqual(config_repo_root(cfg).resolve(), ROOT.resolve())
        paths = [
            config_source_metadata(cfg),
            config_policy(cfg),
            config_tokenizer(cfg),
        ]
        for path in paths:
            self.assertTrue(path.exists(), path)
            self.assertEqual(path.resolve().relative_to(ROOT).parts[0], "dataset_v2")

    def test_stale_v1_repo_root_argument_is_ignored(self):
        cfg = load_config()
        expected = ROOT / cfg["source_metadata"]
        actual = resolve_config_path(cfg["source_metadata"], V1_WINDOWS_REPO)
        self.assertEqual(actual.resolve(), expected.resolve())

    def test_catalog_has_correct_wikimedia_page_base_urls(self):
        cfg = load_config()
        meta = json.loads(config_source_metadata(cfg).read_text(encoding="utf-8"))
        by_id = {s["id"]: s for s in meta["sources"]}
        expected = {
            "wikipedia_fa_official_20261001": "https://fa.wikipedia.org/wiki/",
            "fawikibooks_20261101": "https://fa.wikibooks.org/wiki/",
            "fawikisource_20261101": "https://fa.wikisource.org/wiki/",
            "fawikinews_20261101": "https://fa.wikinews.org/wiki/",
        }
        for source_id, base in expected.items():
            with self.subTest(source=source_id):
                s = by_id[source_id]
                self.assertEqual(s["page_base_url"], base)
                self.assertTrue(any(f["options"].get("page_base_url") == base
                                    for f in s["files"]))

    def test_launchers_use_v2_path_and_no_unsafe_git_operations(self):
        for name in ("run_v2.cmd", "run_v2_research.cmd", "run_v2_tests.cmd"):
            with self.subTest(launcher=name):
                data = (ROOT / "dataset_v2" / "windows" / name).read_bytes()
                text = data.decode("utf-8")
                self.assertIn(V2_WINDOWS_REPO, text)
                self.assertNotIn(f'set "REPO={V1_WINDOWS_REPO}"', text)
                self.assertNotIn("git checkout", text)
                self.assertNotIn("git switch", text)
                self.assertNotIn("git pull", text)
                self.assertIn("git rev-parse --abbrev-ref HEAD", text)
                self.assertIn("git status --porcelain", text)


if __name__ == "__main__":
    unittest.main()
