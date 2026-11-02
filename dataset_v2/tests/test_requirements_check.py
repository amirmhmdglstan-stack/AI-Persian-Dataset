#!/usr/bin/env python3
"""Core-only dependency policy and enabled-optional-dependency tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dataset_v2" / "pipeline"))
import check_requirements as CR  # noqa: E402


def meta():
    return {"sources": [
        {"id": "txt_source", "profile": "A", "download": True,
         "files": [{"format": "txt", "extractor": "txt_lines", "required": True}]},
        {"id": "parquet_source", "profile": "A", "download": True,
         "files": [{"format": "parquet", "extractor": "parquet", "required": True}]},
        {"id": "optional_7z", "profile": "A", "download": True,
         "files": [{"format": "github_api_7z", "extractor": "csv_from_7z", "required": False}]},
        {"id": "quarantined_parquet", "profile": "C", "download": False,
         "files": [{"format": "parquet", "extractor": "parquet", "required": True}]},
    ]}


class TestRequirementsPolicy(unittest.TestCase):
    def test_tokenizers_0233_is_supported_without_downgrade(self):
        self.assertTrue(CR.version_in_range("0.23.3"))
        self.assertTrue(CR.version_in_range("0.22.2"))
        self.assertFalse(CR.version_in_range("0.19.9"))
        self.assertFalse(CR.version_in_range("0.24.0"))
        self.assertFalse(CR.version_in_range(None))

    def test_default_profile_a_checks_only_enabled_parquet(self):
        files = CR.enabled_files(meta(), ["A"], include_optional=False)
        self.assertEqual(len(files), 2)
        self.assertEqual(CR.required_optional_packages(files), ["pyarrow"])

    def test_optional_7z_is_checked_only_with_include_optional(self):
        files = CR.enabled_files(meta(), ["A"], include_optional=True)
        self.assertEqual(set(CR.required_optional_packages(files)),
                         {"pyarrow", "py7zr"})

    def test_quarantine_and_only_sources_are_excluded(self):
        self.assertEqual(CR.enabled_files(meta(), ["C"], include_optional=True), [])
        files = CR.enabled_files(meta(), ["A"], include_optional=True,
                                 only_sources=["txt_source"])
        self.assertEqual(CR.required_optional_packages(files), [])

    def test_core_requirements_file_has_no_optional_or_heavyweight_packages(self):
        lines = [ln.strip() for ln in (ROOT / "dataset_v2" / "requirements.txt")
                 .read_text(encoding="utf-8").splitlines()
                 if ln.strip() and not ln.strip().startswith("#")]
        self.assertEqual(lines, ["tokenizers>=0.20,<0.24"])
        all_files = [
            ROOT / "dataset_v2" / "requirements.txt",
            ROOT / "dataset_v2" / "requirements-parquet.txt",
            ROOT / "dataset_v2" / "requirements-7z.txt",
            ROOT / "dataset_v2" / "requirements-optional.txt",
        ]
        forbidden = ("torch", "tensorflow", "pandas", "numpy", "datasets")
        for path in all_files:
            with self.subTest(path=path.name):
                code = "\n".join(
                    ln.strip().lower() for ln in path.read_text(encoding="utf-8").splitlines()
                    if ln.strip() and not ln.strip().startswith("#"))
                for package in forbidden:
                    self.assertNotIn(package, code)

    def test_default_launchers_do_not_install_optional_packages(self):
        for name in ("run_v2.cmd", "run_v2_research.cmd", "run_v2_tests.cmd"):
            with self.subTest(launcher=name):
                text = (ROOT / "dataset_v2" / "windows" / name).read_text(encoding="utf-8")
                self.assertIn("requirements.txt", text)
                self.assertNotIn("requirements-optional.txt", text)
                self.assertNotIn("requirements-parquet.txt", text)
                self.assertNotIn("requirements-7z.txt", text)
                self.assertNotIn("pip install pyarrow", text)
                self.assertNotIn("pip install py7zr", text)
                self.assertIn("check_requirements.py", text)


if __name__ == "__main__":
    unittest.main()
