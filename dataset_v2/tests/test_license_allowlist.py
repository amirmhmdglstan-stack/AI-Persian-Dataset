#!/usr/bin/env python3
"""Conservative Wikisource licence-template allowlist tests (synthetic pages)."""
from __future__ import annotations

import bz2
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dataset_v2" / "pipeline"))
import extract_sources as EX  # noqa: E402
from common import config_policy, load_config  # noqa: E402

CFG = load_config(ROOT / "dataset_v2" / "configs" / "dataset_config.json")
POLICY = json.loads(config_policy(CFG).read_text(encoding="utf-8"))


class DummyLogger:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


def source():
    return {
        "id": "wikisource_fixture", "name": "fixture", "profile": "A",
        "decision": "include", "domain": "literary",
        "license_content": "per-page verified", "copyright_holder": "fixture",
        "content_rights_verified": True,
        "url": "https://dumps.wikimedia.org/fawikisource/20261101/",
        "page_base_url": "https://fa.wikisource.org/wiki/",
    }


def xml_with(template: str, page_id: str = "1") -> str:
    return f"""<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">
<page><title>صفحه {page_id}</title><ns>0</ns><id>{page_id}</id><revision><text>{template}
این یک متن فارسی آزمایشی است که برای بررسی پروانه و حقوق متن کافی به نظر میرسد.</text></revision></page>
</mediawiki>"""


def context(tmp: Path, path: Path):
    return EX.ExtractContext(
        source=source(), options={
            "require_free_license_template": True, "namespace": 0,
            "min_raw_chars": 10, "include_title": True,
            "exclude_disambiguation": True,
            "page_base_url": "https://fa.wikisource.org/wiki/",
        }, raw_paths=[path], parts_dir=None, workdir=tmp,
        logger=DummyLogger(), policy=POLICY)


class TestLicenseAllowlist(unittest.TestCase):
    def setUp(self):
        self.tmp_obj = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmp_obj.name)

    def tearDown(self):
        self.tmp_obj.cleanup()

    def test_no_broad_free_license_regex_remains(self):
        self.assertFalse(hasattr(EX, "RE_FREE_LICENSE"))
        self.assertTrue(POLICY["global_rules"]["require_explicit_license_allowlist_for_wikisource"])
        self.assertTrue(POLICY["global_rules"]["do_not_weaken_license_checks_for_corpus_size"])

    def test_verified_pd_cc_by_cc_by_sa_gfdl_templates_accepted(self):
        cases = {
            "{{PD-old}}": "Public domain",
            "{{PD-Iran}}": "Public domain in Iran",
            "{{CC-BY-4.0}}": "CC BY 4.0",
            "{{CC-BY-SA-4.0}}": "CC BY-SA 4.0",
            "{{GFDL}}": "GNU Free Documentation License",
        }
        for template, expected in cases.items():
            with self.subTest(template=template):
                status, info = EX.classify_wikisource_license(template, POLICY)
                self.assertEqual(status, "accepted", info)
                self.assertEqual(info["reason"], "exact_allowlist_match")
                self.assertIn(expected, info["license_id"])

    def test_rejected_nc_nd_generic_and_unknown_templates(self):
        cases = {
            "{{CC-BY-NC-SA-4.0}}": "non_commercial",
            "{{CC-BY-NC-4.0}}": "non_commercial",
            "{{CC-BY-ND-4.0}}": "no_derivatives",
            "{{CC-BY-NC-ND-4.0}}": "non_commercial",
            "{{Copyright}}": "generic_copyright",
            "{{حق نشر}}": "generic_copyright",
            "{{مجوز}}": "generic_permission",
            "{{پروانه}}": "generic_permission",
            "{{CC-BY-9.9}}": "unverified_template",
            "{{UnknownLicense}}": "unverified_template",
        }
        for template, reason in cases.items():
            with self.subTest(template=template):
                status, info = EX.classify_wikisource_license(template, POLICY)
                self.assertEqual(status, "rejected", info)
                self.assertEqual(info["reason"], reason)
                self.assertTrue(info["template"])

    def test_no_license_like_template_is_rejected(self):
        status, info = EX.classify_wikisource_license("{{Infobox}} متن فارسی", POLICY)
        self.assertEqual(status, "no_template")
        self.assertEqual(info["reason"], "no_verified_license_template")

    def test_license_container_templates_are_ignored(self):
        raw = "{{license container begin}}{{PD-old}}{{license container end}} متن فارسی"
        status, info = EX.classify_wikisource_license(raw, POLICY)
        self.assertEqual(status, "accepted", info)
        self.assertEqual(info["template"], "PD-old")

    def test_actual_matched_template_and_exact_rejection_reason_in_metadata(self):
        path = self.tmp / "licenses.xml.bz2"
        body = ("<mediawiki xmlns=\"http://www.mediawiki.org/xml/export-0.11/\">"
                "<page><title>صفحه پذیرفته</title><ns>0</ns><id>1</id><revision><text>"
                "{{CC-BY-SA-4.0|author=fixture}} این متن فارسی آزمایشی است و پروانه مشخصی دارد."
                "</text></revision></page>"
                "<page><title>صفحه مردود</title><ns>0</ns><id>2</id><revision><text>"
                "{{CC-BY-NC-SA-4.0}} این متن فارسی آزمایشی است ولی پروانه ناسازگار دارد."
                "</text></revision></page></mediawiki>")
        with bz2.open(path, "wt", encoding="utf-8") as f:
            f.write(body)
        ctx = context(self.tmp, path)
        docs = list(EX.extract_mediawiki_xml_bz2(ctx))
        self.assertEqual(len(docs), 1)
        meta = docs[0]["meta"]
        self.assertEqual(meta["license_template"], "CC-BY-SA-4.0")
        self.assertEqual(meta["license_template_normalized"], "cc-by-sa-4.0")
        self.assertIn("CC BY-SA 4.0", meta["license_template_license_id"])
        self.assertEqual(meta["license_verification_method"], "explicit_allowlist")
        exact = "license_template_rejected:cc-by-nc-sa-4.0:non_commercial"
        self.assertIn(exact, ctx.stats["skipped_by_reason"])
        rejection = ctx.stats["license_template_rejections"][exact]
        self.assertEqual(rejection["template"], "CC-BY-NC-SA-4.0")
        self.assertEqual(rejection["reason"], "non_commercial")


if __name__ == "__main__":
    unittest.main()
