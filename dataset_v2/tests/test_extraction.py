#!/usr/bin/env python3
"""Small parsing tests for the streaming extractors (synthetic fixtures only)."""
from __future__ import annotations

import bz2
import csv
import gzip
import io
import json
import sys
import tarfile
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


def source(**kw):
    base = {
        "id": "fixture", "name": "fixture", "profile": "A", "decision": "include",
        "domain": "encyclopedia", "license_content": "CC0-1.0 test fixture",
        "copyright_holder": "unit test", "content_rights_verified": True,
        "url": "https://fa.wikipedia.org/",
        "page_base_url": "https://fa.wikipedia.org/wiki/",
    }
    base.update(kw)
    return base


def context(tmp: Path, paths, options, src=None, policy=None):
    return EX.ExtractContext(
        source=src or source(), options=options, raw_paths=list(paths),
        parts_dir=None, workdir=tmp, logger=DummyLogger(),
        policy=policy or POLICY)


def add_tar_member(tf: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


def mediawiki_xml(title: str, text: str, page_id: str = "101") -> str:
    return f"""<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">
<page><title>{title}</title><ns>0</ns><id>{page_id}</id><revision><text>{text}</text></revision></page>
</mediawiki>"""


class TestExtraction(unittest.TestCase):
    def setUp(self):
        self.tmp_obj = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmp_obj.name)

    def tearDown(self):
        self.tmp_obj.cleanup()

    def test_wikitext_conversion_keeps_only_piped_link_label(self):
        raw = "{{PD-old}} این [[تهران|پایتخت ایران]] است. <ref>منبع</ref> و '''متن''' دارد."
        text = EX.wikitext_to_text(raw)
        self.assertIn("پایتخت ایران", text)
        self.assertNotIn("تهران|", text)
        self.assertNotIn("{{", text)
        self.assertNotIn("<ref>", text)

    def test_mediawiki_xml_bz2_streaming_and_license_filter(self):
        xml = """<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">
<page><title>صفحه آزمایشی</title><ns>0</ns><id>101</id><revision><text>{{PD-old}}
این یک متن فارسی آزمایشی است که در ویکی\u200cپدیا قرار دارد. [[تهران|پایتخت ایران]] شهری بزرگ است.</text></revision></page>
<page><title>صفحه بدون پروانه</title><ns>0</ns><id>102</id><revision><text>
این صفحه هیچ پروانه\u200cای ندارد و باید کنار گذاشته شود تا حقوق متن بررسی شود.</text></revision></page>
</mediawiki>"""
        path = self.tmp / "small.xml.bz2"
        with bz2.open(path, "wt", encoding="utf-8") as f:
            f.write(xml)
        ctx = context(self.tmp, [path], {
            "require_free_license_template": True, "min_raw_chars": 10,
            "namespace": 0, "include_title": True, "exclude_disambiguation": True,
            "page_base_url": "https://fa.wikisource.org/wiki/"})
        docs = list(EX.extract_mediawiki_xml_bz2(ctx))
        self.assertEqual(len(docs), 1)
        self.assertIn("پایتخت ایران", docs[0]["text"])
        self.assertTrue(docs[0]["meta"]["free_license_template"])
        self.assertEqual(docs[0]["meta"]["license_template"], "PD-old")
        self.assertIn("license_template_no_template:none:no_verified_license_template",
                      ctx.stats["skipped_by_reason"])
        self.assertTrue(docs[0]["meta"]["page_url"].startswith(
            "https://fa.wikisource.org/wiki/"))

    def test_mediawiki_page_base_urls_for_four_projects(self):
        domains = {
            "wikipedia": "https://fa.wikipedia.org/wiki/",
            "wikibooks": "https://fa.wikibooks.org/wiki/",
            "wikisource": "https://fa.wikisource.org/wiki/",
            "wikinews": "https://fa.wikinews.org/wiki/",
        }
        for project, base in domains.items():
            with self.subTest(project=project):
                path = self.tmp / f"{project}.xml.bz2"
                xml = mediawiki_xml(
                    "صفحه آزمایشی",
                    "این یک متن فارسی آزمایشی کافی برای استخراج است و هیچ الگوی پروانه ندارد.")
                with bz2.open(path, "wt", encoding="utf-8") as f:
                    f.write(xml)
                ctx = context(self.tmp, [path], {
                    "namespace": 0, "min_raw_chars": 10, "include_title": True,
                    "exclude_disambiguation": True, "page_base_url": base})
                docs = list(EX.extract_mediawiki_xml_bz2(ctx))
                self.assertEqual(len(docs), 1)
                self.assertEqual(docs[0]["meta"]["page_base_url"], base)
                self.assertTrue(docs[0]["meta"]["page_url"].startswith(base))
                if project != "wikipedia":
                    self.assertFalse(docs[0]["meta"]["page_url"].startswith(
                        "https://fa.wikipedia.org/wiki/"))

    def test_mediawiki_refuses_missing_page_base_url(self):
        path = self.tmp / "missing-base.xml.bz2"
        with bz2.open(path, "wt", encoding="utf-8") as f:
            f.write(mediawiki_xml("صفحه", "این متن فارسی آزمایشی است."))
        src = source(page_base_url=None)
        ctx = context(self.tmp, [path], {"namespace": 0, "min_raw_chars": 10}, src=src)
        with self.assertRaisesRegex(RuntimeError, "no page_base_url"):
            list(EX.extract_mediawiki_xml_bz2(ctx))

    def test_txt_lines_one_document_per_natural_sentence(self):
        path = self.tmp / "sentences.txt"
        path.write_text("سلام، حال شما چطور است؟\nامروز هوا آفتابی است.\nکتاب را خواندم.\n",
                        encoding="utf-8")
        ctx = context(self.tmp, [path], {"one_doc_per_line": True, "min_line_chars": 5})
        docs = list(EX.extract_txt_lines(ctx))
        self.assertEqual(len(docs), 3)
        self.assertEqual(docs[0]["text"], "سلام، حال شما چطور است؟")

    def test_csv_mapping(self):
        path = self.tmp / "articles.csv"
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "title", "article"])
            w.writerow(["1", "شهر تهران", "تهران پایتخت ایران است و شهری بزرگ محسوب می\u200cشود."])
            w.writerow(["2", "زبان فارسی", "زبان فارسی در ایران رایج است و تاریخی طولانی دارد."])
        ctx = context(self.tmp, [path], {
            "delimiter": ",", "text_column": "article",
            "title_column": "title", "id_column": "id", "meta_columns": []})
        docs = list(EX.extract_csv(ctx))
        self.assertEqual(len(docs), 2)
        self.assertIn("شهر تهران", docs[0]["text"])
        self.assertEqual(docs[1]["title"], "زبان فارسی")

    def test_jsonl_gz_excludes_synthetic_query_fields(self):
        path = self.tmp / "docs.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(json.dumps({"id": "a", "text": "این متن طبیعی فارسی است.",
                                "query_questions": ["پرسش مصنوعی"]}, ensure_ascii=False) + "\n")
        ctx = context(self.tmp, [path], {
            "text_fields": ["text"], "exclude_field_prefixes": ["query_", "judge_"],
            "id_field": "id"})
        docs = list(EX.extract_jsonl(ctx))
        self.assertEqual(len(docs), 1)
        self.assertNotIn("مصنوعی", docs[0]["text"])
        self.assertIn("query_", docs[0]["meta"]["excluded_synthetic_prefixes"][0])

    def test_parquet_excludes_generated_query_columns(self):
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore
        except ImportError:
            self.skipTest("optional pyarrow not installed")
        path = self.tmp / "rows.parquet"
        table = pa.table({
            "id": ["1", "2"],
            "title": ["عنوان یک", "عنوان دو"],
            "text": ["این یک متن طبیعی فارسی است.", "متن دیگری به زبان فارسی نوشته شده است."],
            "query_questions": [["پرسش مصنوعی"], ["پرسش دیگر"]],
        })
        pq.write_table(table, path)
        ctx = context(self.tmp, [path], {
            "text_fields": ["text"], "title_field": "title", "id_field": "id",
            "exclude_field_prefixes": ["query_", "judge_"], "batch_size": 1})
        docs = list(EX.extract_parquet(ctx))
        self.assertEqual(len(docs), 2)
        self.assertNotIn("مصنوعی", " ".join(d["text"] for d in docs))
        self.assertIn("query_", docs[0]["meta"]["excluded_synthetic_prefixes"][0])

    def test_public_domain_allowlist_in_txt_tree(self):
        path = self.tmp / "ganjoor.tar.gz"
        with tarfile.open(path, "w:gz") as tf:
            add_tar_member(tf, "repo/txt/hafez/ghazal1.txt",
                           "الا یا ایها الساقی ادر کاسا و ناولها\nخطاب به ساقی است.\n".encode("utf-8"))
            add_tar_member(tf, "repo/txt/modern/poem.txt",
                           "این اثر متعلق به شاعری معاصر است.\n".encode("utf-8"))
        ctx = context(self.tmp, [path], {
            "subdir": "txt", "allowed_dirs": ["hafez"], "min_chars": 10,
            "one_doc_per_file": True})
        docs = list(EX.extract_zip_txt_tree(ctx))
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["meta"]["book"], "hafez")
        self.assertIn("book_not_in_public_domain_allowlist", ctx.stats["skipped_by_reason"])


if __name__ == "__main__":
    unittest.main()
