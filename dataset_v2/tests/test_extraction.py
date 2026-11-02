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
from common import load_config  # noqa: E402

CFG = load_config(ROOT / "dataset_v2" / "configs" / "dataset_config.json")


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
    }
    base.update(kw)
    return base


def context(tmp: Path, paths, options, src=None):
    return EX.ExtractContext(
        source=src or source(), options=options, raw_paths=list(paths),
        parts_dir=None, workdir=tmp, logger=DummyLogger())


def add_tar_member(tf: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


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
            "namespace": 0, "include_title": True, "exclude_disambiguation": True})
        docs = list(EX.extract_mediawiki_xml_bz2(ctx))
        self.assertEqual(len(docs), 1)
        self.assertIn("پایتخت ایران", docs[0]["text"])
        self.assertTrue(docs[0]["meta"]["free_license_template"])
        self.assertIn("no_free_license_template", ctx.stats["skipped_by_reason"])

    def test_tar_tsv_tatoeba_attribution_only_in_metadata(self):
        path = self.tmp / "sentences_detailed.tar.bz2"
        data = ("1\tfas\tسلام، امروز هوا خوب است.\tali\t2020-01-01\t2020-01-01\n"
                "2\teng\tHello, the weather is fine.\tbob\t2020-01-01\t2020-01-01\n").encode("utf-8")
        with tarfile.open(path, "w:bz2") as tf:
            add_tar_member(tf, "fas_sentences_detailed.tsv", data)
        ctx = context(self.tmp, [path], {
            "member_pattern": "*fas*sentences*.tsv", "lang_code": "fas",
            "columns": ["id", "lang", "text", "username", "date_added", "date_modified"]})
        docs = list(EX.extract_tar_tsv(ctx))
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["text"], "سلام، امروز هوا خوب است.")
        self.assertEqual(docs[0]["meta"]["username_for_attribution_only"], "ali")
        self.assertIn("Tatoeba contributor ali", docs[0]["meta"]["attribution"])
        self.assertNotIn("ali", docs[0]["text"])

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
            self.skipTest("pyarrow not installed")
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
