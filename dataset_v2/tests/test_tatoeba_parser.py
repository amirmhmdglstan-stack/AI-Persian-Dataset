#!/usr/bin/env python3
"""Synthetic tests for the verified Tatoeba `pes` detailed export parser."""
from __future__ import annotations

import bz2
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
from common import config_source_metadata, load_config  # noqa: E402

FIELDS = ["id", "lang", "text", "username", "date_added", "date_last_modified"]


class DummyLogger:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


def source():
    return {
        "id": "tatoeba_fa_cc_by", "name": "Tatoeba fixture", "profile": "A",
        "decision": "include", "domain": "dialogue",
        "license_content": "CC BY 2.0 FR by default; some CC0",
        "copyright_holder": "Tatoeba contributors",
        "content_rights_verified": True,
        "url": "https://downloads.tatoeba.org/exports/per_language/pes/",
    }


def context(tmp: Path, paths, options):
    return EX.ExtractContext(
        source=source(), options=options, raw_paths=list(paths),
        parts_dir=None, workdir=tmp, logger=DummyLogger())


def row(*values) -> str:
    return "\t".join(str(v) for v in values) + "\n"


def tar_member(tf: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


class TestTatoebaParser(unittest.TestCase):
    def setUp(self):
        self.tmp_obj = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmp_obj.name)

    def tearDown(self):
        self.tmp_obj.cleanup()

    def options(self, **extra):
        opts = {"language_code": "pes", "columns": list(FIELDS),
                "one_doc_per_line": True}
        opts.update(extra)
        return opts

    def test_catalog_pins_verified_pes_export_and_real_csv_fallback(self):
        cfg = load_config(ROOT / "dataset_v2" / "configs" / "dataset_config.json")
        meta = json.loads(config_source_metadata(cfg).read_text(encoding="utf-8"))
        s = next(x for x in meta["sources"] if x["id"] == "tatoeba_fa_cc_by")
        primary, fallback = s["files"]
        self.assertEqual(primary["filename"], "pes_sentences_detailed.tsv.bz2")
        self.assertTrue(primary["url"].endswith(
            "per_language/pes/pes_sentences_detailed.tsv.bz2"))
        self.assertEqual(primary["format"], "tsv.bz2")
        self.assertEqual(primary["extractor"], "tatoeba_detailed")
        self.assertEqual(primary["options"]["language_code"], "pes")
        self.assertEqual(primary["observed_size_bytes"], 695149)
        self.assertEqual(primary["snapshot_date"], "2026-10-31")
        self.assertTrue(primary["mutable_weekly_snapshot"])
        self.assertIsNone(primary["reported_size_bytes"])
        self.assertEqual(fallback["filename"], "sentences_detailed.tar.bz2")
        self.assertFalse(fallback["required"])
        self.assertEqual(fallback["options"]["fallback_member"], "sentences_detailed.csv")
        self.assertNotIn("fas", primary["filename"])
        self.assertNotIn("fas", json.dumps(primary["options"], ensure_ascii=False))
        self.assertNotIn("fas", json.dumps(fallback["options"], ensure_ascii=False))

    def test_mysql_outfile_escapes_and_null(self):
        line = "\t".join([
            "101", "pes", "سلام\\tدوستان و مسیر C:\\\\temp", "ali",
            "2026-10-31 06:39:00", "\\N",
        ])
        fields = EX.parse_mysql_outfile_fields(line)
        self.assertEqual(len(fields), 6)
        self.assertEqual(fields[0], "101")
        self.assertEqual(fields[1], "pes")
        self.assertEqual(fields[2], "سلام\tدوستان و مسیر C:\\temp")
        self.assertEqual(fields[3], "ali")
        self.assertEqual(fields[4], "2026-10-31 06:39:00")
        self.assertIsNone(fields[5])
        with self.assertRaises(ValueError):
            EX.parse_mysql_outfile_fields(
                "\t".join(["1", "pes", "bad\\qescape", "ali", "\\N", "\\N"]))

    def test_pes_only_bzip2_tsv_fields_metadata_and_attribution(self):
        path = self.tmp / "pes_sentences_detailed.tsv.bz2"
        lines = [
            row("101", "pes", "سلام، امروز هوا خوب است.", "ali",
                "2026-10-31 06:39:00", "2026-10-31 06:39:00"),
            row("102", "eng", "Hello, the weather is fine.", "bob", "\\N", "\\N"),
            row("103", "fas", "این کد زبان قدیمی باید رد شود.", "cam", "\\N", "\\N"),
            row("104", "pes", "جمله\\tبا یک tab درون متن", "سارا",
                "2026-10-31 06:39:00", "2026-11-01 06:39:00"),
            row("105", "pes", "جمله بدون نام مشارکت\u200cکننده", "\\N",
                "2026-10-31 06:39:00", "\\N"),
        ]
        with bz2.open(path, "wt", encoding="utf-8", newline="") as f:
            f.writelines(lines)
        ctx = context(self.tmp, [path], self.options())
        docs = list(EX.extract_tatoeba_detailed(ctx))
        self.assertEqual(len(docs), 3)
        self.assertGreaterEqual(ctx.stats["skipped_by_reason"].get("other_language", 0), 2)
        by_id = {d["meta"]["tatoeba_sentence_id"]: d for d in docs}
        first = by_id["101"]
        self.assertEqual(first["text"], "سلام، امروز هوا خوب است.")
        self.assertEqual(first["meta"]["tatoeba_language_code"], "pes")
        self.assertEqual(first["meta"]["tatoeba_username_for_attribution_only"], "ali")
        self.assertEqual(first["meta"]["tatoeba_date_added"], "2026-10-31 06:39:00")
        self.assertEqual(first["meta"]["tatoeba_date_last_modified"], "2026-10-31 06:39:00")
        self.assertEqual(first["meta"]["source_filename"], "pes_sentences_detailed.tsv.bz2")
        self.assertEqual(first["meta"]["raw_file"], "pes_sentences_detailed.tsv.bz2")
        self.assertIn("Tatoeba contributor ali", first["meta"]["attribution"])
        self.assertIn("CC BY 2.0 FR", first["meta"]["attribution"])
        self.assertTrue(first["meta"]["attribution_url"].endswith("/101"))
        self.assertTrue(first["meta"]["attribution_required"])
        self.assertFalse(first["meta"]["attribution_in_training_text"])
        self.assertNotIn("ali", first["text"])
        self.assertNotIn("CC BY", first["text"])
        self.assertNotIn("101", first["text"])
        self.assertIn("\t", by_id["104"]["text"])
        self.assertEqual(by_id["105"]["meta"]["tatoeba_username_for_attribution_only"], "unknown")
        self.assertNotIn("\\N", by_id["105"]["text"])

    def test_fas_code_is_refused(self):
        path = self.tmp / "pes_sentences_detailed.tsv.bz2"
        with bz2.open(path, "wt", encoding="utf-8", newline="") as f:
            f.write(row("1", "fas", "یک جمله فارسی", "ali", "\\N", "\\N"))
        ctx = context(self.tmp, [path], {"lang_code": "fas", "columns": list(FIELDS)})
        with self.assertRaisesRegex(RuntimeError, "not 'fas'"):
            list(EX.extract_tatoeba_detailed(ctx))

    def test_real_all_language_csv_member_fallback(self):
        path = self.tmp / "sentences_detailed.tar.bz2"
        csv_data = (
            row("201", "pes", "این جمله از عضو CSV است.", "reza", "\\N", "\\N")
            + row("202", "eng", "This row is not Persian.", "bob", "\\N", "\\N")
        ).encode("utf-8")
        fake_tsv = row("301", "pes", "این عضو خیالی نباید خوانده شود.", "ali", "\\N", "\\N").encode("utf-8")
        with tarfile.open(path, "w:bz2") as tf:
            tar_member(tf, "sentences_detailed.csv", csv_data)
            tar_member(tf, "fas_sentences_detailed.tsv", fake_tsv)
        ctx = context(self.tmp, [path], self.options(
            fallback_member=EX.TATOEBA_ALL_LANGUAGE_MEMBER))
        docs = list(EX.extract_tatoeba_detailed(ctx))
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["text"], "این جمله از عضو CSV است.")
        self.assertEqual(docs[0]["meta"]["member"], "sentences_detailed.csv")
        self.assertEqual(docs[0]["meta"]["source_filename"], "sentences_detailed.tar.bz2")
        self.assertNotIn("301", " ".join(d["meta"]["tatoeba_sentence_id"] for d in docs))
        self.assertEqual(ctx.stats["skipped_by_reason"].get("other_language", 0), 1)

    def test_fallback_without_real_csv_member_is_rejected(self):
        path = self.tmp / "sentences_detailed.tar.bz2"
        with tarfile.open(path, "w:bz2") as tf:
            tar_member(tf, "fas_sentences_detailed.tsv", b"1\tfa\told\n")
        ctx = context(self.tmp, [path], self.options())
        with self.assertRaisesRegex(RuntimeError, "sentences_detailed.csv"):
            list(EX.extract_tatoeba_detailed(ctx))
        self.assertIn("missing_sentences_detailed_csv_member", ctx.stats["skipped_by_reason"])


if __name__ == "__main__":
    unittest.main()
