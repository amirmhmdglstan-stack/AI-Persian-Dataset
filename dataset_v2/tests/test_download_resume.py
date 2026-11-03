#!/usr/bin/env python3
"""Synthetic download resume/checkpoint tests; no real network is used."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dataset_v2" / "pipeline"))
import download_sources as DS  # noqa: E402
from common import sha256_file  # noqa: E402

URL = "https://downloads.tatoeba.org/exports/per_language/pes/pes_sentences_detailed.tsv.bz2"
OTHER_URL = "https://example.org/different-snapshot.bz2"
LAST_MODIFIED = "Sat, 31 Oct 2026 06:39:00 GMT"


class DummyLogger:
    def __init__(self):
        self.warnings = []

    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        self.warnings.append(" ".join(str(x) for x in a))


class FakeResponse:
    def __init__(self, data: bytes, headers=None, status: int = 200):
        self._io = io.BytesIO(data)
        self.headers = headers or {}
        self.status = status

    def read(self, n: int = -1) -> bytes:
        return self._io.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest_record(dest: Path, data: bytes, *, url: str = URL, **extra):
    rec = {
        "path": str(dest), "filename": dest.name, "source_id": "tatoeba_fa_cc_by",
        "url": url, "size": len(data), "sha1": None, "sha256": sha(data),
        "status": "downloaded", "mutable_weekly_snapshot": True,
        "snapshot_date": "2026-10-31", "snapshot_last_modified": LAST_MODIFIED,
        "checksum_algorithm": "sha256",
    }
    rec.update(extra)
    return rec


class TestDownloadResume(unittest.TestCase):
    def setUp(self):
        self.tmp_obj = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmp_obj.name)
        self.logger = DummyLogger()

    def tearDown(self):
        self.tmp_obj.cleanup()

    def no_network(self, req, timeout=None):
        raise AssertionError("urlopen must not be called for a verified ordinary resume")

    def test_manifest_identity_requires_url_and_filename(self):
        dest = self.tmp / "pes_sentences_detailed.tsv.bz2"
        data = b"valid"
        manifest = {"sources": {"tatoeba_fa_cc_by": {"files": [
            manifest_record(dest, data),
            manifest_record(dest, data, url=OTHER_URL, filename="different.bz2"),
        ]}}}
        self.assertIsNotNone(DS.manifest_record_for(manifest, "tatoeba_fa_cc_by", URL, dest))
        self.assertIsNone(DS.manifest_record_for(manifest, "tatoeba_fa_cc_by", OTHER_URL, dest))
        self.assertIsNone(DS.manifest_record_for(manifest, "another_source", URL, dest))

    def test_valid_completed_file_skips_without_network(self):
        dest = self.tmp / "pes_sentences_detailed.tsv.bz2"
        data = b"verified weekly snapshot"
        dest.write_bytes(data)
        rec = manifest_record(dest, data)
        with patch("download_sources.urllib.request.urlopen", new=self.no_network):
            out = DS.download_one(URL, dest, logger=self.logger,
                                  manifest_record=rec,
                                  mutable_weekly_snapshot=True,
                                  catalog_snapshot_date="2026-10-31",
                                  source_id="tatoeba_fa_cc_by")
        self.assertEqual(out["status"], "already_verified_from_manifest")
        self.assertEqual(out["size"], len(data))
        self.assertEqual(out["sha256"], sha(data))
        self.assertEqual(out["snapshot_date"], "2026-10-31")
        self.assertTrue(out["mutable_weekly_snapshot"])
        self.assertFalse(out["resumed"])

    def test_corrupted_completed_file_is_not_trusted(self):
        dest = self.tmp / "pes_sentences_detailed.tsv.bz2"
        original = b"verified weekly snapshot"
        corrupted = b"X" * len(original)  # same length, different bytes
        self.assertEqual(len(original), len(corrupted))
        dest.write_bytes(corrupted)
        rec = manifest_record(dest, original)
        seen = []

        def fake(req, timeout=None):
            seen.append(req)
            return FakeResponse(original, headers={"Last-Modified": LAST_MODIFIED})

        with patch("download_sources.urllib.request.urlopen", new=fake):
            out = DS.download_one(URL, dest, logger=self.logger,
                                  manifest_record=rec,
                                  mutable_weekly_snapshot=True,
                                  catalog_snapshot_date="2026-10-31")
        self.assertEqual(len(seen), 1)
        self.assertEqual(dest.read_bytes(), original)
        self.assertEqual(out["status"], "downloaded")
        self.assertEqual(out["sha256"], sha(original))
        self.assertTrue(any("manifest identity failed" in w for w in self.logger.warnings))

    def test_interrupted_partial_resumes_with_range(self):
        dest = self.tmp / "pes_sentences_detailed.tsv.bz2"
        part = dest.with_name(dest.name + ".part")
        full = b"12345678"
        part.write_bytes(full[:4])
        part.with_name(part.name + ".json").write_text(json.dumps({
            "url": URL, "size": 4, "sha256": sha(full[:4]),
            "last_modified": LAST_MODIFIED, "etag": ""}))
        seen = []

        def fake(req, timeout=None):
            seen.append(req)
            return FakeResponse(full[4:], headers={
                "Content-Range": "bytes 4-7/8", "Content-Length": "4",
                "Last-Modified": LAST_MODIFIED,
            }, status=206)

        with patch("download_sources.urllib.request.urlopen", new=fake):
            out = DS.download_one(URL, dest, expected_size=8, logger=self.logger,
                                  mutable_weekly_snapshot=True,
                                  catalog_snapshot_date="2026-10-31")
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].get_header("Range"), "bytes=4-")
        self.assertEqual(dest.read_bytes(), full)
        self.assertFalse(part.exists())
        self.assertTrue(out["resumed"])
        self.assertEqual(out["size"], 8)
        self.assertEqual(out["snapshot_date"], "2026-10-31")
        self.assertEqual(out["catalog_snapshot_date"], "2026-10-31")

    def test_forced_refresh_redownloads_verified_file(self):
        dest = self.tmp / "pes_sentences_detailed.tsv.bz2"
        old = b"verified weekly snapshot"
        new = b"new weekly snapshot!!!!!"  # same length as old
        self.assertEqual(len(old), len(new))
        dest.write_bytes(old)
        rec = manifest_record(dest, old)
        seen = []

        def fake(req, timeout=None):
            seen.append(req)
            return FakeResponse(new, headers={"Last-Modified": "Sat, 07 Nov 2026 06:39:00 GMT"})

        with patch("download_sources.urllib.request.urlopen", new=fake):
            out = DS.download_one(URL, dest, logger=self.logger,
                                  manifest_record=rec, force=True,
                                  mutable_weekly_snapshot=True,
                                  source_id="tatoeba_fa_cc_by")
        self.assertEqual(len(seen), 1)
        self.assertEqual(dest.read_bytes(), new)
        self.assertEqual(out["status"], "downloaded")
        self.assertTrue(out["force_refresh"])
        self.assertEqual(out["snapshot_date"], "2026-11-07")
        self.assertEqual(out["sha256"], sha(new))

    def test_filename_alone_is_not_identity(self):
        dest = self.tmp / "pes_sentences_detailed.tsv.bz2"
        data = b"verified weekly snapshot"
        dest.write_bytes(data)
        rec = manifest_record(dest, data, url=OTHER_URL)
        seen = []

        def fake(req, timeout=None):
            seen.append(req)
            return FakeResponse(data, headers={"Last-Modified": LAST_MODIFIED})

        with patch("download_sources.urllib.request.urlopen", new=fake):
            out = DS.download_one(URL, dest, logger=self.logger,
                                  manifest_record=rec,
                                  mutable_weekly_snapshot=True)
        self.assertEqual(len(seen), 1)
        self.assertEqual(out["status"], "downloaded")
        self.assertTrue(any("does not match URL + filename" in w
                            or "no matching URL+filename manifest" in w
                            for w in self.logger.warnings))

    def test_main_loads_existing_manifest_and_passes_matching_record(self):
        work = self.tmp / "work"
        work.mkdir()
        config_path = self.tmp / "config.json"
        meta_path = self.tmp / "metadata.json"
        filename = "pes_sentences_detailed.tsv.bz2"
        data = b"verified weekly snapshot"
        config = {
            "config_version": "test", "default_profiles": ["A"],
            "workdir": str(work), "min_free_gb": 0,
            "source_metadata": str(meta_path),
        }
        config_path.write_text(json.dumps(config), encoding="utf-8")
        meta = {"sources": [{
            "id": "tatoeba_fa_cc_by", "name": "fixture", "profile": "A",
            "download": True, "decision": "include",
            "license_content": "CC BY 2.0 FR", "files": [{
                "url": URL, "filename": filename, "format": "tsv.bz2",
                "extractor": "tatoeba_detailed", "required": True,
                "reported_size_bytes": None, "sha1": None,
                "snapshot_date": "2026-10-31", "mutable_weekly_snapshot": True,
                "options": {},
            }],
        }]}
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        dest = work / "raw" / "tatoeba_fa_cc_by" / filename
        manifest = {"version": DS.MANIFEST_VERSION, "sources": {
            "tatoeba_fa_cc_by": {"files": [manifest_record(dest, data)]}}}
        (work / "download_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        seen = {}

        def fake_download(url, dest, **kwargs):
            seen.update({"url": url, "dest": dest,
                         "manifest_record": kwargs.get("manifest_record"),
                         "force": kwargs.get("force"),
                         "mutable": kwargs.get("mutable_weekly_snapshot"),
                         "catalog_date": kwargs.get("catalog_snapshot_date")})
            return manifest_record(dest, data, **{"status": "already_verified_from_manifest"})

        argv = ["download_sources.py", "--config", str(config_path),
                "--metadata", str(meta_path), "--workdir", str(work),
                "--profiles", "A", "--min-free-gb", "0", "--timeout", "1"]
        with patch.object(sys, "argv", argv), \
             patch.object(DS, "download_one", new=fake_download), \
             patch("download_sources.urllib.request.urlopen", new=self.no_network), \
             contextlib.redirect_stdout(io.StringIO()):
            DS.main()
        self.assertEqual(seen["url"], URL)
        self.assertEqual(seen["dest"].name, filename)
        self.assertIsNotNone(seen["manifest_record"])
        self.assertEqual(seen["manifest_record"]["url"], URL)
        self.assertEqual(seen["manifest_record"]["sha256"], sha(data))
        self.assertFalse(seen["force"])
        self.assertTrue(seen["mutable"])
        self.assertEqual(seen["catalog_date"], "2026-10-31")
        written = json.loads((work / "download_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(written["mutable_snapshot_policy"]["identity"],
                         "source URL + destination filename + actual SHA-256 + actual size")

    def test_first_download_records_then_ordinary_resume_skips_snapshot(self):
        dest = self.tmp / "pes_sentences_detailed.tsv.bz2"
        data = b"weekly snapshot bytes"
        seen = []

        def fake(req, timeout=None):
            seen.append(req)
            return FakeResponse(data, headers={"Last-Modified": LAST_MODIFIED,
                                              "ETag": '"pes-20261031"'})

        with patch("download_sources.urllib.request.urlopen", new=fake):
            first = DS.download_one(URL, dest, logger=self.logger,
                                    mutable_weekly_snapshot=True,
                                    catalog_snapshot_date="2026-10-31",
                                    source_id="tatoeba_fa_cc_by")
        self.assertEqual(sha256_file(dest), sha(data))
        self.assertEqual(first["snapshot_date"], "2026-10-31")
        self.assertEqual(first["snapshot_last_modified"], LAST_MODIFIED)
        self.assertEqual(first["snapshot_etag"], '"pes-20261031"')
        self.assertEqual(first["sha256"], sha(data))
        self.assertEqual(first["size"], len(data))

        with patch("download_sources.urllib.request.urlopen", new=self.no_network):
            second = DS.download_one(URL, dest, logger=self.logger,
                                     manifest_record=first,
                                     mutable_weekly_snapshot=True,
                                     source_id="tatoeba_fa_cc_by")
        self.assertEqual(second["status"], "already_verified_from_manifest")
        self.assertEqual(second["snapshot_date"], "2026-10-31")
        self.assertEqual(len(seen), 1)


if __name__ == "__main__":
    unittest.main()
