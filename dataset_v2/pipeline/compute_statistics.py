#!/usr/bin/env python3
"""Compute measured V2 statistics directly from the final files.

All numeric corpus values in the output are MEASURED from the files on disk.
Estimated full-corpus expectations live in reports/source_research.md and are
never mixed into this file without an explicit label.  Exact token counts use
the unchanged V1 32K byte-level BPE tokenizer.  Both content tokens and counts
including one <|endoftext|> token per document are reported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DOC_SEPARATOR, config_tokenizer, configure_logging, load_config, now_utc,
    peak_rss_mb, sha256_file,
)

RE_PERSIAN_LETTER = re.compile("[آ-یءأإئؤۀٔ]")
RE_LATIN_LETTER = re.compile(r"[A-Za-z]")
RE_PERSIAN_DIGIT = re.compile("[۰-۹]")
RE_LATIN_DIGIT = re.compile("[0-9]")
RE_WORD = re.compile(r"\S+")
RE_SENT = re.compile(r"[.!؟]+")
RE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ufffd]")


class HashStore:
    """Disk-backed exact-duplicate hash set; bounded RAM.

    Tracks global uniqueness and per-file duplicates separately so that a
    multi-million-document corpus never requires an in-memory Python set.
    """

    def __init__(self, path: Path):
        if path.exists():
            path.unlink()
        self.conn = sqlite3.connect(str(path))
        self.conn.execute("PRAGMA journal_mode=OFF")
        self.conn.execute("PRAGMA synchronous=OFF")
        self.conn.execute("PRAGMA cache_size=-32768")
        self.conn.execute("CREATE TABLE hashes (h BLOB PRIMARY KEY, first_file TEXT, first_split TEXT)")
        self.conn.execute("CREATE TABLE file_hashes (file_key TEXT, h BLOB, PRIMARY KEY(file_key, h)) WITHOUT ROWID")

    def add(self, text: str, file_key: str, split: str) -> tuple[bool, bool, str]:
        h = hashlib.blake2b(re.sub(r"\s+", " ", text).encode("utf-8"),
                            digest_size=24).digest()
        g = self.conn.execute(
            "INSERT OR IGNORE INTO hashes(h, first_file, first_split) VALUES(?,?,?)",
            (h, file_key, split))
        global_new = g.rowcount > 0
        first_file = file_key
        if not global_new:
            row = self.conn.execute("SELECT first_file FROM hashes WHERE h=?", (h,)).fetchone()
            first_file = row[0] if row else "unknown"
        f = self.conn.execute(
            "INSERT OR IGNORE INTO file_hashes(file_key, h) VALUES(?,?)", (file_key, h))
        return global_new, f.rowcount > 0, first_file

    def first_file(self, text: str) -> str | None:
        h = hashlib.blake2b(re.sub(r"\s+", " ", text).encode("utf-8"),
                            digest_size=24).digest()
        row = self.conn.execute("SELECT first_file FROM hashes WHERE h=?", (h,)).fetchone()
        return row[0] if row else None

    def count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM hashes").fetchone()[0])

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()


def iter_documents(path: Path) -> Iterator[str]:
    """Stream documents separated by a line containing exactly <|endoftext|>."""
    buf: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.rstrip("\n") == DOC_SEPARATOR:
                doc = "".join(buf).strip("\n")
                if doc.strip():
                    yield doc
                buf = []
            else:
                buf.append(line)
    tail = "".join(buf).strip("\n")
    if tail.strip():
        yield tail


def scan_file(path: Path, tokenizer, hash_store: HashStore, profile: str, split: str) -> dict:
    file_key = f"profile_{profile}/{split}.txt"
    docs = chars = words = persian_words = lines = sentences = 0
    persian_letters = latin_letters = persian_digits = latin_digits = zwnj = 0
    min_len, max_len, len_sum = 10 ** 12, 0, 0
    dup_in_file = dup_global = 0
    tokens_content = 0
    t0 = time.time()
    size = path.stat().st_size
    with open(path, "rb") as f:
        bom = f.read(3).startswith(b"\xef\xbb\xbf")
    for text in iter_documents(path):
        docs += 1
        n = len(text)
        chars += n
        len_sum += n
        min_len = min(min_len, n)
        max_len = max(max_len, n)
        lines += text.count("\n") + 1
        words += len(RE_WORD.findall(text))
        persian_words += sum(1 for w in RE_WORD.findall(text) if RE_PERSIAN_LETTER.search(w))
        persian_letters += len(RE_PERSIAN_LETTER.findall(text))
        latin_letters += len(RE_LATIN_LETTER.findall(text))
        persian_digits += len(RE_PERSIAN_DIGIT.findall(text))
        latin_digits += len(RE_LATIN_DIGIT.findall(text))
        zwnj += text.count("\u200c")
        sentences += len([s for s in RE_SENT.split(text) if s.strip()])
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        tokens_content += len(ids)
        global_new, file_new, _first = hash_store.add(text, file_key, split)
        if not file_new:
            dup_in_file += 1
        if not global_new:
            dup_global += 1
    return {
        "path": str(path), "profile": profile, "split": split,
        "bytes": size, "bom": bom,
        "documents": docs, "characters": chars, "words": words,
        "persian_words": persian_words, "sentences": sentences, "lines": lines,
        "persian_letters": persian_letters, "latin_letters": latin_letters,
        "persian_digits": persian_digits, "latin_digits": latin_digits,
        "zwnj": zwnj,
        "characters_mean_per_doc": round(len_sum / max(1, docs), 1),
        "characters_min_doc": 0 if docs == 0 else min_len,
        "characters_max_doc": max_len,
        "unique_documents_in_file": docs - dup_in_file,
        "exact_duplicate_documents_in_file": dup_in_file,
        "exact_duplicate_documents_global": dup_global,
        "tokens_content_excluding_document_end": tokens_content,
        "document_end_token_count": docs,
        "tokens_including_document_end": tokens_content + docs,
        "characters_per_token": round(chars / max(1, tokens_content), 4),
        "bytes_per_token": round(size / max(1, tokens_content), 4),
        "tokens_per_1000_characters": round(1000 * tokens_content / max(1, chars), 2),
        "latin_letter_ratio_of_letters": round(
            latin_letters / max(1, persian_letters + latin_letters), 4),
        "persian_letter_ratio_of_letters": round(
            persian_letters / max(1, persian_letters + latin_letters), 4),
        "persian_letter_ratio_of_characters": round(
            persian_letters / max(1, chars), 4),
        "scan_seconds": round(time.time() - t0, 3),
    }


def read_manifests(outdir: Path, profiles: list[str]) -> dict:
    by_source: dict[str, dict] = defaultdict(lambda: {"documents": 0, "characters": 0, "by_split": defaultdict(int)})
    by_license: dict[str, dict] = defaultdict(lambda: {"documents": 0, "characters": 0})
    by_domain: dict[str, dict] = defaultdict(lambda: {"documents": 0, "characters": 0})
    by_profile: dict[str, dict] = defaultdict(lambda: {"documents": 0, "characters": 0})
    for profile in profiles:
        path = outdir / f"profile_{profile}" / "documents.jsonl"
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                sid, lic, dom = d["source_id"], d["license"], d["domain"]
                n = int(d.get("characters", 0))
                by_source[sid]["documents"] += 1
                by_source[sid]["characters"] += n
                by_source[sid]["by_split"][d["split"]] += 1
                by_license[lic]["documents"] += 1
                by_license[lic]["characters"] += n
                by_domain[dom]["documents"] += 1
                by_domain[dom]["characters"] += n
                by_profile[profile]["documents"] += 1
                by_profile[profile]["characters"] += n
    return {
        "by_source": {k: {"documents": v["documents"], "characters": v["characters"],
                           "by_split": dict(v["by_split"])} for k, v in by_source.items()},
        "by_license": dict(by_license),
        "by_domain": dict(by_domain),
        "by_profile": dict(by_profile),
    }


def source_index_stats(workdir: Path) -> dict:
    path = workdir / "corpus_index.sqlite3"
    if not path.exists():
        return {"available": False, "reason": "corpus_index.sqlite3 not found"}
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    out = {}
    for row in conn.execute(
            "SELECT source_id, "
            "SUM(CASE WHEN status='kept' AND selected=1 THEN 1 ELSE 0 END) AS selected_docs, "
            "SUM(CASE WHEN status='kept' AND selected=1 THEN chars ELSE 0 END) AS selected_chars, "
            "SUM(CASE WHEN status='rejected' THEN 1 ELSE 0 END) AS rejected_docs, "
            "SUM(CASE WHEN status='exact_duplicate' THEN 1 ELSE 0 END) AS exact_dups, "
            "SUM(CASE WHEN status='near_duplicate' THEN 1 ELSE 0 END) AS near_dups "
            "FROM docs GROUP BY source_id ORDER BY source_id"):
        out[row["source_id"]] = dict(row)
    conn.close()
    return {"available": True, "by_source": out}


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute measured statistics for the final V2 corpus")
    ap.add_argument("--config", default=None)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--reports-dir", default=None)
    ap.add_argument("--tokenizer", default=None, help="override baseline tokenizer JSON path")
    ap.add_argument("--profiles", nargs="*", default=None)
    ap.add_argument("--scope", default="full_run", choices=["full_run", "small_sample"])
    ap.add_argument("--out", default=None, help="output JSON path")
    args = ap.parse_args()

    cfg = load_config(args.config)
    workdir = Path(args.workdir) if args.workdir else Path(cfg["workdir"])
    outdir = Path(args.outdir) if args.outdir else Path(cfg["outdir"])
    reports_dir = Path(args.reports_dir) if args.reports_dir else outdir / "reports"
    tok_path = Path(args.tokenizer) if args.tokenizer else config_tokenizer(cfg)
    profiles = args.profiles or cfg.get("default_profiles", ["A"])
    logger = configure_logging(workdir, "dataset_v2.stats")
    reports_dir.mkdir(parents=True, exist_ok=True)

    try:
        from tokenizers import Tokenizer  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "Exact token counts require the 'tokenizers' package. Install with: "
            "D:\\GPT\\venv\\Scripts\\python.exe -m pip install tokenizers") from e
    if not tok_path.exists():
        raise FileNotFoundError(f"Baseline tokenizer not found: {tok_path}")
    tokenizer = Tokenizer.from_file(str(tok_path))
    eot_id = tokenizer.token_to_id(DOC_SEPARATOR)
    eot_ids = tokenizer.encode(DOC_SEPARATOR, add_special_tokens=False).ids
    if eot_id != 0 or eot_ids != [0]:
        raise RuntimeError(f"Baseline tokenizer special token check failed: id={eot_id}, ids={eot_ids}")

    hash_store = HashStore(workdir / "statistics_hashes.sqlite3")
    files: dict[str, dict] = {}
    t0 = time.time()
    for profile in profiles:
        pdir = outdir / f"profile_{profile}"
        for split in ("train", "validation", "test"):
            path = pdir / f"{split}.txt"
            if not path.exists():
                raise FileNotFoundError(f"Missing final split: {path}")
            files[f"profile_{profile}/{split}.txt"] = scan_file(path, tokenizer, hash_store, profile, split)
            logger.info("scanned %s: %s docs / %s tokens", path.name,
                        f"{files[f'profile_{profile}/{split}.txt']['documents']:,}",
                        f"{files[f'profile_{profile}/{split}.txt']['tokens_content_excluding_document_end']:,}")
    unique_docs = hash_store.count()
    hash_store.close()
    scan_seconds = round(time.time() - t0, 3)

    totals = {k: sum(v[k] for v in files.values())
              for k in ("documents", "characters", "words", "persian_words", "sentences", "lines",
                        "persian_letters", "latin_letters", "persian_digits", "latin_digits",
                        "zwnj", "tokens_content_excluding_document_end",
                        "document_end_token_count", "tokens_including_document_end", "bytes")}
    min_doc = min(v["characters_min_doc"] for v in files.values() if v["documents"])
    max_doc = max(v["characters_max_doc"] for v in files.values())
    checksums_path = outdir / "checksums.json"
    checksum_report = {"available": checksums_path.exists(), "matches": {}, "mismatches": []}
    if checksums_path.exists():
        recorded = json.loads(checksums_path.read_text(encoding="utf-8")).get("files", {})
        for key, rec in recorded.items():
            p = outdir / key
            if not p.exists():
                checksum_report["mismatches"].append({"file": key, "reason": "missing"})
                continue
            actual = sha256_file(p)
            ok = actual == rec.get("sha256")
            checksum_report["matches"][key] = ok
            if not ok:
                checksum_report["mismatches"].append({"file": key, "expected": rec.get("sha256"), "actual": actual})

    manifest_stats = read_manifests(outdir, profiles)
    index_stats = source_index_stats(workdir)
    quality_path = reports_dir / "quality_report.json"
    build_path = reports_dir / "build_report.json"
    bench_path = reports_dir / "pipeline_benchmark.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else {}
    build = json.loads(build_path.read_text(encoding="utf-8")) if build_path.exists() else {}
    benchmark = json.loads(bench_path.read_text(encoding="utf-8")) if bench_path.exists() else {}

    result = {
        "dataset": "Persian Dataset V2",
        "measurement_scope": args.scope,
        "measurement_status": "measured_from_final_files_on_disk",
        "generated_at_utc": now_utc(),
        "seed": cfg.get("seed"),
        "config_version": cfg.get("config_version"),
        "profiles": profiles,
        "files": files,
        "totals": {
            **totals,
            "unique_documents": unique_docs,
            "characters_mean_per_doc": round(totals["characters"] / max(1, totals["documents"]), 1),
            "characters_min_doc": min_doc,
            "characters_max_doc": max_doc,
            "characters_per_token": round(totals["characters"] / max(1, totals["tokens_content_excluding_document_end"]), 4),
            "bytes_per_token": round(totals["bytes"] / max(1, totals["tokens_content_excluding_document_end"]), 4),
            "persian_letter_ratio_of_letters": round(
                totals["persian_letters"] / max(1, totals["persian_letters"] + totals["latin_letters"]), 4),
            "latin_letter_ratio_of_letters": round(
                totals["latin_letters"] / max(1, totals["persian_letters"] + totals["latin_letters"]), 4),
            "persian_letter_ratio_of_characters": round(
                totals["persian_letters"] / max(1, totals["characters"]), 4),
            "zwnj_frequency": totals["zwnj"],
            "dataset_file_sizes_bytes": {k: v["bytes"] for k, v in files.items()},
        },
        "tokenizer": {
            "path": str(tok_path),
            "sha256": sha256_file(tok_path),
            "type": "byte-level BPE (unchanged V1 baseline)",
            "vocab_size": tokenizer.get_vocab_size(),
            "special_token": DOC_SEPARATOR,
            "special_token_id": eot_id,
            "special_token_single_token": eot_ids == [0],
            "method": "exact: Tokenizer.encode() over every final document, add_special_tokens=False",
            "document_ending_tokens_note": "Content token counts EXCLUDE document separators. One <|endoftext|> token per document is reported separately and in tokens_including_document_end. Separator lines are not tokenized as text.",
        },
        "splits": {
            split: {
                "documents": sum(v["documents"] for k, v in files.items() if v["split"] == split),
                "characters": sum(v["characters"] for k, v in files.items() if v["split"] == split),
                "tokens_content": sum(v["tokens_content_excluding_document_end"] for k, v in files.items() if v["split"] == split),
                "tokens_including_document_end": sum(v["tokens_including_document_end"] for k, v in files.items() if v["split"] == split),
            } for split in ("train", "validation", "test")
        },
        "source_distribution": manifest_stats["by_source"],
        "license_distribution": manifest_stats["by_license"],
        "domain_distribution": manifest_stats["by_domain"],
        "profile_distribution": manifest_stats["by_profile"],
        "source_index_statistics": index_stats,
        "quality_and_dedup": {
            "rejected_documents": quality.get("documents_rejected"),
            "rejection_reasons": quality.get("rejection_reasons", {}),
            "exact_duplicates_removed": quality.get("deduplication", {}).get("exact_duplicates_removed"),
            "near_duplicates_removed": quality.get("deduplication", {}).get("near_duplicates_removed"),
            "cleaning_statistics": quality.get("cleaning_statistics", {}),
            "manual_audit_file": quality.get("manual_audit", {}).get("path"),
        },
        "checksums": checksum_report,
        "processing_duration_seconds": build.get("processing_duration_seconds"),
        "statistics_scan_seconds": scan_seconds,
        "peak_rss_mb": benchmark.get("peak_rss_mb", peak_rss_mb()),
        "peak_rss_note": benchmark.get("peak_rss_note", "Best-effort; None when the OS cannot report it."),
        "split_report": build.get("split", {}),
        "balance_report": build.get("balance", {}),
        "estimated_values": {},
        "estimate_note": "No estimated number is mixed into this JSON. Full-corpus expectations are explicitly labelled estimates in reports/source_research.md and reports/model_data_balance.md.",
        "limitations": quality.get("limitations", []),
    }
    out = Path(args.out) if args.out else reports_dir / "dataset_statistics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    logger.info("statistics: %s docs / %s chars / %s content tokens -> %s",
                f"{totals['documents']:,}", f"{totals['characters']:,}",
                f"{totals['tokens_content_excluding_document_end']:,}", out)
    print(json.dumps({
        "scope": args.scope,
        "documents": totals["documents"],
        "unique_documents": unique_docs,
        "characters": totals["characters"],
        "words": totals["words"],
        "content_tokens": totals["tokens_content_excluding_document_end"],
        "tokens_including_document_end": totals["tokens_including_document_end"],
        "characters_per_token": result["totals"]["characters_per_token"],
        "bytes_per_token": result["totals"]["bytes_per_token"],
        "output": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
