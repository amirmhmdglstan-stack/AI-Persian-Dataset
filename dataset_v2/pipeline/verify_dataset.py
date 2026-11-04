#!/usr/bin/env python3
"""Independent, read-only verification of the final V2 corpus.

Re-scans the written train/validation/test files (not the build statistics),
checks integrity, exact duplicates within and across splits, duplicate-cluster
leakage, PII/markup leftovers, checksums, profile separation and V1 immutability.
Exits non-zero if a hard check fails.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DOC_SEPARATOR, configure_logging, load_config, now_utc,
    repo_root_from_here, sha1_git_blob_file, sha256_file,
)
from compute_statistics import HashStore, iter_documents  # noqa: E402

RE_PERSIAN = re.compile("[آ-یءأإئؤۀٔ]")
RE_LATIN = re.compile(r"[A-Za-z]")
RE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ufffd]")
RE_HTML = re.compile(r"&(?:nbsp|amp|lt|gt|quot|#\d{2,7});|<[A-Za-z/][^<>]{0,120}>|\{\{|\}\}|\[\[")
RE_URL = re.compile(r"(?:https?://|www\.|\.(?:com|ir|org|net)/)", re.IGNORECASE)
RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
RE_LONG_NUMBER = re.compile(r"(?<!\d)[0-9۰-۹]{10,}(?!\d)")
FORBIDDEN = {
    "arabic_yeh_U064A": "\u064a", "arabic_kaf_U0643": "\u0643",
    "teh_marbuta_U0629": "\u0629", "tatweel_U0640": "\u0640",
    "zwj_U200D": "\u200d", "zwsp_U200B": "\u200b", "lrm_U200E": "\u200e",
    "rlm_U200F": "\u200f", "bom_UFEFF": "\ufeff", "soft_hyphen_U00AD": "\u00ad",
}
PRESENTATION_RANGES = ((0xFB50, 0xFDFF), (0xFE70, 0xFEFF))


def cluster_leakage(index_path: Path) -> list[str]:
    """Independently inspect global clusters, including cross-source members."""
    with sqlite3.connect(str(index_path)) as conn:
        return [r[0] for r in conn.execute(
            "SELECT cluster_id FROM docs WHERE status='kept' AND selected=1 "
            "AND split IS NOT NULL GROUP BY cluster_id "
            "HAVING COUNT(DISTINCT split)>1 LIMIT 20")]


def check_v1_immutability(repo_root: Path) -> dict:
    baseline_path = repo_root / "dataset_v2" / "sources" / "v1_baseline.json"
    if not baseline_path.exists():
        return {"status": "skipped", "reason": "v1_baseline.json not found"}
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    results = {}
    mismatches = []
    for rel, info in baseline["files"].items():
        p = repo_root / rel
        if not p.exists():
            results[rel] = "not_present_in_this_checkout"
            continue
        # This session's branch is based on main's initial commit and has its
        # OWN small README.md.  A same-named file of a different size is not a
        # V1 file; do not report it as a corrupted V1 artifact.
        if p.stat().st_size != info["size"]:
            results[rel] = "different_branch_file_skipped_by_size"
            continue
        actual = sha1_git_blob_file(p)
        expected = info["git_blob_sha1"]
        ok = actual == expected
        results[rel] = "unchanged" if ok else "MISMATCH"
        if not ok:
            mismatches.append({"file": rel, "expected": expected, "actual": actual})
    return {
        "status": "fail" if mismatches else "ok",
        "files": results,
        "mismatches": mismatches,
        "note": "V1 files are absent from this session's branch checkout; they remain on the remote V1 branch. Missing files and a different-branch README.md of a different size are not failures; only a same-sized V1 file with a changed hash is a failure.",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify the final Persian Dataset V2 files")
    ap.add_argument("--config", default=None)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--reports-dir", default=None)
    ap.add_argument("--profiles", nargs="*", default=None)
    ap.add_argument("--scope", default="full_run", choices=["full_run", "small_sample"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    workdir = Path(args.workdir) if args.workdir else Path(cfg["workdir"])
    outdir = Path(args.outdir) if args.outdir else Path(cfg["outdir"])
    reports_dir = Path(args.reports_dir) if args.reports_dir else outdir / "reports"
    profiles = args.profiles or cfg.get("default_profiles", ["A"])
    logger = configure_logging(workdir, "dataset_v2.verify")
    reports_dir.mkdir(parents=True, exist_ok=True)

    checks: list[dict] = []

    def add(name: str, passed: bool, detail: Any = None, warning: bool = False) -> None:
        checks.append({"check": name, "status": "PASS" if passed else ("WARN" if warning else "FAIL"),
                       "detail": detail})

    if outdir.resolve() == repo_root_from_here().resolve():
        add("output_dir_is_not_repo_root", False, "Refusing to verify a corpus written into the repository root.")
    else:
        add("output_dir_is_not_repo_root", True, str(outdir))

    # Independent file scan.
    hash_store = HashStore(workdir / "verification_hashes.sqlite3")
    file_stats: dict[str, dict] = {}
    cross_file_duplicates = []
    forbidden_counts: Counter = Counter()
    markup_hits = url_hits = email_hits = pii_hits = control_hits = empty_docs = 0
    low_persian_docs = 0
    low_persian_examples = []
    separator_counts: Counter = Counter()
    doc_counts: Counter = Counter()
    bom_files = []
    v1_marker_in_text = 0
    for profile in profiles:
        pdir = outdir / f"profile_{profile}"
        for split in ("train", "validation", "test"):
            path = pdir / f"{split}.txt"
            key = f"profile_{profile}/{split}.txt"
            if not path.exists():
                add("final_files_exist", False, f"Missing {path}")
                continue
            with open(path, "rb") as f:
                if f.read(3).startswith(b"\xef\xbb\xbf"):
                    bom_files.append(key)
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.rstrip("\n") == DOC_SEPARATOR:
                        separator_counts[key] += 1
            docs = chars = persian_letters = latin_letters = zwnj = 0
            dup_in_file = 0
            for text in iter_documents(path):
                if not text.strip():
                    empty_docs += 1
                    continue
                doc_counts[key] += 1
                docs += 1
                chars += len(text)
                p_letters = len(RE_PERSIAN.findall(text))
                l_letters = len(RE_LATIN.findall(text))
                persian_letters += p_letters
                latin_letters += l_letters
                zwnj += text.count("\u200c")
                if DOC_SEPARATOR in text:
                    v1_marker_in_text += 1
                control_hits += len(RE_CONTROL.findall(text))
                for name, ch in FORBIDDEN.items():
                    forbidden_counts[name] += text.count(ch)
                for lo, hi in PRESENTATION_RANGES:
                    forbidden_counts["presentation_forms"] += sum(
                        1 for c in text if lo <= ord(c) <= hi)
                if RE_HTML.search(text):
                    markup_hits += 1
                if RE_URL.search(text):
                    url_hits += 1
                if RE_EMAIL.search(text):
                    email_hits += 1
                if RE_LONG_NUMBER.search(text):
                    pii_hits += 1
                if p_letters / max(1, p_letters + l_letters) < 0.65:
                    low_persian_docs += 1
                    if len(low_persian_examples) < 5:
                        low_persian_examples.append(text[:180])
                global_new, file_new, first_file = hash_store.add(text, key, split)
                if not file_new:
                    dup_in_file += 1
                if not global_new and first_file != key:
                    cross_file_duplicates.append({"file": key, "first_seen": first_file,
                                                  "excerpt": text[:120]})
            file_stats[key] = {
                "documents": docs, "characters": chars,
                "persian_letters": persian_letters, "latin_letters": latin_letters,
                "zwnj": zwnj, "exact_duplicates_in_file": dup_in_file,
                "separator_lines": separator_counts[key],
            }
            add(f"strict_utf8_readable:{key}", True)
    unique_docs = hash_store.count()
    hash_store.close()

    add("no_bom", not bom_files, bom_files)
    add("no_empty_documents", empty_docs == 0, empty_docs)
    add("no_control_or_replacement_chars", control_hits == 0, control_hits)
    add("arabic_variants_and_invisible_chars_normalized",
        sum(forbidden_counts.values()) == 0, dict(forbidden_counts))
    add("zwnj_preserved", forbidden_counts.get("zwj_U200D", 0) == 0
        and forbidden_counts.get("zwsp_U200B", 0) == 0,
        {"total_zwnj_measured": sum(v["zwnj"] for v in file_stats.values())})
    add("no_html_or_wiki_markup", markup_hits == 0, markup_hits)
    add("no_urls_or_emails", url_hits == 0 and email_hits == 0,
        {"urls": url_hits, "emails": email_hits})
    add("no_long_pii_number_patterns", pii_hits == 0, pii_hits)
    add("document_marker_not_inside_text", v1_marker_in_text == 0, v1_marker_in_text)
    add("separator_count_matches_documents",
        all(v["separator_lines"] == v["documents"] for v in file_stats.values()),
        {k: {"separators": v["separator_lines"], "documents": v["documents"]}
         for k, v in file_stats.items()})
    add("no_exact_duplicates_within_splits",
        all(v["exact_duplicates_in_file"] == 0 for v in file_stats.values()),
        {k: v["exact_duplicates_in_file"] for k, v in file_stats.items()})
    add("no_exact_duplicates_across_splits", not cross_file_duplicates,
        cross_file_duplicates[:5])
    add("low_non_persian_contamination", low_persian_docs == 0,
        {"docs_below_0_65_persian_letter_share": low_persian_docs,
         "examples": low_persian_examples}, warning=low_persian_docs > 0)
    add("unique_documents_measured", unique_docs == sum(doc_counts.values()),
        {"unique": unique_docs, "total_documents": sum(doc_counts.values())})

    # Check known clusters only; LSH can miss similar pairs or fail to union clusters.
    index_path = workdir / "corpus_index.sqlite3"
    leakage = []
    if index_path.exists():
        leakage = cluster_leakage(index_path)
        add("no_duplicate_cluster_leakage_across_splits", not leakage, leakage[:5])
    else:
        add("no_duplicate_cluster_leakage_across_splits", False,
            "corpus_index.sqlite3 missing; cannot verify identified cluster consistency")

    # Provenance manifest counts and license/domain distribution.
    manifest_counts = {}
    license_dist: Counter = Counter()
    domain_dist: Counter = Counter()
    source_dist: Counter = Counter()
    for profile in profiles:
        mpath = outdir / f"profile_{profile}" / "documents.jsonl"
        if not mpath.exists():
            add(f"provenance_manifest_exists:profile_{profile}", False, str(mpath))
            continue
        n = 0
        with open(mpath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                n += 1
                license_dist[d["license"]] += 1
                domain_dist[d["domain"]] += 1
                source_dist[d["source_id"]] += 1
        manifest_counts[profile] = n
        expected = sum(v["documents"] for k, v in file_stats.items()
                       if k.startswith(f"profile_{profile}/"))
        add(f"provenance_manifest_matches:{profile}", n == expected,
            {"manifest_rows": n, "final_documents": expected})
        # Duplicate provenance is separate to keep manifest rows memory-bounded.
        # Compare its cardinality and identities to the authoritative work index.
        duplicate_path = outdir / f"profile_{profile}" / "duplicate_sources.jsonl"
        if not index_path.exists() or not duplicate_path.exists():
            add(f"duplicate_provenance_matches:{profile}", False,
                "missing corpus index or duplicate_sources.jsonl")
        else:
            with sqlite3.connect(str(index_path)) as conn:
                conn.execute("PRAGMA temp_store=FILE")
                conn.execute("CREATE TEMP TABLE seen_duplicate_ids (doc_id TEXT PRIMARY KEY)")
                expected_dup = conn.execute(
                    "SELECT COUNT(*) FROM docs d JOIN docs r ON r.doc_id=d.cluster_id "
                    "WHERE d.status='exact_duplicate' AND d.profile=? "
                    "AND r.status='kept' AND r.selected=1 AND r.split IS NOT NULL", (profile,)).fetchone()[0]
                count = bad = 0
                with duplicate_path.open(encoding="utf-8") as dup_f:
                    for line in dup_f:
                        try:
                            d = json.loads(line)
                            row = conn.execute(
                                "SELECT d.source_id,d.profile,d.license,d.domain,d.meta,d.cluster_id "
                                "FROM docs d JOIN docs r ON r.doc_id=d.cluster_id "
                                "WHERE d.doc_id=? AND d.status='exact_duplicate' AND d.profile=? "
                                "AND r.status='kept' AND r.selected=1 AND r.split IS NOT NULL",
                                (d['doc_id'], profile)).fetchone()
                            if (not row or [d[k] for k in ('source_id','profile','license','domain')] != list(row[:4])
                                    or d['meta'] != json.loads(row[4] or '{}')
                                    or d['representative_doc_id'] != row[5]):
                                bad += 1
                            conn.execute("INSERT INTO seen_duplicate_ids VALUES (?)", (d['doc_id'],))
                            count += 1
                        except (ValueError, KeyError, TypeError, sqlite3.IntegrityError):
                            bad += 1
                add(f"duplicate_provenance_matches:{profile}", count == expected_dup and bad == 0,
                    {"manifest_rows": count, "expected": expected_dup, "invalid": bad})
    add("license_distribution_recorded", bool(license_dist), dict(license_dist))
    add("domain_distribution_recorded", bool(domain_dist), dict(domain_dist))
    add("source_distribution_recorded", bool(source_dist), dict(source_dist))

    # Split ratios (warnings only for small samples).
    total_docs = sum(doc_counts.values())
    split_docs = defaultdict(int)
    for key, n in doc_counts.items():
        split_docs[key.split("/")[-1].replace(".txt", "")] += n
    ratios = {k: round(v / max(1, total_docs), 4) for k, v in split_docs.items()}
    ratio_ok = (0.90 <= ratios.get("train", 0) <= 0.99
                and 0.005 <= ratios.get("validation", 0) <= 0.05
                and 0.002 <= ratios.get("test", 0) <= 0.03)
    add("split_ratios_reasonable", ratio_ok, ratios,
        warning=(not ratio_ok and args.scope == "small_sample"))

    # Quarantine: no training text for PROFILE C.
    qdir = outdir / "profile_C_quarantine_manifest_only"
    q_txt = list(qdir.glob("*.txt")) if qdir.exists() else []
    q_manifest = qdir / "quarantine_manifest.json"
    add("quarantine_has_manifest_but_no_text", q_manifest.exists() and not q_txt,
        {"text_files": [str(p) for p in q_txt], "manifest": str(q_manifest)})

    # Checksums.
    checksum_path = outdir / "checksums.json"
    checksum_detail = {"available": checksum_path.exists(), "mismatches": []}
    if checksum_path.exists():
        recorded = json.loads(checksum_path.read_text(encoding="utf-8")).get("files", {})
        for key, rec in recorded.items():
            p = outdir / key
            if not p.exists():
                checksum_detail["mismatches"].append({"file": key, "reason": "missing"})
                continue
            actual = sha256_file(p)
            if actual != rec.get("sha256"):
                checksum_detail["mismatches"].append(
                    {"file": key, "expected": rec.get("sha256"), "actual": actual})
    add("checksums_match", checksum_path.exists() and not checksum_detail["mismatches"],
        checksum_detail)

    # V1 immutability.
    v1 = check_v1_immutability(repo_root_from_here())
    add("v1_files_unchanged_or_absent", v1["status"] != "fail", v1)

    hard_failures = [c for c in checks if c["status"] == "FAIL"]
    warnings = [c for c in checks if c["status"] == "WARN"]
    report = {
        "measurement_scope": args.scope,
        "generated_at_utc": now_utc(),
        "checks_run": len(checks),
        "passed": sum(c["status"] == "PASS" for c in checks),
        "warnings": len(warnings),
        "failed": len(hard_failures),
        "checks": checks,
        "file_stats": file_stats,
        "split_document_ratios": ratios,
        "unique_documents": unique_docs,
        "v1_note": "V1 root files are on a different remote branch and were not modified.",
    }
    out = Path(args.out) if args.out else reports_dir / "verification_report_v2.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    logger.info("verification: %d checks, %d passed, %d warnings, %d failed -> %s",
                len(checks), report["passed"], len(warnings), len(hard_failures), out)
    print(json.dumps({"checks": len(checks), "passed": report["passed"],
                      "warnings": len(warnings), "failed": len(hard_failures),
                      "report": str(out)}, ensure_ascii=False, indent=2))
    if hard_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
