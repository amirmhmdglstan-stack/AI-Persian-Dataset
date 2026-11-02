#!/usr/bin/env python3
"""Orchestrate Persian Dataset V2: normalize -> quality -> dedup -> balance -> split -> write.

Scalability properties:
* one document at a time; no full-corpus lists;
* normalized documents live in size-bounded JSONL shards on disk;
* exact/near dedup index and audit samples live in SQLite;
* source balancing and splitting use streaming SQL cursors;
* checkpoint.json allows resuming after interruption;
* final text is written to a separate V2 directory, never to V1 root files.

PROFILE A and PROFILE B are written to physically separate directories.
PROFILE C is never written as training text; only a quarantine manifest is
emitted.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DOC_SEPARATOR, assert_safe_output_dir, atomic_write_json, atomic_write_text,
    config_policy, config_source_metadata, configure_logging, json_dumps,
    load_config, now_utc, peak_rss_mb, read_jsonl_at, repo_root_from_here,
    require_free_space, safe_excerpt, sha256_file, stable_hash,
)
from deduplicate import SQLiteCorpusIndex  # noqa: E402
from filter_quality import audit_category, evaluate  # noqa: E402
from normalize_persian import detect_language, normalize  # noqa: E402
from split_dataset import assign_splits  # noqa: E402

DOC_JOIN = f"\n\n{DOC_SEPARATOR}\n\n"
RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
RE_LONG_NUMBER = re.compile(r"\b\d{8,}\b")


class ShardWriter:
    """Append normalized JSONL records to bounded-size shards; resumable."""

    def __init__(self, directory: Path, max_bytes: int, state: dict | None = None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.path: Path | None = None
        self.fh = None
        self.size = 0
        self.next_index = int((state or {}).get("shard_next_index", 0))
        resume_path = (state or {}).get("current_shard")
        if resume_path and Path(resume_path).exists():
            self.path = Path(resume_path)
            self.fh = open(self.path, "ab")
            self.size = self.path.stat().st_size
        else:
            self._rotate()

    def _rotate(self) -> None:
        if self.fh:
            self.fh.close()
        while True:
            candidate = self.directory / f"shard-{self.next_index:06d}.jsonl"
            self.next_index += 1
            if not candidate.exists() or candidate.stat().st_size == 0:
                self.path = candidate
                break
        self.fh = open(self.path, "ab")
        self.size = self.path.stat().st_size

    def write(self, record: dict) -> tuple[str, int, int]:
        data = json.dumps(record, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")).encode("utf-8") + b"\n"
        if self.size + len(data) > self.max_bytes and self.size > 0:
            self._rotate()
        assert self.fh is not None and self.path is not None
        offset = self.size
        self.fh.write(data)
        self.fh.flush()
        self.size += len(data)
        return str(self.path.resolve()), offset, len(data)

    def state(self) -> dict:
        return {"current_shard": str(self.path) if self.path else None,
                "shard_next_index": self.next_index, "shard_size": self.size}

    def close(self) -> None:
        if self.fh:
            self.fh.close()
            self.fh = None


def redact_excerpt(text: str, limit: int = 400) -> str:
    s = safe_excerpt(text or "", limit)
    s = RE_EMAIL.sub("[EMAIL_REDACTED]", s)
    s = RE_LONG_NUMBER.sub("[NUMBER_REDACTED]", s)
    return s


def write_checkpoint(path: Path, state: dict) -> None:
    atomic_write_json(path, state)


def load_catalog(path: Path) -> tuple[dict, dict]:
    meta = json.loads(path.read_text(encoding="utf-8"))
    return {s["id"]: s for s in meta["sources"]}, meta


def fallback_source(source_id: str, profile_hint: str, rec: dict) -> dict:
    """Allow small local integration fixtures that are not in the catalog."""
    profile = rec.get("profile") or profile_hint or "C"
    return {
        "id": source_id, "name": source_id, "profile": profile,
        "decision": "include" if profile in ("A", "B") else "quarantine",
        "domain": rec.get("domain", "encyclopedia"),
        "license_content": rec.get("license", "UNVERIFIED local fixture"),
        "copyright_holder": "local test fixture",
        "content_rights_verified": bool(rec.get("content_rights_verified", False)),
        "notes": "Local integration-test fixture; not part of the production catalog.",
    }


def iter_extracted_files(root: Path) -> list[Path]:
    files = sorted(root.glob("*/*.jsonl")) + sorted(root.glob("*.jsonl"))
    unique = []
    seen = set()
    for p in files:
        rp = str(p.resolve())
        if rp not in seen and not p.name.startswith("_"):
            seen.add(rp)
            unique.append(p)
    return unique


def apply_balance(index: SQLiteCorpusIndex, config: dict, logger,
                  scope: str = "full_run") -> dict:
    """Deterministic soft caps by source and domain; never duplicates documents.

    For ``scope="full_run"`` the configured source/domain caps are ALWAYS
    applied, including when the accepted corpus is smaller than the configured
    target.  The effective budget is the lesser of the configured target and the
    actual available characters; unfilled capacity is reported, not padded.
    ``scope="small_sample"`` is the only scope allowed to bypass caps for small
    synthetic integration tests.  Intended source percentages are unchanged.
    """
    if scope not in ("full_run", "small_sample"):
        raise ValueError(f"Unknown balance scope {scope!r}")
    conn = index.conn
    available = conn.execute(
        "SELECT COUNT(*) AS n, COALESCE(SUM(chars),0) AS c FROM docs WHERE status='kept'").fetchone()
    available_docs, available_chars = int(available["n"]), int(available["c"])
    target_budget = int(config.get("target_char_budget", 480_000_000))
    budget = min(target_budget, available_chars)
    caps = config.get("source_caps", {})
    domains = config.get("domain_targets", {})
    default_cap = float(config.get("default_source_cap", 0.05))
    default_domain = float(config.get("default_domain_target", 0.02))

    if scope == "small_sample":
        # Explicit small-test scope only: keep every quality-approved,
        # deduplicated document. Production full runs must not use this branch.
        conn.execute("UPDATE docs SET selected=1 WHERE status='kept'")
        conn.commit()
        sel = conn.execute("SELECT COUNT(*) AS n, COALESCE(SUM(chars),0) AS c "
                           "FROM docs WHERE status='kept' AND selected=1").fetchone()
        by_source = {}
        for row in conn.execute(
                "SELECT source_id, COUNT(*) AS n, SUM(chars) AS c FROM docs "
                "WHERE status='kept' AND selected=1 GROUP BY source_id"):
            sid = row["source_id"]
            by_source[sid] = {
                "available_documents": int(row["n"]), "available_characters": int(row["c"]),
                "selected_documents": int(row["n"]), "selected_characters": int(row["c"]),
                "cap_share": caps.get(sid, default_cap),
                "cap_characters": int(budget * float(caps.get(sid, default_cap))),
                "caps_applied": False,
            }
        selected_by_source = {r[0]: r[1] for r in conn.execute(
            "SELECT source_id, COUNT(*) FROM docs WHERE status='kept' AND selected=1 GROUP BY source_id")}
        selected_by_domain = {r[0]: r[1] for r in conn.execute(
            "SELECT domain, COUNT(*) FROM docs WHERE status='kept' AND selected=1 GROUP BY domain")}
        selected_docs, selected_chars = int(sel["n"]), int(sel["c"])
        logger.info("balance scope=small_sample: selected all %s docs / %s chars; caps bypassed for test scope only",
                    f"{selected_docs:,}", f"{selected_chars:,}")
        return {
            "method": "small_sample: all quality-approved deduplicated documents kept; caps NOT applied",
            "scope": scope,
            "balance_scope_behavior": "small_sample_caps_bypassed_for_synthetic_tests_only",
            "caps_applied": False,
            "source_percentages_changed": False,
            "target_char_budget": target_budget,
            "effective_budget_characters": budget,
            "available_documents": available_docs,
            "available_characters": available_chars,
            "selected_documents": selected_docs,
            "selected_characters": selected_chars,
            "unfilled_characters": max(0, budget - selected_chars),
            "skipped_by_cap": {},
            "source_cap_shares": dict(caps),
            "domain_target_shares": dict(domains),
            "selected_by_source": dict(selected_by_source),
            "selected_by_domain": dict(selected_by_domain),
            "by_source": by_source,
            "note": ("Small synthetic test scope. A full_run ALWAYS applies source/domain caps, "
                     "even below the configured target. No document is duplicated or padded."),
        }

    # full_run: caps are applied regardless of budget pressure.
    conn.execute("UPDATE docs SET selected=0 WHERE status='kept'")
    conn.commit()
    used_source: Counter = Counter()
    used_domain: Counter = Counter()
    selected_by_source: Counter = Counter()
    selected_by_domain: Counter = Counter()
    skipped: Counter = Counter()
    batch: list[tuple[str]] = []
    selected_docs = selected_chars = 0
    for row in conn.execute(
            "SELECT doc_id, source_id, domain, chars FROM docs WHERE status='kept' ORDER BY doc_id"):
        sid, dom, chars = row["source_id"], row["domain"] or "other", int(row["chars"])
        source_limit = budget * float(caps.get(sid, default_cap))
        domain_limit = budget * float(domains.get(dom, default_domain))
        if used_source[sid] + chars > source_limit:
            skipped["source_cap"] += 1
            continue
        if used_domain[dom] + chars > domain_limit:
            skipped["domain_cap"] += 1
            continue
        if selected_chars + chars > budget:
            skipped["total_budget"] += 1
            continue
        used_source[sid] += chars
        used_domain[dom] += chars
        selected_by_source[sid] += 1
        selected_by_domain[dom] += 1
        selected_docs += 1
        selected_chars += chars
        batch.append((row["doc_id"],))
        if len(batch) >= 1000:
            conn.executemany("UPDATE docs SET selected=1 WHERE doc_id=?", batch)
            batch = []
    if batch:
        conn.executemany("UPDATE docs SET selected=1 WHERE doc_id=?", batch)
    conn.commit()

    by_source = {}
    for row in conn.execute(
            "SELECT source_id, COUNT(*) AS n, SUM(chars) AS c FROM docs "
            "WHERE status='kept' GROUP BY source_id"):
        sid = row["source_id"]
        sel = conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(chars),0) AS c FROM docs "
            "WHERE status='kept' AND selected=1 AND source_id=?", (sid,)).fetchone()
        by_source[sid] = {
            "available_documents": int(row["n"]), "available_characters": int(row["c"]),
            "selected_documents": int(sel["n"]), "selected_characters": int(sel["c"]),
            "cap_share": caps.get(sid, default_cap),
            "cap_characters": int(budget * float(caps.get(sid, default_cap))),
            "caps_applied": True,
        }
    report = {
        "method": "streaming deterministic soft caps applied after global quality filtering and deduplication",
        "scope": scope,
        "balance_scope_behavior": "full_run_caps_always_applied_even_when_available_chars_below_target",
        "caps_applied": True,
        "source_percentages_changed": False,
        "target_char_budget": target_budget,
        "effective_budget_characters": budget,
        "available_documents": available_docs,
        "available_characters": available_chars,
        "selected_documents": selected_docs,
        "selected_characters": selected_chars,
        "unfilled_characters": max(0, budget - selected_chars),
        "skipped_by_cap": dict(skipped),
        "source_cap_shares": dict(caps),
        "domain_target_shares": dict(domains),
        "selected_by_source": dict(selected_by_source),
        "selected_by_domain": dict(selected_by_domain),
        "by_source": by_source,
        "note": ("Effective budget = min(configured target, actual available chars). Caps are "
                 "investigation settings, not mandatory quotas. No document was duplicated or "
                 "padded. Unfilled capacity is reported rather than filled with quarantined text; "
                 "the configured source percentages were not changed."),
    }
    logger.info("balance scope=full_run: selected %s docs / %s chars of %s available chars; caps applied, unfilled %s",
                f"{selected_docs:,}", f"{selected_chars:,}", f"{available_chars:,}",
                f"{report['unfilled_characters']:,}")
    return report

def document_text(index: SQLiteCorpusIndex, doc_id: str) -> str:
    row = index.conn.execute(
        "SELECT shard, offset, byte_len FROM docs WHERE doc_id=?", (doc_id,)).fetchone()
    if not row or not row["shard"] or not row["byte_len"]:
        return ""
    try:
        return read_jsonl_at(row["shard"], int(row["offset"]), int(row["byte_len"])).get("text", "")
    except (OSError, ValueError, json.JSONDecodeError):
        return ""


def write_audit_samples(index: SQLiteCorpusIndex, out_path: Path, config: dict) -> dict:
    """Write human-review samples from every source, with PII-redacted excerpts."""
    per_category = int(config.get("audit_samples_per_category", 5))
    limit = int(config.get("audit_excerpt_chars", 400))
    categories = ["good", "rejected", "borderline", "near_duplicate_example",
                  "formal_prose", "informal_or_dialogue"]
    sources = [r[0] for r in index.conn.execute(
        "SELECT DISTINCT source_id FROM audit WHERE source_id IS NOT NULL ORDER BY source_id")]
    lines = [
        "=" * 78,
        "PERSIAN DATASET V2 - MANUAL AUDIT SAMPLES",
        f"Generated: {now_utc()}",
        "Excerpts are truncated and PII-redacted. Automated quality scores do NOT",
        "establish semantic correctness, factual accuracy, or legal safety.",
        "Review good, rejected, borderline, near-duplicate, formal and informal",
        "documents from EVERY source before training.",
        "=" * 78, "",
    ]
    counts: Counter = Counter()
    for sid in sources:
        lines.append(f"SOURCE: {sid}")
        lines.append("-" * 78)
        for cat in categories:
            rows = index.conn.execute(
                "SELECT doc_id, category, reason, excerpt, paired_doc_id, meta FROM audit "
                "WHERE source_id=? AND category LIKE ? ORDER BY id LIMIT ?",
                (sid, f"%{cat}%", per_category)).fetchall()
            if not rows:
                continue
            lines.append(f"  CATEGORY: {cat} ({len(rows)} sample(s))")
            for r in rows:
                meta = {}
                try:
                    meta = json.loads(r["meta"] or "{}")
                except json.JSONDecodeError:
                    pass
                doc = index.conn.execute(
                    "SELECT split, status, quality FROM docs WHERE doc_id=?", (r["doc_id"],)).fetchone()
                excerpt = r["excerpt"] or document_text(index, r["doc_id"])
                excerpt = redact_excerpt(excerpt, limit)
                lines.append(f"    doc_id={r['doc_id']} reason={r['reason']} "
                             f"category={r['category']} split={doc['split'] if doc else 'n/a'} "
                             f"quality={doc['quality'] if doc else 'n/a'}")
                lines.append(f"    license={meta.get('license', 'unknown')} "
                             f"domain={meta.get('domain', 'unknown')}")
                lines.append(f"    text: {excerpt}")
                if r["paired_doc_id"]:
                    paired = redact_excerpt(document_text(index, r["paired_doc_id"]), limit)
                    lines.append(f"    paired_survivor={r['paired_doc_id']}")
                    lines.append(f"    survivor_text: {paired}")
                lines.append("")
                counts[cat] += 1
        lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out_path, "\n".join(lines))
    return {"samples_written": int(sum(counts.values())), "by_category": dict(counts),
            "sources": sources, "path": str(out_path)}


def write_final_outputs(index: SQLiteCorpusIndex, outdir: Path, profiles: list[str],
                        catalog_meta: dict, config: dict, logger) -> dict:
    """Write per-profile train/validation/test files and provenance manifests."""
    sep = DOC_JOIN
    result: dict[str, Any] = {"profiles": {}, "files": {}}
    for profile in profiles:
        pdir = outdir / f"profile_{profile}"
        pdir.mkdir(parents=True, exist_ok=True)
        handles = {}
        for split in ("train", "validation", "test"):
            tmp = pdir / f"{split}.txt.part"
            handles[split] = open(tmp, "w", encoding="utf-8", newline="\n")
        manifest_tmp = pdir / "documents.jsonl.part"
        manifest_f = open(manifest_tmp, "w", encoding="utf-8", newline="\n")
        counts: Counter = Counter()
        chars: Counter = Counter()
        try:
            rows = index.conn.execute(
                "SELECT doc_id, source_id, profile, license, domain, shard, offset, byte_len, "
                "chars, quality, cluster_id, split, meta FROM docs "
                "WHERE status='kept' AND selected=1 AND split IN ('train','validation','test') "
                "AND profile=? ORDER BY split, doc_id", (profile,))
            for row in rows:
                rec = read_jsonl_at(row["shard"], int(row["offset"]), int(row["byte_len"]))
                text = rec.get("text", "")
                if not text:
                    continue
                split = row["split"]
                handles[split].write(text)
                handles[split].write(sep)
                counts[split] += 1
                chars[split] += len(text)
                manifest = {
                    "doc_id": row["doc_id"], "source_id": row["source_id"],
                    "profile": row["profile"], "license": row["license"],
                    "domain": row["domain"], "split": split,
                    "characters": len(text), "quality": row["quality"],
                    "cluster_id": row["cluster_id"],
                    "meta": json.loads(row["meta"] or "{}"),
                }
                manifest_f.write(json.dumps(manifest, ensure_ascii=False, sort_keys=True,
                                            separators=(",", ":")) + "\n")
        finally:
            for f in handles.values():
                f.flush()
                os.fsync(f.fileno())
                f.close()
            manifest_f.flush()
            os.fsync(manifest_f.fileno())
            manifest_f.close()
        for split in ("train", "validation", "test"):
            tmp = pdir / f"{split}.txt.part"
            final = pdir / f"{split}.txt"
            os.replace(tmp, final)
            result["files"][f"profile_{profile}/{split}.txt"] = {
                "bytes": final.stat().st_size, "sha256": sha256_file(final),
                "documents": counts[split], "characters": chars[split],
                "document_separator": DOC_SEPARATOR,
            }
        os.replace(manifest_tmp, pdir / "documents.jsonl")
        result["files"][f"profile_{profile}/documents.jsonl"] = {
            "bytes": (pdir / "documents.jsonl").stat().st_size,
            "sha256": sha256_file(pdir / "documents.jsonl"),
        }
        result["profiles"][profile] = {
            "directory": str(pdir), "documents_by_split": dict(counts),
            "characters_by_split": dict(chars),
        }
        logger.info("profile %s: %s", profile, dict(counts))

    # Quarantine manifest only: NO text from PROFILE C is written.
    qdir = outdir / "profile_C_quarantine_manifest_only"
    qdir.mkdir(parents=True, exist_ok=True)
    quarantined = [{"id": s["id"], "name": s["name"], "reason": s["notes"],
                    "license_content": s["license_content"],
                    "privacy_risk": s["privacy_risk"]}
                   for s in catalog_meta["sources"] if s["profile"] == "C"]
    atomic_write_json(qdir / "quarantine_manifest.json", {
        "generated_at_utc": now_utc(),
        "policy": "Do not train on any PROFILE C source until rights, provenance and privacy issues are resolved.",
        "sources": quarantined,
        "text_files_written": 0,
    })
    result["quarantine_manifest"] = str(qdir / "quarantine_manifest.json")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the Persian Dataset V2 corpus")
    ap.add_argument("--config", default=None)
    ap.add_argument("--metadata", default=None)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--extracted", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--reports-dir", default=None,
                    help="where to write reports (default: <outdir>/reports)")
    ap.add_argument("--profiles", nargs="*", default=None)
    ap.add_argument("--only-sources", nargs="*", default=None)
    ap.add_argument("--max-docs", type=int, default=None,
                    help="process at most N documents total; small tests only")
    ap.add_argument("--resume", action="store_true", help="resume from checkpoint.json")
    ap.add_argument("--force", action="store_true", help="discard existing index/shards/checkpoint")
    ap.add_argument("--scope", default="full_run",
                    choices=["full_run", "small_sample"], help="label for honest benchmark reporting")
    ap.add_argument("--min-free-gb", type=float, default=None,
                    help="override configured disk-space safety threshold (small tests only)")
    args = ap.parse_args()

    t0 = time.time()
    cfg = load_config(args.config)
    meta_path = Path(args.metadata) if args.metadata else config_source_metadata(cfg)
    policy_path = config_policy(cfg)
    workdir = Path(args.workdir) if args.workdir else Path(cfg["workdir"])
    extracted_root = Path(args.extracted) if args.extracted else workdir / "extracted"
    outdir = Path(args.outdir) if args.outdir else Path(cfg["outdir"])
    reports_dir = Path(args.reports_dir) if args.reports_dir else outdir / "reports"
    logger = configure_logging(workdir, "dataset_v2.build")
    min_free = args.min_free_gb if args.min_free_gb is not None else float(cfg.get("min_free_gb", 20))
    require_free_space(workdir, min_free, logger)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    profiles = args.profiles or cfg.get("default_profiles", ["A"])
    for p in profiles:
        if p not in policy["profiles"]:
            raise ValueError(f"Unknown profile {p!r}; allowed: {list(policy['profiles'])}")
    if "C" in profiles:
        raise ValueError("PROFILE C is quarantine-only; build_corpus.py refuses to train on it.")
    assert_safe_output_dir(outdir, repo_root_from_here(),
                           policy["global_rules"]["v1_root_filenames"])
    catalog, catalog_meta = load_catalog(meta_path)
    reports_dir.mkdir(parents=True, exist_ok=True)

    index_path = workdir / "corpus_index.sqlite3"
    normalized_dir = workdir / "normalized"
    checkpoint_path = workdir / "checkpoint.json"
    if args.force:
        for suffix in ("", "-wal", "-shm"):
            p = Path(str(index_path) + suffix)
            if p.exists():
                p.unlink()
        if normalized_dir.exists():
            shutil.rmtree(normalized_dir)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        checkpoint = {"processed_sources": [], "last_source": None, "last_line": 0,
                      "processed_docs": 0, "stats": {}}
    elif args.resume and checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    elif index_path.exists() and not args.resume:
        raise RuntimeError(
            f"Corpus index already exists at {index_path}. Use --resume to continue "
            "or --force to rebuild from scratch. Refusing to mix runs silently.")
    else:
        checkpoint = {"processed_sources": [], "last_source": None, "last_line": 0,
                      "processed_docs": 0, "stats": {}}

    index = SQLiteCorpusIndex(index_path, cfg, reset=args.force)
    writer = ShardWriter(normalized_dir, int(cfg.get("shard_bytes", 67108864)),
                         checkpoint.get("shard_state"))

    stats = checkpoint.get("stats") or {
        "documents_seen": 0, "documents_accepted": 0, "documents_rejected": 0,
        "exact_duplicates": 0, "near_duplicates": 0, "normalized_characters": 0,
        "persian_letters": 0, "latin_letters": 0, "latin_words": 0, "persian_words": 0,
        "persian_digits": 0, "latin_digits": 0, "zwnj": 0,
        "quality_score_sum": 0.0, "cleaning_removed_chars": 0,
        "by_source": {}, "rejection_reasons": {}, "language_reasons": {},
    }
    # defaultdict(Counter) needs explicit conversion of loaded dict values.
    by_source = defaultdict(Counter, {k: Counter(v) for k, v in stats["by_source"].items()})
    rejection_reasons = Counter(stats.get("rejection_reasons", {}))
    language_reasons = Counter(stats.get("language_reasons", {}))
    processed_sources = list(dict.fromkeys(checkpoint.get("processed_sources", [])))
    last_source = checkpoint.get("last_source")
    last_line = int(checkpoint.get("last_line", 0))
    processed_docs = int(checkpoint.get("processed_docs", 0))

    files = iter_extracted_files(extracted_root)
    if not files:
        raise FileNotFoundError(
            f"No extracted JSONL files found under {extracted_root}. "
            "Run extract_sources.py first, or pass --extracted for a small test.")
    logger.info("found %d extracted source files", len(files))
    seed = cfg.get("seed", "persian-dataset-v2")
    checkpoint_every = int(cfg.get("checkpoint_every_docs", 5000))
    log_every = int(cfg.get("log_every_docs", 10000))

    for path in files:
        source_id = path.stem
        parent = path.parent.name if path.parent.name in ("A", "B", "C") else ""
        if args.only_sources and source_id not in args.only_sources:
            continue
        if source_id in processed_sources:
            logger.info("[skip] %s already processed (checkpoint)", source_id)
            continue
        # Peek at the first record for fallback metadata when the source is a local fixture.
        with open(path, "r", encoding="utf-8") as f:
            first_line = f.readline()
        if not first_line.strip():
            continue
        first = json.loads(first_line)
        src = catalog.get(source_id) or fallback_source(source_id, parent, first)
        if src["profile"] not in profiles:
            continue
        if src["decision"] not in policy["profiles"][src["profile"]]["allowed_decisions"]:
            logger.warning("[skip] %s: decision %s not allowed for profile %s",
                           source_id, src["decision"], src["profile"])
            continue
        if src["profile"] == "A" and policy["profiles"]["A"]["require_content_rights_verified"] \
                and not src.get("content_rights_verified"):
            logger.warning("[skip] %s: PROFILE A requires verified content rights", source_id)
            continue
        skip_lines = last_line if last_source == source_id else 0
        logger.info("[process] %s (profile=%s, skip_lines=%d)", source_id, src["profile"], skip_lines)
        line_no = 0
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if line_no <= skip_lines:
                    continue
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid extracted JSONL at {path}:{line_no}: {e}") from e
                processed_docs += 1
                stats["documents_seen"] += 1
                by_source[source_id]["seen"] += 1
                raw_text = rec.get("text", "")
                doc_id = stable_hash(source_id + "\x1f" + rec.get("doc_id", str(line_no)),
                                     seed + ":docid")
                if index.has_doc(doc_id):
                    by_source[source_id]["already_indexed"] += 1
                    continue
                norm = normalize(raw_text, cfg)
                lang = detect_language(norm.text, norm.stats, cfg,
                                       domain=src.get("domain", "default"))
                decision = evaluate(norm.text, norm.stats, lang, src, cfg)
                meta = dict(rec.get("meta") or {})
                meta.update({
                    "source_id": source_id, "source_name": src.get("name", source_id),
                    "profile": src["profile"], "license": src["license_content"],
                    "domain": src.get("domain", "encyclopedia"),
                    "attribution": src.get("copyright_holder", ""),
                    "content_rights_verified": bool(src.get("content_rights_verified")),
                    "quality_score": decision.score,
                    "quality_features": decision.features,
                    "language_score": lang.score,
                    "language_reason": lang.reason,
                })
                excerpt = redact_excerpt(raw_text or norm.text,
                                         int(cfg.get("audit_excerpt_chars", 400)))
                if not decision.passed:
                    reason = ";".join(decision.reasons)
                    index.add_rejected(
                        doc_id=doc_id, source_id=source_id, profile=src["profile"],
                        license_=src["license_content"], domain=src.get("domain", "default"),
                        reason=reason, quality=decision.score, excerpt=excerpt,
                        meta=meta, categories=audit_category(decision, src, "rejected"))
                    stats["documents_rejected"] += 1
                    by_source[source_id]["rejected"] += 1
                    for r in decision.reasons:
                        rejection_reasons[r.split(":")[0]] += 1
                    if not lang.is_persian:
                        language_reasons[lang.reason] += 1
                else:
                    h = index.exact_hash(norm.text)
                    exact_of = index.exact_duplicate_of(h)
                    if exact_of:
                        index.add_duplicate(
                            doc_id=doc_id, source_id=source_id, profile=src["profile"],
                            license_=src["license_content"], domain=src.get("domain", "default"),
                            status="exact_duplicate", cluster_id=exact_of,
                            quality=decision.score, excerpt=excerpt,
                            paired_doc_id=exact_of, similarity=1.0, meta=meta)
                        stats["exact_duplicates"] += 1
                        by_source[source_id]["exact_duplicates"] += 1
                    else:
                        sig, n_shingles = index.make_signature(norm.text, seed)
                        near_of, sim, _ = index.near_duplicate_of(sig, n_shingles)
                        if near_of:
                            index.add_duplicate(
                                doc_id=doc_id, source_id=source_id, profile=src["profile"],
                                license_=src["license_content"], domain=src.get("domain", "default"),
                                status="near_duplicate", cluster_id=near_of,
                                quality=decision.score, excerpt=excerpt,
                                paired_doc_id=near_of, similarity=sim, meta=meta)
                            stats["near_duplicates"] += 1
                            by_source[source_id]["near_duplicates"] += 1
                        else:
                            shard, offset, byte_len = writer.write({"doc_id": doc_id, "text": norm.text})
                            index.add_kept(
                                doc_id=doc_id, source_id=source_id, profile=src["profile"],
                                license_=src["license_content"], domain=src.get("domain", "default"),
                                shard=shard, offset=offset, byte_len=byte_len,
                                chars=len(norm.text), quality=decision.score,
                                exact_hash=h, signature=sig, meta=meta,
                                categories=audit_category(decision, src, "kept"))
                            stats["documents_accepted"] += 1
                            stats["normalized_characters"] += len(norm.text)
                            stats["persian_letters"] += norm.stats.get("persian_letters", 0)
                            stats["latin_letters"] += norm.stats.get("latin_letters", 0)
                            stats["latin_words"] += norm.stats.get("latin_words", 0)
                            stats["persian_words"] += norm.stats.get("persian_words", 0)
                            stats["persian_digits"] += norm.stats.get("persian_digits", 0)
                            stats["latin_digits"] += norm.stats.get("latin_digits", 0)
                            stats["zwnj"] += norm.stats.get("zwnj_after", 0)
                            stats["quality_score_sum"] += decision.score
                            stats["cleaning_removed_chars"] += norm.stats.get("removed_chars", 0)
                            by_source[source_id]["accepted"] += 1
                            by_source[source_id]["characters"] += len(norm.text)
                if processed_docs % log_every == 0:
                    logger.info("progress: %s docs seen, %s accepted", f"{processed_docs:,}",
                                f"{stats['documents_accepted']:,}")
                if processed_docs % checkpoint_every == 0:
                    index.commit()
                    checkpoint.update({
                        "processed_sources": processed_sources,
                        "last_source": source_id, "last_line": line_no,
                        "processed_docs": processed_docs, "shard_state": writer.state(),
                        "stats": {**stats, "by_source": {k: dict(v) for k, v in by_source.items()},
                                  "rejection_reasons": dict(rejection_reasons),
                                  "language_reasons": dict(language_reasons)},
                        "updated_at_utc": now_utc(),
                    })
                    write_checkpoint(checkpoint_path, checkpoint)
                if args.max_docs and processed_docs >= args.max_docs:
                    logger.warning("stopping at --max-docs %d", args.max_docs)
                    break
        processed_sources.append(source_id)
        last_source, last_line = source_id, 0
        index.commit()
        checkpoint.update({
            "processed_sources": processed_sources, "last_source": None, "last_line": 0,
            "processed_docs": processed_docs, "shard_state": writer.state(),
            "stats": {**stats, "by_source": {k: dict(v) for k, v in by_source.items()},
                      "rejection_reasons": dict(rejection_reasons),
                      "language_reasons": dict(language_reasons)},
            "updated_at_utc": now_utc(),
        })
        write_checkpoint(checkpoint_path, checkpoint)
        if args.max_docs and processed_docs >= args.max_docs:
            break
    writer.close()
    index.commit()

    balance_report = apply_balance(index, cfg, logger, scope=args.scope)
    split_report = assign_splits(index, cfg, logger=logger)
    outputs = write_final_outputs(index, outdir, profiles, catalog_meta, cfg, logger)
    audit_report = write_audit_samples(index, reports_dir / "sample_audit.txt", cfg)
    duration = round(time.time() - t0, 3)
    sqlite_bytes = index_path.stat().st_size if index_path.exists() else 0
    shard_files = sorted(normalized_dir.glob("shard-*.jsonl")) if normalized_dir.exists() else []
    input_bytes = sum(p.stat().st_size for p in files if p.exists())
    output_bytes = sum(v.get("bytes", 0) for v in outputs["files"].values())
    index_stats = index.all_stats()
    total_chars = max(1, stats["normalized_characters"])
    quality_report = {
        "measurement_scope": args.scope,
        "generated_at_utc": now_utc(),
        "seed": seed,
        "config_version": cfg.get("config_version"),
        "documents_seen": stats["documents_seen"],
        "documents_accepted_after_quality": stats["documents_accepted"],
        "documents_rejected": stats["documents_rejected"],
        "rejection_reasons": dict(rejection_reasons),
        "language_rejection_reasons": dict(language_reasons),
        "deduplication": {
            "exact_duplicates_removed": stats["exact_duplicates"],
            "near_duplicates_removed": stats["near_duplicates"],
            "sqlite_measured": {k: v for k, v in index_stats.items()
                                if k.startswith(("exact_", "near_", "lsh_", "documents_with_"))},
        },
        "cleaning_statistics": {
            "normalized_characters": stats["normalized_characters"],
            "cleaning_removed_chars": stats["cleaning_removed_chars"],
            "persian_letters": stats["persian_letters"],
            "latin_letters": stats["latin_letters"],
            "latin_letter_ratio_of_letters": round(
                stats["latin_letters"] / max(1, stats["persian_letters"] + stats["latin_letters"]), 4),
            "latin_words": stats["latin_words"],
            "persian_words": stats["persian_words"],
            "persian_digits": stats["persian_digits"],
            "latin_digits": stats["latin_digits"],
            "zwnj": stats["zwnj"],
            "mean_quality_score": round(stats["quality_score_sum"] / max(1, stats["documents_accepted"]), 4),
        },
        "by_source": {k: dict(v) for k, v in by_source.items()},
        "manual_audit": audit_report,
        "limitations": [
            "Automated quality scores are surface-form heuristics; they do not establish semantic correctness.",
            "MinHash/LSH near deduplication is approximate; undetected duplicates may remain.",
            "Language detection uses script ratios and function words, not a trained language classifier.",
            "All statistics labelled small_sample are NOT full-corpus results.",
        ],
    }
    benchmark = {
        "measurement_scope": args.scope,
        "generated_at_utc": now_utc(),
        "duration_seconds": duration,
        "documents_seen": stats["documents_seen"],
        "documents_per_second": round(stats["documents_seen"] / max(duration, 0.001), 2),
        "normalized_characters_per_second": round(stats["normalized_characters"] / max(duration, 0.001), 2),
        "peak_rss_mb": peak_rss_mb(),
        "peak_rss_note": "Measured with resource.getrusage on Linux / GetProcessMemoryInfo on Windows; None if unavailable.",
        "sqlite_index_bytes": sqlite_bytes,
        "normalized_shards": len(shard_files),
        "normalized_shard_bytes": sum(p.stat().st_size for p in shard_files),
        "input_extracted_bytes": input_bytes,
        "final_output_bytes": output_bytes,
        "lsh_candidates_verified": index_stats.get("near_dup_candidates_verified", 0),
        "lsh_candidates_rejected": index_stats.get("near_dup_candidates_rejected", 0),
        "documents_with_overloaded_lsh_bands": index_stats.get("documents_with_overloaded_lsh_bands", 0),
        "untested_at_scale": [
            "Full 1.456 GB MediaWiki XML bz2 extraction",
            "Tens of millions of near-dup LSH lookups",
            "Full-corpus tokenization on Windows",
            "Peak RAM on the user's Windows 11 / Python 3.12 machine",
        ],
    }
    build_report = {
        "measurement_scope": args.scope,
        "generated_at_utc": now_utc(),
        "config_version": cfg.get("config_version"),
        "seed": seed,
        "profiles": profiles,
        "python_version": sys.version,
        "platform": sys.platform,
        "processing_duration_seconds": duration,
        "statistics": stats,
        "by_source": {k: dict(v) for k, v in by_source.items()},
        "rejection_reasons": dict(rejection_reasons),
        "balance": balance_report,
        "balance_scope_policy": {
            "scope": args.scope,
            "full_run_caps_always_applied": True,
            "small_sample_caps_bypassed_for_synthetic_tests": True,
            "effective_budget": "min(target_char_budget, actual_available_characters)",
            "source_percentages_changed": False,
            "no_duplication_or_padding": True,
        },
        "split": split_report,
        "outputs": outputs,
        "quality_report": str(reports_dir / "quality_report.json"),
        "sample_audit": audit_report["path"],
        "tokenizer_baseline": cfg.get("tokenizer_baseline"),
        "v1_immutable": True,
    }
    atomic_write_json(reports_dir / "quality_report.json", quality_report)
    atomic_write_json(reports_dir / "pipeline_benchmark.json", benchmark)
    atomic_write_json(reports_dir / "build_report.json", build_report)
    # Close/checkpoint SQLite before hashing the main database file.
    index.close()
    atomic_write_json(outdir / "checksums.json", {
        "generated_at_utc": now_utc(),
        "config_version": cfg.get("config_version"),
        "seed": seed,
        "algorithm": "sha256",
        "files": outputs["files"],
        "index_sqlite3": {"bytes": sqlite_bytes, "sha256": sha256_file(index_path) if index_path.exists() else None},
        "note": "Full-run checksums are generated on the user's PC; do not commit large files to Git.",
    })
    logger.info("build complete in %.1fs; reports in %s", duration, reports_dir)
    print(json.dumps({
        "scope": args.scope,
        "documents_seen": stats["documents_seen"],
        "accepted": stats["documents_accepted"],
        "rejected": stats["documents_rejected"],
        "exact_duplicates": stats["exact_duplicates"],
        "near_duplicates": stats["near_duplicates"],
        "selected_documents": balance_report["selected_documents"],
        "selected_characters": balance_report["selected_characters"],
        "duration_seconds": duration,
        "peak_rss_mb": benchmark["peak_rss_mb"],
        "reports_dir": str(reports_dir),
        "outdir": str(outdir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
