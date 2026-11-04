#!/usr/bin/env python3
"""Disk-backed exact deduplication and approximate near-match indexing.

* Exact duplicates: BLAKE2b hash of the original extracted text, scoped by profile.
* Near duplicates: deterministic one-permutation MinHash (32 bins) over 5-word
  shingles, with 8 LSH bands stored in SQLite.  Candidate lists are capped so
  RAM stays bounded even for hundreds of millions of documents.
* Near matches are retained and can be assigned a common cluster for splitting.
  LSH is approximate and does not establish absence of cross-split leakage.

MinHash/LSH can miss matches and group distinct texts; it is a review and
split-clustering aid, not a claim of perfect duplicate detection.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
from pathlib import Path
from typing import Iterable

from normalize_persian import dedup_key, words_for_shingles

SENTINEL = (1 << 59) - 1  # unfilled MinHash bin


class SQLiteCorpusIndex:
    """SQLite index of document metadata, exact hashes, near-dup bands, audits."""

    def __init__(self, path: str | Path, config: dict, *, reset: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if reset and self.path.exists():
            for suffix in ("", "-wal", "-shm"):
                p = Path(str(self.path) + suffix)
                if p.exists():
                    p.unlink()
        self.cfg = config.get("dedup", {})
        self.num_bins = int(self.cfg.get("num_bins", 32))
        self.band_size = int(self.cfg.get("band_size", 4))
        self.shingle_n = int(self.cfg.get("shingle_n", 5))
        self.max_words = int(self.cfg.get("max_words", 2000))
        self.min_shingles = int(self.cfg.get("min_shingles_for_near_dup", 8))
        self.threshold = float(self.cfg.get("similarity_threshold", 0.75))
        self.max_candidates = int(self.cfg.get("max_candidates_per_band", 200))
        self.commit_every = int(self.cfg.get("commit_every", 1000))
        self._ops = 0
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute(f"PRAGMA cache_size=-{int(self.cfg.get('sqlite_cache_mb', 32)) * 1024}")
        self.conn.execute("PRAGMA temp_store=FILE")
        self.conn.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        c = self.conn
        c.executescript("""
        CREATE TABLE IF NOT EXISTS docs (
            doc_id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            profile TEXT NOT NULL,
            license TEXT,
            domain TEXT,
            shard TEXT NOT NULL DEFAULT '',
            offset INTEGER NOT NULL DEFAULT 0,
            byte_len INTEGER NOT NULL DEFAULT 0,
            chars INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL,
            reject_reason TEXT,
            quality REAL,
            exact_hash BLOB,
            signature BLOB,
            cluster_id TEXT,
            split TEXT,
            selected INTEGER NOT NULL DEFAULT 0,
            meta TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_docs_status_split ON docs(status, split);
        CREATE INDEX IF NOT EXISTS idx_docs_cluster ON docs(cluster_id);
        CREATE INDEX IF NOT EXISTS idx_docs_source ON docs(source_id, status);
        CREATE TABLE IF NOT EXISTS exact (
            hash BLOB PRIMARY KEY,
            doc_id TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS bands (
            band_idx INTEGER NOT NULL,
            band_key BLOB NOT NULL,
            doc_id TEXT NOT NULL,
            PRIMARY KEY (band_idx, band_key, doc_id)
        ) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS idx_bands_lookup ON bands(band_idx, band_key, doc_id);
        CREATE TABLE IF NOT EXISTS audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT,
            source_id TEXT,
            category TEXT,
            reason TEXT,
            excerpt TEXT,
            paired_doc_id TEXT,
            meta TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_audit_category ON audit(category, doc_id);
        CREATE TABLE IF NOT EXISTS stats (k TEXT PRIMARY KEY, v TEXT);
        CREATE TABLE IF NOT EXISTS clusters (
            source_id TEXT NOT NULL,
            cluster_id TEXT NOT NULL,
            chars INTEGER NOT NULL,
            h INTEGER NOT NULL,
            split TEXT,
            PRIMARY KEY (source_id, cluster_id)
        );
        CREATE INDEX IF NOT EXISTS idx_clusters_order ON clusters(source_id, h, cluster_id);
        """)
        cols = [r[1] for r in c.execute("PRAGMA table_info(docs)")]
        if "selected" not in cols:
            c.execute("ALTER TABLE docs ADD COLUMN selected INTEGER NOT NULL DEFAULT 0")
        c.commit()

    # ------------------------------ signatures ---------------------------- #

    def make_signature(self, text: str, seed: str = "") -> tuple[bytes, int]:
        words = words_for_shingles(text)
        sampled = False
        if len(words) > self.max_words:
            step = len(words) / self.max_words
            words = [words[int(i * step)] for i in range(self.max_words)]
            sampled = True
        n_shingles = max(0, len(words) - self.shingle_n + 1)
        sig = [SENTINEL + 1] * self.num_bins
        if n_shingles <= 0:
            return struct.pack(f"<{self.num_bins}Q", *sig), 0
        person = b"persian-minh"[:16]
        for i in range(n_shingles):
            shingle = " ".join(words[i:i + self.shingle_n])
            data = (seed + "\x1f" + shingle).encode("utf-8")
            h = int.from_bytes(hashlib.blake2b(data, digest_size=8, person=person).digest(), "big")
            b = h >> (64 - 5)  # top 5 bits -> 32 bins
            v = h & SENTINEL
            if v < sig[b]:
                sig[b] = v
        return struct.pack(f"<{self.num_bins}Q", *sig), n_shingles

    def signature_bands(self, sig: bytes) -> list[tuple[int, bytes]]:
        vals = struct.unpack(f"<{self.num_bins}Q", sig)
        out = []
        for i in range(0, self.num_bins, self.band_size):
            chunk = b"|".join(str(v).encode("ascii") for v in vals[i:i + self.band_size])
            key = hashlib.blake2b(chunk, digest_size=12, person=b"persian-lsh"[:16]).digest()
            out.append((i // self.band_size, key))
        return out

    def signature_similarity(self, a: bytes, b: bytes) -> float:
        ua = struct.unpack(f"<{self.num_bins}Q", a)
        ub = struct.unpack(f"<{self.num_bins}Q", b)
        equal = comparable = 0
        for x, y in zip(ua, ub):
            if x > SENTINEL and y > SENTINEL:
                continue  # both empty: not evidence of similarity
            comparable += 1
            if x == y:
                equal += 1
        return round(equal / max(1, comparable), 4)

    def exact_hash(self, text: str, profile: str = "") -> bytes:
        # Length-prefix the profile so A/B rights are never silently merged.
        payload = len(profile.encode("utf-8")).to_bytes(4, "big") + profile.encode("utf-8") + dedup_key(text).encode("utf-8")
        return hashlib.blake2b(payload, digest_size=24,
                               person=b"persian-exac"[:16]).digest()

    # ------------------------------ lookups ------------------------------- #

    def has_doc(self, doc_id: str) -> bool:
        return self.conn.execute("SELECT 1 FROM docs WHERE doc_id=?", (doc_id,)).fetchone() is not None

    def exact_duplicate_of(self, h: bytes) -> str | None:
        row = self.conn.execute("SELECT doc_id FROM exact WHERE hash=?", (h,)).fetchone()
        return row["doc_id"] if row else None

    def near_duplicate_of(self, sig: bytes, n_shingles: int) -> tuple[str | None, float, int]:
        if n_shingles < self.min_shingles:
            return None, 0.0, 0
        cands: set[str] = set()
        overloaded = 0
        for band_idx, band_key in self.signature_bands(sig):
            rows = self.conn.execute(
                "SELECT doc_id FROM bands WHERE band_idx=? AND band_key=? "
                "ORDER BY doc_id LIMIT ?",
                (band_idx, band_key, self.max_candidates + 1)).fetchall()
            if len(rows) > self.max_candidates:
                overloaded += 1
                self.increment_stat("lsh_bands_over_candidate_cap", 1)
            for row in rows[:self.max_candidates]:
                cands.add(row["doc_id"])
        if overloaded:
            self.increment_stat("documents_with_overloaded_lsh_bands", 1)
        best_sim, best_doc = 0.0, None
        for doc_id in sorted(cands):
            row = self.conn.execute(
                "SELECT signature FROM docs WHERE doc_id=? AND status='kept'", (doc_id,)).fetchone()
            if not row or not row["signature"]:
                continue
            sim = self.signature_similarity(sig, row["signature"])
            if sim > best_sim:
                best_sim, best_doc = sim, doc_id
            if sim >= self.threshold:
                self.increment_stat("near_dup_candidates_verified", 1)
                return doc_id, sim, len(cands)
        self.increment_stat("near_dup_candidates_rejected", 1)
        return None, best_sim, len(cands)

    # ------------------------------ writes -------------------------------- #

    def _bump(self) -> None:
        self._ops += 1
        if self._ops >= self.commit_every:
            self.conn.commit()
            self._ops = 0

    def add_rejected(self, *, doc_id: str, source_id: str, profile: str,
                     license_: str, domain: str, reason: str, quality: float,
                     excerpt: str, meta: dict | None = None, categories: Iterable[str] = ()) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO docs(doc_id,source_id,profile,license,domain,status,"
            "reject_reason,quality,cluster_id,meta) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (doc_id, source_id, profile, license_, domain, "rejected", reason,
             quality, doc_id, json.dumps(meta or {}, ensure_ascii=False, sort_keys=True)))
        self.record_audit(doc_id, ",".join(categories) or "rejected", reason,
                          excerpt, None, meta, source_id=source_id)
        self.increment_stat("rejected_documents", 1)
        self.increment_stat("reject_reason:" + reason.split(":")[0], 1)
        self._bump()

    def add_kept(self, *, doc_id: str, source_id: str, profile: str,
                 license_: str, domain: str, shard: str, offset: int,
                 byte_len: int, chars: int, quality: float, exact_hash: bytes,
                 signature: bytes, meta: dict | None = None,
                 categories: Iterable[str] = (), cluster_id: str | None = None) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO docs(doc_id,source_id,profile,license,domain,shard,offset,"
            "byte_len,chars,status,quality,exact_hash,signature,cluster_id,split,selected,meta) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,?)",
            (doc_id, source_id, profile, license_, domain, shard, offset, byte_len,
             chars, "kept", quality, exact_hash, signature, cluster_id or doc_id, None,
             json.dumps(meta or {}, ensure_ascii=False, sort_keys=True)))
        self.conn.execute("INSERT OR IGNORE INTO exact(hash,doc_id) VALUES(?,?)",
                          (exact_hash, doc_id))
        for band_idx, band_key in self.signature_bands(signature):
            self.conn.execute("INSERT OR IGNORE INTO bands(band_idx,band_key,doc_id) VALUES(?,?,?)",
                              (band_idx, band_key, doc_id))
        self.record_audit(doc_id, ",".join(categories) or "good", "kept", "", None, meta,
                          source_id=source_id)
        self.increment_stat("kept_documents", 1)
        self._bump()

    def add_duplicate(self, *, doc_id: str, source_id: str, profile: str,
                      license_: str, domain: str, status: str, cluster_id: str,
                      quality: float, excerpt: str, paired_doc_id: str,
                      similarity: float, meta: dict | None = None) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO docs(doc_id,source_id,profile,license,domain,status,"
            "reject_reason,quality,cluster_id,meta) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (doc_id, source_id, profile, license_, domain, status,
             f"{status}:{paired_doc_id}:{similarity}", quality, cluster_id,
             json.dumps(meta or {}, ensure_ascii=False, sort_keys=True)))
        self.record_audit(doc_id, "near_duplicate_example", status, excerpt,
                          paired_doc_id, meta, source_id=source_id)
        self.increment_stat("exact_duplicates" if status == "exact_duplicate" else "near_duplicates", 1)
        self._bump()

    def record_audit(self, doc_id: str, category: str, reason: str, excerpt: str,
                     paired_doc_id: str | None, meta: dict | None = None,
                     source_id: str | None = None) -> None:
        self.conn.execute(
            "INSERT INTO audit(doc_id,source_id,category,reason,excerpt,paired_doc_id,meta) "
            "VALUES(?,?,?,?,?,?,?)",
            (doc_id, source_id, category, reason, excerpt, paired_doc_id,
             json.dumps(meta or {}, ensure_ascii=False, sort_keys=True)))

    def increment_stat(self, key: str, amount: int = 1) -> None:
        row = self.conn.execute("SELECT v FROM stats WHERE k=?", (key,)).fetchone()
        value = int(row["v"]) if row else 0
        self.conn.execute("INSERT OR REPLACE INTO stats(k,v) VALUES(?,?)", (key, str(value + amount)))

    def set_stat(self, key: str, value) -> None:
        self.conn.execute("INSERT OR REPLACE INTO stats(k,v) VALUES(?,?)",
                          (key, json.dumps(value, ensure_ascii=False, sort_keys=True)))

    def get_stat(self, key: str, default=None):
        row = self.conn.execute("SELECT v FROM stats WHERE k=?", (key,)).fetchone()
        if not row:
            return default
        try:
            return json.loads(row["v"])
        except json.JSONDecodeError:
            return row["v"]

    def all_stats(self) -> dict:
        out = {}
        for row in self.conn.execute("SELECT k,v FROM stats ORDER BY k"):
            try:
                out[row["k"]] = json.loads(row["v"])
            except (json.JSONDecodeError, TypeError):
                out[row["k"]] = row["v"]
        return out

    def commit(self) -> None:
        self.conn.commit()
        self._ops = 0

    def close(self) -> None:
        self.commit()
        self.conn.close()
