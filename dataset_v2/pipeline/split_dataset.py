#!/usr/bin/env python3
"""Deterministic, cluster-consistent train/validation/test assignment.

Splitting happens AFTER exact deduplication, approximate near-match clustering,
and balancing. Members of an identified cluster share one split. LSH can miss
matches or fail to connect overlapping clusters; this is not proof that all near
matches are isolated across train/validation/test.

Clusters are ordered by a seeded BLAKE2b hash within each source and assigned
to meet per-source character quotas without large overshoot.  A post-pass adds
minimum val/test representation only for sources with many clusters (or, on a
tiny test corpus, globally), and reports all shortfalls.  No full-corpus list
is materialized: updates are streamed through SQLite in small batches.
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from typing import Any

from common import load_config, stable_int
from deduplicate import SQLiteCorpusIndex


def _set_split(conn, split: str, cluster_id: str) -> None:
    conn.execute("UPDATE clusters SET split=? WHERE cluster_id=?", (split, cluster_id))
    conn.execute(
        "UPDATE docs SET split=? WHERE status='kept' AND selected=1 AND cluster_id=?",
        (split, cluster_id))


def assign_splits(index: SQLiteCorpusIndex, config: dict, *,
                  logger: logging.Logger | None = None) -> dict:
    seed = config.get("seed", "persian-dataset-v2")
    ratios = config.get("split_ratios", {"train": 0.97, "validation": 0.02, "test": 0.01})
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError(f"split_ratios must sum to 1.0, got {sum(ratios.values())}")
    min_forced = int(config.get("min_clusters_for_forced_representation", 30))
    conn = index.conn
    conn.execute("DELETE FROM clusters")
    conn.commit()

    # 1. One row per global cluster, stratified by its kept representative source.
    n_clusters = 0
    for row in conn.execute(
            "SELECT d.cluster_id, r.source_id, SUM(d.chars) AS chars FROM docs d "
            "JOIN docs r ON r.doc_id=d.cluster_id "
            "WHERE d.status='kept' AND d.selected=1 GROUP BY d.cluster_id, r.source_id"):
        h = stable_int(row["cluster_id"] + "\x1f" + row["source_id"], seed + ":split")
        conn.execute(
            "INSERT OR REPLACE INTO clusters(source_id,cluster_id,chars,h,split) VALUES(?,?,?,?,NULL)",
            (row["source_id"], row["cluster_id"], int(row["chars"]), h))
        n_clusters += 1
    conn.commit()

    totals = {r["source_id"]: {"chars": int(r["chars"]), "clusters": int(r["n"])}
              for r in conn.execute(
                  "SELECT source_id, SUM(chars) AS chars, COUNT(*) AS n FROM clusters GROUP BY source_id")}

    # 2. Streaming per-source quota assignment, with bounded overshoot.
    for source_id in sorted(totals):
        info = totals[source_id]
        val_target = info["chars"] * ratios["validation"]
        test_target = info["chars"] * ratios["test"]
        val_used = test_used = assigned_chars = 0
        batch: list[tuple[str, str]] = []
        for row in conn.execute(
                "SELECT cluster_id, chars FROM clusters WHERE source_id=? ORDER BY h, cluster_id",
                (source_id,)):
            cid, chars = row["cluster_id"], int(row["chars"])
            remaining = info["chars"] - assigned_chars
            val_need = val_target - val_used
            test_need = test_target - test_used
            if val_need > 0 and (chars <= val_need * 1.25 or remaining <= val_need):
                split = "validation"
                val_used += chars
            elif test_need > 0 and (chars <= test_need * 1.25 or remaining <= test_need + max(0, val_need)):
                split = "test"
                test_used += chars
            else:
                split = "train"
            assigned_chars += chars
            batch.append((split, cid))
            if len(batch) >= 1000:
                conn.executemany("UPDATE clusters SET split=? WHERE cluster_id=?", batch)
                conn.executemany(
                    "UPDATE docs SET split=? WHERE status='kept' AND selected=1 AND cluster_id=?", batch)
                batch = []
        if batch:
            conn.executemany("UPDATE clusters SET split=? WHERE cluster_id=?", batch)
            conn.executemany(
                "UPDATE docs SET split=? WHERE status='kept' AND selected=1 AND cluster_id=?", batch)
    conn.commit()

    # 3. Minimum source representation for genuinely large sources.
    source_forced = []
    for source_id, info in sorted(totals.items()):
        if info["clusters"] < min_forced:
            continue
        present = {r[0] for r in conn.execute(
            "SELECT DISTINCT split FROM clusters WHERE source_id=?", (source_id,))}
        missing = {"validation", "test"} - present
        if not missing:
            continue
        donors = [r[0] for r in conn.execute(
            "SELECT cluster_id FROM clusters WHERE source_id=? AND split='train' "
            "ORDER BY chars, h, cluster_id LIMIT ?", (source_id, len(missing)))]
        for split, cid in zip(sorted(missing), donors):
            _set_split(conn, split, cid)
        if donors:
            source_forced.append({"source_id": source_id, "splits_added": sorted(missing),
                                  "clusters": donors})
    conn.commit()

    # 4. Tiny-corpus fallback: make sure all three split files can exist.
    global_forced = None
    counts = {r[0]: int(r[1]) for r in conn.execute(
        "SELECT split, COUNT(*) FROM clusters GROUP BY split")}
    missing_global = {"train", "validation", "test"} - set(counts)
    if missing_global and n_clusters >= 3:
        donor = max(counts, key=counts.get) if counts else "train"
        donors = [r[0] for r in conn.execute(
            "SELECT cluster_id FROM clusters WHERE split=? ORDER BY chars, h, cluster_id LIMIT ?",
            (donor, len(missing_global)))]
        for split, cid in zip(sorted(missing_global), donors):
            _set_split(conn, split, cid)
        global_forced = {"donor_split": donor, "splits_added": sorted(missing_global),
                         "clusters": donors,
                         "note": "Only for tiny/small-sample corpora; full runs should meet quotas naturally."}
        conn.commit()

    # 5. Verify cluster consistency and summarize.
    leaked = [r[0] for r in conn.execute(
        "SELECT cluster_id FROM docs WHERE status='kept' AND selected=1 AND split IS NOT NULL "
        "GROUP BY cluster_id HAVING COUNT(DISTINCT split) > 1 LIMIT 20")]
    documents_assigned = int(conn.execute(
        "SELECT COUNT(*) FROM docs WHERE status='kept' AND selected=1 AND split IS NOT NULL").fetchone()[0])
    by_split = {}
    for row in conn.execute(
            "SELECT split, COUNT(*) AS docs, SUM(chars) AS chars FROM docs "
            "WHERE status='kept' AND selected=1 GROUP BY split"):
        by_split[row["split"] or "unassigned"] = {
            "documents": int(row["docs"]), "characters": int(row["chars"])}
    by_source_split = defaultdict(lambda: defaultdict(lambda: {"documents": 0, "characters": 0}))
    for row in conn.execute(
            "SELECT source_id, split, COUNT(*) AS docs, SUM(chars) AS chars FROM docs "
            "WHERE status='kept' AND selected=1 GROUP BY source_id, split"):
        cell = by_source_split[row["source_id"]][row["split"] or "unassigned"]
        cell["documents"] = int(row["docs"])
        cell["characters"] = int(row["chars"])
    total_chars = sum(v["characters"] for v in by_split.values())
    for v in by_split.values():
        v["share_of_characters"] = round(v["characters"] / max(1, total_chars), 4)

    shortfalls = []
    for source_id, info in sorted(totals.items()):
        present = {r[0] for r in conn.execute(
            "SELECT DISTINCT split FROM clusters WHERE source_id=?", (source_id,))}
        if not {"train", "validation", "test"}.issubset(present):
            shortfalls.append({
                "source_id": source_id, "clusters": info["clusters"],
                "splits_present": sorted(present),
                "reason": "source_has_too_few_clusters_for_all_splits_or_quota_not_met",
            })

    report: dict[str, Any] = {
        "method": "seeded BLAKE2b ordering of duplicate clusters, per-source character quotas with bounded overshoot, cluster-consistent assignment after global deduplication",
        "seed": seed,
        "requested_ratios": ratios,
        "min_clusters_for_forced_representation": min_forced,
        "clusters": n_clusters,
        "clusters_assigned": n_clusters,
        "documents_assigned": documents_assigned,
        "by_split": by_split,
        "by_source_split": {s: dict(d) for s, d in by_source_split.items()},
        "forced_minimum_representation_sources": source_forced,
        "global_tiny_corpus_representation": global_forced,
        "shortfalls": shortfalls,
        "shortfall_note": "Shortfalls are reported, never filled by duplicating documents.",
        "cluster_leakage_examples": leaked,
        "cluster_leakage_count": len(leaked),
    }
    if leaked:
        raise RuntimeError(f"Split leakage detected for {len(leaked)} duplicate clusters: {leaked[:5]}")
    if logger:
        logger.info("splits: %s", json.dumps(by_split, ensure_ascii=False))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Assign deterministic V2 splits to an existing corpus index")
    ap.add_argument("--index", required=True, help="path to corpus_index.sqlite3")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    idx = SQLiteCorpusIndex(args.index, cfg)
    try:
        report = assign_splits(idx, cfg)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        idx.close()


if __name__ == "__main__":
    main()
