#!/usr/bin/env python3
"""
build_dataset.py — clean, filter, deduplicate, balance and split the corpus.

Stages
------
1. clean   : for every extracted document -> unicode/character normalization,
             markup and boilerplate removal, line-level noise removal.
2. filter  : language + quality gate (persian_norm.quality_check), with a
             per-source minimum length.
3. dedup   : exact duplicates (hash of aggressively normalized text) and
             near-duplicates (64-bit SimHash, 4x16-bit LSH banding, Hamming<=3).
4. budget  : because a Git repository has a practical size ceiling, the corpus
             is sampled down to a size budget with per-source caps; Wikipedia is
             sampled by quality score, everything else by stable hash order.
5. split   : deterministic 98.5/1.5 train/validation split on the document hash
             (the split happens *after* dedup, so the two sides are disjoint).

Intermediate files go to --tmpdir. The final train/validation files are written
into --outdir.

Usage:
    python3 pipeline/build_dataset.py \
        --extracted /tmp/work/extracted --tmpdir /tmp/work/build \
        --outdir . --budget 45000000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import persian_norm as P  # noqa: E402

# --------------------------------------------------------------------------- #
# Per-source configuration
# --------------------------------------------------------------------------- #
# collapse_phrases: remove back-to-back repeated multi-word phrases (a wiki
#            link-flattening artefact). Off for verse, where refrains are real.
# min_chars: shorter documents are rejected for this source. QA/dialogue items
#            are legitimately short, prose sources are not.
# cap:       maximum share of the total character budget (weights are renormalized
#            if a source cannot fill its quota).
# score:     "quality" ranks documents by the quality score (used for Wikipedia),
#            "hash" takes a uniform stable sample.
SOURCE_PROFILES = {
    # cap = maximum share of the character budget for that source
    # score = "quality" (rank by quality score) or "hash" (uniform stable sample)
    "wikipedia_fa":        {"cap": 0.650, "score": "quality", "collapse_phrases": True},
    "persian_news":        {"cap": 0.100, "score": "hash", "collapse_phrases": True},
    "pn_summary_news":     {"cap": 0.080, "score": "hash", "collapse_phrases": True},
    "persian_poetry":      {"cap": 0.020, "score": "hash"},
    "ganjoor_classics":    {"cap": 0.030, "score": "hash"},
    "parsinlu_entailment": {"cap": 0.025, "score": "hash", "collapse_phrases": True},
    "peqa_dialogue":       {"cap": 0.025, "score": "hash"},
    "parsinlu_rc_passage": {"cap": 0.020, "score": "hash", "collapse_phrases": True},
    "parsinlu_mc":         {"cap": 0.006, "score": "hash", "collapse_phrases": True},
    "parsinlu_qa":         {"cap": 0.004, "score": "hash", "collapse_phrases": True},
}
DEFAULT_PROFILE = {"cap": 0.02, "score": "hash"}

VAL_FRACTION = 0.02           # 2 % of documents go to validation
SEED = "persian-gpt-100m-v1"  # stable hashing salt (makes runs reproducible)
DOC_SEPARATOR = "\n\n<|endoftext|>\n\n"


def stable_hash(s: str, salt: str = SEED) -> str:
    return hashlib.sha1((salt + "\x1f" + s).encode("utf-8")).hexdigest()


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Quality score (only used to prioritise Wikipedia sampling)
# --------------------------------------------------------------------------- #
def quality_score(text: str, cp: dict) -> float:
    """Crude 0..1 quality proxy: length, information density, Persian-ness."""
    length = min(len(text) / 2500.0, 1.0)
    cr = P.compress_ratio(text) if len(text) > 600 else 2.6
    density = 1.0 if 1.9 <= cr <= 3.4 else max(0.0, 1.0 - abs(cr - 2.6) / 2.0)
    letters = cp["persian_letters"] + cp["latin_letters"]
    persian = cp["persian_letters"] / max(letters, 1)
    digits = 1.0 - min(cp["digits"] / max(cp["non_space"], 1) / 0.22, 1.0)
    return 0.40 * length + 0.30 * density + 0.20 * persian + 0.10 * digits


# --------------------------------------------------------------------------- #
# Stage 1+2: clean and filter one source
# --------------------------------------------------------------------------- #
def clean_source(src_path: str, src: str, cleaned_path: str, prof: dict):
    """Normalize + quality-filter one source. Returns (index rows, stats)."""
    drops = Counter()
    rows = []          # (id, offset, length, exact_hash, simhash, quality)
    n_raw = n_clean_chars = 0
    t0 = time.time()

    with open(src_path, encoding="utf-8") as fin, \
            open(cleaned_path, "wb") as fout:
        for line in fin:
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_raw += 1
            text = P.normalize(doc["text"], collapse_phrases=prof.get("collapse_phrases", False))
            text = P.clean_lines(text)
            ok, reason = P.quality_check(
                text,
                min_chars=prof.get("min_chars", 280),
                min_letters=prof.get("min_letters", 100),
                min_sentences=prof.get("min_sentences", 3),
                filters=prof.get("filters"),
            )
            if not ok:
                drops[reason] += 1
                continue
            cp = P.char_profile(text)
            sig, nsh = P.near_dup_signature(text)
            offset = fout.tell()          # byte offset of the document record
            fout.write(json.dumps(text, ensure_ascii=False).encode("utf-8") + b"\n")
            length = len(text)
            n_clean_chars += length
            rows.append({
                "id": doc["id"],
                "src": src,
                "off": offset,
                "len": length,
                "h": sha1(P.dedup_key(text)),
                "sig": sig,
                "nsh": nsh,
                "q": round(quality_score(text, cp), 4),
            })
    dt = time.time() - t0
    stats = {
        "source": src,
        "raw_docs": n_raw,
        "kept_after_quality": len(rows),
        "kept_chars": n_clean_chars,
        "dropped_by_reason": dict(drops.most_common()),
        "seconds": round(dt, 1),
    }
    print(f"  [{src:<22}] raw {n_raw:>8,} -> kept {len(rows):>8,} "
          f"({n_clean_chars:>12,} chars, {dt:>5.1f}s)  drops={dict(drops.most_common(4))}")
    return rows, stats


# --------------------------------------------------------------------------- #
# Stage 3: deduplication
# --------------------------------------------------------------------------- #
def deduplicate(rows: list, bucket_cap: int = 20000):
    """Exact + near-duplicate removal.

    Exact duplicates  : identical SHA-1 of the aggressively normalized text.
    Near duplicates   : one-permutation MinHash (16 buckets); candidates come
                        from 4-band LSH, and a candidate is a duplicate when at
                        least 3 of the 4 bands agree (>= 12/16 buckets). Short
                        documents (< 8 shingles) are only checked exactly,
                        because their signatures are too sparse to be reliable.

    Rows arrive pre-sorted by quality, so the surviving copy of a duplicate
    cluster is the highest-quality copy.
    """
    seen_exact = set()
    buckets = defaultdict(list)      # band key -> [idx into kept]
    kept = []
    sigs = []
    exact_dups = near_dups = empty = 0

    for row in rows:
        if row["len"] == 0:
            empty += 1
            continue
        h = row["h"]
        if h in seen_exact:
            exact_dups += 1
            continue

        sig = row["sig"]
        band_bytes = P.signature_bands(sig)
        is_dup = False
        if row["nsh"] >= 8:
            cands = set()
            for bi, band in enumerate(band_bytes):
                lst = buckets.get((bi, band))
                if lst and len(lst) <= bucket_cap:
                    cands.update(lst)
            for idx in cands:
                if P.signature_match(sig, sigs[idx]) >= P.MIN_BANDS_MATCH * P.BAND_LEN:
                    is_dup = True
                    break
        if is_dup:
            near_dups += 1
            continue

        seen_exact.add(h)
        idx = len(kept)
        kept.append(row)
        sigs.append(row["sig"])
        for bi, band in enumerate(band_bytes):
            buckets[(bi, band)].append(idx)

    stats = {
        "input_docs": len(rows),
        "exact_duplicates": exact_dups,
        "near_duplicates": near_dups,
        "empty": empty,
        "output_docs": len(kept),
    }
    return kept, stats


# --------------------------------------------------------------------------- #
# Stage 4: budget selection
# --------------------------------------------------------------------------- #
# Wikipedia (and any large "quality"-sampled source) is sampled per length band so
# that the shipped corpus mirrors the natural mix of article lengths instead of
# collapsing onto a few thousand very long encyclopedic articles.
LENGTH_BANDS = [("short", 0, 800, 0.15),
                ("medium", 800, 2500, 0.40),
                ("long", 2500, 8000, 0.35),
                ("very_long", 8000, 10 ** 9, 0.10)]


def stratified_quality_order(rows: list):
    """Return (ordered_rows, picked_rows) using length-band quotas inside the quota.

    The caller passes a quota; here we only need the ordering per band, so bands
    are concatenated in a fixed order and the quota split is applied by the
    caller through band boundaries recorded as attributes.
    """
    bands = {name: [] for name, _, _, _ in LENGTH_BANDS}
    for r in rows:
        L = r["len"]
        for name, lo, hi, _w in LENGTH_BANDS:
            if lo <= L < hi:
                r["band"] = name
                bands[name].append(r)
                break
    for name in bands:
        bands[name].sort(key=lambda r: (-r["q"], stable_hash(r["id"])))
    return bands


def select_to_budget(all_rows: list, total_budget: int):
    """Pick documents per source up to its cap; Wikipedia ranked by quality."""
    by_src = defaultdict(list)
    for r in all_rows:
        by_src[r["src"]].append(r)

    total_chars = sum(r["len"] for r in all_rows)
    scale = min(1.0, total_budget / max(total_chars, 1))

    selected = []
    per_source = {}
    remaining_partners = []      # (src, rows, target, filled)

    for src, rows in by_src.items():
        prof = SOURCE_PROFILES.get(src, DEFAULT_PROFILE)
        target = int(total_budget * prof["cap"]) if scale < 1.0 else sum(r["len"] for r in rows)
        picked, used = [], 0
        if prof["score"] == "quality":
            bands = stratified_quality_order(rows)
            order = [r for name, _lo, _hi, _w in LENGTH_BANDS for r in bands[name]]
            # quota per length band (weights renormalized over bands that exist)
            weights = {name: w for name, _lo, _hi, w in LENGTH_BANDS if bands[name]}
            wsum = sum(weights.values()) or 1.0
            band_report = {}
            for name, _lo, _hi, _w in LENGTH_BANDS:
                if not bands[name]:
                    continue
                band_target = target * weights[name] / wsum
                band_used, band_docs = 0, 0
                for r in bands[name]:
                    if band_used + r["len"] > band_target and band_used > 0:
                        continue
                    picked.append(r)
                    band_used += r["len"]
                    band_docs += 1
                    if band_used >= band_target:
                        break
                used += band_used
                band_report[name] = {"documents": band_docs, "characters": band_used,
                                     "weight": round(weights[name] / wsum, 4)}
        else:
            order = sorted(rows, key=lambda r: stable_hash(r["id"]))
            for r in order:
                if used + r["len"] > target and picked:
                    continue
                picked.append(r)
                used += r["len"]
                if used >= target:
                    break
        selected.extend(picked)
        per_source[src] = {
            "available_docs": len(rows),
            "available_chars": sum(r["len"] for r in rows),
            "cap_share": prof["cap"],
            "quota_chars": target,
            "selected_docs": len(picked),
            "selected_chars": used,
            "sampling": prof["score"],
        }
        if prof["score"] == "quality":
            per_source[src]["length_band_sampling"] = band_report
        remaining_partners.append([src, order, target, used, picked])

    # 4b. redistribute unused budget to sources that still have material
    used_total = sum(v["selected_chars"] for v in per_source.values())
    slack = total_budget - used_total
    if slack > 0:
        # never let one source dominate: give leftovers to the biggest provider
        # that still has unselected documents, proportional to remaining stock
        pool = [(s, order, picked) for (s, order, target, used, picked) in remaining_partners
                if len(picked) < len(order)]
        pool.sort(key=lambda x: -(len(x[1]) - len(x[2])))
        for src, order, picked in pool:
            if slack <= 0:
                break
            chosen_ids = {id(r) for r in picked}
            added, used = 0, 0
            for r in order:
                if id(r) in chosen_ids:
                    continue
                if used + r["len"] > slack:
                    continue
                picked.append(r)
                used += r["len"]
                added += 1
                if used >= slack:
                    break
            selected.extend([r for r in picked[-added:]] if added else [])
            per_source[src]["selected_docs"] = len(picked)
            per_source[src]["selected_chars"] += used
            per_source[src]["budget_overflow_fill"] = used
            slack -= used
    return selected, per_source


# --------------------------------------------------------------------------- #
# Stage 5: write train/validation
# --------------------------------------------------------------------------- #
def write_split(selected: list, cleaned_paths: dict, outdir: str, manifest_path: str | None = None):
    os.makedirs(outdir, exist_ok=True)
    train_path = os.path.join(outdir, "train.txt")
    val_path = os.path.join(outdir, "validation.txt")

    ordered = sorted(selected, key=lambda r: stable_hash(r["id"], SEED + ":order"))
    n_val = 0
    handles = {src: open(p, "rb") for src, p in cleaned_paths.items()}
    manifest = open(manifest_path, "w", encoding="utf-8") if manifest_path else None
    try:
      with open(train_path, "w", encoding="utf-8", newline="\n") as ftrain, \
              open(val_path, "w", encoding="utf-8", newline="\n") as fval:
        for r in ordered:
            f = handles[r["src"]]
            f.seek(r["off"])
            text = json.loads(f.readline())
            h = stable_hash(r["id"], SEED + ":split")
            to_val = (int(h[:8], 16) / 0xFFFFFFFF) < VAL_FRACTION
            out = ftrain if not to_val else fval
            out.write(text)
            out.write(DOC_SEPARATOR)
            if to_val:
                n_val += 1
            if manifest is not None:
                manifest.write(f"{r['src']}\t{r['id']}\t{r['len']}\t"
                               f"{'val' if to_val else 'train'}\n")
    finally:
        for f in handles.values():
            f.close()
        if manifest is not None:
            manifest.close()
    return {"train_docs": len(ordered) - n_val, "val_docs": n_val}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extracted", required=True)
    ap.add_argument("--tmpdir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--budget", type=int, default=45_000_000,
                    help="character budget for the shipped corpus")
    ap.add_argument("--report", default=None, help="where to write the build report JSON")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    os.makedirs(args.tmpdir, exist_ok=True)
    cleaned_dir = os.path.join(args.tmpdir, "cleaned")
    os.makedirs(cleaned_dir, exist_ok=True)

    src_files = sorted(f for f in os.listdir(args.extracted) if f.endswith(".jsonl")
                       and not f.startswith("_"))
    if args.only:
        src_files = [f for f in src_files if f[:-6] in args.only]

    print(f"Stage 1-2: clean + filter ({len(src_files)} sources)")
    all_rows, clean_stats, cleaned_paths = [], [], {}
    for fn in src_files:
        src = fn[:-6]
        prof = SOURCE_PROFILES.get(src, DEFAULT_PROFILE)
        cv = os.path.join(cleaned_dir, fn)
        thr = {**P.PROFILES.get(src, P.PROFILES["persian_news"]), **prof}
        rows, st = clean_source(os.path.join(args.extracted, fn), src, cv, thr)
        cleaned_paths[src] = cv
        clean_stats.append(st)
        all_rows.extend(rows)

    print(f"\nStage 3: deduplication over {len(all_rows):,} documents")
    # deterministic, quality-first order so the survivor of a dup cluster is the
    # best copy: higher quality score first, then stable hash
    all_rows.sort(key=lambda r: (-r["q"], stable_hash(r["id"])))
    kept, dedup_stats = deduplicate(all_rows)
    print(f"  exact={dedup_stats['exact_duplicates']:,} "
          f"near={dedup_stats['near_duplicates']:,} -> kept {len(kept):,}")

    print(f"\nStage 4: budget selection (target {args.budget:,} chars)")
    selected, per_source = select_to_budget(kept, args.budget)
    sel_chars = sum(r["len"] for r in selected)
    print(f"  selected {len(selected):,} docs / {sel_chars:,} chars")
    for s, v in sorted(per_source.items(), key=lambda kv: -kv[1]["selected_chars"]):
        print(f"    {s:<22} {v['selected_docs']:>8,} docs {v['selected_chars']:>12,} chars")

    print("\nStage 5: writing train/validation")
    manifest_path = os.path.join(args.tmpdir, "selected_manifest.tsv")
    split = write_split(selected, cleaned_paths, args.outdir, manifest_path)
    print(f"  train={split['train_docs']:,}  validation={split['val_docs']:,}")

    report = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pipeline": {
            "budget_chars": args.budget,
            "val_fraction": VAL_FRACTION,
            "seed": SEED,
            "doc_separator": "<|endoftext|>",
            "normalization": "pipeline/persian_norm.py",
            "filters": P.FILTERS,
        },
        "cleaning": clean_stats,
        "dedup": dedup_stats,
        "selection": per_source,
        "split": split,
    }
    out_report = args.report or os.path.join(args.tmpdir, "build_report.json")
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nBuild report: {out_report}")


if __name__ == "__main__":
    sys.exit(main())
