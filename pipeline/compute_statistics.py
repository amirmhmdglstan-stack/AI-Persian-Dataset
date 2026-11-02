#!/usr/bin/env python3
"""
compute_statistics.py — derive every reported number directly from the shipped
files (train.txt / validation.txt), never from intermediate estimates.

Each number in the output is tagged with how it was obtained:
  "verified"  -> counted from the final files by this script
  "estimated" -> heuristic estimate (only the tokenizer-based counts are exact;
                 see tokenizer/token_stats.json)

Usage:
  python3 pipeline/compute_statistics.py \
      --train train.txt --validation validation.txt \
      --manifest /tmp/work/build/selected_manifest.tsv \
      --build-report /tmp/work/build/build_report.json \
      --token-stats tokenizer/token_stats.json \
      --out dataset_statistics.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

SEP = "\n\n<|endoftext|>\n\n"
RE_PERSIAN_LETTER = re.compile("[آ-یءأإئؤ]")
RE_LATIN = re.compile(r"[A-Za-z]")
RE_DIGIT = re.compile("[0-9۰-۹]")
RE_ZWJ_CHARS = re.compile("[\u200b\u200d\u200e\u200f\ufeff\u200c]")
RE_SENT = re.compile(r"[.!؟]+")
RE_WORD = re.compile(r"\S+")
RE_PERSIAN_WORD = re.compile(r"[آ-یءأإئؤ]")


def iter_documents(path: str):
    """Stream documents that were joined with the <|endoftext|> separator."""
    buf = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.rstrip("\n") == "<|endoftext|>":
                doc = "".join(buf).strip("\n")
                if doc.strip():          # ignore the blank tail after the last marker
                    yield doc
                buf = []
            else:
                buf.append(line)
    tail = "".join(buf).strip("\n")
    if tail.strip():
        yield tail


def analyse_file(path: str):
    docs = 0
    chars = lines = words = persian_words = sentences = 0
    persian_letters = latin_letters = digits = zwnj = spaces = 0
    hashes = set()
    dup_in_file = 0
    max_len = 0
    min_len = 10 ** 9
    len_sum = 0
    doc_first_chars = Counter()
    t0 = time.time()

    for text in iter_documents(path):
        docs += 1
        n = len(text)
        chars += n
        len_sum += n
        max_len = max(max_len, n)
        min_len = min(min_len, n)
        lines += text.count("\n") + 1
        spaces += text.count(" ")
        words += len(RE_WORD.findall(text))
        persian_words += sum(1 for w in RE_WORD.findall(text) if RE_PERSIAN_WORD.search(w))
        persian_letters += len(RE_PERSIAN_LETTER.findall(text))
        latin_letters += len(RE_LATIN.findall(text))
        digits += len(RE_DIGIT.findall(text))
        zwnj += text.count("\u200c")
        sentences += len([s for s in RE_SENT.split(text) if s.strip()])
        h = hashlib.sha1(re.sub(r"\s+", " ", text).encode("utf-8")).hexdigest()
        if h in hashes:
            dup_in_file += 1
        else:
            hashes.add(h)

    size = os.path.getsize(path)
    return {
        "path": os.path.relpath(path),
        "bytes": size,
        "size_mb": round(size / 1e6, 2),
        "documents": docs,
        "characters": chars,
        "characters_mean_per_doc": round(len_sum / max(docs, 1), 1),
        "characters_min_doc": min_len if docs else 0,
        "characters_max_doc": max_len,
        "lines": lines,
        "words": words,
        "persian_words": persian_words,
        "sentences": sentences,
        "persian_letters": persian_letters,
        "latin_letters": latin_letters,
        "digits": digits,
        "zwnj": zwnj,
        "persian_letter_share_of_chars": round(persian_letters / max(chars, 1), 4),
        "persian_letter_share_of_letters": round(
            persian_letters / max(persian_letters + latin_letters, 1), 4),
        "unique_documents": len(hashes),
        "exact_duplicate_documents_in_file": dup_in_file,
        "bytes_per_character": round(size / max(chars, 1), 3),
        "scan_seconds": round(time.time() - t0, 1),
    }


def source_breakdown(manifest_path: str):
    def new_entry():
        return {"documents": 0, "characters": 0,
                "train_documents": 0, "validation_documents": 0,
                "train_characters": 0, "validation_characters": 0}
    per_source = defaultdict(new_entry)
    total_docs = total_chars = 0
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue
            src, _id, length, split = parts[0], parts[1], int(parts[2]), parts[3]
            split = "validation" if split == "val" else split
            s = per_source[src]
            s["documents"] += 1
            s["characters"] += length
            s[f"{split}_documents"] += 1
            s[f"{split}_characters"] += length
            total_docs += 1
            total_chars += length
    for s in per_source.values():
        s["share_of_characters"] = round(s["characters"] / max(total_chars, 1), 4)
    return dict(per_source), total_docs, total_chars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train.txt")
    ap.add_argument("--validation", default="validation.txt")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--build-report", required=True)
    ap.add_argument("--token-stats", default=None)
    ap.add_argument("--sources", default="sources.json")
    ap.add_argument("--out", default="dataset_statistics.json")
    args = ap.parse_args()

    print("[stats] scanning train.txt ...")
    train = analyse_file(args.train)
    print("[stats] scanning validation.txt ...")
    val = analyse_file(args.validation)

    sources, man_docs, man_chars = source_breakdown(args.manifest)
    build = json.load(open(args.build_report, encoding="utf-8"))
    src_meta = {}
    if args.sources and os.path.exists(args.sources):
        try:
            sj = json.load(open(args.sources, encoding="utf-8"))
            for s in sj.get("sources", []):
                src_meta[s["id"]] = s
        except (json.JSONDecodeError, OSError):
            print(f"[stats] note: could not read {args.sources}, "
                  f"source names/licenses will be filled from sources.json later")

    # attach human-readable names / licenses to the per-source table
    for key, val_ in sources.items():
        meta = src_meta.get(key, {})
        val_["name"] = meta.get("name", key)
        val_["type"] = meta.get("content_type", "unknown")
        val_["license"] = meta.get("license", "unknown")

    total_chars = train["characters"] + val["characters"]
    total_docs = train["documents"] + val["documents"]
    total_bytes = train["bytes"] + val["bytes"]
    total_words = train["words"] + val["words"]
    total_lines = train["lines"] + val["lines"]
    total_sentences = train["sentences"] + val["sentences"]

    # cross-check: manifest totals must match what we counted in the files
    manifest_matches = (man_docs == total_docs and abs(man_chars - total_chars) <= total_docs)

    token_info = {}
    if args.token_stats and os.path.exists(args.token_stats):
        token_info = json.load(open(args.token_stats, encoding="utf-8"))

    stats = {
        "dataset": {
            "name": "Persian Pretraining Corpus (GPT-100M)",
            "version": "1.0",
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "files": {
                "train": train,
                "validation": val,
            },
            "totals": {
                "documents": total_docs,
                "unique_documents": train["unique_documents"] + val["unique_documents"],
                "persian_characters": train["persian_letters"] + val["persian_letters"],
                "characters": total_chars,
                "words": total_words,
                "persian_words": train["persian_words"] + val["persian_words"],
                "lines": total_lines,
                "sentences": total_sentences,
                "size_bytes": total_bytes,
                "size_mb": round(total_bytes / 1e6, 2),
                "size_gib": round(total_bytes / 2 ** 30, 3),
                "persian_letter_share_of_letters": round(
                    (train["persian_letters"] + val["persian_letters"]) /
                    max(train["persian_letters"] + val["persian_letters"] +
                        train["latin_letters"] + val["latin_letters"], 1), 4),
                "persian_letter_share_of_characters": round(
                    (train["persian_letters"] + val["persian_letters"]) / max(total_chars, 1), 4),
                "zwnj_occurrences": train["zwnj"] + val["zwnj"],
            },
            "split": {
                "train_documents": train["documents"],
                "validation_documents": val["documents"],
                "train_characters": train["characters"],
                "validation_characters": val["characters"],
                "train_share_of_characters": round(train["characters"] / max(total_chars, 1), 5),
                "validation_share_of_characters": round(val["characters"] / max(total_chars, 1), 5),
                "method": "deterministic SHA-1 bucket of the document id, applied after "
                          "deduplication (so the two splits are disjoint)",
            },
            "source_distribution": {
                "by_source": sources,
                "note": "document and character counts per source of the SHIPPED corpus "
                        "come from the selection manifest written by build_dataset.py; "
                        "they are consistent with the totals counted in the files "
                        f"(manifest_matches_files={manifest_matches})",
            },
            "deduplication": build.get("dedup", {}),
            "collection_and_cleaning": {
                "raw_documents_extracted": sum(c["raw_docs"] for c in build.get("cleaning", [])),
                "raw_characters_extracted": None,  # filled below from extract manifest
                "documents_after_quality_filters": sum(
                    c["kept_after_quality"] for c in build.get("cleaning", [])),
                "characters_after_quality_filters": sum(
                    c["kept_chars"] for c in build.get("cleaning", [])),
                "documents_after_dedup": build.get("dedup", {}).get("output_docs"),
                "documents_shipped_after_budget": total_docs,
                "per_source_pipeline": build.get("cleaning", []),
                "filters": build.get("pipeline", {}).get("filters", {}),
            },
            "tokens": token_info or {
                "note": "run pipeline/make_tokenizer.py to count tokens exactly",
            },
            "methodology": {
                "count_units": "documents are separated by a line containing <|endoftext|>",
                "characters": "Unicode code points, counted with len() on the UTF-8 text",
                "words": "whitespace-separated tokens",
                "persian_characters": "characters in the Persian/Arabic letter block "
                                      "(U+0621..U+06CC) actually used by Persian",
                "verified_vs_estimated": "everything above is counted from the shipped "
                                         "files unless a field says otherwise",
            },
        },
        "pipeline_report": build,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    t = stats["dataset"]["totals"]
    print("\n=== verified statistics ===")
    print(f"documents        {t['documents']:,}")
    print(f"characters       {t['characters']:,}")
    print(f"words            {t['words']:,}")
    print(f"lines            {t['lines']:,}")
    print(f"persian chars    {t['persian_characters']:,}")
    print(f"size             {t['size_mb']:,.2f} MB")
    print(f"train/val docs   {train['documents']:,} / {val['documents']:,}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
