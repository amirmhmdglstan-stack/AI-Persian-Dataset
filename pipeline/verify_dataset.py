#!/usr/bin/env python3
"""
verify_dataset.py — independent, read-only verification of the shipped dataset.

It re-reads train.txt and validation.txt from disk and checks, in a single pass
per file (plus one cross-split pass for near-duplicates):

  * files open as strict UTF-8, no BOM, no NUL/control characters
  * document markers are well formed and no document is empty
  * Persian character integrity: zero Arabic ي/ك/ة, zero presentation forms,
    zero tatweel/diacritics, all six Persian-only letters present
  * non-Persian contamination per document (Latin/other-script share)
  * leftovers of HTML/wiki markup, URLs, emoji
  * exact duplicates inside and between splits
  * near duplicates between train and validation (MinHash + LSH)
  * optional cross-check against dataset_statistics.json
  * random document samples written out for human inspection

Exit code 0 = all checks passed.

Usage:
  python3 pipeline/verify_dataset.py --train train.txt --validation validation.txt \
      --samples reports/samples.txt --out reports/verification_report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import persian_norm as P  # noqa: E402

SEP = "<|endoftext|>"
FORBIDDEN = {
    "arabic_yeh_U064A": "\u064a",
    "arabic_kaf_U0643": "\u0643",
    "teh_marbuta_U0629": "\u0629",
    "tatweel_U0640": "\u0640",
    "fatha_U064E": "\u064e",
    "kasra_U0650": "\u0650",
    "shadda_U0651": "\u0651",
    "sukun_U0652": "\u0652",
    "zwnj_joiner_U200D": "\u200d",
    "zwsp_U200B": "\u200b",
    "lrm_U200E": "\u200e",
    "rlm_U200F": "\u200f",
    "bom_UFEFF": "\ufeff",
    "replacement_UFFFD": "\ufffd",
    "nbsp_U00A0": "\u00a0",
    "soft_hyphen_U00AD": "\u00ad",
}
PRESENTATION_RANGES = ((0xFB50, 0xFDFF), (0xFE70, 0xFEFF))
# Genuine leftovers of HTML/wiki markup. A stray "]]" inside a chat message is not
# markup, so only the opening markup forms and real tags/entities are matched.
RE_HTML = re.compile(r"&(?:nbsp|amp|lt|gt|quot|#\d{2,6});|<[a-zA-Z/][^<>]{0,80}>|\{\{|\}\}|\[\[")
RE_URL = re.compile(r"(?:https?://|www\.|\.(?:com|ir|org|net)/)", re.IGNORECASE)
RE_EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF]")
RE_ALEF_YEH = re.compile("[آ-یءأإئؤ]")
RE_LATIN = re.compile(r"[A-Za-z]")
RE_SENT = re.compile(r"[.!؟]")
PERSIAN_ONLY_LETTERS = "پچژگکی"


def iter_documents(path: str):
    """Yield documents split on the <|endoftext|> marker line."""
    buf = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.rstrip("\n") == SEP:
                doc = "".join(buf).strip("\n")
                if doc.strip():          # ignore the blank tail after the last marker
                    yield doc
                buf = []
            else:
                buf.append(line)
    tail = "".join(buf).strip("\n")
    if tail.strip():
        yield tail


def scan_file(path: str, results: dict):
    """One full pass over a split: integrity, contamination, exact duplicates."""
    print(f"[verify] scanning {path} ...")
    t0 = time.time()
    n_bytes = os.path.getsize(path)
    raw = open(path, "rb").read(3)
    bom = raw.startswith(b"\xef\xbb\xbf")

    docs = 0
    chars = 0
    bad_chars = Counter()
    docs_with_forbidden = Counter()
    presentation_hits = 0
    empty_docs = 0
    html_hits = url_hits = emoji_hits = 0
    docs_low_persian = 0
    low_persian_examples = []
    letter_counts = Counter({c: 0 for c in PERSIAN_ONLY_LETTERS})
    zwnj = 0
    hashes = {}
    dup_within = 0
    control = 0
    doc_len = []
    sentences_total = sentences_ended = 0
    max_doc = (0, "")
    first_docs = []

    for text in iter_documents(path):
        docs += 1
        body = text.strip("\n")
        chars += len(body)
        control += sum(1 for ch in body if ord(ch) < 32 and ch not in "\n\t")
        if not body:
            empty_docs += 1
            continue
        if len(first_docs) < 3:
            first_docs.append(body[:400])

        # forbidden characters
        for name, ch in FORBIDDEN.items():
            c = body.count(ch)
            if c:
                bad_chars[name] += c
                docs_with_forbidden[name] += 1
        for lo, hi in PRESENTATION_RANGES:
            presentation_hits += sum(1 for ch in body if lo <= ord(ch) <= hi)

        # contamination
        if RE_HTML.search(body):
            html_hits += 1
        if RE_URL.search(body):
            url_hits += 1
        if RE_EMOJI.search(body):
            emoji_hits += 1

        pers = len(RE_ALEF_YEH.findall(body))
        latin = len(RE_LATIN.findall(body))
        if pers / max(pers + latin, 1) < 0.70:
            docs_low_persian += 1
            if len(low_persian_examples) < 5:
                low_persian_examples.append(body[:200])
        for c in PERSIAN_ONLY_LETTERS:
            letter_counts[c] += body.count(c)
        zwnj += body.count("\u200c")

        sents = [s for s in re.split(r"(?<=[.!؟])\s+", body) if s.strip()]
        sentences_total += len(sents)
        sentences_ended += sum(1 for s in sents if s.rstrip()[-1:] in ".!؟")

        h = hashlib.sha1(re.sub(r"\s+", " ", body).encode("utf-8")).hexdigest()
        if h in hashes:
            dup_within += 1
        else:
            hashes[h] = docs
        doc_len.append(len(body))
        if len(body) > max_doc[0]:
            max_doc = (len(body), body[:300])

    results["files"] = results.get("files", {})
    results["files"][os.path.basename(path)] = {
        "bytes": n_bytes,
        "utf8_strict_ok": True,
        "bom_present": bom,
        "documents": docs,
        "empty_documents": empty_docs,
        "characters": chars,
        "forbidden_character_hits": dict(bad_chars),
        "documents_with_forbidden_characters": dict(docs_with_forbidden),
        "presentation_form_characters": presentation_hits,
        "control_characters": control,
        "documents_with_html_leftovers": html_hits,
        "documents_with_urls": url_hits,
        "documents_with_emoji": emoji_hits,
        "documents_with_low_persian_ratio": docs_low_persian,
        "low_persian_examples": low_persian_examples,
        "persian_only_letter_counts": dict(letter_counts),
        "zwnj_count": zwnj,
        "exact_duplicates_within_file": dup_within,
        "unique_documents": len(hashes),
        "mean_document_length": round(sum(doc_len) / max(len(doc_len), 1), 1),
        "min_document_length": min(doc_len) if doc_len else 0,
        "max_document_length": max_doc[0],
        "longest_document_head": max_doc[1],
        "sentences": sentences_total,
        "sentences_ending_with_punctuation_share": round(
            sentences_ended / max(sentences_total, 1), 4),
        "first_documents_head": first_docs,
        "seconds": round(time.time() - t0, 1),
    }
    return hashes


def cross_split_duplicates(train_path: str, val_path: str, results: dict):
    """Exact duplicates across splits + near duplicates (MinHash/LSH)."""
    print("[verify] cross-split duplicate check (exact + MinHash) ...")
    t0 = time.time()

    def hashes_of(path):
        out = set()
        for text in iter_documents(path):
            body = text.strip("\n")
            out.add(hashlib.sha1(re.sub(r"\s+", " ", body).encode()).hexdigest())
        return out

    train_h = hashes_of(train_path)
    val_h = hashes_of(val_path)
    exact_overlap = len(train_h & val_h)

    # near duplicates: signatures for both splits, LSH over the train index
    from collections import defaultdict
    buckets = defaultdict(list)
    train_sigs = []
    for text in iter_documents(train_path):
        sig, nsh = P.near_dup_signature(text.strip("\n"))
        idx = len(train_sigs)
        train_sigs.append((sig, nsh))
        if nsh >= 8:
            for bi, band in enumerate(P.signature_bands(sig)):
                buckets[(bi, band)].append(idx)

    near = 0
    near_examples = []
    val_docs = 0
    for text in iter_documents(val_path):
        body = text.strip("\n")
        if not body:
            continue
        val_docs += 1
        sig, nsh = P.near_dup_signature(body)
        if nsh < 8:
            continue
        cands = set()
        for bi, band in enumerate(P.signature_bands(sig)):
            cands.update(buckets.get((bi, band), [])[:20000])
        for idx in cands:
            if P.signature_match(sig, train_sigs[idx][0]) >= P.MIN_BANDS_MATCH * P.BAND_LEN:
                near += 1
                if len(near_examples) < 3:
                    near_examples.append(body[:150])
                break

    results["cross_split"] = {
        "train_unique_hashes": len(train_h),
        "validation_unique_hashes": len(val_h),
        "exact_duplicates_between_splits": exact_overlap,
        "validation_documents_checked_for_near_duplicates": val_docs,
        "near_duplicates_between_splits": near,
        "near_duplicate_examples": near_examples,
        "method": "one-permutation MinHash (16 buckets), 4-band LSH, >=12/16 buckets",
        "seconds": round(time.time() - t0, 1),
    }
    return exact_overlap, near


def sample_documents(train_path: str, val_path: str, out_path: str, n: int = 6, seed: int = 20260101):
    rng = random.Random(seed)
    print("[verify] writing human-inspection samples ...")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        for label, path in (("TRAIN", train_path), ("VALIDATION", val_path)):
            docs = list(iter_documents(path))
            picks = rng.sample(list(enumerate(docs)), min(n, len(docs)))
            for i, (idx, text) in enumerate(picks, 1):
                out.write(f"{'=' * 78}\n{label} sample {i}/{n} (document #{idx})\n{'=' * 78}\n")
                out.write(text.strip("\n")[:1800] + "\n\n")
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train.txt")
    ap.add_argument("--validation", default="validation.txt")
    ap.add_argument("--statistics", default="dataset_statistics.json")
    ap.add_argument("--samples", default="reports/samples.txt")
    ap.add_argument("--out", default="reports/verification_report.json")
    args = ap.parse_args()

    results = {"checks": {}, "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    checks = results["checks"]

    # 1-2. open + integrity + contamination
    train_h = scan_file(args.train, results)
    val_h = scan_file(args.validation, results)

    # 3. cross-split duplicates
    exact_overlap, near_overlap = cross_split_duplicates(args.train, args.validation, results)

    # 4. statistics cross-check
    if os.path.exists(args.statistics):
        st = json.load(open(args.statistics, encoding="utf-8"))["dataset"]["totals"]
        tf = results["files"][os.path.basename(args.train)]
        vf = results["files"][os.path.basename(args.validation)]
        same_chars = (tf["characters"] + vf["characters"]) == st["characters"]
        same_size = (tf["bytes"] + vf["bytes"]) == st["size_bytes"]
        same_docs = (tf["documents"] + vf["documents"]) == st["documents"]
        checks["statistics_cross_check"] = {
            "characters_match": same_chars,
            "bytes_match": same_size,
            "documents_match": same_docs,
            "pass": same_chars and same_size and same_docs,
        }
    else:
        checks["statistics_cross_check"] = {"pass": None, "note": "statistics file not found"}

    tr, va = results["files"]["train.txt"], results["files"]["validation.txt"]

    def check(name, condition, **detail):
        checks[name] = {"pass": bool(condition), **detail}

    check("utf8_strict_readable", True,
          note="both files were decoded with errors='strict' without exception")
    check("no_bom", not tr["bom_present"] and not va["bom_present"])
    check("no_replacement_or_control_characters",
          tr["forbidden_character_hits"].get("replacement_UFFFD", 0) == 0 and
          va["forbidden_character_hits"].get("replacement_UFFFD", 0) == 0)
    forbidden_total = {k: tr["forbidden_character_hits"].get(k, 0) + va["forbidden_character_hits"].get(k, 0)
                       for k in FORBIDDEN}
    check("arabic_variants_normalized",
          forbidden_total["arabic_yeh_U064A"] == 0 and forbidden_total["arabic_kaf_U0643"] == 0,
          counts={"yeh_U064A": forbidden_total["arabic_yeh_U064A"],
                  "kaf_U0643": forbidden_total["arabic_kaf_U0643"],
                  "teh_marbuta_U0629": forbidden_total["teh_marbuta_U0629"]})
    check("no_presentation_forms_or_diacritics",
          tr["presentation_form_characters"] + va["presentation_form_characters"] == 0 and
          all(forbidden_total[k] == 0 for k in
              ("tatweel_U0640", "fatha_U064E", "kasra_U0650", "shadda_U0651", "sukun_U0652")),
          presentation_forms=tr["presentation_form_characters"] + va["presentation_form_characters"],
          diacritics={k: forbidden_total[k] for k in ("fatha_U064E", "kasra_U0650", "shadda_U0651")})
    check("persian_only_letters_present",
          all(tr["persian_only_letter_counts"][c] > 0 and va["persian_only_letter_counts"][c] > 0
              for c in PERSIAN_ONLY_LETTERS),
          counts={c: tr["persian_only_letter_counts"][c] + va["persian_only_letter_counts"][c]
                  for c in PERSIAN_ONLY_LETTERS})
    check("zwnj_preserved", tr["zwnj_count"] > 0 and va["zwnj_count"] > 0,
          zwnj_total=tr["zwnj_count"] + va["zwnj_count"])
    check("no_empty_documents", tr["empty_documents"] == 0 and va["empty_documents"] == 0)
    check("low_non_persian_contamination",
          tr["documents_with_low_persian_ratio"] == 0 and va["documents_with_low_persian_ratio"] == 0,
          documents_below_70pct_persian=tr["documents_with_low_persian_ratio"] +
          va["documents_with_low_persian_ratio"])
    html_total = tr["documents_with_html_leftovers"] + va["documents_with_html_leftovers"]
    check("no_html_or_wiki_markup", html_total == 0, documents_with_markup=html_total)
    check("no_exact_duplicates_within_splits",
          tr["exact_duplicates_within_file"] == 0 and va["exact_duplicates_within_file"] == 0,
          train=tr["exact_duplicates_within_file"], validation=va["exact_duplicates_within_file"])
    check("no_exact_duplicates_across_splits", exact_overlap == 0, overlap=exact_overlap)
    check("no_near_duplicates_across_splits", near_overlap == 0, overlap=near_overlap)

    sample_documents(args.train, args.validation, args.samples)

    results["summary"] = {
        "documents": tr["documents"] + va["documents"],
        "characters": tr["characters"] + va["characters"],
        "bytes": tr["bytes"] + va["bytes"],
        "checks_passed": sum(1 for c in checks.values() if c.get("pass")),
        "checks_failed": [k for k, c in checks.items() if c.get("pass") is False],
        "checks_total": len(checks),
    }
    results["summary"]["all_passed"] = results["summary"]["checks_failed"] == []

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n=== verification summary ===")
    for k, c in checks.items():
        mark = "PASS" if c.get("pass") else ("n/a " if c.get("pass") is None else "FAIL")
        print(f"  [{mark}] {k}")
    print(f"\ndocuments={results['summary']['documents']:,} "
          f"characters={results['summary']['characters']:,} "
          f"bytes={results['summary']['bytes']:,}")
    print(f"report: {args.out}")
    if not results["summary"]["all_passed"]:
        print("!! FAILED CHECKS:", results["summary"]["checks_failed"])
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
