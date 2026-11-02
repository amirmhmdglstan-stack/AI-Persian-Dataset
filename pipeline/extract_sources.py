#!/usr/bin/env python3
"""
extract_sources.py — turn each upstream repository into a uniform JSONL document
stream:  {"src": <str>, "id": <str>, "text": <str>, "meta": {...}}

Nothing is cleaned here beyond structural unpacking (joining title + body,
un-escaping a ZWNJ marker, dropping index pages). Character normalization and
quality filtering happen in build_dataset.py so that every source goes through
exactly the same gate.

Usage:
    python3 pipeline/extract_sources.py --workdir /tmp/work --out /tmp/work/extracted
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys

csv.field_size_limit(10 ** 9)

OUT_FIELDS = ("src", "id", "text", "meta")


class Writer:
    def __init__(self, outdir: str, src: str):
        self.src = src
        self.n = 0
        self.chars = 0
        os.makedirs(outdir, exist_ok=True)
        self.path = os.path.join(outdir, f"{src}.jsonl")
        self.fh = open(self.path, "w", encoding="utf-8")

    def add(self, doc_id: str, text: str, meta: dict | None = None):
        if not text:
            return
        self.fh.write(json.dumps(
            {"src": self.src, "id": str(doc_id), "text": text, "meta": meta or {}},
            ensure_ascii=False) + "\n")
        self.n += 1
        self.chars += len(text)

    def close(self):
        self.fh.close()
        print(f"  {self.src:<28} docs={self.n:>9,}  chars={self.chars:>12,}  -> {self.path}")
        return {"source": self.src, "raw_docs": self.n, "raw_chars": self.chars}


# --------------------------------------------------------------------------- #
# 1. Persian Wikipedia (hsmirzaie/fawiki: id, title, article CSV)
# --------------------------------------------------------------------------- #
def extract_fawiki(workdir: str, outdir: str):
    path = os.path.join(workdir, "fawiki.csv")
    w = Writer(outdir, "wikipedia_fa")
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        r = csv.reader(f)
        header = next(r)
        idx = {name: i for i, name in enumerate(header)}
        for row in r:
            if len(row) < 3:
                continue
            title = row[idx["title"]].strip()
            body = row[idx["article"]]
            w.add(f"fa-{row[idx['id']]}", f"{title}\n\n{body}", {"title": title})
    return [w.close()]


# --------------------------------------------------------------------------- #
# 2. ParsiNLU (persiannlp/parsinlu) — reading comprehension, multiple choice,
#    textual entailment.  Questions/answers are human-written/annotated.
# --------------------------------------------------------------------------- #
def extract_parsinlu(workdir: str, outdir: str):
    base = os.path.join(workdir, "parsinlu", "data")
    stats = []

    # --- reading comprehension -> passages (prose) + question/answer pairs ---
    w_pass = Writer(outdir, "parsinlu_rc_passage")
    w_qa = Writer(outdir, "parsinlu_qa")
    seen_q = set()
    for split in ("train", "eval", "test"):
        p = os.path.join(base, "reading_comprehension", f"{split}.jsonl")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                passage = (d.get("passage") or "").strip()
                question = (d.get("question") or "").strip()
                answers = d.get("answers") or []
                ans = ""
                if answers and isinstance(answers[0], (list, tuple)) and len(answers[0]) > 1:
                    ans = str(answers[0][1]).strip()
                if passage:
                    w_pass.add(f"rc-{split}-{i}", passage, {"url": d.get("url", "")})
                if question and ans:
                    key = question + "\x00" + ans
                    if key in seen_q:
                        continue
                    seen_q.add(key)
                    w_qa.add(f"rcqa-{split}-{i}",
                             f"پرسش: {question}\n\nپاسخ: {ans}",
                             {"url": d.get("url", "")})
    stats += [w_pass.close(), w_qa.close()]

    # --- multiple choice -> educational questions with options ---
    w_mc = Writer(outdir, "parsinlu_mc")
    seen = set()
    p = os.path.join(base, "multiple-choice", "train.jsonl")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                q = (d.get("question") or "").strip()
                cands = [str(c).strip() for c in (d.get("candidates") or [])]
                ans = str(d.get("answer", "")).strip()
                if not q or len(cands) < 2:
                    continue
                key = q + "\x00" + "|".join(cands)
                if key in seen:
                    continue
                seen.add(key)
                opts = "\n".join(f"{j + 1}) {c}" for j, c in enumerate(cands))
                w_mc.add(f"mc-{i}",
                         f"پرسش: {q}\n\nگزینه‌ها:\n{opts}\n\nپاسخ درست: {ans}",
                         {"category": d.get("category", "")})
    stats.append(w_mc.close())

    # --- textual entailment -> premise/hypothesis/label ---
    w_ent = Writer(outdir, "parsinlu_entailment")
    label_fa = {"c": "نتیجه می‌شود", "n": "خنثی", "e": "تناقض دارد"}
    # NB: test_translation.tsv is excluded on purpose: its pairs are translated,
    # i.e. not native Persian (see README "known limitations").
    for rel in ("merged_with_farstail/train.tsv", "merged_with_farstail/dev.tsv",
                "merged_with_farstail/test_natural.tsv", "merged_with_farstail/test_farstail.tsv"):
        p = os.path.join(base, "entailment", rel)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for i, row in enumerate(csv.reader(f, delimiter="\t")):
                if len(row) < 2:
                    continue
                # The merged files pack "premise<sep>hypothesis" into one column
                # followed by the label; older files use three columns.
                if "<sep>" in row[0] or len(row) == 2:
                    pair, lab = row[0], row[-1]
                    if "<sep>" not in pair:
                        continue
                    prem, hyp = pair.split("<sep>", 1)
                else:
                    prem, hyp, lab = row[0], row[1], row[2]
                prem, hyp = prem.strip(), hyp.strip()
                if not prem or not hyp:
                    continue
                w_ent.add(f"ent-{os.path.basename(p)}-{i}",
                          f"متن نخست: {prem}\n\nمتن دوم: {hyp}\n\nرابطه: {label_fa.get(lab, 'خنثی')}")
    stats.append(w_ent.close())
    return stats


# --------------------------------------------------------------------------- #
# 3. PeQa (ZahraArshia/PeQa) — public Persian chat pairs -> dialogue blocks
# --------------------------------------------------------------------------- #
def extract_peqa(workdir: str, outdir: str, block_chars: int = 1400):
    path = os.path.join(workdir, "PeQa", "PeQa-dataset", "cleaned-data", "formatted_data.txt")
    w = Writer(outdir, "peqa_dialogue")
    block, block_len, block_i, pair_i = [], 0, 0, 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            a, b = parts[0].strip(), parts[1].strip()
            if not a or not b:
                continue
            # PeQa joins words with "_" instead of a space; restore the space.
            a, b = a.replace("_", " "), b.replace("_", " ")
            block.append(f"کاربر: {a}\nکاربر: {b}")
            block_len += len(a) + len(b) + 14
            pair_i += 1
            if block_len >= block_chars:
                w.add(f"peqa-{block_i}", "\n".join(block), {"pairs": len(block)})
                block, block_len, block_i = [], 0, block_i + 1
    if block_len > 0:
        w.add(f"peqa-{block_i}", "\n".join(block))
    return [w.close()]


# --------------------------------------------------------------------------- #
# 4. Ganjoor classical literature (ganjoor/ganjoor-tex, txt/ tree)
#    Public-domain classical works (Rumi, Hafez, Saadi, Attar, ...).
# --------------------------------------------------------------------------- #
_RE_HEADER = re.compile(r"^(?:رباعی|غزل|قصیده|مثنوی|قطعه|شماره|بیت)\b.{0,60}$")


def extract_ganjoor(workdir: str, outdir: str, block_chars: int = 3200):
    root = os.path.join(workdir, "ganjoor-tex", "txt")
    w = Writer(outdir, "ganjoor_classics")
    files = sorted(glob.glob(os.path.join(root, "**", "*.txt"), recursive=True))
    buf, buf_len, block_i = [], 0, 0
    current_book = None
    for fp in files:
        rel = os.path.relpath(fp, root)
        book = os.path.dirname(rel)
        try:
            raw = open(fp, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        lines = [ln.strip() for ln in raw.split("\n")]
        # drop the "رباعی شماره ۱۲: <first verse>"-style catalogue header
        lines = [ln for ln in lines if not _RE_HEADER.match(ln)]
        body = "\n".join(ln for ln in lines if ln)
        if len(body) < 40:
            continue
        if buf and (book != current_book or buf_len + len(body) > block_chars):
            w.add(f"ganjoor-{block_i}", "\n\n".join(buf), {"book": current_book})
            buf, buf_len, block_i = [], 0, block_i + 1
        current_book = book
        buf.append(body)
        buf_len += len(body)
    if buf:
        w.add(f"ganjoor-{block_i}", "\n\n".join(buf), {"book": current_book})
    return [w.close()]


# --------------------------------------------------------------------------- #
# 5. Chronological Persian Poetry (aghasemi/ChronologicalPersianPoetryDataset)
# --------------------------------------------------------------------------- #
def extract_poetry(workdir: str, outdir: str, block_chars: int = 3200):
    path = os.path.join(workdir, "ChronologicalPersianPoetryDataset", "poems.tsv")
    w = Writer(outdir, "persian_poetry")
    buf, buf_len, block_i, poet_cur = [], 0, 0, None
    seen = set()
    with open(path, encoding="utf-8", errors="replace") as f:
        r = csv.reader(f, delimiter="\t")
        next(r, None)
        for row in r:
            if len(row) < 3:
                continue
            poet, century, poem = row[0].strip(), row[1].strip(), row[2].strip()
            h = hash(poem)
            if h in seen:                     # intra-file exact repeats
                continue
            seen.add(h)
            if not poem:
                continue
            if buf and (poet != poet_cur or buf_len + len(poem) > block_chars):
                w.add(f"poem-{block_i}", "\n\n".join(buf), {"poet": poet_cur})
                buf, buf_len, block_i = [], 0, block_i + 1
            poet_cur = poet
            buf.append(poem)
            buf_len += len(poem) + 2
    if buf:
        w.add(f"poem-{block_i}", "\n\n".join(buf), {"poet": poet_cur})
    return [w.close()]


# --------------------------------------------------------------------------- #
# 6. Persian news (milad-4274/persian_news: title + description)
# --------------------------------------------------------------------------- #
def extract_persian_news(workdir: str, outdir: str):
    path = os.path.join(workdir, "persian_news", "main_news.csv")
    w = Writer(outdir, "persian_news")
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        r = csv.reader(f)
        header = next(r)
        idx = {name: i for i, name in enumerate(header)}
        for row in r:
            if len(row) < 7:
                continue
            title = row[idx["title"]].strip()
            desc = row[idx["description"]].strip()
            site = row[idx["site"]].strip()
            link = row[idx["link"]].strip()
            if not desc:
                continue
            w.add(f"news-{link or title[:40]}", f"{title}\n\n{desc}",
                  {"site": site, "url": link})
    return [w.close()]


# --------------------------------------------------------------------------- #
# 7. pn-summary (hooshvare/pn-summary): Iranian news articles + human summaries
# --------------------------------------------------------------------------- #
_ZWNJ_MARK = re.compile(r"\[ZWNJ\]")


def extract_pn_summary(workdir: str, outdir: str):
    root = os.path.join(workdir, "pn-summary", "summarization")
    w = Writer(outdir, "pn_summary_news")
    seen_ids = set()
    files = sorted(glob.glob(os.path.join(root, "**", "*prediction*.csv"), recursive=True))
    for fp in files:
        with open(fp, newline="", encoding="utf-8", errors="replace") as f:
            r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            header = next(r, None)
            if not header or "article" not in header:
                continue
            idx = {name: i for i, name in enumerate(header)}
            for row in r:
                if len(row) != len(header):
                    continue
                did = row[idx["id"]].strip()
                if did in seen_ids:
                    continue
                seen_ids.add(did)
                title = row[idx["title"]].strip()
                art = _ZWNJ_MARK.sub("\u200c", row[idx["article"]]).strip()
                if len(art) < 120:
                    continue
                link = row[idx["link"]].strip() if "link" in idx else ""
                w.add(f"pn-{did}", f"{title}\n\n{art}", {"url": link})
    return [w.close()]


# --------------------------------------------------------------------------- #
EXTRACTORS = {
    "wikipedia_fa": extract_fawiki,
    "parsinlu": extract_parsinlu,
    "peqa_dialogue": extract_peqa,
    "ganjoor_classics": extract_ganjoor,
    "persian_poetry": extract_poetry,
    "persian_news": extract_persian_news,
    "pn_summary_news": extract_pn_summary,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True, help="directory containing the cloned upstream repos")
    ap.add_argument("--out", required=True, help="output directory for extracted JSONL")
    ap.add_argument("--only", nargs="*", default=None, help="subset of extractors to run")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    manifest, stats = {}, []
    for name, fn in EXTRACTORS.items():
        if args.only and name not in args.only:
            continue
        print(f"[extract] {name} ...")
        try:
            st = fn(args.workdir, args.out)
        except FileNotFoundError as e:
            print(f"  !! skipped ({e})")
            continue
        stats.extend(st)
        manifest[name] = st

    with open(os.path.join(args.out, "_extract_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"sources": stats}, f, ensure_ascii=False, indent=2)
    print("\nExtraction totals:")
    for s in stats:
        print(f"  {s['source']:<26} {s['raw_docs']:>9,} docs {s['raw_chars']:>13,} chars")


if __name__ == "__main__":
    sys.exit(main())
