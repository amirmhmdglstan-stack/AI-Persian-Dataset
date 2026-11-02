#!/usr/bin/env python3
"""
make_sources_json.py — emit sources.json: every upstream corpus used here, with
its origin, license, collection facts and the exact number of documents/chars it
contributed to the shipped dataset.

Numbers come from the build report (+ the extraction manifest) so they are
measured, not typed by hand.

Usage:
  python3 pipeline/make_sources_json.py \
      --build-report /tmp/work/build/build_report.json \
      --extract-manifest /tmp/work/extracted/_extract_manifest.json \
      --out sources.json
"""
from __future__ import annotations

import argparse
import json
import time

COLLECTED_AT = "2026-11-01"

# Static, human-verified metadata for every upstream corpus.
METADATA = {
    "wikipedia_fa": {
        "name": "Persian Wikipedia — cleaned article dump (fa.wikipedia.org)",
        "name_fa": "ویکی‌پدیای فارسی — نسخه پاک‌سازی‌شده",
        "content_type": "encyclopedia / factual articles",
        "topics": ["science", "history", "geography", "culture", "technology",
                   "biography", "society", "religion", "sport", "art"],
        "url": "https://github.com/hsmirzaie/fawiki",
        "upstream": "fa.wikipedia.org dump dated 2021-02-01 (865k articles)",
        "license": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "license_notes": "Text is Wikipedia content, licensed CC BY-SA 4.0. The GitHub "
                         "repository only redistributes a cleaned dump and adds no "
                         "license of its own; attribution must be given to Wikipedia "
                         "and contributors, and share-alike applies to derived data.",
        "copyright_holder": "Wikimedia Foundation / Wikipedia contributors",
        "access_method": "git clone of a public repository (no live crawling)",
        "commercial_use": "allowed with attribution (share-alike)",
        "language_variety": "standard Iranian Persian, encyclopedic register",
    },
    "parsinlu_qa": {
        "name": "ParsiNLU — reading comprehension question/answer pairs",
        "name_fa": "پارسی‌نلو — پرسش و پاسخ درک مطلب",
        "content_type": "questions and answers",
        "topics": ["news", "general knowledge"],
        "url": "https://github.com/persiannlp/parsinlu",
        "upstream": "ParsiNLU (LREC 2020 / TACL) release",
        "license": "CC BY-NC-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "license_notes": "Non-commercial. Questions were written by native-speaker "
                         "annotators; passages come from Persian news sites and remain "
                         "the property of their publishers.",
        "copyright_holder": "ParsiNLU authors; passages © original publishers",
        "access_method": "git clone of a public repository",
        "commercial_use": "not allowed (NC)",
        "language_variety": "standard Persian, concise questions",
    },
    "parsinlu_mc": {
        "name": "ParsiNLU — multiple-choice questions (educational)",
        "name_fa": "پارسی‌نلو — پرسش‌های چندگزینه‌ای",
        "content_type": "educational questions and answers",
        "topics": ["math and logic", "science", "general knowledge"],
        "url": "https://github.com/persiannlp/parsinlu",
        "upstream": "ParsiNLU multiple-choice split",
        "license": "CC BY-NC-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "license_notes": "Non-commercial. Questions originate from Persian educational "
                         "blogs and Q&A sites; answer options were added by annotators.",
        "copyright_holder": "ParsiNLU authors",
        "access_method": "git clone of a public repository",
        "commercial_use": "not allowed (NC)",
        "language_variety": "standard Persian, school-level register",
    },
    "parsinlu_entailment": {
        "name": "ParsiNLU / FarsTail — textual entailment pairs",
        "name_fa": "پارسی‌نلو و فارس‌تیل — زوج‌های استلزام متنی",
        "content_type": "questions and answers / reasoning pairs",
        "topics": ["science", "news", "daily life", "general knowledge"],
        "url": "https://github.com/persiannlp/parsinlu",
        "upstream": "ParsiNLU merge of the FarsTail and ParsiNLU entailment sets "
                    "(the machine-translated test split is excluded)",
        "license": "CC BY-NC-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "license_notes": "Non-commercial.",
        "copyright_holder": "ParsiNLU and FarsTail authors",
        "access_method": "git clone of a public repository",
        "commercial_use": "not allowed (NC)",
        "language_variety": "standard Persian",
    },
    "parsinlu_rc_passage": {
        "name": "ParsiNLU — reading-comprehension passages (news prose)",
        "name_fa": "پارسی‌نلو — متن‌های درک مطلب",
        "content_type": "news and general articles",
        "topics": ["news", "politics", "society", "international"],
        "url": "https://github.com/persiannlp/parsinlu",
        "upstream": "passages underlying the ParsiNLU reading-comprehension split",
        "license": "CC BY-NC-SA 4.0 (annotations); passages © original publishers",
        "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "license_notes": "Passages were taken from Persian news websites and are "
                         "reproduced here only for research use.",
        "copyright_holder": "original news publishers",
        "access_method": "git clone of a public repository",
        "commercial_use": "not allowed (NC)",
        "language_variety": "standard Persian, journalistic register",
    },
    "peqa_dialogue": {
        "name": "PeQa — Persian chat/dialogue pairs",
        "name_fa": "پکا — جفت‌های گفتگوی فارسی",
        "content_type": "everyday conversations and dialogue",
        "topics": ["daily life", "religion", "sport", "politics", "humour"],
        "url": "https://github.com/ZahraArshia/PeQa",
        "upstream": "public Telegram channel conversations collected by the PeQa authors",
        "license": "MIT",
        "license_url": "https://opensource.org/licenses/MIT",
        "license_notes": "Repository content is MIT. The conversations are public "
                         "messages; user handles were stripped before release upstream. "
                         "Informal (colloquial) Persian — a small share of this corpus.",
        "copyright_holder": "PeQa authors / original message authors",
        "access_method": "git clone of a public repository",
        "commercial_use": "allowed (MIT) for the repository content",
        "language_variety": "colloquial Tehrani Persian, chat register",
    },
    "ganjoor_classics": {
        "name": "Ganjoor — classical Persian literature (divans)",
        "name_fa": "گنجور — ادبیات کلاسیک فارسی",
        "content_type": "literature and poetry",
        "topics": ["poetry", "mysticism", "ethics", "epic", "storytelling"],
        "url": "https://github.com/ganjoor/ganjoor-tex",
        "upstream": "Ganjoor classical texts (Rumi, Hafez, Saadi, Attar, Ferdowsi, "
                    "Khayyam, Nezami, Sanaee, Babaafzal, Baba Taher, Shah Nematollah ...)",
        "license": "GPL-3.0 (repository); the underlying works are public domain",
        "license_url": "https://www.gnu.org/licenses/gpl-3.0.html",
        "license_notes": "The works themselves are centuries old and in the public domain. "
                         "The repository's typesetting/data files are GPL-3.0.",
        "copyright_holder": "public domain (works) / Ganjoor (compilation)",
        "access_method": "git clone of a public repository",
        "commercial_use": "allowed for the public-domain works",
        "language_variety": "classical Persian verse",
    },
    "persian_poetry": {
        "name": "Chronological Persian Poetry Dataset",
        "name_fa": "مجموعه شعر فارسی بر پایه سده",
        "content_type": "literature and poetry",
        "topics": ["poetry", "classical literature"],
        "url": "https://github.com/aghasemi/ChronologicalPersianPoetryDataset",
        "upstream": "classical Persian poems labelled by poet and century",
        "license": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "license_notes": "Underlying poems are public domain; the compilation is CC BY-SA 4.0.",
        "copyright_holder": "public domain (works) / dataset authors (compilation)",
        "access_method": "git clone of a public repository",
        "commercial_use": "allowed with attribution (share-alike)",
        "language_variety": "classical Persian verse",
    },
    "persian_news": {
        "name": "Persian News Dataset (5 Iranian news agencies)",
        "name_fa": "دیتاست اخبار فارسی",
        "content_type": "news and general articles",
        "topics": ["politics", "sport", "economy", "society", "culture", "international"],
        "url": "https://github.com/milad-4274/persian_news",
        "upstream": "13.5k news items (title + lead) from Iranian agencies",
        "license": "none declared",
        "license_url": "",
        "license_notes": "The repository ships no license file, and the article text is "
                         "copyright of the original news agencies. Included for "
                         "non-commercial research/education only; remove this source "
                         "(see README) if you need a purely permissive corpus.",
        "copyright_holder": "original news agencies",
        "access_method": "git clone of a public repository",
        "commercial_use": "not granted",
        "language_variety": "standard Persian, journalistic register",
    },
    "pn_summary_news": {
        "name": "pn-summary — Iranian news articles with human summaries",
        "name_fa": "پی‌ان‌سامری — اخبار فارسی",
        "content_type": "news and general articles",
        "topics": ["news", "economy", "energy", "city", "sport", "culture"],
        "url": "https://github.com/hooshvare/pn-summary",
        "upstream": "news articles published by Iranian agencies (ISNA, IRNA, SHANA, "
                    "IMNA, YJC ...) with human-written summaries",
        "license": "MIT (repository); article text © the publishers",
        "license_url": "https://opensource.org/licenses/MIT",
        "license_notes": "The dataset packaging/code is MIT. The news articles remain "
                         "the property of their publishers and are included for "
                         "non-commercial research only.",
        "copyright_holder": "original news publishers",
        "access_method": "git clone of a public repository",
        "commercial_use": "not granted for the article text",
        "language_variety": "standard Persian, journalistic register",
        "quality_outcome": "REJECTED by the quality gate: the upstream release has all "
                           "sentence punctuation stripped (0 of 11,185 articles contain "
                           "a period, question mark or exclamation mark), so the text is "
                           "unusable as sentence-level training data. It contributes 0 "
                           "documents to the shipped corpus; the source is documented "
                           "here for transparency.",
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-report", required=True)
    ap.add_argument("--extract-manifest", required=True)
    ap.add_argument("--out", default="sources.json")
    args = ap.parse_args()

    build = json.load(open(args.build_report, encoding="utf-8"))
    extract = json.load(open(args.extract_manifest, encoding="utf-8"))

    raw = {s["source"]: s for s in extract.get("sources", [])}
    clean = {s["source"]: s for s in build.get("cleaning", [])}
    sel = build.get("selection", {})

    sources = []
    for sid, meta in METADATA.items():
        entry = {
            "id": sid,
            **meta,
            "collected_at": COLLECTED_AT,
            "collection_method": "downloaded from a public GitHub repository; no live "
                                 "websites were crawled, no robots.txt rules were "
                                 "circumvented, no access control bypassed, and no "
                                 "personal data was collected",
            "counts": {
                "raw_documents_extracted": raw.get(sid, {}).get("raw_docs", 0),
                "raw_characters_extracted": raw.get(sid, {}).get("raw_chars", 0),
                "documents_after_quality_filter": clean.get(sid, {}).get("kept_after_quality", 0),
                "characters_after_quality_filter": clean.get(sid, {}).get("kept_chars", 0),
                "documents_in_dataset": sel.get(sid, {}).get("selected_docs", 0),
                "characters_in_dataset": sel.get(sid, {}).get("selected_chars", 0),
            },
            "cleaning_dropped_by_reason": clean.get(sid, {}).get("dropped_by_reason", {}),
        }
        if entry["counts"]["documents_in_dataset"] == 0:
            entry["status"] = "rejected_by_quality_gate"
        else:
            entry["status"] = "included"
        if "cap_share" in sel.get(sid, {}):
            entry["counts"]["sampling"] = sel[sid].get("sampling")
            entry["counts"]["budget_cap_share"] = sel[sid].get("cap_share")
        sources.append(entry)

    # aggregate pipeline counts
    raw_docs = sum(s["counts"]["raw_documents_extracted"] for s in sources)
    raw_chars = sum(s["counts"]["raw_characters_extracted"] for s in sources)
    clean_docs = sum(s["counts"]["documents_after_quality_filter"] for s in sources)
    clean_chars = sum(s["counts"]["characters_after_quality_filter"] for s in sources)
    shipped_docs = sum(s["counts"]["documents_in_dataset"] for s in sources)
    shipped_chars = sum(s["counts"]["characters_in_dataset"] for s in sources)

    out = {
        "dataset": "Persian Pretraining Corpus (GPT-100M)",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "collection_date": COLLECTED_AT,
        "collector": "autonomous data-collection agent (Arena.ai Agent Mode)",
        "language": "Persian (Farsi), standard Iranian Persian",
        "number_of_sources": len(sources),
        "access_scope": "Only public GitHub repositories were used. No website was "
                        "crawled; no login-gated, paywalled or private material is "
                        "included; no personal information is stored.",
        "aggregate": {
            "raw_documents_extracted": raw_docs,
            "raw_characters_extracted": raw_chars,
            "documents_after_quality_filter": clean_docs,
            "characters_after_quality_filter": clean_chars,
            "documents_in_dataset": shipped_docs,
            "characters_in_dataset": shipped_chars,
            "deduplication": build.get("dedup", {}),
        },
        "licensing_summary": {
            "permissive_or_share_alike": ["wikipedia_fa", "ganjoor_classics",
                                          "persian_poetry", "peqa_dialogue"],
            "non_commercial": ["parsinlu_qa", "parsinlu_mc", "parsinlu_entailment",
                               "parsinlu_rc_passage"],
            "publisher_copyright_research_only": ["persian_news", "pn_summary_news"],
            "note": "The corpus mixes licenses. It is released for research and "
                    "non-commercial use; commercial use requires removing the "
                    "non-commercial and publisher-copyright sources (see README.txt).",
        },
        "sources": sources,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"wrote {args.out}: {len(sources)} sources, "
          f"{shipped_docs:,} documents / {shipped_chars:,} chars shipped")


if __name__ == "__main__":
    main()
