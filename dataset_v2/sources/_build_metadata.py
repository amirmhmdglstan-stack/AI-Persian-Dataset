#!/usr/bin/env python3
"""Regenerate the V2 machine-readable source catalog and its CSV exports.

This file is the *authoritative* catalog generator.  ``source_catalog.csv`` and
``licensing_matrix.csv`` are derived from ``source_metadata.json``; do not edit
the CSVs by hand.  Run:

    python dataset_v2/sources/_build_metadata.py

The catalog records rights that apply to the ACTUAL TEXT, not merely to a
repository or a dataset compilation.  It is research documentation, not legal
advice.  See reports/licensing_report.md for the reasoning.
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))

# Local citation index used by the research/licensing reports.  Every id maps
# to an exact source URL; reports cite it as [id](url).  Fetched dataset cards
# and dump indexes are also linked directly in DIRECT_REFS below.
RESEARCH_REFS = {
    "1": "https://huggingface.co/datasets/SLPL/naab",
    "2": "https://www.researchgate.net/publication/363084493_naab_A_ready-to-use_plug-and-play_corpus_for_Farsi",
    "4": "https://nlpdataset.ir/farsi/raw_text_corpora.html",
    "5": "https://www.mozillafoundation.org/en/blog/common-voice-18-dataset-release/",
    "6": "https://github.com/common-voice/common-voice",
    "7": "https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use",
    "8": "https://dumps.wikimedia.org/legal.html",
    "9": "https://en.wikipedia.org/wiki/Template:PD-Iran",
    "10": "https://tatoeba.org/en/downloads",
    "11": "https://tatoeba.org/en/terms_of_use",
    "12": "https://en.wikisource.org/wiki/Wikisource:COPY",
    "13": "https://commoncrawl.org/terms-of-use",
    "14": "https://huggingface.co/datasets/Targoman/TLPC",
    "15": "https://www.kaggle.com/datasets/ehsankhani/hamshahri-corpus",
    "16": "https://huggingface.co/datasets/universitytehran/ESPOSITO",
    "17": "https://huggingface.co/datasets/MohammadJRanjbar/PersianPunc",
    "18": "https://www.bbc.co.uk/archiveservices/written-archives-centre/licensing-and-copyright",
    "19": "https://huggingface.co/datasets/SLPL/naab-raw",
    "20": "https://en.wikipedia.org/wiki/Persian_Wikipedia",
    "21": "https://www.gutenberg.org/ebooks/10315",
    "22": "https://huggingface.co/datasets/saeedzou/persianvox_all",
    "23": "https://huggingface.co/datasets/MahtaFetrat/Mana-TTS",
    "24": "https://huggingface.co/datasets/n9e6y/persian-dehkhoda-dictionary",
    "25": "https://www.harakenzo.com/en/service/iran/",
    "26": "https://groups.google.com/g/common-crawl/c/wZwjDoxEilw",
    "27": "https://www.researchgate.net/publication/398476043_Persian-Phi_Efficient_Cross-Lingual_Adaptation_of_Compact_LLMs_via_Curriculum_Learning",
    "28": "https://huggingface.co/datasets/shekar-ai/Shiraz",
}

DIRECT_REFS = {
    "wikimedia_fawiki_index": "https://dumps.wikimedia.org/fawiki/20261001/dumpstatus.json",
    "wikimedia_books_index": "https://dumps.wikimedia.org/fawikibooks/20261101/dumpstatus.json",
    "wikimedia_source_index": "https://dumps.wikimedia.org/fawikisource/20261101/dumpstatus.json",
    "wikimedia_news_index": "https://dumps.wikimedia.org/fawikinews/20261101/dumpstatus.json",
    "naab_card": "https://huggingface.co/datasets/SLPL/naab/raw/main/README.md",
    "shiraz_card": "https://huggingface.co/datasets/shekar-ai/Shiraz/raw/main/README.md",
    "shiraz_files": "https://huggingface.co/api/datasets/shekar-ai/Shiraz/tree/main/data",
    "naab_files": "https://huggingface.co/api/datasets/SLPL/naab/tree/main/data",
    "ganjoor_repo": "https://github.com/ganjoor/ganjoor-tex",
    "poetry_repo": "https://github.com/aghasemi/ChronologicalPersianPoetryDataset",
    "parsinlu_repo": "https://github.com/persiannlp/parsinlu",
    "peqa_repo": "https://github.com/ZahraArshia/PeQa",
    "fawiki_mirror_repo": "https://github.com/hsmirzaie/fawiki",
    "mirastext_repo": "https://github.com/miras-tech/MirasText",
    "persian_raw_text_repo": "https://github.com/persiannlp/persian-raw-text",
    "common_voice_repo": "https://github.com/common-voice/common-voice",
    "tatoeba_downloads": "https://tatoeba.org/en/downloads",
    "v1_sources_json": "https://github.com/amirmhmdglstan-stack/AI-Persian-Dataset/blob/arena/f01a4e1e-ai-persian-dataset/sources.json",
    "v1_readme": "https://github.com/amirmhmdglstan-stack/AI-Persian-Dataset/blob/arena/f01a4e1e-ai-persian-dataset/README.txt",
}

WIKI = "https://dumps.wikimedia.org"
CV_BASE = "https://raw.githubusercontent.com/common-voice/common-voice/ea5d6ee99d8135946343a1d6d0616f0b85b00860/server/data/fa/"
GANJOOR_SHA = "d1f89705d6ad8bff40a6221c03cec90d3bcc3a5e"
POETRY_SHA = "e58102757808e6e1cb33a63084f5f383d86ec499"
PARSINLU_SHA = "a3b08d0b93ca84e0017e2f490a2aa3ab36c79885"
FAWIKI_MIRROR_SHA = "2d2b27aa622430077b90a33dd788181a5703ba41"
PEQA_SHA = "e3f4a923e32e718b2afd040f9a6c11608e22b8c1"
PN_SUMMARY_SHA = "152c7ccc9afe8b47b6b59f608be634e10e88ae8b"
PERSIAN_NEWS_SHA = "a529ab74a95ad6bf964141d1468b86f7e30ebb7f"


def github_tarball(repo: str, sha: str) -> str:
    return f"https://codeload.github.com/{repo}/tar.gz/{sha}"


def file_entry(url: str, filename: str, fmt: str, extractor: str, *, size: int | None = None,
               sha1: str | None = None, options: dict | None = None, required: bool = True) -> dict:
    d = {
        "url": url,
        "filename": filename,
        "format": fmt,
        "extractor": extractor,
        "reported_size_bytes": size,
        "sha1": sha1,
        "required": required,
        "options": options or {},
    }
    return d


def source(**kw) -> dict:
    base = {
        "profile": "C",
        "decision": "quarantine",
        "download": False,
        "train": False,
        "format": "unknown",
        "approx_size": "unknown",
        "doc_count_estimate": None,
        "availability": "unknown",
        "update_date": "unknown",
        "license_compilation": "unknown",
        "license_content": "unknown",
        "content_rights_verified": False,
        "copyright_holder": "unknown",
        "attribution_required": "unknown",
        "share_alike": "unknown",
        "non_commercial": "unknown",
        "commercial_use": "unknown",
        "redistribution": "unknown",
        "privacy_risk": "unknown",
        "language_purity": "unknown",
        "lm_suitability": "no",
        "rights_verification": "unverified",
        "refs": [],
        "notes": "",
        "v1_filtered_chars": None,
        "v1_shipped_chars": None,
        "v1_shipped_docs": None,
        "files": [],
    }
    base.update(kw)
    if base["decision"] in ("include", "include_research"):
        base["download"] = True
        base["train"] = True
    return base


SOURCES = [
    # ---------------- PROFILE A: verified / permissive core ----------------
    source(
        id="wikipedia_fa_official_20261001",
        name="Persian Wikipedia official articles dump (2026-10-01)",
        profile="A", decision="include",
        domain="encyclopedia",
        description="Official Wikimedia current-revisions dump; primary modern factual Persian prose.",
        url=f"{WIKI}/fawiki/20261001/",
        format="MediaWiki XML bz2", approx_size="1.456 GB compressed",
        doc_count_estimate=1100753, availability="public HTTP, dated dump",
        update_date="2026-10-01",
        license_compilation="CC BY-SA 4.0 / GFDL",
        license_content="CC BY-SA 4.0 / GFDL",
        content_rights_verified=True,
        copyright_holder="Wikipedia contributors",
        attribution_required="yes (link to article/history or contributor list)",
        share_alike="yes for redistributed adaptations",
        non_commercial="no", commercial_use="yes with CC BY-SA conditions",
        redistribution="yes with attribution and share-alike",
        privacy_risk="low", language_purity="Persian, some technical Latin terms",
        lm_suitability="yes", rights_verification="official Wikimedia ToU and dump index",
        refs=["7", "8", "20", "wikimedia_fawiki_index"],
        notes="Use the dated 20261001 pages-articles file, not the mutable 'latest' symlink. CC BY-SA share-alike may affect redistributed derived corpus; this is not legal advice.",
        files=[file_entry(
            f"{WIKI}/fawiki/20261001/fawiki-20261001-pages-articles.xml.bz2",
            "fawiki-20261001-pages-articles.xml.bz2", "bz2", "mediawiki_xml_bz2",
            size=1456390218, sha1="9b49c6e5988cf28ff9b618574e997cabd0df57f4",
            options={"skip_redirects": True, "namespace": 0, "min_raw_chars": 200,
                     "include_title": True, "exclude_disambiguation": True})],
    ),
    source(
        id="fawikibooks_20261101",
        name="Persian Wikibooks official articles dump (2026-11-01)",
        profile="A", decision="include", domain="educational",
        description="Free textbooks/how-to prose; small but genuinely educational modern Persian.",
        url=f"{WIKI}/fawikibooks/20261101/",
        format="MediaWiki XML bz2", approx_size="7.09 MB compressed",
        availability="public HTTP, dated dump", update_date="2026-11-01",
        license_compilation="CC BY-SA 4.0 / GFDL", license_content="CC BY-SA 4.0 / GFDL",
        content_rights_verified=True, copyright_holder="Wikibooks contributors",
        attribution_required="yes", share_alike="yes", non_commercial="no",
        commercial_use="yes with conditions", redistribution="yes with conditions",
        privacy_risk="low", language_purity="Persian", lm_suitability="yes",
        rights_verification="official Wikimedia ToU and dump index",
        refs=["7", "8", "wikimedia_books_index"],
        notes="Small source; useful as explanatory/educational register, not as a volume filler.",
        files=[file_entry(
            f"{WIKI}/fawikibooks/20261101/fawikibooks-20261101-pages-articles.xml.bz2",
            "fawikibooks-20261101-pages-articles.xml.bz2", "bz2", "mediawiki_xml_bz2",
            size=7086756, sha1="ce11959c5edc617641760bff7574b7777889b5d8",
            options={"skip_redirects": True, "namespace": 0, "min_raw_chars": 120, "include_title": True})],
    ),
    source(
        id="fawikisource_20261101",
        name="Persian Wikisource official articles dump (2026-11-01)",
        profile="A", decision="include", domain="literary",
        description="Public-domain or freely licensed source texts; literary/historical prose and verse.",
        url=f"{WIKI}/fawikisource/20261101/",
        format="MediaWiki XML bz2", approx_size="47.72 MB compressed",
        availability="public HTTP, dated dump", update_date="2026-11-01",
        license_compilation="CC BY-SA 4.0 / GFDL",
        license_content="per-page: public domain or compatible free license; page contributions CC BY-SA 4.0",
        content_rights_verified=True, copyright_holder="original authors / Wikisource contributors",
        attribution_required="yes", share_alike="yes for wiki contributions",
        non_commercial="no", commercial_use="yes for PD/free pages with conditions",
        redistribution="yes with conditions", privacy_risk="low",
        language_purity="Persian, some Arabic religious quotations", lm_suitability="yes_conditional",
        rights_verification="official Wikisource copyright policy and dump index",
        refs=["7", "8", "12", "wikimedia_source_index"],
        notes="Extractor keeps only pages with an explicit PD/CC/GPL-compatible license template; unlabeled pages are skipped and counted. Classical register, not modern prose.",
        files=[file_entry(
            f"{WIKI}/fawikisource/20261101/fawikisource-20261101-pages-articles.xml.bz2",
            "fawikisource-20261101-pages-articles.xml.bz2", "bz2", "mediawiki_xml_bz2",
            size=47719980, sha1="6408e20973ff086a49d31b215d7242a0d76cc76b",
            options={"skip_redirects": True, "namespace": 0, "min_raw_chars": 120,
                     "include_title": True, "require_free_license_template": True})],
    ),
    source(
        id="fawikinews_20261101",
        name="Persian Wikinews official articles dump (2026-11-01)",
        profile="A", decision="include", domain="news",
        description="Collaboratively written news-style Persian prose; small, modern register.",
        url=f"{WIKI}/fawikinews/20261101/",
        format="MediaWiki XML bz2", approx_size="2.63 MB compressed",
        availability="public HTTP, dated dump", update_date="2026-11-01",
        license_compilation="CC BY 2.5", license_content="CC BY 2.5",
        content_rights_verified=True, copyright_holder="Wikinews contributors",
        attribution_required="yes", share_alike="no (CC BY, not BY-SA)",
        non_commercial="no", commercial_use="yes with attribution",
        redistribution="yes with attribution", privacy_risk="low",
        language_purity="Persian", lm_suitability="yes",
        rights_verification="official Wikimedia dump legal page",
        refs=["8", "wikimedia_news_index"],
        notes="Wikinews is licensed CC BY 2.5, NOT CC BY-SA 4.0; keep its license tag distinct in the manifest.",
        files=[file_entry(
            f"{WIKI}/fawikinews/20261101/fawikinews-20261101-pages-articles.xml.bz2",
            "fawikinews-20261101-pages-articles.xml.bz2", "bz2", "mediawiki_xml_bz2",
            size=2627858, sha1="263fa26c8655f63a6e0481d86147e1a2aec709a7",
            options={"skip_redirects": True, "namespace": 0, "min_raw_chars": 120, "include_title": True})],
    ),
    source(
        id="common_voice_fa_sentences",
        name="Mozilla Common Voice Persian sentence files (CC0 text)",
        profile="A", decision="include", domain="dialogue",
        description="Natural short Persian sentences contributed to / collected for Common Voice; text only, no audio.",
        url="https://github.com/common-voice/common-voice/tree/main/server/data/fa",
        format="plain text lines", approx_size="about 2.8 MB across 21 files",
        doc_count_estimate=None, availability="public GitHub raw files, commit-pinned",
        update_date="2022-08-05 (pinned commit ea5d6ee)",
        license_compilation="MPL-2.0 repository software",
        license_content="CC0 1.0 for sentence text (repository README states sentence text is CC0)",
        content_rights_verified=True, copyright_holder="public-domain dedication by contributors",
        attribution_required="no (CC0), but voluntary credit recommended",
        share_alike="no", non_commercial="no", commercial_use="yes",
        redistribution="yes", privacy_risk="low", language_purity="Persian short sentences",
        lm_suitability="yes_small_sentences",
        rights_verification="Common Voice repository README + Mozilla release statement",
        refs=["5", "6", "common_voice_repo"],
        notes="One natural sentence per document; no artificial merging. Do not use audio or speaker metadata. Short-sentence quality profile applies.",
        files=[file_entry(CV_BASE + fname, fname, "txt", "txt_lines",
                          options={"one_doc_per_line": True, "min_line_chars": 10,
                                   "source_label": "common_voice_fa_sentences"})
               for fname in (["sentence-collector.txt"] + [f"sentences{i:02d}.txt" for i in range(1, 21)])],
    ),
    source(
        id="tatoeba_fa_cc_by",
        name="Tatoeba Persian sentences (CC BY 2.0 FR, author attribution)",
        profile="A", decision="include", domain="dialogue",
        description="Crowd-sourced natural Persian example sentences; detailed export retains author name for attribution.",
        url="https://tatoeba.org/en/downloads",
        format="tar.bz2 containing per-language TSV", approx_size="weekly all-language export (tens of MB)",
        availability="public weekly export; not immutable", update_date="weekly (record download date)",
        license_compilation="CC BY 2.0 FR", license_content="CC BY 2.0 FR by default; some CC0",
        content_rights_verified=True, copyright_holder="individual Tatoeba contributors",
        attribution_required="yes: author name must be retained",
        share_alike="no", non_commercial="no", commercial_use="yes with attribution",
        redistribution="yes with author attribution", privacy_risk="low (public pseudonyms in manifest only)",
        language_purity="Persian short sentences, some translated phrasing", lm_suitability="yes_small_sentences",
        rights_verification="official Tatoeba downloads and terms of use",
        refs=["10", "11", "tatoeba_downloads"],
        notes="The CC0-only export does NOT include Persian; use detailed export and store usernames in the manifest (not in training text). One sentence per document.",
        files=[file_entry("https://downloads.tatoeba.org/exports/sentences_detailed.tar.bz2",
                          "sentences_detailed.tar.bz2", "tar.bz2", "tar_tsv",
                          options={"member_pattern": "*fas*sentences*.tsv", "lang_code": "fas",
                                   "columns": ["id", "lang", "text", "username", "date_added", "date_modified"],
                                   "one_doc_per_line": True})],
    ),
    source(
        id="ganjoor_classics_pd",
        name="Ganjoor classical works, public-domain-only subset",
        profile="A", decision="include", domain="literary",
        description="Classical Persian poetry/prose from ganjoor-tex; only pre-modern poet directories in the allowlist.",
        url="https://github.com/ganjoor/ganjoor-tex",
        format="GitHub tarball of .txt tree", approx_size="about 289 MB repository",
        doc_count_estimate=2292, availability="public GitHub, commit-pinned", update_date="2026-09-26",
        license_compilation="GPL-3.0 repository/data files",
        license_content="underlying classical works are public domain; transcription/typesetting may be GPL-3.0",
        content_rights_verified=True, copyright_holder="classical authors (PD) / Ganjoor contributors (transcription)",
        attribution_required="yes (Ganjoor + GPL notice for redistributed data files)",
        share_alike="GPL-3.0 may apply to redistributed data files", non_commercial="no",
        commercial_use="yes for PD works; GPL obligations for data files", redistribution="yes with GPL/PD notices",
        privacy_risk="low", language_purity="classical Persian", lm_suitability="yes_conditional",
        rights_verification="GitHub GPL-3.0 license file + classical author list",
        refs=["9", "25", "ganjoor_repo", "v1_sources_json"],
        notes="Only the 12 classical directories present at the pinned commit are allowed. A modern poet's work must NOT be added until its rights are verified. One source file per document; no cross-file stitching.",
        files=[file_entry(github_tarball("ganjoor/ganjoor-tex", GANJOOR_SHA),
                          "ganjoor-tex.tar.gz", "tar.gz", "zip_txt_tree",
                          options={"subdir": "txt", "allowed_dirs": ["abusaeed", "attar", "babaafzal",
                                   "babataher", "bahaee", "eraghi", "hafez", "khayyam", "moulavi",
                                   "saadi", "sanaee", "shabestari"],
                                   "one_doc_per_file": True, "min_chars": 40})],
    ),
    source(
        id="persian_poetry_ccbysa",
        name="Chronological Persian Poetry Dataset (CC BY-SA 4.0)",
        profile="A", decision="include", domain="literary",
        description="Public-domain classical poems with an explicitly CC BY-SA 4.0 compilation.",
        url="https://github.com/aghasemi/ChronologicalPersianPoetryDataset",
        format="GitHub tarball containing poems.tsv", approx_size="about 35 MB repository",
        doc_count_estimate=14348, availability="public GitHub, commit-pinned", update_date="2021-01-31",
        license_compilation="CC BY-SA 4.0", license_content="public-domain classical poems",
        content_rights_verified=True, copyright_holder="classical poets (PD) / dataset compiler",
        attribution_required="yes", share_alike="yes for compilation", non_commercial="no",
        commercial_use="yes with conditions", redistribution="yes with conditions",
        privacy_risk="low", language_purity="classical Persian verse", lm_suitability="yes_conditional",
        rights_verification="GitHub SPDX CC-BY-SA-4.0 + dataset README",
        refs=["poetry_repo", "v1_sources_json"],
        notes="One poem per document; no artificial 3,200-character blocks. V1 merged unrelated poems; V2 does not.",
        files=[file_entry(github_tarball("aghasemi/ChronologicalPersianPoetryDataset", POETRY_SHA),
                          "ChronologicalPersianPoetryDataset.tar.gz", "tar.gz", "tsv",
                          options={"member_pattern": "*poems.tsv", "delimiter": "\t", "has_header": True,
                                   "columns": ["poet", "century", "poem"], "text_column": "poem",
                                   "id_prefix": "poem"})],
    ),
    source(
        id="wikipedia_fa_2021_mirror_recovery",
        name="V1 Wikipedia 2021 cleaned CSV mirror (optional recovery source)",
        profile="A", decision="include", domain="encyclopedia",
        description="The 865k-article cleaned CSV used by V1; recoverable from split 7z parts in a GitHub mirror.",
        url="https://github.com/hsmirzaie/fawiki",
        format="split 7z -> CSV", approx_size="about 186 MB repository; about 1.1 GB 7z",
        doc_count_estimate=864889, availability="public GitHub API; py7zr needed", update_date="2021-02-14",
        license_compilation="no repository license", license_content="CC BY-SA 4.0 / GFDL Wikipedia text",
        content_rights_verified=True, copyright_holder="Wikipedia contributors",
        attribution_required="yes", share_alike="yes", non_commercial="no",
        commercial_use="yes with conditions", redistribution="yes with conditions",
        privacy_risk="low", language_purity="encyclopedic Persian", lm_suitability="yes",
        rights_verification="Wikimedia ToU; V1 sources.json; mirror README",
        refs=["7", "fawiki_mirror_repo", "v1_sources_json"],
        v1_filtered_chars=457109769, v1_shipped_chars=35298596, v1_shipped_docs=26850,
        notes="Optional. Prefer the official 2026 dump for modern Persian. This source is included to audit V1's claim that 538M filtered characters were recoverable.",
        files=[file_entry(f"https://api.github.com/repos/hsmirzaie/fawiki/contents?ref={FAWIKI_MIRROR_SHA}",
                          "fawiki.7z", "github_api_7z", "csv_from_7z", required=False,
                          options={"filename_pattern": "fawiki.7z.0*", "inner_member": "fawiki.csv",
                                   "delimiter": ",", "columns": ["id", "title", "article"],
                                   "title_column": "title", "text_column": "article", "id_column": "id"})],
    ),
    # ---------------- PROFILE B: research-only ----------------
    source(
        id="parsinlu_suite_nc",
        name="ParsiNLU QA / multiple-choice / entailment annotations",
        profile="B", decision="include_research", domain="qa",
        description="Explicitly CC BY-NC-SA 4.0 Persian task annotations. Research-only, separate output.",
        url="https://github.com/persiannlp/parsinlu",
        format="GitHub tarball of JSONL/TSV", approx_size="about 28 MB repository",
        doc_count_estimate=14000, availability="public GitHub, commit-pinned", update_date="2021-03-29",
        license_compilation="CC BY-NC-SA 4.0", license_content="CC BY-NC-SA 4.0 annotations",
        content_rights_verified=True, copyright_holder="ParsiNLU authors",
        attribution_required="yes", share_alike="yes", non_commercial="yes",
        commercial_use="no", redistribution="yes only under NC-SA and research conditions",
        privacy_risk="low", language_purity="Persian QA / educational / reasoning", lm_suitability="research_only",
        rights_verification="repository LICENSE file",
        refs=["parsinlu_repo", "v1_sources_json"],
        v1_filtered_chars=4061854, v1_shipped_chars=1827229, v1_shipped_docs=6379,
        notes="Excludes reading-comprehension passages taken from publisher-copyright news. Do not include synthetic task markers in the main corpus; keep natural text fields only.",
        files=[file_entry(github_tarball("persiannlp/parsinlu", PARSINLU_SHA),
                          "parsinlu.tar.gz", "tar.gz", "parsinlu_suite",
                          options={"include": ["reading_comprehension", "multiple_choice", "entailment"],
                                   "exclude_translated_test": True})],
    ),
    # ---------------- PROFILE C: quarantine / exclusion ----------------
    source(id="v1_persian_news", name="V1 Persian news corpus (5 agencies)", profile="C",
           decision="quarantine", domain="news", description="Publisher-copyright news articles; no license declared.",
           url="https://github.com/milad-4274/persian_news", format="CSV", approx_size="about 13 MB repository",
           doc_count_estimate=13455, availability="public GitHub", update_date="2020-11-13",
           license_compilation="none declared", license_content="publisher copyright",
           copyright_holder="original news agencies", attribution_required="unclear",
           commercial_use="no", redistribution="not granted", privacy_risk="medium",
           language_purity="Persian news", lm_suitability="no", rights_verification="no license file",
           refs=["v1_sources_json"], v1_filtered_chars=25251850, v1_shipped_chars=4499846, v1_shipped_docs=1757,
           notes="Quarantine: public download does not grant rights to underlying articles. Do not train on it."),
    source(id="v1_pn_summary_news", name="V1 pn-summary news articles", profile="C",
           decision="quarantine", domain="news", description="MIT repository but publisher-copyright news text; V1 quality gate also rejected it.",
           url="https://github.com/hooshvare/pn-summary", format="TSV/CSV", approx_size="about 44 MB repository",
           doc_count_estimate=93207, availability="Google Drive links in repository", update_date="2024-01-20",
           license_compilation="MIT repository", license_content="publisher copyright",
           copyright_holder="news publishers", commercial_use="no", redistribution="not granted",
           privacy_risk="medium", lm_suitability="no", rights_verification="MIT does not cover article text",
           refs=["v1_sources_json"], v1_filtered_chars=0, v1_shipped_chars=0, v1_shipped_docs=0,
           notes="Quarantine: underlying rights unverified and punctuation was stripped upstream."),
    source(id="v1_peqa_dialogue", name="V1 PeQa Twitter-derived dialogue", profile="C",
           decision="quarantine", domain="dialogue", description="14M public tweets/replies; MIT repository does not license the tweets.",
           url="https://github.com/ZahraArshia/PeQa", format="tab-separated chat pairs",
           approx_size="about 3 MB released subset; full 14M pairs external",
           doc_count_estimate=436072, availability="public GitHub + Google Drive", update_date="2022-08-05",
           license_compilation="MIT repository", license_content="original social-media posts; rights unverified",
           copyright_holder="individual posters", attribution_required="unclear", commercial_use="unclear",
           redistribution="unclear", privacy_risk="high", language_purity="informal Persian tweets",
           lm_suitability="no", rights_verification="MIT software license + README",
           refs=["peqa_repo", "v1_sources_json"], v1_filtered_chars=4778181, v1_shipped_chars=1124702, v1_shipped_docs=791,
           notes="Quarantine: identifiable personal social-media content and unverified underlying rights."),
    source(id="slpl_naab", name="SLPL/naab cleaned Persian web corpus", profile="C",
           decision="quarantine", domain="mixed_web", description="130 GB / ~250M paragraphs / ~15B words; mixed Common Crawl, news, Telegram, private company corpus, blogs.",
           url="https://huggingface.co/datasets/SLPL/naab", format="126 train .txt shards + 3 test shards",
           approx_size="137 GB", doc_count_estimate=32672067,
           availability="public Hugging Face; individual 1GB shards", update_date="2022-08-29 (card)",
           license_compilation="MIT metadata; card says 'mit?'", license_content="underlying web/Telegram/private texts; not licensed by the compilation",
           copyright_holder="many third parties", attribution_required="unclear", commercial_use="unclear",
           redistribution="unclear", privacy_risk="high", language_purity="Persian, formal + informal",
           lm_suitability="no_until_cleared", rights_verification="card explicitly takes no responsibility for personal information",
           refs=["1", "2", "4", "19", "naab_card", "naab_files"],
           notes="QUARANTINE. Do not download the full 137 GB. MIT metadata and 'mit?' card do not license crawled pages, Telegram messages, or formerly private data."),
    source(id="shekar_ai_shiraz", name="shekar-ai/Shiraz Persian web corpus", profile="C",
           decision="quarantine", domain="mixed_web", description="6.25M rows / 5.97 GB across 14 sites; contains BBC Persian, Civilica, blogs, generated retrieval queries.",
           url="https://huggingface.co/datasets/shekar-ai/Shiraz", format="Parquet",
           approx_size="5.97 GB", doc_count_estimate=6254329, availability="public Hugging Face",
           update_date="2026 (card; active)", license_compilation="ODC-BY 1.0 compilation",
           license_content="underlying web content remains subject to original authors' rights and site terms",
           copyright_holder="many third parties / BBC / publishers", attribution_required="yes for compilation",
           commercial_use="no blanket permission", redistribution="not granted for underlying text",
           privacy_risk="medium", language_purity="Persian web/news/blog prose",
           lm_suitability="no_until_cleared", rights_verification="dataset card explicitly disclaims granting underlying content rights",
           refs=["28", "18", "shiraz_card", "shiraz_files"],
           notes="QUARANTINE. ODC-BY covers selection/annotations only; BBC and Civilica content has separate rights. Synthetic query columns must never enter pretraining."),
    source(id="targoman_tlpc", name="Targoman Large Persian Corpus (TLPC)", profile="C",
           decision="quarantine", domain="mixed_web", description="98.6 GB / 75M documents / 41B claimed tokens scraped from 800+ sites; CC BY-NC-SA 4.0 compilation.",
           url="https://huggingface.co/datasets/Targoman/TLPC", format="per-site JSONL.gz",
           approx_size="98.6 GB", doc_count_estimate=75000000, availability="public Hugging Face",
           update_date="2024-08-05 statistics; 2026 card", license_compilation="CC BY-NC-SA 4.0",
           license_content="scraped third-party web pages plus reader comments with commenter names/dates",
           copyright_holder="800+ sites and individual commenters", attribution_required="yes",
           share_alike="yes", non_commercial="yes", commercial_use="only ISTI-approved companies",
           redistribution="NC-SA only", privacy_risk="high", language_purity="Persian, formal + colloquial",
           lm_suitability="no_until_pii_and_rights_cleared", rights_verification="card documents named reader comments",
           refs=["14", "27", "13"],
           notes="QUARANTINE by default, despite a clear NC license: documented personal data (names/dates in comments) and uncertain third-party rights. Conditional PROFILE B only after documented PII scrubbing and review."),
    source(id="mirastext", name="MirasText news corpus", profile="C", decision="quarantine",
           domain="news", description="2.8M articles / 1.4B words from 250+ news sites; password-controlled full release.",
           url="https://github.com/miras-tech/MirasText", format="password-protected ZIP",
           approx_size="12 GB in naab", doc_count_estimate=2835414, availability="request password by email",
           update_date="2018/2020", license_compilation="MIT repository",
           license_content="publisher-copyright news articles", copyright_holder="250+ news sites",
           commercial_use="no", redistribution="not granted", privacy_risk="medium",
           lm_suitability="no", rights_verification="MIT repo does not license articles; access control present",
           refs=["4", "mirastext_repo"],
           notes="QUARANTINE. Do not request or bypass the archive password from this pipeline."),
    source(id="persian_raw_text_80gb", name="persiannlp/persian-raw-text 80 GB mixture", profile="C",
           decision="quarantine", domain="mixed_web", description="65 GB Common Crawl + MirasText + W2C + Wikipedia + Leipzig + VOA + poems + TEP; no license.",
           url="https://github.com/persiannlp/persian-raw-text", format="large .txt links",
           approx_size="about 70-80 GB", doc_count_estimate=13500000, availability="Google Storage links",
           update_date="2020-08-28", license_compilation="none declared",
           license_content="mixed; mostly third-party web text", copyright_holder="many third parties",
           commercial_use="unclear", redistribution="unclear", privacy_risk="high",
           lm_suitability="no", rights_verification="no license file", refs=["4", "13", "persian_raw_text_repo"],
           notes="QUARANTINE. No deduplication upstream, mixed rights, Common Crawl terms push copyright responsibility to the user."),
    source(id="commoncrawl_derived_fa", name="OSCAR-fa / CC-100-fa / CulturaX-fa / FineWeb-2-fa family", profile="C",
           decision="quarantine", domain="mixed_web", description="Common-Crawl-derived Persian subsets; dataset licenses do not grant rights to crawled pages.",
           url="https://commoncrawl.org/terms-of-use", format="JSONL/Parquet/text", approx_size="GB to TB",
           availability="public mirrors", license_compilation="varies (often CC BY-SA / ODC-BY)",
           license_content="third-party web pages", copyright_holder="original site owners",
           commercial_use="unclear", redistribution="unclear", privacy_risk="high",
           lm_suitability="no_until_cleared", rights_verification="Common Crawl ToU",
           refs=["13", "26"],
           notes="QUARANTINE. Common Crawl explicitly states crawled content may carry separate terms and users must respect third-party rights."),
    source(id="hamshahri_corpus", name="Hamshahri Persian news corpus", profile="C", decision="quarantine",
           domain="news", description="Newspaper articles used for IR/NLP research; license unknown on redistributions.",
           url="https://www.kaggle.com/datasets/ehsankhani/hamshahri-corpus", format="XML-like .ham files",
           approx_size="hundreds of MB", availability="Kaggle / research request", license_compilation="unknown",
           license_content="Hamshahri newspaper copyright", copyright_holder="Hamshahri",
           commercial_use="no", redistribution="unclear", privacy_risk="medium", lm_suitability="no",
           rights_verification="license unknown", refs=["15", "4"],
           notes="QUARANTINE: unknown license and publisher copyright."),
    source(id="bijankhan_corpus", name="Bijankhan / Peykare tagged corpus", profile="C", decision="quarantine",
           domain="mixed", description="~2.6M-word tagged corpus widely used in Persian NLP; rights and redistribution terms not established here.",
           url="https://nlpdataset.ir/farsi/raw_text_corpora.html", format="tagged text",
           approx_size="small relative to web corpora", availability="research distribution",
           license_compilation="unverified", license_content="unverified", commercial_use="unclear",
           redistribution="unclear", privacy_risk="low", lm_suitability="no_until_cleared",
           rights_verification="only secondary listings inspected", refs=["4", "17"],
           notes="QUARANTINE pending authoritative license documentation."),
    source(id="lscp_corpus", name="Large Scale Colloquial Persian (LSCP)", profile="C", decision="quarantine",
           domain="dialogue", description="27M casual Persian sentences; used as an informal component of naab.",
           url="https://iasbs.ac.ir/~ansari/lscp/", format="derived tree/annotations",
           approx_size="2.3 GB in naab", availability="academic project page",
           license_compilation="unverified", license_content="web/social text, unverified",
           commercial_use="unclear", redistribution="unclear", privacy_risk="high",
           lm_suitability="no_until_cleared", rights_verification="not established", refs=["1", "naab_card"],
           notes="QUARANTINE: underlying casual-text provenance and rights were not verified."),
    source(id="persian_poems_corpus_unlicensed", name="amnghd/Persian_poems_corpus", profile="C",
           decision="quarantine", domain="literary", description="Large poem collection with no repository license.",
           url="https://github.com/amnghd/Persian_poems_corpus", format="text tree", approx_size="about 46 MB",
           availability="public GitHub", license_compilation="none declared",
           license_content="classical PD poems but compilation unverified", commercial_use="unclear",
           redistribution="unclear", privacy_risk="low", lm_suitability="no_until_cleared",
           rights_verification="no license file", refs=["4"],
           notes="QUARANTINE; use the CC BY-SA Chronological dataset instead."),
    source(id="persianpunc", name="PersianPunc punctuation-restoration corpus", profile="C",
           decision="quarantine", domain="mixed", description="17M samples aggregated from Bijankhan, medical QA, Wikipedia, and other corpora.",
           url="https://huggingface.co/datasets/MohammadJRanjbar/PersianPunc", format="Parquet",
           approx_size="~823K released samples; 17M full", availability="public Hugging Face",
           license_compilation="CC BY 4.0", license_content="mixed upstream corpora with differing rights",
           commercial_use="unclear", redistribution="unclear", privacy_risk="medium",
           lm_suitability="no_until_cleared", rights_verification="compilation license only", refs=["17"],
           notes="QUARANTINE: CC BY 4.0 compilation does not settle Bijankhan/medical QA rights; task-specific sentence fragments."),
    source(id="esposito_parallel", name="Esposito English-Persian scientific parallel corpus", profile="C",
           decision="quarantine", domain="scientific_parallel", description="3.5M EN-FA parallel sentences; CC BY-NC-SA 4.0.",
           url="https://huggingface.co/datasets/universitytehran/ESPOSITO", format="parallel TSV",
           approx_size="1.43 GB", availability="public Hugging Face", license_compilation="CC BY-NC-SA 4.0",
           license_content="Scientific Information Database text", non_commercial="yes",
           commercial_use="no", privacy_risk="low", lm_suitability="no",
           rights_verification="dataset card", refs=["16"],
           notes="EXCLUDE: bilingual/parallel objective, NC, and not naturally occurring Persian-only documents."),
    source(id="persian_gk_chat", name="mshojaei77/persian-gk synthetic chat", profile="C",
           decision="quarantine", domain="instruction", description="~5.9K Persian conversations in ChatML; CC BY 4.0.",
           url="https://huggingface.co/datasets/mshojaei77/persian-gk", format="ChatML",
           approx_size="small", availability="public Hugging Face", license_compilation="CC BY 4.0",
           license_content="generated/edited conversation data", privacy_risk="medium",
           lm_suitability="no", rights_verification="secondary model card",
           refs=["28"], notes="EXCLUDE from V2 pretraining: instruction/synthetic chat belongs to the later conversational stage."),
    source(id="persian_tts_transcripts", name="PersianVox / ManaTTS / ParsiGoo transcripts", profile="C",
           decision="quarantine", domain="speech", description="ASR/TTS transcripts mined from in-the-wild audio or magazines; CC BY/CC0 claims vary.",
           url="https://huggingface.co/datasets/saeedzou/persianvox_all", format="Parquet with audio",
           approx_size="112 GB / 86 hours", availability="public Hugging Face",
           license_compilation="CC BY 4.0 or CC0 1.0", license_content="web audio/video or magazine text; rights unclear",
           privacy_risk="high", lm_suitability="no", rights_verification="cards describe in-the-wild mining",
           refs=["22", "23", "5"],
           notes="EXCLUDE: audio-derived short utterances, possible personal data, not natural documents."),
    source(id="persian_dehkhoda_dictionary", name="Dehkhoda dictionary dump", profile="C",
           decision="quarantine", domain="reference", description="Lexicographic entries, CC BY-NC-SA 4.0 compilation.",
           url="https://huggingface.co/datasets/n9e6y/persian-dehkhoda-dictionary", format="dictionary entries",
           approx_size="100K-1M entries", availability="public Hugging Face",
           license_compilation="CC BY-NC-SA 4.0", license_content="dictionary text; NC",
           non_commercial="yes", commercial_use="no", privacy_risk="low", lm_suitability="no",
           rights_verification="dataset card", refs=["24"],
           notes="EXCLUDE: dictionary fragments are not natural prose and NC applies."),
    source(id="persian_wikiquote_wiktionary", name="Persian Wikiquote / Wiktionary dumps", profile="C",
           decision="quarantine", domain="reference", description="Quotations and dictionary entries; CC BY-SA but not natural documents and may quote copyrighted works.",
           url="https://dumps.wikimedia.org/fawiktionary/", format="MediaWiki XML bz2",
           availability="public dumps", license_compilation="CC BY-SA 4.0 / GFDL",
           license_content="quotes may have third-party rights", privacy_risk="low",
           lm_suitability="no", rights_verification="Wikimedia licensing", refs=["7", "8"],
           notes="EXCLUDE from pretraining; not natural continuous Persian prose."),
    source(id="iranian_official_legal_text", name="Iranian laws / official gazette bulk text", profile="C",
           decision="quarantine", domain="legal", description="Official legal texts may be excluded from copyright, but no authoritative bulk open-licensed release was verified.",
           url="https://en.wikipedia.org/wiki/Template:PD-Iran", format="unknown",
           availability="not established", license_compilation="unverified",
           license_content="possibly public domain official texts", privacy_risk="low",
           lm_suitability="no_until_cleared", rights_verification="secondary legal summaries only",
           refs=["9", "25"],
           notes="QUARANTINE pending an authoritative government source and rights statement."),
    source(id="project_gutenberg_persian", name="Project Gutenberg Persian-language / Persian literature books", profile="C",
           decision="quarantine", domain="literary", description="Mostly English translations of classical Persian works; public domain in USA but not Persian-language prose.",
           url="https://www.gutenberg.org/ebooks/10315", format="eBook text",
           availability="public", license_compilation="Project Gutenberg license",
           license_content="public domain in USA", privacy_risk="low", lm_suitability="no",
           rights_verification="ebook page", refs=["21"],
           notes="EXCLUDE: not Persian-language modern prose. A few PD Persian books on Internet Archive need OCR and per-work rights review; not part of V2."),
    source(id="voa_persian_corpus", name="VOA Persian 2003-2008 corpus", profile="C",
           decision="quarantine", domain="news", description="7.9M-word VOA corpus; possible US-government public-domain status not authoritatively verified here.",
           url="https://jon.dehdari.org/corpora/", format="plain text", approx_size="66 MB",
           availability="academic mirror", license_compilation="unverified",
           license_content="VOA content; rights unverified", privacy_risk="low",
           lm_suitability="no_until_cleared", rights_verification="secondary listing", refs=["4"],
           notes="QUARANTINE pending an authoritative rights statement."),
]


def catalog() -> dict:
    return {
        "catalog_version": "2.0.0",
        "research_date_utc": "2026-11-02",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "description": "Persian Dataset V2 source and licensing catalog. Rights describe the actual text, not merely a repository/compilation. Not legal advice.",
        "profiles": {
            "A": "PERMISSIVE CORE - only sources with verified underlying content rights",
            "B": "RESEARCH ONLY - clearly documented non-commercial sources, physically separate output",
            "C": "QUARANTINE - unknown/conflicting rights, provenance or privacy concerns; do not train",
        },
        "research_refs": RESEARCH_REFS,
        "direct_refs": DIRECT_REFS,
        "sources": SOURCES,
    }


CATALOG_COLUMNS = [
    "id", "name", "profile", "decision", "domain", "format", "approx_size",
    "doc_count_estimate", "availability", "update_date", "license_compilation",
    "license_content", "content_rights_verified", "commercial_use", "privacy_risk",
    "lm_suitability", "rights_verification", "url", "v1_filtered_chars",
    "v1_shipped_chars", "v1_shipped_docs", "notes",
]

LICENSING_COLUMNS = [
    "id", "name", "profile", "decision", "underlying_text", "copyright_holder",
    "license_compilation", "license_content", "attribution_required", "share_alike",
    "non_commercial", "commercial_use", "redistribution", "privacy_risk",
    "rights_verification", "rationale", "refs",
]


def _csv_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict, bool)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def export_csvs(meta: dict) -> None:
    with open(os.path.join(HERE, "source_catalog.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CATALOG_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for s in meta["sources"]:
            w.writerow({k: _csv_value(s.get(k)) for k in CATALOG_COLUMNS})
    with open(os.path.join(HERE, "licensing_matrix.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LICENSING_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for s in meta["sources"]:
            row = dict(s)
            row["underlying_text"] = s["description"]
            row["rationale"] = s["notes"]
            row["refs"] = ", ".join(str(r) for r in s.get("refs", []))
            w.writerow({k: _csv_value(row.get(k)) for k in LICENSING_COLUMNS})


def write_json(meta: dict) -> None:
    tmp = os.path.join(HERE, "source_metadata.json.part")
    out = os.path.join(HERE, "source_metadata.json")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, out)


def main() -> None:
    meta = catalog()
    write_json(meta)
    export_csvs(meta)
    print(f"wrote {len(meta['sources'])} sources; profiles: "
          f"A={sum(s['profile']=='A' for s in SOURCES)}, "
          f"B={sum(s['profile']=='B' for s in SOURCES)}, "
          f"C={sum(s['profile']=='C' for s in SOURCES)}")


if __name__ == "__main__":
    main()
