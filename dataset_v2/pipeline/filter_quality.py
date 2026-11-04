#!/usr/bin/env python3
"""Explainable, deterministic quality gate for Persian Dataset V2.

IMPORTANT: these are surface-form heuristics.  They can detect broken layout,
repetition, non-Persian script, PII-like strings and boilerplate.  They do NOT
establish factual accuracy, semantic correctness, absence of bias, or legal
safety.  Representative manual audit samples are generated separately.
"""
from __future__ import annotations

import re
import zlib
from dataclasses import dataclass, field
from typing import Any

from normalize_persian import (
    LanguageResult, RE_LATIN_LETTER, RE_PERSIAN_LETTER, RE_ANY_DIGIT,
)

RE_SENTENCE_END = re.compile(r"[.!؟]")
RE_SENT_PART = re.compile(r"(?<=[.!؟])\s+|\n+")
RE_PUNCT = re.compile(r"[،؛؟!.:,;«»\(\)\[\]\{\}\"'\-–—…%٪٫٬/]")
RE_PIPE = re.compile(r"\|")
RE_HTML_LEFTOVER = re.compile(r"&(?:nbsp|amp|lt|gt|quot);|<[A-Za-z/][^<>]{0,100}>|\{\{|\}\}|\[\[")
RE_SPAM = re.compile(
    r"همین\s*حالا\s*کلیک|تخفیف\s*ویژه|فروش\s*ویژه|خرید\s*آنلاین|"
    r"برای\s*کسب\s*اطلاعات\s*بیشتر\s*کلیک|تبلیغات|اسپانسر|"
    r"click\s*here|buy\s*now", re.IGNORECASE)
RE_REPEATED_LINE = re.compile(r"^\s*([\w\u200c]{2,})(?:\s+\1){2,}\s*$", re.MULTILINE)
RE_PERSIAN_WORD = re.compile(r"[آ-یءأإئؤ]")


@dataclass
class QualityDecision:
    passed: bool
    score: float
    reasons: list[str] = field(default_factory=list)
    features: dict = field(default_factory=dict)
    category: str = "good"


def sentence_stats(text: str) -> dict:
    parts = [p.strip() for p in RE_SENT_PART.split(text) if p.strip()]
    if not parts:
        return {"count": 0, "avg_chars": 0.0, "short_ratio": 1.0, "ended_ratio": 0.0}
    lens = [len(p) for p in parts]
    ended = sum(1 for p in parts if p.rstrip().endswith((".", "!", "؟")))
    return {
        "count": len(parts),
        "avg_chars": round(sum(lens) / len(parts), 1),
        "short_ratio": round(sum(1 for n in lens if n < 15) / len(parts), 4),
        "ended_ratio": round(ended / len(parts), 4),
    }


def repeated_paragraph_ratio(text: str) -> float:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paras) < 2:
        return 0.0
    return round(1.0 - len(set(paras)) / len(paras), 4)


def repeated_sentence_ratio(text: str) -> float:
    parts = [p.strip() for p in RE_SENT_PART.split(text) if p.strip()]
    if len(parts) < 3:
        return 0.0
    return round(1.0 - len(set(parts)) / len(parts), 4)


def ngram_repeat_ratio(text: str, n: int = 5, max_words: int = 4000) -> float:
    words = text.split()
    if len(words) > max_words:
        step = len(words) / max_words
        words = [words[int(i * step)] for i in range(max_words)]
    if len(words) < n + 1:
        return 0.0
    counts: dict[str, int] = {}
    for i in range(len(words) - n + 1):
        g = " ".join(words[i:i + n])
        counts[g] = counts.get(g, 0) + 1
    return round(max(counts.values()) / (len(words) - n + 1), 4)


def compress_ratio(text: str) -> float:
    if not text:
        return 0.0
    return round(len(text) / max(1, len(zlib.compress(text.encode("utf-8"), 6))), 3)


def _quality_profile(domain: str, config: dict) -> dict:
    profiles = config.get("quality_profiles", {})
    return profiles.get(domain, profiles.get("default", {}))


def evaluate(text: str, norm_stats: dict, language: LanguageResult,
             source_meta: dict, config: dict) -> QualityDecision:
    """Return an explainable accept/reject decision and a 0..1 quality score."""
    cfg = config.get("quality_global", {})
    prof = _quality_profile(source_meta.get("domain", "default"), config)
    reasons: list[str] = []
    chars = len(text)
    nonspace = max(1, chars - text.count(" ") - text.count("\n"))

    if not text.strip():
        return QualityDecision(False, 0.0, ["empty_document"], {}, "rejected")
    if norm_stats.get("pii_total", 0) > 0:
        reasons.append("pii_detected")
    if language is not None and not language.is_persian:
        reasons.append("non_persian:" + language.reason)
    if norm_stats.get("removed_ratio", 0.0) > config.get("normalization", {}).get("max_removal_ratio", 0.25):
        reasons.append("excessive_normalization_removal")
    if chars < prof.get("min_chars", 200):
        reasons.append("too_short")
    if chars > cfg.get("max_chars", 60000):
        reasons.append("too_long")
    if norm_stats.get("persian_letters", 0) < prof.get("min_persian_letters", 80):
        reasons.append("few_persian_letters")
    if norm_stats.get("latin_letter_ratio_of_letters", 0.0) > cfg.get("max_latin_letter_share", 0.30):
        reasons.append("high_latin_contamination")
    digit_share = norm_stats.get("digits", 0) / nonspace
    if digit_share > cfg.get("max_digit_share", 0.25):
        reasons.append("numeric_garbage")
    punct_share = len(RE_PUNCT.findall(text)) / nonspace
    if punct_share > cfg.get("max_punctuation_share", 0.20):
        reasons.append("excessive_punctuation")
    pipe_share = len(RE_PIPE.findall(text)) / nonspace
    if pipe_share > cfg.get("max_pipe_share", 0.005):
        reasons.append("table_debris")

    ss = sentence_stats(text)
    if ss["count"] < prof.get("min_sentences", 2):
        reasons.append("too_few_sentences")
    if ss["avg_chars"] > prof.get("max_avg_sentence_len", 400):
        reasons.append("broken_sentences")
    if ss["short_ratio"] > prof.get("max_short_sentence_ratio", 0.60):
        reasons.append("fragmented")
    if prof.get("require_sentence_end", True) and not RE_SENTENCE_END.search(text):
        reasons.append("no_sentence_end")
    if prof.get("require_sentence_end", True) and ss["ended_ratio"] < 0.25 and chars > 350:
        reasons.append("broken_sentence_boundaries")

    rep_para = repeated_paragraph_ratio(text)
    rep_ngram = ngram_repeat_ratio(text)
    rep_sentence = repeated_sentence_ratio(text)
    max_ngram = prof.get("max_ngram_repeat_ratio", cfg.get("max_ngram_repeat_ratio", 0.15))
    if rep_para > cfg.get("max_repeated_paragraph_ratio", 0.25):
        reasons.append("repeated_paragraphs")
    if rep_ngram > max_ngram:
        reasons.append("repetitive_ngrams")
    if rep_sentence > cfg.get("max_repeated_sentence_ratio", 0.25):
        reasons.append("repeated_sentences")
    if RE_REPEATED_LINE.search(text):
        reasons.append("repeated_line_words")

    cr = compress_ratio(text) if chars > 600 else 0.0
    if cr and cr > cfg.get("max_compress_ratio", 6.0):
        reasons.append("low_information_density")

    boilerplate_lines = norm_stats.get("boilerplate_lines_removed", 0)
    original_lines = max(1, norm_stats.get("original_lines", 1))
    boiler_rate = boilerplate_lines / original_lines
    if boiler_rate > config.get("normalization", {}).get("max_line_boilerplate_ratio", 0.40):
        reasons.append("navigation_boilerplate")
    spam_hits = len(RE_SPAM.findall(text))
    if spam_hits >= 3:
        reasons.append("spam_like_text")
    if RE_HTML_LEFTOVER.search(text):
        reasons.append("markup_leftover")

    lang_score = language.score if language is not None else 0.0
    ends_well = text.rstrip().endswith((".", "!", "؟", "»", ")", "]")) or not prof.get("require_sentence_end", True)
    features = {
        "chars": chars,
        "length_score": round(min(1.0, chars / 2000.0), 4),
        "persian_ratio": norm_stats.get("persian_letter_ratio_of_letters", 0.0),
        "latin_letter_ratio": norm_stats.get("latin_letter_ratio_of_letters", 0.0),
        "latin_word_ratio": norm_stats.get("latin_word_ratio", 0.0),
        "sentence_count": ss["count"],
        "avg_sentence_chars": ss["avg_chars"],
        "sentence_ended_ratio": ss["ended_ratio"],
        "short_sentence_ratio": ss["short_ratio"],
        "paragraph_count": norm_stats.get("paragraphs", 1),
        "repeated_paragraph_ratio": rep_para,
        "repeated_sentence_ratio": rep_sentence,
        "max_5gram_share": rep_ngram,
        "boilerplate_line_rate": round(boiler_rate, 4),
        "spam_hits": spam_hits,
        "digit_share": round(digit_share, 4),
        "punctuation_share": round(punct_share, 4),
        "compression_ratio": cr,
        "language_consistency": lang_score,
        "document_completeness": 1.0 if ends_well else 0.45,
        "zwnj_count": norm_stats.get("zwnj_after", 0),
        "removed_ratio": norm_stats.get("removed_ratio", 0.0),
        "pii_total": norm_stats.get("pii_total", 0),
    }
    repetition = max(rep_para, rep_ngram, rep_sentence)
    score = (
        0.20 * features["length_score"]
        + 0.15 * features["persian_ratio"]
        + 0.12 * features["language_consistency"]
        + 0.12 * features["sentence_ended_ratio"]
        + 0.08 * features["document_completeness"]
        + 0.10 * (1.0 - min(1.0, repetition / max(0.01, max_ngram)))
        + 0.08 * (1.0 - min(1.0, boiler_rate))
        + 0.05 * (1.0 - min(1.0, spam_hits / 5.0))
        + 0.05 * (1.0 - min(1.0, features["digit_share"] / max(0.01, cfg.get("max_digit_share", 0.25))))
        + 0.05 * (1.0 - min(1.0, features["punctuation_share"] / max(0.01, cfg.get("max_punctuation_share", 0.20))))
    )
    score = round(max(0.0, min(1.0, score)), 4)
    if score < cfg.get("min_quality_score", 0.35):
        reasons.append("low_quality_score")

    passed = not reasons
    lo, hi = cfg.get("borderline_score_band", [0.35, 0.55])
    if not passed:
        category = "rejected"
    elif lo <= score <= hi:
        category = "borderline"
    else:
        category = "good"
    return QualityDecision(passed, score, reasons, features, category)


def audit_category(decision: QualityDecision, source_meta: dict, duplicate_status: str = "kept") -> list[str]:
    """Categories used by the manual audit sampler (a document may have several)."""
    cats = [decision.category]
    domain = source_meta.get("domain", "default")
    if domain in ("dialogue", "qa"):
        cats.append("informal_or_dialogue")
    if domain in ("encyclopedia", "educational", "news"):
        cats.append("formal_prose")
    if duplicate_status in ("exact_duplicate", "near_duplicate"):
        cats.append("near_duplicate_example")
    return list(dict.fromkeys(cats))
