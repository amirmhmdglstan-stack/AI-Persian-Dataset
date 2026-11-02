#!/usr/bin/env python3
"""Improved Persian normalization, language scoring and deduplication keys.

Differences from V1's ``persian_norm.py``:

* Latin-script technical abbreviations and proper names INSIDE genuine Persian
  prose are preserved, measured and reported instead of silently deleted.
* Persian digits are preserved; Latin digits are preserved; Arabic-Indic digits
  are mapped to Persian digits.  V1 converted all digits to Persian.
* PII-like strings (e-mail, phone, national-ID patterns) are detected and the
  WHOLE document is rejected by the quality gate; they are never quietly kept.
* Whole-line boilerplate is removed, but short natural dialogue/verse lines are
  not deleted merely for being short.
* Every removal is counted.  A document losing more than ``max_removal_ratio``
  of its characters is rejected with an explicit reason.
* Language detection combines script ratios with Persian function-word hits,
  Persian-only letters and Arabic-marker words; it is not just a Unicode check.

This module is standard-library only and Windows/Python 3.12 compatible.
"""
from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

ZWNJ = "\u200c"
ZWJ = "\u200d"

PERSIAN_LETTERS = set("آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیءأإئؤۀٔ")
PERSIAN_SPECIFIC = set("پچژگ")
PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
PERSIAN_PUNCT = "،؛؟«»٪٫٬"
LATIN_PUNCT = "!.:,;?()[]{}\"'-–—…/%&=+*#@~|^$`\\"
ALLOWED_EXTRA = set(PERSIAN_PUNCT + LATIN_PUNCT + " \n\t" + ZWNJ + PERSIAN_DIGITS)
ALLOWED_CHARS = PERSIAN_LETTERS | ALLOWED_EXTRA | set("0123456789") | set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")

# Arabic letter variants -> Persian standard.  Counts are reported.
CHAR_MAP = {
    "\u064a": "\u06cc",  # Arabic yeh -> Persian yeh
    "\u0649": "\u06cc",  # alef maksura -> yeh
    "\u0643": "\u06a9",  # Arabic kaf -> Persian kaf
    "\u06aa": "\u06a9",
    "\u0629": "\u0647",  # teh marbuta -> heh
    "\u06c0": "\u0647",
    "\u06d5": "\u0647",
    "\u06be": "\u0647",
    "\u06c1": "\u0647",
    "\u0640": "",        # tatweel
}
for i, ch in enumerate(ARABIC_INDIC_DIGITS):
    CHAR_MAP[ch] = PERSIAN_DIGITS[i]

DIACRITICS = dict.fromkeys(
    [ord(c) for c in "\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652"
                     "\u0653\u0654\u0655\u0656\u0657\u0658\u0659\u065a"
                     "\u065b\u065c\u065d\u065e\u065f\u0670"], None)
INVISIBLE = dict.fromkeys(
    [ord(c) for c in "\u200b\u200d\u200e\u200f\u202a\u202b\u202c\u202d"
                     "\u202e\u2066\u2067\u2068\u2069\ufeff\u00ad\u061c"], None)

_BASE_NAMES = {
    "ALEF": "ا", "ALEF WITH MADDA ABOVE": "آ", "ALEF WITH HAMZA ABOVE": "أ",
    "ALEF WITH HAMZA BELOW": "إ", "ALEF WITH WAVY HAMZA ABOVE": "ٱ",
    "BEH": "ب", "TEH": "ت", "THEH": "ث", "JEEM": "ج", "HAH": "ح",
    "KHAH": "خ", "DAL": "د", "THAL": "ذ", "REH": "ر", "ZAIN": "ز",
    "SEEN": "س", "SHEEN": "ش", "SAD": "ص", "DAD": "ض", "TAH": "ط",
    "ZAH": "ظ", "AIN": "ع", "GHAIN": "غ", "FEH": "ف", "QAF": "ق",
    "KAF": "ک", "KEHEH": "ک", "GAF": "گ", "LAM": "ل", "MEEM": "م",
    "NOON": "ن", "HEH": "ه", "TEH MARBUTA": "ه", "WAW": "و",
    "YEH": "ی", "FARSI YEH": "ی", "ALEF MAKSURA": "ی", "HAMZA": "ء",
    "TCHEH": "چ", "JEH": "چ", "PEH": "پ", "VEH": "و",
}
_FORM_SUFFIX = re.compile(r"\s+(?:ISOLATED|FINAL|INITIAL|MEDIAL)\s+FORM$")


def _presentation_map() -> dict[int, str]:
    table: dict[int, str] = {}
    for cp in range(0xFB50, 0xFE00):
        ch = chr(cp)
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        if "ARABIC" not in name:
            continue
        base = _FORM_SUFFIX.sub("", name)
        if base.startswith("ARABIC LETTER "):
            key = base[len("ARABIC LETTER "):]
            if key in _BASE_NAMES:
                table[cp] = _BASE_NAMES[key]
        elif base.startswith("ARABIC LIGATURE "):
            key = base[len("ARABIC LIGATURE "):]
            parts = key.split(" WITH ")
            if len(parts) > 1 and parts[0] == "LAM" and parts[1] in _BASE_NAMES:
                table[cp] = "ل" + _BASE_NAMES[parts[1]]
    return table


PRESENTATION_MAP = _presentation_map()
_TRANSLATE = {ord(k): v for k, v in CHAR_MAP.items()}
_TRANSLATE.update(DIACRITICS)
_TRANSLATE.update(INVISIBLE)

RE_PERSIAN_LETTER = re.compile("[آ-یءأإئؤۀٔ]")
RE_PERSIAN_SPECIFIC = re.compile("[پچژگ]")
RE_LATIN_LETTER = re.compile(r"[A-Za-z]")
RE_PERSIAN_DIGIT = re.compile("[۰-۹]")
RE_LATIN_DIGIT = re.compile("[0-9]")
RE_ANY_DIGIT = re.compile("[0-9۰-۹]")
RE_ZWNJ = re.compile("\u200c")
RE_WORD = re.compile(r"\S+")
RE_SENT_SPLIT = re.compile(r"[.!؟?؛;\n]+")

RE_REF = re.compile(r"<ref[^>]{0,200}>.*?</ref>|<ref[^>]{0,200}/>", re.DOTALL | re.IGNORECASE)
RE_HTML_TAG = re.compile(r"</?[A-Za-z][^<>]{0,300}?>")
RE_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
RE_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
RE_WIKI_FILE = re.compile(r"\[\[(?:پرونده|تصویر|File|Image):[^\]]{0,400}\]\]", re.IGNORECASE)
RE_WIKI_CATEGORY = re.compile(r"\[\[(?:رده|Category):[^\]]{0,300}\]\]", re.IGNORECASE)
RE_WIKI_LINK = re.compile(r"\[\[(?:[^\]|]{0,200}\|)?([^\]]{0,300})\]\]")
RE_EXTERNAL_LINK = re.compile(r"\[(?:https?://|www\.)\S+?\s+([^\]]{1,300})\]", re.IGNORECASE)
RE_TEMPLATE_SIMPLE = re.compile(r"\{\{[^{}]{0,400}\}\}")
RE_TABLE = re.compile(r"\{\|.*?\|\}", re.DOTALL)
RE_HEADING = re.compile(r"^\s*={2,6}\s*(.*?)\s*={2,6}\s*$", re.MULTILINE)
RE_EMPTY_PARENS = re.compile(r"[([{][ \t]*[،؛:.!?\-–—]{0,3}[ \t]*[)\]}]")
RE_EMPTY_QUOTES = re.compile(r"«\s*[،؛:\-]{0,2}\s*»")
RE_MULTISPACE = re.compile(r"[ \t\u00a0\u2000-\u200a\u202f\u205f]+")
RE_SPACE_BEFORE_PUNCT = re.compile(r"[ \t]+([،؛؟!.:,»\)\],;])")
RE_SPACE_AFTER_OPEN = re.compile(r"([«\(\[{])[ \t]+")
RE_SPACE_AROUND_ZWNJ = re.compile(r"[ \t]*\u200c[ \t]*")
RE_MISSING_SPACE = re.compile(r"([.!؟؛،:؛])(?=[آ-یءأإئؤ])")
RE_TRAILING_WS = re.compile(r"[ \t]+\n")
RE_MULTINEWLINE = re.compile(r"\n{3,}")
RE_REPEAT_PUNCT = re.compile(r"([!؟؟.,،؛:])\1{1,}")
RE_BROKEN_ENCODING = re.compile(r"(?:Ø§|Ù…|ÛŒ|â€|Ã©|Ø¨|Ù†)")
RE_MOJIBAKE_MARKERS = ("Ø", "Ù", "Û", "â€", "Ã")

BOILERPLATE_PATTERNS = [
    r"اشتراک\s*گذاری", r"کد\s*خبر", r"منبع\s*خبر", r"کلیه\s*حقوق",
    r"تمام\s*حقوق", r"تمامی\s*حقوق", r"حقوق\s*محفوظ", r"بازگشت\s*به\s*بالا",
    r"مطالب\s*مرتبط", r"برچسب\s*ها", r"نظر\s*شما", r"دیدگاه\s*شما",
    r"نوشته\s*شده\s*توسط", r"دنبال\s*کنید", r"کلیک\s*کنید", r"اینجا\s*کلیک",
    r"لینک\s*کوتاه", r"پرینت\s*این\s*صفحه", r"کانال\s*ما",
    r"تلگرام|اینستاگرام|واتساپ|واتس\u200cاپ|توییتر|فیسبوک",
    r"©|copyright|all rights reserved|rss feed|subscribe",
    r"^\s*[|\-–—_=*#]{3,}\s*$",
    r"^\s*(?:صفحه|page)\s*[0-9۰-۹]+\s*(?:از|of)\s*[0-9۰-۹]+\s*$",
]
RE_BOILERPLATE = re.compile("|".join(BOILERPLATE_PATTERNS), re.IGNORECASE)

# PII detection on a temporary ASCII-digit copy.
RE_PHONE = re.compile(r"(?:\+98|0098|098|0)?9[0-9]{9}|(?:\+98|0)?[1-9][0-9]{2}\s?[0-9]{3}\s?[0-9]{4}")
RE_NATIONAL_ID = re.compile(r"(?<!\d)[0-9]{10}(?!\d)")
RE_LONG_CARD = re.compile(r"(?<!\d)(?:[0-9]{4}[ -]?){3}[0-9]{4}(?!\d)")

PERSIAN_STOPWORDS_DEFAULT = [
    "است", "بود", "شد", "شود", "هستند", "این", "آن", "که", "با", "به", "از",
    "در", "و", "برای", "من", "ما", "یا", "را", "هم", "نیز", "ولی", "اما",
    "اگر", "چون", "باید", "وجود", "دارد", "بوده", "خواهد", "توان", "باشد",
]
ARABIC_MARKERS_DEFAULT = ["الذی", "هذا", "هذه", "في", "على", "كانت", "التي", "ذلك", "كان", "من"]


@dataclass
class NormalizationResult:
    text: str
    stats: dict = field(default_factory=dict)


@dataclass
class LanguageResult:
    is_persian: bool
    score: float
    features: dict
    reason: str


def _strip_templates_nested(text: str) -> tuple[str, int]:
    """Remove {{...}} templates with bounded nesting; return (text, count)."""
    out, depth, start, removed = [], 0, None, 0
    i = 0
    while i < len(text):
        if text.startswith("{{", i):
            if depth == 0:
                start = i
            depth += 1
            i += 2
            continue
        if text.startswith("}}", i) and depth:
            depth -= 1
            i += 2
            if depth == 0:
                removed += 1
                if out and not out[-1].endswith(("\n", " ")):
                    out.append(" ")
                start = None
            continue
        if depth == 0:
            out.append(text[i])
        i += 1
    if depth:  # unterminated template: preserve conservative remainder marker
        out.append(" ")
    return "".join(out), removed


def _strip_markup(text: str, stats: dict) -> str:
    def sub(pattern, repl, s, key):
        s2, n = pattern.subn(repl, s)
        stats[key] = stats.get(key, 0) + n
        return s2

    text = sub(RE_REF, " ", text, "refs_removed")
    text = sub(RE_HTML_COMMENT, " ", text, "html_comments_removed")
    text = sub(RE_WIKI_FILE, " ", text, "wiki_files_removed")
    text = sub(RE_WIKI_CATEGORY, " ", text, "wiki_categories_removed")
    text = sub(RE_TABLE, " ", text, "tables_removed")
    text, n_templates = _strip_templates_nested(text)
    stats["templates_removed"] = stats.get("templates_removed", 0) + n_templates
    text = sub(RE_WIKI_LINK, r"\1", text, "wiki_links_flattened")
    text = sub(RE_EXTERNAL_LINK, r"\1", text, "external_links_flattened")
    text = sub(RE_HTML_TAG, " ", text, "html_tags_removed")
    entities = len(re.findall(r"&(?:[A-Za-z]{2,12}|#\d{1,7}|#x[0-9A-Fa-f]{1,7});", text))
    text = html.unescape(text)
    stats["html_entities_unescaped"] = stats.get("html_entities_unescaped", 0) + entities
    text = sub(RE_EMAIL, " ", text, "emails_removed")
    text = sub(RE_URL, " ", text, "urls_removed")
    return text


def _repair_mojibake(text: str, stats: dict) -> str:
    if not RE_BROKEN_ENCODING.search(text) and not any(m in text for m in RE_MOJIBAKE_MARKERS):
        return text
    if RE_PERSIAN_LETTER.search(text):
        return text  # already Persian; do not risk corrupting it
    try:
        candidate = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text
    if len(RE_PERSIAN_LETTER.findall(candidate)) > len(RE_PERSIAN_LETTER.findall(text)):
        stats["mojibake_repaired"] = 1
        return candidate
    return text


def _pii_hits(text: str) -> dict:
    ascii_digits = text.translate(str.maketrans(PERSIAN_DIGITS, "0123456789"))
    return {
        "email": len(RE_EMAIL.findall(ascii_digits)),
        "phone": len(RE_PHONE.findall(ascii_digits)),
        "national_id": len(RE_NATIONAL_ID.findall(ascii_digits)),
        "card_like": len(RE_LONG_CARD.findall(ascii_digits)),
    }


def normalize(text: str, config: dict | None = None, *, collapse_adjacent_duplicates: bool = False) -> NormalizationResult:
    """Normalize one Persian document and return explicit removal statistics."""
    cfg = (config or {}).get("normalization", {})
    stats: dict[str, Any] = {
        "original_chars": len(text or ""),
        "original_lines": len((text or "").split("\n")),
    }
    if not text:
        stats.update({"final_chars": 0, "removed_chars": 0, "removed_ratio": 0.0})
        return NormalizationResult("", stats)

    stats["arabic_variants_normalized"] = sum(text.count(c) for c in CHAR_MAP if c in text)
    stats["presentation_forms_normalized"] = sum(text.count(chr(cp)) for cp in PRESENTATION_MAP)
    stats["diacritics_removed"] = sum(text.count(chr(cp)) for cp in DIACRITICS)
    stats["invisible_removed"] = sum(text.count(chr(cp)) for cp in INVISIBLE)
    stats["zwnj_before"] = text.count(ZWNJ)
    stats["replacement_chars_before"] = text.count("\ufffd")

    if cfg.get("repair_mojibake", True):
        text = _repair_mojibake(text, stats)
    text = html.unescape(text)
    text = text.translate(PRESENTATION_MAP)
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_TRANSLATE)
    # PII is detected BEFORE markup/URL/e-mail stripping; the quality gate
    # rejects the WHOLE document.  Removal is only a second safety layer.
    pii = _pii_hits(text)
    text = _strip_markup(text, stats)
    stats["pii_hits"] = pii
    stats["pii_total"] = sum(pii.values())
    # Optional second safety layer: remove PII-like substrings.  The document is
    # still rejected, so no private text can silently survive.
    if cfg.get("strip_pii", True) and stats["pii_total"]:
        text = RE_EMAIL.sub(" ", text)
        text = re.sub(r"[0-9۰-۹+][0-9۰-۹\s\-]{8,}[0-9۰-۹]", " ", text)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = RE_HEADING.sub(r"\1", text)
    text = RE_SPACE_AROUND_ZWNJ.sub(ZWNJ, text)
    text = RE_MULTISPACE.sub(" ", text)
    text = RE_SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = RE_SPACE_AFTER_OPEN.sub(r"\1", text)
    text = RE_MISSING_SPACE.sub(r"\1 ", text)
    text = RE_EMPTY_PARENS.sub(" ", text)
    text = RE_EMPTY_QUOTES.sub(" ", text)
    if cfg.get("collapse_repeated_punctuation", True):
        text, n = RE_REPEAT_PUNCT.subn(r"\1", text)
        stats["repeated_punctuation_collapsed"] = n

    if collapse_adjacent_duplicates:
        # Conservative: only exact immediate repeats of a 3+ char token.
        text, n = re.subn(r"(\w[\w\u200c]{2,})(?:\s+\1)+", r"\1", text)
        stats["adjacent_duplicates_collapsed"] = n

    # Whole-line boilerplate only.  A long prose line that merely CONTAINS a
    # boilerplate phrase is kept; deleting it would silently destroy valuable
    # Persian content.  Short navigation lines are still dropped.
    kept_lines, boiler, kept_hits = [], 0, 0
    for line in text.split("\n"):
        if RE_BOILERPLATE.search(line):
            persian_in_line = len(RE_PERSIAN_LETTER.findall(line))
            if len(line) > 200 and persian_in_line > 80:
                kept_hits += 1
                kept_lines.append(line.strip(" \t"))
                continue
            boiler += 1
            continue
        kept_lines.append(line.strip(" \t"))
    stats["boilerplate_lines_removed"] = boiler
    stats["boilerplate_hits_in_long_prose_kept"] = kept_hits
    text = "\n".join(kept_lines)

    # Character whitelist: preserve Latin script inside genuine Persian text.
    before_filter = len(text)
    disallowed = sum(1 for ch in text if ch not in ALLOWED_CHARS)
    text = "".join(ch if ch in ALLOWED_CHARS else " " for ch in text)
    stats["disallowed_chars_replaced"] = disallowed
    text = RE_MULTISPACE.sub(" ", text)
    text = RE_TRAILING_WS.sub("\n", text)
    text = RE_MULTINEWLINE.sub("\n\n", text)
    text = "\n".join(ln.strip() for ln in text.split("\n"))
    text = text.strip()
    stats["chars_after_whitelist"] = before_filter
    stats["final_chars"] = len(text)
    stats["removed_chars"] = max(0, stats["original_chars"] - len(text))
    stats["removed_ratio"] = round(stats["removed_chars"] / max(1, stats["original_chars"]), 4)

    words = RE_WORD.findall(text)
    persian_letters = len(RE_PERSIAN_LETTER.findall(text))
    latin_letters = len(RE_LATIN_LETTER.findall(text))
    persian_digits = len(RE_PERSIAN_DIGIT.findall(text))
    latin_digits = len(RE_LATIN_DIGIT.findall(text))
    latin_words = [w for w in words if RE_LATIN_LETTER.search(w) and not RE_PERSIAN_LETTER.search(w)]
    persian_words = [w for w in words if RE_PERSIAN_LETTER.search(w)]
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    sentences = [s.strip() for s in RE_SENT_SPLIT.split(text) if s.strip()]
    stats.update({
        "persian_letters": persian_letters,
        "latin_letters": latin_letters,
        "persian_specific_letters": len(RE_PERSIAN_SPECIFIC.findall(text)),
        "persian_digits": persian_digits,
        "latin_digits": latin_digits,
        "digits": persian_digits + latin_digits,
        "zwnj_after": len(RE_ZWNJ.findall(text)),
        "words": len(words),
        "persian_words": len(persian_words),
        "latin_words": len(latin_words),
        "latin_word_ratio": round(len(latin_words) / max(1, len(words)), 4),
        "latin_letter_ratio_of_letters": round(latin_letters / max(1, persian_letters + latin_letters), 4),
        "persian_letter_ratio_of_letters": round(persian_letters / max(1, persian_letters + latin_letters), 4),
        "persian_letter_ratio_of_nonspace": round(persian_letters / max(1, len(text.replace(" ", "").replace("\n", ""))), 4),
        "paragraphs": max(1, len(paragraphs)),
        "sentences": max(1, len(sentences)),
        "avg_sentence_chars": round(sum(len(s) for s in sentences) / max(1, len(sentences)), 1),
    })
    return NormalizationResult(text, stats)


def detect_language(text: str, stats: dict, config: dict | None = None,
                    domain: str = "default") -> LanguageResult:
    """Heuristic Persian detection using script, function words and markers.

    Short natural sentences (dialogue/QA/verse) often contain no function word,
    so they are accepted on lexical evidence (Persian-only letter OR function
    word) plus script ratios and Latin/Arabic checks.  This is deliberately NOT
    a claim about semantic correctness; it is a reproducible script/register
    filter.  Manual audit samples remain necessary.
    """
    cfg = (config or {}).get("language", {})
    short_or_special = len(text) < 120 or domain in ("dialogue", "qa", "literary")
    words = [w.strip("،؛؟!.،:؛«»()[]\"'") for w in RE_WORD.findall(text.lower())]
    persian_letters = stats.get("persian_letters", 0)
    latin_letters = stats.get("latin_letters", 0)
    letters = max(1, persian_letters + latin_letters)
    persian_share = persian_letters / letters
    persian_nonspace = stats.get("persian_letter_ratio_of_nonspace", 0.0)
    latin_word_share = stats.get("latin_word_ratio", 0.0)
    stopwords = set(cfg.get("persian_stopwords", PERSIAN_STOPWORDS_DEFAULT))
    stopword_hits = sum(1 for w in words if w in stopwords)
    specific = stats.get("persian_specific_letters", 0)
    lowered = text.lower()
    arabic_markers = sum(1 for m in cfg.get("arabic_markers", ARABIC_MARKERS_DEFAULT) if m in lowered)
    lexical_evidence = stopword_hits > 0 or specific > 0 or len(text) >= 180
    min_share = cfg.get("min_persian_letter_share_of_letters", 0.65)
    min_nonspace = cfg.get("min_persian_letter_share_of_nonspace", 0.30)
    max_latin_words = cfg.get("max_latin_word_share", 0.35)
    # A genuine long Persian sentence may contain NO پچژگ but many function
    # words; requiring both signals would create false non-Persian rejections.
    min_stop = 0 if (short_or_special or specific >= 1) else cfg.get("min_persian_stopword_hits", 1)
    min_specific = 0 if (short_or_special or stopword_hits >= 1) else cfg.get("min_persian_specific_letters", 1)

    score = 0.0
    score += 0.40 * min(1.0, persian_share / max(0.01, min_share))
    score += 0.20 * min(1.0, persian_nonspace / max(0.01, min_nonspace))
    score += 0.15 * min(1.0, stopword_hits / max(1, cfg.get("min_persian_stopword_hits", 1)))
    score += 0.10 * (1.0 if specific >= cfg.get("min_persian_specific_letters", 1) else 0.0)
    score += 0.05 * (1.0 if lexical_evidence else 0.0)
    score += 0.10 * max(0.0, 1.0 - latin_word_share / max(0.01, max_latin_words))
    score -= 0.25 * min(1.0, arabic_markers / 2.0)
    score = round(max(0.0, min(1.0, score)), 4)

    reasons = []
    if persian_share < min_share:
        reasons.append("persian_letter_share_below_threshold")
    if persian_nonspace < min_nonspace:
        reasons.append("persian_nonspace_share_below_threshold")
    if latin_word_share > max_latin_words:
        reasons.append("too_many_latin_only_words")
    if stopword_hits < min_stop:
        reasons.append("no_persian_function_words")
    if specific < min_specific:
        reasons.append("no_persian_specific_letters")
    if not lexical_evidence:
        reasons.append("no_persian_lexical_evidence")
    if arabic_markers >= 2 and stopword_hits == 0:
        reasons.append("arabic_marker_words")
    is_persian = not reasons
    features = {
        "persian_letter_share_of_letters": round(persian_share, 4),
        "persian_letter_share_of_nonspace": round(persian_nonspace, 4),
        "latin_word_share": round(latin_word_share, 4),
        "persian_stopword_hits": stopword_hits,
        "persian_specific_letters": specific,
        "arabic_marker_hits": arabic_markers,
        "lexical_evidence": bool(lexical_evidence),
        "short_or_special_domain": bool(short_or_special),
    }
    return LanguageResult(is_persian, score, features, ";".join(reasons) or "ok")


def dedup_key(text: str) -> str:
    """Aggressively normalized form for exact duplicate detection."""
    t = text.lower()
    t = t.translate(str.maketrans(PERSIAN_DIGITS, "0123456789"))
    t = re.sub(r"[\s\u200c\u200d،؛.,!؟?:;\"'«»\(\)\[\]\{\}\-–—…]+", "", t)
    return t


def words_for_shingles(text: str) -> list[str]:
    t = text.lower().translate(str.maketrans(PERSIAN_DIGITS, "0123456789"))
    return re.sub(r"[\s\u200c\u200d،؛.,!؟?:;\"'«»\(\)\[\]\{\}\-–—…]+", " ", t).split()
