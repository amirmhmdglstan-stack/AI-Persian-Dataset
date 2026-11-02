"""
persian_norm.py — Persian text normalization, quality gating and fingerprints.

Used by build_dataset.py. Standard library only (no numpy/pandas needed) so the
pipeline runs anywhere.

Design notes
------------
* Normalization is *conservative*: it fixes character-level noise and encoding
  damage but never rewrites words or invents text.
* ZWNJ (U+200C, "نیم‌فاصله") is a meaningful Persian character and is PRESERVED.
* Persian digits ۰۱۲۳۴۵۶۷۸۹ are canonical: Latin and Arabic-Indic digits are
  mapped onto them so the model sees a single digit alphabet.
* Arabic letter variants (ي ك ة ى) are mapped to Persian counterparts (ی ک ه ی).
* Quality filters are heuristics; every threshold lives in FILTERS / PROFILES and
  can be audited or tuned.
"""

from __future__ import annotations

import re
import struct
import unicodedata
import zlib

# --------------------------------------------------------------------------- #
# Character tables
# --------------------------------------------------------------------------- #

ZWNJ = "\u200c"

# Persian alphabet (letters that may legitimately appear)
PERSIAN_LETTERS = set("آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیءأإئؤۀٔ")
PERSIAN_DIGITS = set("۰۱۲۳۴۵۶۷۸۹")
PERSIAN_PUNCT = "،؛؟«»!.:,()[]-–—…%٫٬٪"

ALLOWED_EXTRA = set(PERSIAN_PUNCT + " \n" + ZWNJ)
ALLOWED = "".join(sorted(PERSIAN_LETTERS | PERSIAN_DIGITS | ALLOWED_EXTRA))

# Arabic -> Persian letter variants (applied via str.translate)
_LETTER_MAP_STR = {
    "\u064a": "\u06cc", "\u0649": "\u06cc",           # ي , ى -> ی
    "\u0643": "\u06a9", "\u06aa": "\u06a9",           # ك -> ک
    "\u0629": "\u0647", "\u06c0": "\u0647",           # ة , ۀ -> ه
    "\u06d5": "\u0647", "\u06be": "\u0647", "\u06c1": "\u0647",
    "\u0640": "",                                     # tatweel
}
LETTER_MAP = {ord(k): v for k, v in _LETTER_MAP_STR.items()}

# Diacritics (harakat) and Quranic marks: optional in Persian and a major source
# of spelling inconsistency in web text -> removed.
DIACRITICS = dict.fromkeys(
    [ord(c) for c in "\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652"
                     "\u0653\u0654\u0655\u0656\u0657\u0658\u0659\u065a"
                     "\u065b\u065c\u065d\u065e\u065f\u0670"],
    None,
)

# Invisible / bidi control characters always dropped (ZWNJ is kept separately)
INVISIBLE = dict.fromkeys(
    [ord(c) for c in "\u200b\u200d\u200e\u200f\u202a\u202b\u202c\u202d"
                     "\u202e\u2066\u2067\u2068\u2069\ufeff\u00ad\u061c"],
    None,
)

# Arabic presentation forms (U+FB50-U+FDFF, U+FE70-U+FEFF) must be folded onto
# base letters BEFORE NFKC: Unicode's own decomposition maps some Persian forms
# to the wrong base letter (e.g. U+FBF5 decomposes to U+0684 rather than چ).
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


def _build_presentation_map() -> dict:
    table = {}
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


PRESENTATION_MAP = _build_presentation_map()

_SYMBOL_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"   # emoji / pictographs
    "\u2600-\u27BF"           # misc symbols, dingbats
    "\u2B00-\u2BFF"           # arrows, stars
    "\uFE0F\u20E3"            # variation selector-16, keycap
    "\u2190-\u21FF"           # arrows
    "\u2500-\u257F"           # box drawing
    "]+",
    flags=re.UNICODE,
)

_DIGIT_MAP = {ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")}
_DIGIT_MAP.update({ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")})
_PERSIAN_DIGIT_TABLE = {ord(str(i)): c for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")}

# Anything not in ALLOWED becomes a space (single C-level regex pass)
_NOT_ALLOWED_RE = re.compile("[^" + re.escape(ALLOWED) + "]+")

# HTML / wiki / web markup
_RE_REF = re.compile(r"<ref[^>]{0,200}>.*?</ref>|<ref[^>]{0,200}/>", re.DOTALL)
_RE_FILE = re.compile(r"\[\[(?:پرونده|تصویر|File|Image):[^\]]{0,300}\]\]", re.IGNORECASE)
_RE_HTML_TAG = re.compile(r"</?[a-zA-Z][^<>]{0,200}?>")
_RE_HTML_ENT = re.compile(r"&(?:[a-zA-Z]{2,10}|#\d{1,6}|#x[0-9a-fA-F]{1,6});")
_RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")
_RE_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_RE_WIKI_TMPL = re.compile(r"\{\{[^{}]{0,300}\}\}")
_RE_WIKI_LINK = re.compile(r"\[\[(?:[^\]|]{0,120}\|)?([^\]]{0,200})\]\]")

_RE_MULTISPACE = re.compile(r"[ \t\u00a0\u2000-\u200a\u202f\u205f]+")
_RE_MULTINEWLINE = re.compile(r"\n{3,}")
_RE_SPACE_BEFORE_PUNCT = re.compile(r"\s+([،؛؟!.:»\)\],])")
_RE_SPACE_AFTER_OPEN = re.compile(r"([«\(\[])\s+")
_RE_SPACE_AROUND_ZWNJ = re.compile(r"\s*\u200c\s*")
_RE_MISSING_SPACE = re.compile(r"([.!؟؛،:])(?=[آ-یءأإئؤ])")
_RE_TRAILING_WS = re.compile(r"[ \t]+\n")

# Lines that are web boilerplate, not prose
BOILERPLATE_PATTERNS = [
    r"اشتراک\s*گذاری", r"کد\s*خبر", r"منبع\s*خبر", r"کلیه\s*حقوق",
    r"تمام\s*حقوق", r"تمامی\s*حقوق", r"حقوق\s*محفوظ", r"بازگشت\s*به\s*بالا",
    r"مطالب\s*مرتبط", r"برچسب\s*ها", r"نظر\s*شما", r"دیدگاه\s*شما",
    r"نوشته\s*شده\s*توسط", r"دنبال\s*کنید", r"کلیک\s*کنید", r"اینجا\s*کلیک",
    r"لینک\s*کوتاه", r"پرینت\s*این\s*صفحه", r"کانال\s*ما",
    r"تلگرام|اینستاگرام|واتساپ|واتس‌اپ|توییتر|فیسبوک",
    r"©|copyright|all rights reserved|rss feed|subscribe",
    r"^\s*[|\-–—_=*#]{3,}\s*$",
    r"^\s*(?:صفحه|page)\s*[0-9۰-۹]+\s*(?:از|of)\s*[0-9۰-۹]+\s*$",
]
_RE_BOILERPLATE = re.compile("|".join(BOILERPLATE_PATTERNS), re.IGNORECASE)

# Punctuation debris repaired after the allowed-character pass. Upstream cleaning
# (Wikipedia dump, news archives) leaves behind empty brackets, doubled commas and
# orphan colons, e.g. "(به انگلیسی: )" or "( ، )".
_RE_EMPTY_PARENS = re.compile(r"[([{]\s*[،؛:.\-–—]{0,3}\s*[)\]}]")
_RE_EMPTY_QUOTES = re.compile(r"«\s*[،؛:\-]{0,2}\s*»")
_RE_PUNCT_FIXES = [
    (re.compile(r"،(?:\s*،)+"), "،"),
    (re.compile(r":\s*([،؛.!؟)\]])"), r"\1"),
    (re.compile(r"([،؛:])\s*(\)|\])"), r"\1"),
    (re.compile(r"([\[({])\s*([،؛:])"), r"\1"),
    (re.compile(r"([،؛:])\s*([.!؟])"), r"\2"),
    (re.compile(r"\s+([،؛.؟!:])"), r"\1"),
    (re.compile(r"(^|\n)\s*[،؛:\-–—]+\s*"), r"\1"),
    (re.compile(r"([،؛])\s*\n"), r"\n"),
    (re.compile(r"([.!؟])\s*[.!؟]+"), r"\1"),
]

# Upstream wiki-link flattening ("[[فیلم ترسناک|ترسناک]]") leaves the same word
# twice in a row: "فیلم ترسناک ترسناک". Collapse those immediate repeats.
_RE_ADJACENT_DUP = re.compile(r"(\w[\w\u200c]{2,})(?:\s+\1)+")
# The same artefact also duplicates short multi-word phrases
# ("دی سی کامیکس دی سی کامیکس", "اغوا کننده اغوا کننده"). Only applied to prose
# sources: in verse, repeated phrases can be a deliberate refrain.
_RE_PHRASE_DUP = re.compile(r"(\w[\w\u200c]{1,}(?:[\s\u200c]+\w[\w\u200c]{1,}){1,2})(?:\s+\1)+")

_RE_SENT_SPLIT = re.compile(r"[.!؟?؛;\n]+")
_RE_PERSIAN_LETTER = re.compile("[آ-یءأإئؤ]")
_RE_LATIN = re.compile(r"[A-Za-z]")
_RE_DIGIT = re.compile("[0-9۰-۹]")
_RE_QUOTE = re.compile("[«»\"']")
_RE_PIPE = re.compile(r"\|")

# --------------------------------------------------------------------------- #
# Tunable filter thresholds
# --------------------------------------------------------------------------- #

FILTERS = {
    "max_chars": 60000,
    "min_persian_letter_ratio": 0.80,   # among letters
    "max_latin_ratio": 0.18,            # among letters
    "max_digit_ratio": 0.22,            # among non-space chars
    "min_avg_sentence_len": 22,
    "max_avg_sentence_len": 320,
    "max_short_sentence_ratio": 0.45,   # sentences shorter than 15 chars
    "max_pipe_ratio": 0.004,
    "max_quote_ratio": 0.030,
    "max_ngram_repeat_ratio": 0.12,     # share of the most frequent 4-gram
    "max_compress_ratio": 6.0,          # above => mostly digits/noise
    "max_dup_line_ratio": 0.30,
    # Classical poetry has no sentence-final punctuation and short verse lines,
    # so those two structural checks are switched off for poetry sources.
    "check_sentence_shape": True,
    "require_sentence_end": True,
}

# Per-source overrides for the structural minimums
PROFILES = {
    "wikipedia_fa":        {"min_chars": 320, "min_letters": 120, "min_sentences": 3},
    "persian_news":        {"min_chars": 280, "min_letters": 120, "min_sentences": 2},
    "pn_summary_news":     {"min_chars": 280, "min_letters": 120, "min_sentences": 2},
    "persian_poetry":      {"min_chars": 320, "min_letters": 120, "min_sentences": 1,
                            "filters": {"check_sentence_shape": False,
                                        "require_sentence_end": False,
                                        "max_ngram_repeat_ratio": 0.25}},
    "ganjoor_classics":    {"min_chars": 320, "min_letters": 120, "min_sentences": 1,
                            "filters": {"check_sentence_shape": False,
                                        "require_sentence_end": False,
                                        "max_ngram_repeat_ratio": 0.25}},
    "parsinlu_entailment": {"min_chars": 120, "min_letters": 40,  "min_sentences": 1,
                            "filters": {"min_avg_sentence_len": 12, "max_short_sentence_ratio": 0.70}},
    "peqa_dialogue":       {"min_chars": 200, "min_letters": 80,  "min_sentences": 1},
    "parsinlu_rc_passage": {"min_chars": 280, "min_letters": 100, "min_sentences": 2},
    "parsinlu_mc":         {"min_chars": 60,  "min_letters": 20,  "min_sentences": 1,
                            "filters": {"min_avg_sentence_len": 12, "max_short_sentence_ratio": 0.70}},
    "parsinlu_qa":         {"min_chars": 40,  "min_letters": 15,  "min_sentences": 1,
                            "filters": {"min_avg_sentence_len": 12, "max_short_sentence_ratio": 0.70}},
}


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #

def _strip_markup(text: str) -> str:
    text = _RE_REF.sub(" ", text)
    text = _RE_FILE.sub(" ", text)
    text = _RE_WIKI_TMPL.sub(" ", text)
    text = _RE_WIKI_LINK.sub(r"\1", text)
    text = _RE_HTML_TAG.sub(" ", text)
    text = _RE_HTML_ENT.sub(" ", text)
    text = _RE_EMAIL.sub(" ", text)
    text = _RE_URL.sub(" ", text)
    text = text.replace("[[", " ").replace("]]", " ").replace("{{", " ").replace("}}", " ")
    return text


def normalize(text: str, collapse_phrases: bool = False) -> str:
    """Normalize one Persian document. Returns '' if nothing usable remains."""
    if not text:
        return ""

    # 1. Fold Arabic presentation forms, then NFKC for remaining compatibility
    #    characters, then Arabic -> Persian letters, drop diacritics.
    text = text.translate(PRESENTATION_MAP)
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(INVISIBLE)
    text = text.translate(DIACRITICS)
    text = text.translate(LETTER_MAP)
    text = _SYMBOL_RE.sub(" ", text)

    # 2. Drop markup, URLs, entities
    text = _strip_markup(text)

    # 3. Whitespace and punctuation spacing
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _RE_MULTISPACE.sub(" ", text)
    text = _RE_SPACE_AROUND_ZWNJ.sub(ZWNJ, text)
    text = _RE_SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _RE_SPACE_AFTER_OPEN.sub(r"\1", text)
    text = _RE_MISSING_SPACE.sub(r"\1 ", text)

    # 4. Canonical digit set (NFKC-independent: Arabic-Indic -> Persian)
    text = text.translate(_DIGIT_MAP)
    text = text.translate(_PERSIAN_DIGIT_TABLE)

    # 5. Drop every character that is not Persian letter/digit/punctuation
    text = _NOT_ALLOWED_RE.sub(" ", text)

    # 6. Collapse words (and, for prose sources, short phrases) duplicated
    #    back-to-back by upstream link flattening
    text = _RE_ADJACENT_DUP.sub(r"\1", text)
    if collapse_phrases:
        text = _RE_PHRASE_DUP.sub(r"\1", text)

    # 7. Repair punctuation debris left by upstream cleaning
    text = _RE_EMPTY_PARENS.sub(" ", text)
    text = _RE_EMPTY_QUOTES.sub(" ", text)
    for pattern, repl in _RE_PUNCT_FIXES:
        text = pattern.sub(repl, text)

    # 8. Drop boilerplate lines (whole lines only, never part of a sentence)
    lines = (ln.strip() for ln in text.split("\n"))
    text = "\n".join(ln for ln in lines if ln and not _RE_BOILERPLATE.search(ln))

    text = _RE_MULTISPACE.sub(" ", text)
    text = _RE_MULTINEWLINE.sub("\n\n", text)
    text = _RE_TRAILING_WS.sub("\n", text)
    return text.strip()


def clean_lines(text: str) -> str:
    """Drop residual noise lines: navigation, captions, single words, debris."""
    out = []
    for ln in text.split("\n"):
        ln = ln.strip(" -–—•*|")
        if not ln:
            continue
        if len(_RE_PERSIAN_LETTER.findall(ln)) < 3:
            continue
        if len(ln) < 12 and not ln.endswith(("!", ".", "؟", ":", "»")):
            continue
        out.append(ln)
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def char_profile(text: str) -> dict:
    n = len(text)
    persian_letters = len(_RE_PERSIAN_LETTER.findall(text))
    latin_letters = len(_RE_LATIN.findall(text))
    digits = len(_RE_DIGIT.findall(text))
    pipes = text.count("|")
    quotes = len(_RE_QUOTE.findall(text))
    non_space = n - text.count(" ") - text.count("\n")
    return {
        "chars": n,
        "non_space": max(non_space, 1),
        "persian_letters": persian_letters,
        "latin_letters": latin_letters,
        "digits": digits,
        "pipes": pipes,
        "quotes": quotes,
    }


def sentence_stats(text: str) -> dict:
    parts = [p for p in _RE_SENT_SPLIT.split(text) if p.strip()]
    if not parts:
        return {"n": 0, "avg": 0.0, "short_ratio": 1.0}
    lens = [len(p) for p in parts]
    short = sum(1 for L in lens if L < 15)
    return {"n": len(lens), "avg": sum(lens) / len(lens), "short_ratio": short / len(lens)}


def dup_line_ratio(text: str) -> float:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) < 4:
        return 0.0
    return 1.0 - (len(set(lines)) / len(lines))


def compress_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(text) / max(len(zlib.compress(text.encode("utf-8"), 6)), 1)


def ngram_repeat_ratio(text: str, n: int = 4, max_words: int = 4000) -> float:
    """Share of the most frequent n-gram among all n-grams (repetition proxy)."""
    words = text.split()
    if len(words) > max_words:
        step = len(words) / max_words
        words = [words[int(i * step)] for i in range(max_words)]
    if len(words) < n + 2:
        return 0.0
    counts = {}
    for i in range(len(words) - n + 1):
        g = " ".join(words[i:i + n])
        counts[g] = counts.get(g, 0) + 1
    return max(counts.values()) / (len(words) - n + 1)


# --------------------------------------------------------------------------- #
# Quality gate
# --------------------------------------------------------------------------- #

def quality_check(text: str, min_chars: int = 280, min_letters: int = 100,
                  min_sentences: int = 3, filters: dict | None = None):
    """Return (ok: bool, reason: str). The first failing check wins."""
    f = FILTERS if not filters else {**FILTERS, **filters}

    if len(text) < min_chars:
        return False, "too_short"
    if len(text) > f["max_chars"]:
        return False, "too_long"

    cp = char_profile(text)
    letters = cp["persian_letters"] + cp["latin_letters"]
    if cp["persian_letters"] < min_letters:
        return False, "few_persian_letters"
    if letters == 0:
        return False, "no_letters"
    if cp["persian_letters"] / letters < f["min_persian_letter_ratio"]:
        return False, "low_persian_ratio"
    if cp["latin_letters"] / letters > f["max_latin_ratio"]:
        return False, "high_latin_ratio"
    if cp["digits"] / cp["non_space"] > f["max_digit_ratio"]:
        return False, "high_digit_ratio"
    if cp["pipes"] / cp["non_space"] > f["max_pipe_ratio"]:
        return False, "table_debris"
    if cp["quotes"] / cp["non_space"] > f["max_quote_ratio"]:
        return False, "excessive_quotes"

    ss = sentence_stats(text)
    if ss["n"] < min_sentences:
        return False, "too_few_sentences"
    if f["check_sentence_shape"]:
        if not (f["min_avg_sentence_len"] <= ss["avg"] <= f["max_avg_sentence_len"]):
            return False, "bad_sentence_length"
        if ss["short_ratio"] > f["max_short_sentence_ratio"]:
            return False, "fragmented"

    if dup_line_ratio(text) > f["max_dup_line_ratio"]:
        return False, "repeated_lines"

    if ngram_repeat_ratio(text) > f["max_ngram_repeat_ratio"]:
        return False, "repetitive_ngrams"

    if len(text) > 600 and compress_ratio(text) > f["max_compress_ratio"]:
        return False, "low_information"

    if f["require_sentence_end"] and not any(p in text for p in ".!؟"):
        return False, "no_sentence_end"

    return True, "ok"


# --------------------------------------------------------------------------- #
# Deduplication fingerprints
# --------------------------------------------------------------------------- #

_RE_DEDUP_STRIP = re.compile(r"[\s\u200c،؛.,!؟?:\"'«»\(\)\[\]\-–—…]+")


def dedup_key(text: str) -> str:
    """Aggressively normalized form used for exact duplicate detection."""
    return _RE_DEDUP_STRIP.sub("", text.lower())


def words_for_shingles(text: str) -> list:
    return _RE_DEDUP_STRIP.sub(" ", text.lower()).split()


# --- exact duplicates ------------------------------------------------------- #

def exact_hash(text: str) -> str:
    return _h64_str(dedup_key(text))


def _h64_str(s: str) -> str:
    return "%016x" % (zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFFFFFFFFFF)


# --- near duplicates: one-permutation MinHash ------------------------------- #
# Each shingle hash is split into bucket id (top MB_BITS bits) and value (rest);
# the smallest value per bucket forms the signature. Two documents are considered
# near-duplicates when most buckets agree. This is O(#shingles) per document,
# single pass, no per-bit loops (much cheaper than a naive 64-bit SimHash).

MB_BITS = 4                     # 16 buckets
MB_MASK = (1 << (32 - MB_BITS)) - 1
MB_NBUCKETS = 1 << MB_BITS
SHINGLE_N = 4
MAX_SHINGLE_WORDS = 600

SIG_BANDS = 4                   # 4 bands of 4 buckets for LSH candidate lookup
BAND_LEN = MB_NBUCKETS // SIG_BANDS
MIN_BANDS_MATCH = 3             # >=3 of 4 bands equal => near-duplicate


_SIG_FMT = "<%dI" % MB_NBUCKETS


def minhash_signature(text: str):
    """One-permutation MinHash over 4-word shingles.

    Returns (packed_signature_bytes, n_shingles). Buckets that received no
    shingle keep the sentinel value MB_MASK + 1.
    """
    words = words_for_shingles(text)
    if len(words) > MAX_SHINGLE_WORDS:
        step = len(words) / MAX_SHINGLE_WORDS
        words = [words[int(i * step)] for i in range(MAX_SHINGLE_WORDS)]
    sig = [MB_MASK + 1] * MB_NBUCKETS
    if len(words) < SHINGLE_N:
        return struct.pack(_SIG_FMT, *sig), 0
    crc = zlib.crc32
    for i in range(len(words) - SHINGLE_N + 1):
        h = crc(" ".join(words[i:i + SHINGLE_N]).encode("utf-8"))
        b = h >> (32 - MB_BITS)
        v = h & MB_MASK
        if v < sig[b]:
            sig[b] = v
    return struct.pack(_SIG_FMT, *sig), len(words) - SHINGLE_N + 1


def signature_match(a: bytes, b: bytes) -> int:
    """Number of identical buckets between two packed signatures."""
    ua = struct.unpack(_SIG_FMT, a)
    ub = struct.unpack(_SIG_FMT, b)
    return sum(1 for x, y in zip(ua, ub) if x == y)


def signature_bands(sig: bytes):
    """4 bands of 4 buckets each (4 bytes per bucket)."""
    step = 4 * BAND_LEN
    return [sig[i * step:(i + 1) * step] for i in range(SIG_BANDS)]


def near_dup_signature(text: str):
    """(packed signature, n_shingles) used by the dedup stage."""
    return minhash_signature(text)


def persian_ratio(text: str) -> float:
    cp = char_profile(text)
    return cp["persian_letters"] / cp["non_space"]
