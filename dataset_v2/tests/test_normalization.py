#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dataset_v2" / "pipeline"))
from common import load_config  # noqa: E402
from normalize_persian import detect_language, normalize, dedup_key  # noqa: E402

CFG = load_config(ROOT / "dataset_v2" / "configs" / "dataset_config.json")


class TestNormalization(unittest.TestCase):
    def test_arabic_variants_and_zwnj(self):
        r = normalize("كتاب يك نيم\u200cفاصله و خانه\u200cها و ٢٠٢٦", CFG)
        self.assertIn("کتاب یک نیم\u200cفاصله", r.text)
        self.assertIn("۲۰۲۶", r.text)
        self.assertNotIn("\u064a", r.text)
        self.assertNotIn("\u0643", r.text)
        self.assertGreaterEqual(r.stats["zwnj_after"], 2)
        self.assertNotEqual(dedup_key("سلام، دنیا!"), dedup_key("سلام دنیا"))

    def test_latin_technical_terms_preserved_and_measured(self):
        text = ("زبان برنامه\u200cنویسی Python در این متن فارسی معرفی می\u200cشود و "
                "کتابخانه\u200cهای NumPy و API نیز به کار رفته\u200cاند.")
        r = normalize(text, CFG)
        for term in ("Python", "NumPy", "API"):
            self.assertIn(term, r.text)
        self.assertGreater(r.stats["latin_letters"], 0)
        self.assertGreater(r.stats["latin_words"], 0)
        self.assertLess(r.stats["latin_word_ratio"], 0.35)
        lang = detect_language(r.text, r.stats, CFG, domain="encyclopedia")
        self.assertTrue(lang.is_persian, lang.reason)

    def test_persian_and_latin_digits_preserved(self):
        r = normalize("سال ۱۴۰۲ و سال 2026 و سال ٢٠٢٦", CFG)
        self.assertIn("۱۴۰۲", r.text)
        self.assertIn("2026", r.text)
        self.assertIn("۲۰۲۶", r.text)
        self.assertGreater(r.stats["persian_digits"], 0)
        self.assertGreater(r.stats["latin_digits"], 0)

    def test_markup_urls_emails_and_pii(self):
        r = normalize(
            "این <b>متن</b> فارسی است. نشانی https://example.ir و رایانامه "
            "test.user@example.com و شماره 09123456789 در متن آمده است.", CFG)
        self.assertNotIn("<b>", r.text)
        self.assertNotIn("https://", r.text)
        self.assertNotIn("test.user@example.com", r.text)
        self.assertNotIn("09123456789", r.text)
        self.assertGreater(r.stats["pii_total"], 0)
        self.assertGreater(r.stats["emails_removed"], 0)
        self.assertGreater(r.stats["html_tags_removed"], 0)

    def test_paragraph_boundaries_and_punctuation(self):
        r = normalize("بند نخست است.\n\n\nبند دوم!!!!  برای آزمایش؟", CFG)
        self.assertIn("\n\n", r.text)
        self.assertIn("!", r.text)
        self.assertNotIn("!!!!", r.text)
        self.assertGreaterEqual(r.stats["paragraphs"], 2)

    def test_language_detection_rejects_english_and_arabic(self):
        en = normalize("This is an English-only document about databases and Python.", CFG)
        self.assertFalse(detect_language(en.text, en.stats, CFG, domain="encyclopedia").is_persian)
        ar = normalize("هذا كتاب جميل في المكتبة وكان اليوم جميلا على الطاولة.", CFG)
        self.assertFalse(detect_language(ar.text, ar.stats, CFG, domain="encyclopedia").is_persian)

    def test_no_silent_large_removal(self):
        r = normalize("متن فارسی کوتاه است و نباید بی\u200cدلیل حذف شود.", CFG)
        self.assertIn("removed_ratio", r.stats)
        self.assertIn("removed_chars", r.stats)
        self.assertIn("disallowed_chars_replaced", r.stats)
        self.assertLess(r.stats["removed_ratio"], 0.25)


if __name__ == "__main__":
    unittest.main()
