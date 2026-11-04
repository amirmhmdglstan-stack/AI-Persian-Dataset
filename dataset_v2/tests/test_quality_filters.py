#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dataset_v2" / "pipeline"))
from common import load_config  # noqa: E402
from filter_quality import evaluate  # noqa: E402
from normalize_persian import detect_language, normalize  # noqa: E402

CFG = load_config(ROOT / "dataset_v2" / "configs" / "dataset_config.json")
SRC = {"domain": "encyclopedia", "profile": "A", "id": "test"}
GOOD = (
    "زبان فارسی یکی از زبان\u200cهای ایرانی است و در ایران، افغانستان و تاجیکستان صحبت می\u200cشود. "
    "این زبان دارای تاریخی طولانی است و آثار بسیاری به آن نوشته شده است. "
    "در دوره\u200cهای گوناگون، نویسندگان و شاعران از آن برای بیان دانش، تاریخ و فرهنگ استفاده کرده\u200cاند. "
    "امروزه متن\u200cهای علمی، آموزشی و خبری فراوانی به فارسی منتشر می\u200cشود و پژوهشگران "
    "برای پردازش زبان طبیعی به پیکره\u200cهای متنی با کیفیت نیاز دارند. "
    "کیفیت یک پیکره به تنوع منابع، صحت نگارش، حذف تکرارها و حفظ نشانه\u200cهای جمله بستگی دارد."
)


def decide(text, domain="encyclopedia"):
    src = {**SRC, "domain": domain}
    r = normalize(text, CFG)
    lang = detect_language(r.text, r.stats, CFG, domain=domain)
    return evaluate(r.text, r.stats, lang, src, CFG), r, lang


class TestQualityFilters(unittest.TestCase):
    def test_good_prose_accepted_with_explainable_features(self):
        d, _, _ = decide(GOOD)
        self.assertTrue(d.passed, d.reasons)
        self.assertGreaterEqual(d.score, 0.35)
        for key in ("persian_ratio", "sentence_ended_ratio", "max_5gram_share",
                    "boilerplate_line_rate", "language_consistency", "document_completeness"):
            self.assertIn(key, d.features)
        self.assertEqual(d.category, "good")

    def test_short_document_rejected(self):
        d, _, _ = decide("سلام")
        self.assertFalse(d.passed)
        self.assertIn("too_short", d.reasons)

    def test_english_only_rejected(self):
        d, _, _ = decide("This is an English-only document with enough words to pass length checks. "
                         "It contains no Persian prose at all and should be rejected by language detection. "
                         "The quality gate must not accept it merely because it has punctuation and sentences.")
        self.assertFalse(d.passed)
        self.assertTrue(any(r.startswith("non_persian") for r in d.reasons))

    def test_pii_rejected(self):
        d, _, _ = decide(GOOD + " رایانامه test.user@example.com نیز در این متن آمده است.")
        self.assertFalse(d.passed)
        self.assertIn("pii_detected", d.reasons)

    def test_repetition_rejected(self):
        sentence = "این جمله تکراری برای آزمایش کیفیت نوشته شده است. "
        d, _, _ = decide(sentence * 30)
        self.assertFalse(d.passed)
        self.assertTrue(any(r in d.reasons for r in
                            ("repetitive_ngrams", "repeated_paragraphs",
                             "repeated_sentences", "low_information_density",
                             "low_quality_score")))

    def test_spam_rejected(self):
        spam = (GOOD + " همین حالا کلیک کنید. تخفیف ویژه. فروش ویژه. "
                "همین حالا کلیک کنید. تبلیغات.")
        d, _, _ = decide(spam)
        self.assertFalse(d.passed)
        self.assertIn("spam_like_text", d.reasons)

    def test_short_natural_dialogue_sentence_accepted(self):
        d, _, _ = decide("سلام، امروز حالت چطوره؟", domain="dialogue")
        self.assertTrue(d.passed, d.reasons)

    def test_latin_technical_terms_not_automatically_rejected(self):
        d, _, _ = decide(GOOD + " واژه\u200cهای Python و API در متن فارسی آمده\u200cاند.")
        self.assertTrue(d.passed, d.reasons)
        self.assertGreater(d.features["latin_word_ratio"], 0)


if __name__ == "__main__":
    unittest.main()
