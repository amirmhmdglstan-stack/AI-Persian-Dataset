#!/usr/bin/env python3
"""Small-sample end-to-end integration test for Persian Dataset V2.

Creates synthetic JSONL test fixtures only (NOT production training data),
runs unit tests, then exercises the full local pipeline:
build_corpus -> compute_statistics -> verify_dataset -> compare_tokenizers.
No external dataset is downloaded.  All outputs are labelled small_sample.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "dataset_v2" / "pipeline"
REPORTS = ROOT / "dataset_v2" / "reports"
FIXTURE = ROOT / "dataset_v2" / "tests" / "fixtures" / "representative_persian_sample.txt"
CONFIG = ROOT / "dataset_v2" / "configs" / "dataset_config.json"
TOKENIZER = ROOT / "dataset_v2" / "tokenizer" / "persian-bpe-32k.json"


def lines() -> list[str]:
    return [ln.strip() for ln in FIXTURE.read_text(encoding="utf-8").splitlines() if ln.strip()]


def paragraph(start: int, count: int, *, punctuation: bool = True) -> str:
    ls = lines()
    picked = [ls[(start + i) % len(ls)] for i in range(count)]
    text = "\n\n".join(picked)
    return text if punctuation else text.replace("؟", "").replace(".", "").replace("!", "")


def make_record(source_id: str, i: int, text: str, domain: str, title: str = "") -> dict:
    return {
        "source_id": source_id,
        "doc_id": f"{source_id}-{i:04d}",
        "title": title,
        "text": text,
        "profile": "A",
        "license": "CC0-1.0 small-sample test fixture",
        "domain": domain,
        "content_rights_verified": True,
        "meta": {
            "source_id": source_id, "profile": "A",
            "license": "CC0-1.0 small-sample test fixture",
            "domain": domain, "attribution": "synthetic unit-test fixture",
            "natural_text_only": True, "raw_doc_id": f"{source_id}-{i:04d}",
            "note": "Synthetic test fixture, not production training data.",
        },
    }


def build_fixtures(extracted: Path) -> int:
    out = extracted / "A"
    out.mkdir(parents=True, exist_ok=True)
    sources: dict[str, list[dict]] = {
        "smoke_encyclopedia": [],
        "smoke_news": [],
        "smoke_dialogue": [],
        "smoke_qa": [],
        "smoke_literary": [],
    }
    L = lines()
    # Formal encyclopedic prose: good, near-duplicate, exact duplicate, Latin,
    # markup, boilerplate, PII, English-only, spam, short and borderline.
    good = paragraph(0, 9)
    sources["smoke_encyclopedia"] += [
        make_record("smoke_encyclopedia", 0, good, "encyclopedia", "زبان فارسی"),
        make_record("smoke_encyclopedia", 1, paragraph(9, 9), "encyclopedia", "جغرافیا"),
        make_record("smoke_encyclopedia", 2, paragraph(18, 9), "encyclopedia", "دانشگاه"),
        make_record("smoke_encyclopedia", 3, paragraph(27, 9), "encyclopedia", "فناوری"),
        make_record("smoke_encyclopedia", 4, paragraph(36, 9), "encyclopedia", "پژوهش"),
        make_record("smoke_encyclopedia", 5, good, "encyclopedia", "تکرار دقیق"),
        make_record("smoke_encyclopedia", 6,
                    good.replace("زبان فارسی", "زبان پارسی").replace("کیفیت پیکره", "کیفیت این پیکره"),
                    "encyclopedia", "تکرار نزدیک"),
        make_record("smoke_encyclopedia", 7,
                    paragraph(45, 8) + " واژه\u200cهای Python و API نیز در متن فارسی آمده\u200cاند.",
                    "encyclopedia", "واژه\u200cهای فنی"),
        make_record("smoke_encyclopedia", 8,
                    "<div>" + paragraph(9, 8) + " نشانی https://example.ir</div>",
                    "encyclopedia", "نشانه\u200cگذاری"),
        make_record("smoke_encyclopedia", 9,
                    paragraph(18, 8) + " رایانامه test.user@example.com و شماره 09123456789.",
                    "encyclopedia", "داده شخصی"),
        make_record("smoke_encyclopedia", 10,
                    "This is an English-only document with many words and punctuation. "
                    "It should be rejected because the objective is Persian-only pretraining. "
                    "There is no Persian prose here at all.",
                    "encyclopedia", "English only"),
        make_record("smoke_encyclopedia", 11,
                    paragraph(27, 6) + " همین حالا کلیک کنید. تخفیف ویژه. فروش ویژه. تبلیغات.",
                    "encyclopedia", "spam"),
        make_record("smoke_encyclopedia", 12, "سلام", "encyclopedia", "کوتاه"),
        make_record("smoke_encyclopedia", 13,
                    paragraph(36, 4, punctuation=False), "encyclopedia", "مرزی"),
        make_record("smoke_encyclopedia", 14,
                    "کد خبر ۱۲۳۴۵\nکلیه حقوق محفوظ است.\n" + paragraph(0, 6),
                    "encyclopedia", "ناوبری"),
    ]
    # Modern news-style prose.
    for i in range(6):
        sources["smoke_news"].append(
            make_record("smoke_news", i, paragraph(45 + i, 7), "news", f"گزارش {i + 1}"))
    sources["smoke_news"].append(
        make_record("smoke_news", 99, "Breaking news in English only. " * 20, "news", "انگلیسی"))
    # Natural short dialogue / QA sentences, each kept as its own document.
    for i in range(14):
        sources["smoke_dialogue"].append(
            make_record("smoke_dialogue", i, L[(36 + i) % len(L)], "dialogue"))
    sources["smoke_dialogue"].append(
        make_record("smoke_dialogue", 98, "This is English only.", "dialogue"))
    sources["smoke_dialogue"].append(
        make_record("smoke_dialogue", 99, "شماره تماس 09123456789 است.", "dialogue"))
    for i in range(8):
        sources["smoke_qa"].append(
            make_record("smoke_qa", i, L[(50 + i) % len(L)], "qa"))
    sources["smoke_qa"].append(
        make_record("smoke_qa", 98, "سلام", "qa"))
    # Literary / verse-like documents with ZWNJ and no mandatory sentence end.
    for i in range(6):
        text = "\n".join(L[(20 + i * 3 + j) % len(L)] for j in range(7))
        sources["smoke_literary"].append(
            make_record("smoke_literary", i, text, "literary", f"دفتر {i + 1}"))
    sources["smoke_literary"].append(
        make_record("smoke_literary", 99, "Only English text in a verse slot.", "literary"))

    n = 0
    for sid, recs in sources.items():
        path = out / f"{sid}.jsonl"
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":")) + "\n")
                n += 1
    return n


def run(cmd: list[str], label: str, log: list[str]) -> None:
    log.append(f"\n===== {label} =====\n$ " + " ".join(cmd) + "\n")
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    log.append(p.stdout)
    if p.stderr:
        log.append("\n[stderr]\n" + p.stderr)
    log.append(f"[exit_code] {p.returncode}\n")
    if p.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {p.returncode}\n"
                           + p.stdout[-4000:] + "\n" + p.stderr[-4000:])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--workdir", default=str(ROOT / "dataset_v2" / ".smoke_work"))
    ap.add_argument("--tokenizer-sample", default=str(FIXTURE),
                    help="Persian sample for the tokenizer comparison; use final train.txt on the user's PC")
    ap.add_argument("--tokenizer-max-chars", type=int, default=20000,
                    help="cap characters used for the small tokenizer comparison")
    args = ap.parse_args()
    py = args.python
    work = Path(args.workdir)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    extracted = work / "extracted"
    outdir = work / "final"
    REPORTS.mkdir(parents=True, exist_ok=True)
    log: list[str] = ["Persian Dataset V2 small-sample integration test\n",
                      f"Python: {py}\n", f"Workdir: {work}\n",
                      "Scope: small_sample. No external corpus was downloaded.\n"]

    n = build_fixtures(extracted)
    log.append(f"Generated {n} synthetic test-fixture documents (NOT production training data).\n")

    # 1. Unit tests.
    run([py, "-m", "unittest", "discover", "-s", "dataset_v2/tests", "-v"],
        "unit tests", log)
    unit_text = "\n".join(log[-3:])
    (REPORTS / "unit_test_output.txt").write_text(unit_text, encoding="utf-8")

    # 2. Full local pipeline on the small fixtures.
    run([py, str(PIPELINE / "build_corpus.py"),
         "--config", str(CONFIG), "--workdir", str(work),
         "--extracted", str(extracted), "--outdir", str(outdir),
         "--reports-dir", str(REPORTS), "--profiles", "A",
         "--force", "--scope", "small_sample", "--min-free-gb", "1"],
        "build_corpus (small sample)", log)
    run([py, str(PIPELINE / "compute_statistics.py"),
         "--config", str(CONFIG), "--workdir", str(work), "--outdir", str(outdir),
         "--reports-dir", str(REPORTS), "--tokenizer", str(TOKENIZER),
         "--profiles", "A", "--scope", "small_sample",
         "--out", str(REPORTS / "dataset_statistics.json")],
        "compute_statistics (measured small sample)", log)
    run([py, str(PIPELINE / "verify_dataset.py"),
         "--config", str(CONFIG), "--workdir", str(work), "--outdir", str(outdir),
         "--reports-dir", str(REPORTS), "--profiles", "A", "--scope", "small_sample",
         "--out", str(REPORTS / "verification_report_v2.json")],
        "verify_dataset (small sample)", log)
    run([py, str(PIPELINE / "compare_tokenizers.py"),
         "--config", str(CONFIG), "--sample", str(args.tokenizer_sample),
         "--tokenizer-baseline", str(TOKENIZER), "--reports-dir", str(REPORTS),
         "--out", str(REPORTS / "tokenizer_comparison.json"),
         "--max-chars", str(args.tokenizer_max_chars), "--scope", "small_sample"],
        "compare_tokenizers (small sample)", log)

    # Small, explicit summary of measured values; full statistics stay in the
    # larger dataset_statistics.json report.
    stats = json.loads((REPORTS / "dataset_statistics.json").read_text(encoding="utf-8"))
    t = stats["totals"]
    summary = {
        "measurement_scope": "small_sample",
        "honesty_note": "These numbers were measured on synthetic test fixtures only. They are NOT full-corpus results.",
        "generated_at_utc": stats["generated_at_utc"],
        "fixture_documents_generated": n,
        "documents_in_final_files": t["documents"],
        "unique_documents": t["unique_documents"],
        "characters": t["characters"],
        "words": t["words"],
        "persian_letters": t["persian_letters"],
        "latin_letters": t["latin_letters"],
        "zwnj": t["zwnj_frequency"],
        "content_tokens_measured_with_v1_32k": t["tokens_content_excluding_document_end"],
        "tokens_including_document_end": t["tokens_including_document_end"],
        "characters_per_token": t["characters_per_token"],
        "bytes_per_token": t["bytes_per_token"],
        "train_documents": stats["splits"]["train"]["documents"],
        "validation_documents": stats["splits"]["validation"]["documents"],
        "test_documents": stats["splits"]["test"]["documents"],
        "source_distribution": stats["source_distribution"],
        "license_distribution": stats["license_distribution"],
        "rejection_reasons": stats["quality_and_dedup"]["rejection_reasons"],
        "exact_duplicates_removed": stats["quality_and_dedup"]["exact_duplicates_removed"],
        "near_duplicates_removed": stats["quality_and_dedup"]["near_duplicates_removed"],
        "verification": {
            "checks": json.loads((REPORTS / "verification_report_v2.json").read_text(
                encoding="utf-8"))["checks_run"],
            "failed": json.loads((REPORTS / "verification_report_v2.json").read_text(
                encoding="utf-8"))["failed"],
        },
        "reports": {
            "dataset_statistics": "dataset_v2/reports/dataset_statistics.json",
            "quality_report": "dataset_v2/reports/quality_report.json",
            "sample_audit": "dataset_v2/reports/sample_audit.txt",
            "pipeline_benchmark": "dataset_v2/reports/pipeline_benchmark.json",
            "verification_report": "dataset_v2/reports/verification_report_v2.json",
            "tokenizer_comparison": "dataset_v2/reports/tokenizer_comparison.json",
        },
    }
    (REPORTS / "small_sample_statistics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksums = outdir / "checksums.json"
    if checksums.exists():
        (REPORTS / "small_sample_checksums.json").write_text(
            checksums.read_text(encoding="utf-8"), encoding="utf-8")
    (REPORTS / "integration_test_output.txt").write_text("".join(log), encoding="utf-8")
    print("SMALL-SAMPLE INTEGRATION TEST PASSED")
    print(json.dumps({
        "fixture_documents": n,
        "final_documents": t["documents"],
        "content_tokens": t["tokens_content_excluding_document_end"],
        "verification_failed": summary["verification"]["failed"],
        "reports": str(REPORTS),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
