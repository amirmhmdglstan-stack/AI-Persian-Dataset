# Persian Pretraining Corpus — GPT-100M

A cleaned, deduplicated, quality-filtered Persian (Farsi) text corpus for training a
~100M-parameter GPT from scratch — plus the reproducible pipeline that built it.

**Start here → [`README.txt`](README.txt)** — full documentation: file format, cleaning
steps, licensing, limitations, and how to rebuild everything.

Every number below is measured by `pipeline/compute_statistics.py` and re-checked by
`pipeline/verify_dataset.py`; see `dataset_statistics.json` and
`reports/verification_report.json`.

## At a glance

| Metric | Value |
|---|---|
| Documents | 36,586 (all unique) |
| Characters | 45.0 M |
| Words | 8.49 M |
| Persian letters | 34.62 M (100 % of all letters — the corpus is pure Persian script) |
| Tokens (measured with the shipped tokenizer) | 11,350,954 |
| Size on disk | 82.11 MB (0.08 GiB) |
| Train / validation | 35,859 / 727 documents (97.9 % / 2.1 %) |
| Sources | 10 upstream corpora, 9 shipped + 1 rejected at the quality gate |
| Leakage between splits | 0 exact, 0 near-duplicate |
| Verification | 14 / 14 checks passed |

## Files

| File | Contents |
|---|---|
| `train.txt` | Training split — UTF-8, documents separated by a `<\|endoftext\|>` line |
| `validation.txt` | Held-out validation split, disjoint and deduplicated |
| `sources.json` | Every upstream corpus: URL, license, license URL, collection date, per-source counts and drop reasons |
| `dataset_statistics.json` | Full statistics (counts, source distribution, dedup, filters, tokens) |
| `README.txt` | Complete documentation (format, cleaning, licensing, limitations, reproduction) |
| `tokenizer/` | 32 k Persian BPE fitted on `train.txt` + exact token counts (optional convenience) |
| `reports/` | `verification_report.json` + `samples.txt` (random documents for human inspection) |
| `pipeline/` | fetch → extract → clean → dedup → split → statistics → tokenizer → verify → docs |

## Where the text comes from

| Source | Share | Type | License |
|---|---|---|---|
| Persian Wikipedia (2021-02-01 dump) | 78.4 % | encyclopedia | CC BY-SA 4.0 |
| Persian news corpus | 10.0 % | news | no license declared (publisher copyright) |
| ParsiNLU (QA, multiple choice, entailment, news passages) | 4.1 % | Q&A, reasoning, news | CC BY-NC-SA 4.0 |
| Ganjoor classical literature | 3.0 % | poetry / classical prose | GPL-3.0 repo, public-domain works |
| Chronological Persian poetry dataset | 2.0 % | poetry | CC BY-SA 4.0 |
| PeQa | 2.5 % | everyday dialogue | MIT |

`pn_summary_news` was downloaded and **rejected**: its upstream release has all sentence
punctuation stripped (0 of 11,185 articles contain a period), so it cannot teach sentence
boundaries. It is documented in `sources.json` as `rejected_by_quality_gate`.

## Pipeline

```bash
WORKDIR=/tmp/work bash pipeline/fetch_sources.sh        # 1. public GitHub sources only
python3 pipeline/extract_sources.py --workdir /tmp/work --out /tmp/work/extracted
python3 pipeline/build_dataset.py --extracted /tmp/work/extracted \
    --tmpdir /tmp/work/build --outdir . --budget 45000000   # 2. clean/dedup/split
python3 pipeline/make_tokenizer.py --out-dir tokenizer      # 3. tokenizer + exact tokens
python3 pipeline/compute_statistics.py --manifest /tmp/work/build/selected_manifest.tsv \
    --build-report /tmp/work/build/build_report.json \
    --token-stats tokenizer/token_stats.json --out dataset_statistics.json
python3 pipeline/verify_dataset.py                          # 4. 14 independent checks
python3 pipeline/make_sources_json.py --build-report /tmp/work/build/build_report.json \
    --extract-manifest /tmp/work/extracted/_extract_manifest.json --out sources.json
python3 pipeline/make_readme.py                             # 5. regenerate README.txt
```

Requirements: Python 3.9+ (standard library); `tokenizers` for step 3; `py7zr` to unpack
the Wikipedia archive. Runtime ≈ 35 minutes on 2 CPU cores.

## Cleaning, in one paragraph

Unicode NFKC, Arabic presentation forms folded to base letters, `ي→ی` / `ك→ک` / `ة→ه`,
digits standardized to `۰۱۲۳۴۵۶۷۸۹`, diacritics and tatweel removed, **ZWNJ preserved**,
markup/URLs/e-mails/entities/templates stripped, empty brackets and doubled punctuation
repaired, web boilerplate lines removed, back-to-back duplicated words and phrases
(wiki-link flattening artefacts) collapsed, then a quality gate on Persian-letter ratio,
sentence shape, fragment ratio, n-gram repetition and compression ratio, exact +
MinHash near-duplicate removal, length-stratified quality sampling for Wikipedia, and a
deterministic document-level train/validation split that is verified to have zero leakage.

## Licensing — read before use

The corpus is a compilation of differently licensed sources (CC BY-SA 4.0, MIT, GPL-3.0,
CC BY-NC-SA 4.0, publisher copyright). **Treat it as research / non-commercial material.**
`README.txt` §6 shows how to rebuild a permissive-only subset (Wikipedia + classical
literature + poetry + dialogue) for commercial use, and how to drop individual sources.

No website was crawled: every corpus was cloned from a public GitHub repository, so no
robots.txt rule was evaded, no access control bypassed and no personal data collected.
