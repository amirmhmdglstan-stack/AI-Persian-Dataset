# Persian Dataset V2 — Licensing Architecture and Decisions

**Date:** 2026-11-02  
**Companion machine-readable files:** `sources/source_catalog.csv`, `sources/licensing_matrix.csv`, `sources/source_metadata.json`, `configs/source_policy.json`.

> **Not legal advice.** These are engineering classifications for a research pipeline. Rights may differ by jurisdiction, by work, by date and by the way a derived corpus or model is distributed. When documentation is missing or conflicting, V2 quarantines the source.

## 1. Governing rule

A public download link is not a license. A repository's MIT/GPL/ODC-BY/CC label does not automatically license every underlying text document. The V2 catalog therefore records **two separate license fields** for every source:

* `license_compilation`: what the dataset/repository/package claims;
* `license_content`: what we could verify about the **actual text**.

`content_rights_verified` is `false` unless authoritative documentation for the text was found. `configs/source_policy.json` makes this an enforced rule: PROFILE A requires verified content rights; PROFILE B requires documented non-commercial terms; PROFILE C is never downloaded or trained on.

## 2. Profiles

| Profile | Name | Default | Output | Training | Redistribution |
|---|---|---|---|---|---|
| A | RIGHTS-CLEARED CORE (LICENSE CONDITIONS APPLY) | yes | `profile_A/` | yes | only under each upstream licence's attribution / share-alike / source-specific conditions |
| B | RESEARCH ONLY | no | `profile_B/` | yes, non-commercial research | only under NC / SA terms |
| C | QUARANTINE | n/a | manifest only, **no text** | **no** | not granted |

`build_corpus.py` refuses `--profiles C` and writes a `profile_C_quarantine_manifest_only/quarantine_manifest.json` listing quarantined sources with reasons. A and B are physically separate directories and every final document has a `documents.jsonl` provenance record containing `source_id`, `license`, `domain`, `profile`, `cluster_id`, quality score and source metadata.

## 3. PROFILE A — rights-cleared, with conditions

| Source | Compilation license | Actual text rights | Conditions / risk |
|---|---|---|---|
| Persian Wikipedia 2026-10-01 official dump | CC BY-SA 4.0 / GFDL | CC BY-SA 4.0 / GFDL [7](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use), [8](https://dumps.wikimedia.org/legal.html) | attribute Wikipedia contributors; indicate modifications; share-alike for redistributed adaptations; keep attribution in manifest |
| Persian Wikibooks 2026-11-01 | CC BY-SA 4.0 / GFDL | same | same |
| Persian Wikisource 2026-11-01 | CC BY-SA 4.0 / GFDL wiki contributions | per-work public domain or compatible free licence [12](https://en.wikisource.org/wiki/Wikisource:COPY) | exact configurable allowlist: PD, CC BY, CC BY-SA, GFDL and other verified templates accepted; NC/ND/incompatible, generic Copyright/مجوز/پروانه, ambiguous and unknown licence-like templates rejected with exact reason; matched template stored in metadata |
| Persian Wikinews 2026-11-01 | CC BY 2.5 | CC BY 2.5 [8](https://dumps.wikimedia.org/legal.html), [31](https://en.wikipedia.org/wiki/Wikinews) | attribution; **no** share-alike, keep license distinct |
| Common Voice Persian sentence text | MPL-2.0 repository code | CC0 1.0 sentence text [5](https://www.mozillafoundation.org/en/blog/common-voice-18-dataset-release/), [6](https://github.com/common-voice/common-voice) | no audio, no speaker metadata; one natural sentence per document |
| Tatoeba Persian sentences | CC BY 2.0 FR weekly export | CC BY 2.0 FR by default [10](https://tatoeba.org/en/downloads), [11](https://tatoeba.org/en/terms_of_use), [32](https://downloads.tatoeba.org/exports/per_language/pes/) | verified `pes_sentences_detailed.tsv.bz2` (695,149 bytes, 2026-10-31) keeps sentence ID, Persian text, `pes` language code, contributor username, dates and source filename in metadata; CC BY 2.0 FR author attribution and licence notice retained for every row; usernames never enter training text; mutable weekly snapshot SHA-256/size recorded in download manifest |
| Ganjoor classical subset | GPL-3.0 data files ([LICENSE](https://github.com/ganjoor/ganjoor-tex/blob/master/LICENSE)) | classical works public domain; transcription/typesetting may be GPL-3.0 [9](https://en.wikipedia.org/wiki/Template:PD-Iran), [25](https://www.harakenzo.com/en/service/iran/) | only 12 classical directories at pinned commit; keep GPL/PD notices; no modern poets |
| Chronological Persian Poetry | CC BY-SA 4.0 ([repo](https://github.com/aghasemi/ChronologicalPersianPoetryDataset)) | public-domain classical poems | attribution + share-alike for compilation; one poem per document |
| V1 Wikipedia 2021 mirror (optional) | no repository license | Wikipedia text CC BY-SA 4.0 / GFDL | same as Wikipedia; used only for V1 recovery audit |

**Share-alike / model-weights caveat:** CC BY-SA 4.0 and GPL-3.0 can impose obligations on redistributed *derived data*. V2 keeps per-document licence tags, source-specific attribution and a provenance manifest. Trained model weights are **NOT** assumed to be automatically free of upstream attribution, share-alike, source-specific or redistribution obligations; whether weights or a redistributed cleaned corpus count as an adaptation is a legal question we do not answer. Review provenance and licence terms before distributing any model.

## 4. PROFILE B — research-only, non-commercial

| Source | Actual text rights | Conditions |
|---|---|---|
| ParsiNLU QA / multiple-choice / entailment | CC BY-NC-SA 4.0 annotations ([LICENSE](https://github.com/persiannlp/parsinlu/blob/master/LICENSE), [30](https://huggingface.co/datasets/persiannlp/parsinlu_query_paraphrasing/blob/main/README.md)) | non-commercial only; attribution + share-alike; separate `profile_B/` output; RC passages from publisher news and machine-translated test splits excluded; no synthetic QA templates |

`windows/run_v2_research.cmd` rebuilds A+B together so global deduplication still covers both profiles, then writes `profile_A/` and `profile_B/` separately. It must not be used for commercial deployment.

## 5. PROFILE C — quarantine / exclusion

| Source | Compilation label | Why it is not trainable in V2 |
|---|---|---|
| SLPL/naab, naab-raw | MIT metadata, card says `mit?` [1](https://huggingface.co/datasets/SLPL/naab), [19](https://huggingface.co/datasets/SLPL/naab-raw) | Common Crawl, MirasText, Telegram channels and formerly private company text; curators disclaim responsibility for personal information; a later paper admits residual identifiable data [29](https://journalaiai.com/index.php/aiai/article/download/7/7/41) |
| shekar-ai/Shiraz | ODC-BY 1.0 compilation [28](https://huggingface.co/datasets/shekar-ai/Shiraz) | card explicitly does not grant underlying web-content rights; BBC/Civilica/blog/publisher content; synthetic query columns; BBC reuse needs permission [18](https://www.bbc.co.uk/archiveservices/written-archives-centre/licensing-and-copyright) |
| Targoman/TLPC | CC BY-NC-SA 4.0 [14](https://huggingface.co/datasets/Targoman/TLPC) | 800+ scraped sites, documented reader comments with commenter names/dates; significant privacy risk; conditional B only after PII scrubbing and rights review |
| MirasText | MIT repository | 250+ news sites; password-controlled archive; do not request or bypass access control from the pipeline |
| persiannlp/persian-raw-text, OSCAR/CC-100/CulturaX/FineWeb-2 fa | no blanket content license | Common Crawl ToU says crawled pages may carry separate terms and users must respect third-party rights [13](https://commoncrawl.org/terms-of-use), [26](https://groups.google.com/g/common-crawl/c/wZwjDoxEilw) |
| Hamshahri, Bijankhan/Peykare, LSCP, VOA, unlicensed poem corpus | mixed / unknown | authoritative text license not established [15](https://www.kaggle.com/datasets/ehsankhani/hamshahri-corpus), [4](https://nlpdataset.ir/farsi/raw_text_corpora.html) |
| V1 `persian_news`, `pn_summary_news` | none declared / MIT repo | publisher-copyright news; V1's research-only label is not a license grant |
| V1 `peqa_dialogue` | MIT repo | Twitter-derived personal posts; privacy and underlying rights unverified |
| Esposito, PersianPunc, PersianVox/ManaTTS/ParsiGoo, Dehkhoda, Persian-GK, Wikiquote/Wiktionary, Gutenberg English translations, Iranian legal bulk text | mixed | NC, bilingual/parallel, task fragments, ASR/audio mining, dictionary entries, synthetic instruction data, quotations, non-Persian text, or unverified bulk rights [16](https://huggingface.co/datasets/universitytehran/ESPOSITO), [17](https://huggingface.co/datasets/MohammadJRanjbar/PersianPunc), [22](https://huggingface.co/datasets/saeedzou/persianvox_all), [23](https://huggingface.co/datasets/MahtaFetrat/Mana-TTS), [24](https://huggingface.co/datasets/n9e6y/persian-dehkhoda-dictionary), [21](https://www.gutenberg.org/ebooks/10315) |

## 6. Privacy and synthetic-text policy

* PII-like e-mails, phone numbers, national-ID/card-like digit strings cause **whole-document rejection**; removal is only a second safety layer. Rejection reasons are counted.
* No private messages, identifiable personal conversations, speaker identities or audio metadata are ingested.
* `query_*` / `judge_*` and other generated fields are excluded from Parquet/JSONL extraction. Shiraz's synthetic retrieval queries are never used.
* V1's synthetic `پرسش:` / `پاسخ:` / `گزینه‌ها:` QA wrappers and artificial 1,400/3,200-character blocks are not reproduced; V2 keeps natural source units.
* Manual audit samples are truncated and PII-redacted; the user must review them before training.

## 7. Attribution and reproducibility

`source_metadata.json` records exact URLs, dated dump SHA-1 checksums, pinned GitHub commit SHAs, license fields, privacy risk, rights-verification status and notes. `documents.jsonl` carries per-document provenance. `checksums.json` records SHA-256 of final files. The V1 tokenizer copy is verified against `sources/v1_baseline.json`. Nothing here is a legal guarantee; unresolved sources remain in PROFILE C.
