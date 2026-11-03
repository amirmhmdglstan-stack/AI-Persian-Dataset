# Persian Dataset V2 — External Source Research Report

**Research date:** 2026-11-02  
**Status:** research and pipeline design, **not** a full-corpus verification.  
**Hardware honesty:** this report was produced on a limited Linux/Python 3.11 machine. No 100 GB+ dataset, no complete Wikipedia dump and no full corpus were downloaded or processed here. Only dataset cards, GitHub metadata/repository files, official Wikimedia dump indexes, the V1 branch's scripts/metadata, a 1.7 MB V1 validation sample, and small synthetic fixtures were inspected. Windows 11 / Python 3.12 execution remains for the user's PC.

> This document is engineering research, not legal advice. A public URL, an MIT/ODC-BY/CC label on a repository, or an empty license field is **not** evidence that every underlying text document is licensed for our use.

---

## 1. Repository state and V1 baseline

The current session branch is `arena/b2bcd196-ai-persian-dataset`, based on `main`'s initial commit `a770ebfc33b4d72c26b9655084b45ec13ef5dacc`. Its working tree initially contained only `README.md`. The V1 corpus is on a **different remote branch**, `arena/f01a4e1e-ai-persian-dataset`, at the exact commit named by the user: `8e35d5e7e72597eda96e83db9c01a1335f235e19`. V1 files were fetched read-only through the GitHub API for inspection; they were not modified, copied into the current branch's root, or recommitted. The only V1 artifact copied into the new V2 tree is the 3.1 MB tokenizer JSON, byte-identical to the V1 blob (`sha256 8bdecc85…`), because it is the required working baseline.

V1's independently verified statistics were 36,592 documents, 44,999,973 characters, 8,493,555 words and 11,350,801 measured content tokens (about 11.39 M including one document-ending token per document). See `sources/v1_baseline.json` and `reports/v1_pipeline_audit.md`.

The V1 `sources.json` reports 538,849,962 characters after quality filtering but only 44,999,973 shipped characters. The 493,849,989-character difference is the "recoverable" pool the V2 task asks us to investigate. It is **not** a legally homogeneous pool.

### 1.1 Breakdown of V1's 538.85 M filtered characters

Measured from V1's `sources.json` (values are V1's own measured filtered/shipped counts, not V2 measurements):

| V1 source group | Filtered chars | Share of filtered | Shipped chars | Unshipped chars | V2 classification |
|---|---:|---:|---:|---:|---|
| Persian Wikipedia 2021 mirror | 457,109,769 | 84.83% | 35,298,596 | 421,811,173 | A — CC BY-SA 4.0 / GFDL |
| Chronological Persian Poetry | 41,305,942 | 7.67% | 899,832 | 40,406,110 | A — CC BY-SA 4.0 compilation / PD works |
| Ganjoor classical subset | 6,342,366 | 1.18% | 1,349,768 | 4,992,598 | A conditional — PD works, GPL-3.0 data files |
| ParsiNLU (QA/MC/entailment/RC) | 4,061,854 | 0.75% | 1,827,229 | 2,234,625 | B / C — CC BY-NC-SA; RC passages publisher-copyright |
| PeQa dialogue | 4,778,181 | 0.89% | 1,124,702 | 3,653,479 | C — Twitter-derived personal content; MIT repo only |
| Persian news (5 agencies) | 25,251,850 | 4.69% | 4,499,846 | 20,752,004 | C — no declared license, publisher copyright |
| pn-summary news | 0 | 0% | 0 | 0 | C — rejected by V1 quality gate; publisher copyright |
| **Total** | **538,849,962** | **100%** | **44,999,973** | **493,849,989** | — |

**Core-like A total:** 504,758,077 filtered characters (93.67%), of which 467,209,881 were unshipped. At V1's measured 3.991 characters/token, that is at most **≈126.5 M estimated tokens** of core-like material overall and **≈117.1 M estimated additional tokens** relative to V1. These are **estimates**, not measured V2 tokens: V2's stricter normalizer, PII gate, quality gate, global deduplication, source caps and fresh 2026 dumps will change them.

**The diversity problem:** 84.83% of V1's filtered characters were Wikipedia. Non-Wikipedia core-like V1 text was only 47,648,308 characters (≈11.9 M estimated tokens). Therefore a *balanced* 100–150 M-token corpus cannot be built from V1's rights-cleared core alone. If Wikipedia is capped near 50%, strict PROFILE A plausibly yields only **≈36–50 M measured tokens** after adding the small verified sources below; reaching 100–150 M in A would push Wikipedia back to roughly 70–85%, defeating the project's main goal. A research-only PROFILE B can add volume but carries non-commercial and privacy constraints.

---

## 2. Primary research leads

### 2.1 SLPL/naab — QUARANTINE

The Hugging Face card describes naab as ~130 GB, ~250 M paragraphs and ~15 B words, with 32,672,067 rows / 137 GB total file size and MIT metadata [1](https://huggingface.co/datasets/SLPL/naab). The raw dataset card is also MIT-labelled [19](https://huggingface.co/datasets/SLPL/naab-raw). The published paper's own license is CC BY-NC 4.0 for the paper, not a grant over the corpus text [2](https://www.researchgate.net/publication/363084493_naab_A_ready-to-use_plug-and-play_corpus_for_Farsi). The dataset card's "Licensing Information" section literally says **`mit?`** and its source list includes Common Crawl, MirasText, W2C, Wikipedia, Leipzig, VOA, poems, TEP, OSCAR-fa, Telegram channels, LSCP and a formerly private ASR Gooyesh Pardaz corpus. It also states that the curators "take no responsibility for personal information included in this corpus" ([naab raw README](https://huggingface.co/datasets/SLPL/naab/raw/main/README.md)). The journal version says filtering attempted to remove phone numbers, addresses and similar data but that identifiable data may still exist [29](https://journalaiai.com/index.php/aiai/article/download/7/7/41).

**Decision: PROFILE C quarantine.** The MIT metadata is ambiguous, does not license crawled third-party pages, Telegram messages or formerly private company data, and privacy risk is documented. Do not download the 126 ~1 GiB train shards on the user's PC ([file list](https://huggingface.co/api/datasets/SLPL/naab/tree/main/data)).

### 2.2 shekar-ai/Shiraz — QUARANTINE

Shiraz has 6,254,329 rows / 5.97 GB across 14 sites: Civilica, fa.wikipedia.org, Mihanblog, Virgool, Blogsky, Tabnak, Blogfa, Hidoctor, Digikala, Zoomit, BBC Persian, Magiran, Tejarate Farda and Irandeserts ([dataset card](https://huggingface.co/datasets/shekar-ai/Shiraz/raw/main/README.md), [28](https://huggingface.co/datasets/shekar-ai/Shiraz)). The compilation is ODC-BY 1.0, but the card explicitly says this covers selection/structure/annotations and **"does not grant rights to the underlying web content itself."** It also contains LLM-generated synthetic retrieval queries, which must never enter V2 pretraining. BBC written material requires permission for reuse [18](https://www.bbc.co.uk/archiveservices/written-archives-centre/licensing-and-copyright). Civilica academic abstracts, news articles and blog posts have separate rights. The Parquet files range from 8 MB to 1.6 GB ([file list](https://huggingface.co/api/datasets/shekar-ai/Shiraz/tree/main/data)).

**Decision: PROFILE C quarantine.** Use the official Wikipedia dump instead of Shiraz's Wikipedia rows; do not ingest BBC/Civilica/blog text or synthetic query columns.

---

## 3. Rights-cleared sources worth using (PROFILE A — licence conditions apply)

Wikimedia's Terms of Use license user-contributed text under CC BY-SA 4.0 and GFDL [7](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use); the official dump legal page confirms that original text is GFDL/CC BY-SA 4.0, with **Wikinews under CC BY 2.5** [8](https://dumps.wikimedia.org/legal.html), [31](https://en.wikipedia.org/wiki/Wikinews). Attribution and share-alike obligations apply; this is not a legal guarantee for every edge case.

| Source | Exact dated file | Compressed size | Rights applying to text | V2 role |
|---|---|---:|---|---|
| Persian Wikipedia 2026-10-01 | [`fawiki-20261001-pages-articles.xml.bz2`](https://dumps.wikimedia.org/fawiki/20261001/fawiki-20261001-pages-articles.xml.bz2) | 1,456,390,218 bytes | CC BY-SA 4.0 / GFDL | primary modern factual prose, capped |
| Persian Wikibooks 2026-11-01 | [`fawikibooks-20261101-pages-articles.xml.bz2`](https://dumps.wikimedia.org/fawikibooks/20261101/fawikibooks-20261101-pages-articles.xml.bz2) | 7,086,756 bytes | CC BY-SA 4.0 / GFDL | educational/explanatory prose |
| Persian Wikisource 2026-11-01 | [`fawikisource-20261101-pages-articles.xml.bz2`](https://dumps.wikimedia.org/fawikisource/20261101/fawikisource-20261101-pages-articles.xml.bz2) | 47,719,980 bytes | per-page PD or compatible free license; wiki contributions CC BY-SA 4.0 | literary/historical prose, PD-filtered |
| Persian Wikinews 2026-11-01 | [`fawikinews-20261101-pages-articles.xml.bz2`](https://dumps.wikimedia.org/fawikinews/20261101/fawikinews-20261101-pages-articles.xml.bz2) | 2,627,858 bytes | CC BY 2.5 (attribution, no share-alike) | modern news-style prose |
| V1 Wikipedia 2021 mirror, optional | [`hsmirzaie/fawiki`](https://github.com/hsmirzaie/fawiki) split 7z | ~186 MB repo | CC BY-SA 4.0 / GFDL; no repo license | audit/recovery of V1's unshipped text |

The dated dump SHA-1 values and exact URLs are recorded in `sources/source_metadata.json` from the official `dumpstatus.json` indexes ([fawiki](https://dumps.wikimedia.org/fawiki/20261001/dumpstatus.json), [fawikibooks](https://dumps.wikimedia.org/fawikibooks/20261101/dumpstatus.json), [fawikisource](https://dumps.wikimedia.org/fawikisource/20261101/dumpstatus.json), [fawikinews](https://dumps.wikimedia.org/fawikinews/20261101/dumpstatus.json)). Persian Wikipedia had about 1.10 M articles in late 2026 [20](https://en.wikipedia.org/wiki/Persian_Wikipedia); the live site API returned 1,100,753 articles during this research ([API](https://fa.wikipedia.org/w/api.php?action=query&meta=siteinfo&siprop=statistics&format=json)).

Wikisource's policy requires every work to be public domain or freely licensed [12](https://en.wikisource.org/wiki/Wikisource:COPY). V2 therefore uses an **exact configurable allowlist** in `configs/source_policy.json`: verified PD, CC BY, CC BY-SA, GFDL and other compatible template names are accepted; NC/ND/incompatible templates, generic `Copyright` / `مجوز` / `پروانه`, ambiguous and unverified licence-like templates are rejected with an exact reason. The actual matched template is stored in document metadata. This check is never weakened for corpus size.

Other A sources:

* **Mozilla Common Voice Persian sentence text (QUARANTINED pending per-file rights review).** Mozilla states Common Voice voice clips and associated text are CC0 [5](https://www.mozillafoundation.org/en/blog/common-voice-18-dataset-release/), and the repository README says sentence text in `server/data` comes from Sentence Collector / Wikipedia extraction and is CC0 even though the web app code is MPL-2.0 [6](https://github.com/common-voice/common-voice). The 21 commit-pinned Persian sentence files (~2.8 MB) are excluded from training pending file-level provenance review; no audio or speaker metadata is used.
* **Tatoeba Persian sentences (CC BY 2.0 FR).** Tatoeba's default sentence licence is CC BY 2.0 FR and author attribution is required [10](https://tatoeba.org/en/downloads), [11](https://tatoeba.org/en/terms_of_use). The official Persian-only directory lists `pes_sentences_detailed.tsv.bz2`, 695,149 bytes, dated 2026-10-31, with detailed fields `id`, `lang`, `text`, `username`, `date_added`, `date_last_modified` [32](https://downloads.tatoeba.org/exports/per_language/pes/). The language code is ISO 639-3 **`pes`**, not `fas`. The CC0-only export does **not** list Persian. V2 prefers this bzip2 TSV file, parses real MySQL OUTFILE tab escapes [34](https://raw.githubusercontent.com/ai-pipestream/turboembed/36930780a93e2ea0a647b56f69d5e0cfaa554091/analysis/train-recipes/src/tatoeba.rs), filters `pes`, keeps sentence ID/text/username/dates/source filename and Tatoeba CC BY 2.0 FR attribution in metadata (usernames never in training text), and emits one sentence per document. The weekly file is mutable: `download_manifest.json` records the observed snapshot date, actual SHA-256 and size; ordinary resume does not refresh a verified snapshot and `--force` is the explicit refresh. If the Persian-only file is unavailable, the optional fallback parses the real all-language `sentences_detailed.tar.bz2` member `sentences_detailed.csv` [33](https://github.com/ai-pipestream/turboembed/releases/tag/data-tatoeba-2026-10-10); it does **not** search for imaginary per-language TSV members.
* **Ganjoor classical subset.** The repository is GPL-3.0 ([LICENSE](https://github.com/ganjoor/ganjoor-tex/blob/master/LICENSE)); the underlying classical works are centuries old. Iranian copyright summaries indicate life+50 years after a 2010 amendment and 30 years for some categories [9](https://en.wikipedia.org/wiki/Template:PD-Iran), [25](https://www.harakenzo.com/en/service/iran/). V2 allows only the 12 classical poet directories at the pinned commit, one natural source file per document, and records the GPL/PD attribution. Modern poets must not be added without rights review.
* **Chronological Persian Poetry (CC BY-SA 4.0).** The GitHub repository declares CC-BY-SA-4.0; underlying classical poems are public domain ([repo](https://github.com/aghasemi/ChronologicalPersianPoetryDataset)). V2 emits one poem per document instead of V1's artificial 3,200-character blocks.
* **V1's unshipped Wikipedia/poetry/Ganjoor remainder.** Recoverable through the same public GitHub sources; see the table in §1.1.

---

## 4. Research-only sources (PROFILE B)

**ParsiNLU** is explicitly CC BY-NC-SA 4.0 ([repository LICENSE](https://github.com/persiannlp/parsinlu/blob/master/LICENSE)); its HF task cards repeat this [30](https://huggingface.co/datasets/persiannlp/parsinlu_query_paraphrasing/blob/main/README.md). V2 places ParsiNLU's natural question/answer/entailment text in a physically separate `profile_B` output, excludes reading-comprehension passages taken from publisher-copyright news, excludes machine-translated test splits, and does **not** wrap text in synthetic `پرسش:`/`پاسخ:`/`گزینه‌ها:` templates. Non-commercial use only.

**Targoman/TLPC** is CC BY-NC-SA 4.0, ~98.6 GB / 75 M documents / 41 B claimed tokens scraped from 800+ sites [14](https://huggingface.co/datasets/Targoman/TLPC), and has been used for Persian-Phi pretraining [27](https://www.researchgate.net/publication/398476043_Persian-Phi_Efficient_Cross-Lingual_Adaptation_of_Compact_LLMs_via_Curriculum_Learning). However, its card documents reader comments with commenter names/dates, i.e. identifiable personal data, and the underlying site rights are not established here. **Default decision: PROFILE C quarantine.** It may become conditional B only after documented PII scrubbing, rights review, and separate output.

Other candidates considered and **not approved**:

| Candidate | Size / nature | Why excluded or quarantined |
|---|---|---|
| MirasText | 2.8 M articles / 1.4 B words from 250+ news sites [4](https://nlpdataset.ir/farsi/raw_text_corpora.html) | MIT repo only; full ZIP is password-controlled by email request; publisher-copyright articles. Do not bypass access control. |
| persiannlp/persian-raw-text | ~70–80 GB, mostly Common Crawl [4](https://nlpdataset.ir/farsi/raw_text_corpora.html) | no license, mixed rights, no dedup; Common Crawl ToU pushes copyright responsibility to the user [13](https://commoncrawl.org/terms-of-use), [26](https://groups.google.com/g/common-crawl/c/wZwjDoxEilw). |
| OSCAR-fa / CC-100-fa / CulturaX-fa / FineWeb-2-fa | GB–TB Common-Crawl-derived subsets | Dataset compilation licenses do not grant rights to crawled pages [13](https://commoncrawl.org/terms-of-use). |
| Hamshahri corpus | newspaper articles [15](https://www.kaggle.com/datasets/ehsankhani/hamshahri-corpus) | license unknown, publisher copyright. |
| Bijankhan / Peykare, LSCP | tagged/colloquial corpora [4](https://nlpdataset.ir/farsi/raw_text_corpora.html) | authoritative license not established; social-text provenance unclear. |
| PeQa, V1 Persian news, pn-summary | see §1.1 | Twitter-derived personal content / no declared license / publisher copyright; V1 MIT or no-license claims do not cover text. |
| Esposito EN↔FA scientific parallel corpus | 3.5 M pairs / 1.43 GB, CC BY-NC-SA 4.0 [16](https://huggingface.co/datasets/universitytehran/ESPOSITO) | bilingual parallel objective, NC, not Persian-only natural documents. |
| PersianPunc | 17 M samples from Bijankhan/medical QA/Wikipedia, CC BY 4.0 compilation [17](https://huggingface.co/datasets/MohammadJRanjbar/PersianPunc) | upstream rights unresolved; task-specific fragments. |
| PersianVox / ManaTTS / ParsiGoo / Common Voice audio mirrors | 112 GB audio-derived transcripts etc. [22](https://huggingface.co/datasets/saeedzou/persianvox_all), [23](https://huggingface.co/datasets/MahtaFetrat/Mana-TTS) | in-the-wild audio/video provenance, short ASR utterances, privacy risk; not natural documents. |
| Dehkhoda dictionary | CC BY-NC-SA 4.0 compilation [24](https://huggingface.co/datasets/n9e6y/persian-dehkhoda-dictionary) | dictionary fragments, NC. |
| Iranian official legal texts | possible PD official texts [9](https://en.wikipedia.org/wiki/Template:PD-Iran) | no authoritative bulk open-licensed release verified here. |
| Project Gutenberg "Persian Literature" | English translations, PD in USA [21](https://www.gutenberg.org/ebooks/10315) | not Persian-language modern prose. |
| VOA Persian corpus, Internet Archive PD books | secondary listings [4](https://nlpdataset.ir/farsi/raw_text_corpora.html) | rights/OCR not verified per work; quarantine. |

---

## 5. Proposed V2 source distribution (targets, not quotas)

`configs/dataset_config.json` encodes **investigation targets** applied only after quality filtering and global deduplication:

| Domain | Target share | Verified sources available in A |
|---|---:|---|
| Modern general / factual prose | 40–60% | 2026 Wikipedia (capped), Wikinews, V1 Wikipedia remainder |
| Educational / scientific / explanatory | 10–20% | Wikibooks, Wikipedia science/technology subsets, ParsiNLU in B |
| Narrative / literary prose | 10–20% (target, not measured) | Wikisource PD-filtered, Ganjoor PD subset; poetry collection quarantined |
| Natural dialogue / QA | 5–15% (target, not measured) | Tatoeba CC BY, ParsiNLU in B; Common Voice quarantined |
| Other high-quality domains | remainder | small Wikimedia projects |

**Honest caveat:** with strict licensing, the 5–15% dialogue target and the 10–20% educational target are unlikely to be reached in PROFILE A: Common Voice/Tatoeba are only a few MB, Wikibooks is 7 MB compressed, and ParsiNLU is non-commercial. The pipeline reports unfilled capacity rather than duplicating documents or admitting quarantined web text. PROFILE B can improve the mix, but remains research-only.

---

## 6. Estimated recoverable text and V2 tokens (ESTIMATES, not measured)

| Quantity | Value | Basis / label |
|---|---:|---|
| V1 quality-filtered characters | 538,849,962 | V1 `sources.json`, measured by V1 |
| V1 shipped characters | 44,999,973 | V1 files, independently verified |
| V1 unshipped characters | 493,849,989 | arithmetic |
| V1 core-like A characters | 504,758,077 | Wikipedia + poetry + Ganjoor |
| V1 non-Wikipedia core-like A characters | 47,648,308 | arithmetic |
| Estimated tokens from all V1 core-like A chars | **≈126.5 M** | 504,758,077 / V1 measured 3.991 chars/token — **estimate** |
| Estimated additional tokens recoverable from V1 A | **≈117.1 M** | 467,209,881 / 3.991 — **estimate** |
| Realistic balanced PROFILE A V2 corpus | **≈36–50 M tokens** | **estimate**, because non-Wiki verified sources are small and Wikipedia is capped |
| PROFILE A upper bound if Wikipedia dominates | **≈100–150 M tokens** | **estimate**; would reintroduce ~70–85% Wikipedia skew |
| PROFILE B (NC, rights-cleared) potential | 100–150 M+ tokens | **speculative**; requires PII/rights work on TLPC/web corpora |

The official 2026 Wikipedia dump is 1.456 GB compressed; its uncompressed article text has **not** been measured here. A rough guess of 1–2 B plain-text characters would be 250–500 M estimated tokens, but that is not a measurement and the V2 Wikipedia cap deliberately limits its share. The user's PC must run `compute_statistics.py` to replace every estimate above with measured values.

---

## 7. Storage and download expectations (estimates)

* PROFILE A downloads: ~1.456 GB Wikipedia + ~57 MB other Wikimedia XML + a weekly Tatoeba export (observed Persian primary ~695 KB; fallback much larger) + ~289 MB Ganjoor repo + optional ~186 MB V1 mirror ≈ **2–3 GB**.
* PROFILE B adds ~28 MB ParsiNLU; TLPC (98.6 GB) and naab (137 GB) are **quarantined** and should not be downloaded. Shiraz is 5.97 GB but also quarantined.
* Working disk estimate for a 100–150 M-token run: raw 2–3 GB, extracted XML/JSONL ~6–12 GB, normalized shards ~2–6 GB, SQLite dedup index ~1–5 GB, final text ~0.8–1.2 GB, reports/checkpoints/logs <1 GB ⇒ **roughly 20–40 GB**, comfortably inside the user's ~194 GB free on D:. No GPU is needed.
* Agent-machine downloads were limited to metadata, small V1 scripts/validation and synthetic fixtures; no complete external dataset was fetched.

---

## 8. Recommendation

1. Run `windows/run_v2_tests.cmd` first on the user's PC to validate Windows/Python 3.12 behavior.
2. Run `windows/run_v2.cmd` for **PROFILE A only** and manually review `sample_audit.txt` before training.
3. Treat the resulting measured token count as the real V2 size; do not assume 100–150 M.
4. If the measured A corpus is too small or too Wikipedia-heavy, the next step is **rights acquisition**, not more scraping: obtain explicit licenses for modern Persian prose, educational text and dialogue. Only then consider a conditional PROFILE B.
5. Keep the V1 32K tokenizer as baseline until `compare_tokenizers.py` is rerun on a bounded 8–10 M-character sample of the final corpus; do not claim full-corpus memory or a 40 M-character bounded sample.

---

## References (citation index)

1. [1](https://huggingface.co/datasets/SLPL/naab) SLPL/naab dataset card.
2. [2](https://www.researchgate.net/publication/363084493_naab_A_ready-to-use_plug-and-play_corpus_for_Farsi) naab paper.
3. [4](https://nlpdataset.ir/farsi/raw_text_corpora.html) Persian raw-text corpus directory.
4. [5](https://www.mozillafoundation.org/en/blog/common-voice-18-dataset-release/) Mozilla Common Voice CC0 release.
5. [6](https://github.com/common-voice/common-voice) Common Voice repository / licensing note.
6. [7](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use) Wikimedia Terms of Use.
7. [8](https://dumps.wikimedia.org/legal.html) Wikimedia dump license information.
8. [9](https://en.wikipedia.org/wiki/Template:PD-Iran) Iran public-domain/copyright summary.
9. [10](https://tatoeba.org/en/downloads) Tatoeba downloads / CC BY 2.0 FR.
10. [11](https://tatoeba.org/en/terms_of_use) Tatoeba terms of use.
11. [12](https://en.wikisource.org/wiki/Wikisource:COPY) Wikisource copyright policy.
12. [13](https://commoncrawl.org/terms-of-use) Common Crawl Terms of Use.
13. [14](https://huggingface.co/datasets/Targoman/TLPC) Targoman/TLPC card.
14. [15](https://www.kaggle.com/datasets/ehsankhani/hamshahri-corpus) Hamshahri corpus, license unknown.
15. [16](https://huggingface.co/datasets/universitytehran/ESPOSITO) Esposito parallel corpus.
16. [17](https://huggingface.co/datasets/MohammadJRanjbar/PersianPunc) PersianPunc.
17. [18](https://www.bbc.co.uk/archiveservices/written-archives-centre/licensing-and-copyright) BBC written archives licensing.
18. [19](https://huggingface.co/datasets/SLPL/naab-raw) SLPL/naab-raw card.
19. [20](https://en.wikipedia.org/wiki/Persian_Wikipedia) Persian Wikipedia article count.
20. [21](https://www.gutenberg.org/ebooks/10315) Project Gutenberg Persian literature (English translations).
21. [22](https://huggingface.co/datasets/saeedzou/persianvox_all) PersianVox-All.
22. [23](https://huggingface.co/datasets/MahtaFetrat/Mana-TTS) ManaTTS.
23. [24](https://huggingface.co/datasets/n9e6y/persian-dehkhoda-dictionary) Dehkhoda dictionary.
24. [25](https://www.harakenzo.com/en/service/iran/) Iran copyright overview.
25. [26](https://groups.google.com/g/common-crawl/c/wZwjDoxEilw) Common Crawl clarification on crawled-content rights.
26. [27](https://www.researchgate.net/publication/398476043_Persian-Phi_Efficient_Cross-Lingual_Adaptation_of_Compact_LLMs_via_Curriculum_Learning) Persian-Phi / TLPC usage.
27. [28](https://huggingface.co/datasets/shekar-ai/Shiraz) shekar-ai/Shiraz.
28. [29](https://journalaiai.com/index.php/aiai/article/download/7/7/41) naab journal paper (privacy filtering statement).
29. [30](https://huggingface.co/datasets/persiannlp/parsinlu_query_paraphrasing/blob/main/README.md) ParsiNLU CC BY-NC-SA 4.0 card.
30. [31](https://en.wikipedia.org/wiki/Wikinews) Wikinews CC BY 2.5 content licence.
31. [32](https://downloads.tatoeba.org/exports/per_language/pes/) Official Tatoeba Persian-only per-language export index (verified filename, size and snapshot date).
32. [33](https://github.com/ai-pipestream/turboembed/releases/tag/data-tatoeba-2026-10-10) Archived Tatoeba weekly release documenting the real `sentences_detailed.csv` member.
33. [34](https://raw.githubusercontent.com/ai-pipestream/turboembed/36930780a93e2ea0a647b56f69d5e0cfaa554091/analysis/train-recipes/src/tatoeba.rs) Strict reader documenting Tatoeba MySQL OUTFILE fields and escapes.

Fetched directly (not search-result citations): [naab raw card](https://huggingface.co/datasets/SLPL/naab/raw/main/README.md), [Shiraz raw card](https://huggingface.co/datasets/shekar-ai/Shiraz/raw/main/README.md), [Shiraz file list](https://huggingface.co/api/datasets/shekar-ai/Shiraz/tree/main/data), [naab file list](https://huggingface.co/api/datasets/SLPL/naab/tree/main/data), [fawiki dump status](https://dumps.wikimedia.org/fawiki/20261001/dumpstatus.json), [fawikibooks dump status](https://dumps.wikimedia.org/fawikibooks/20261101/dumpstatus.json), [fawikisource dump status](https://dumps.wikimedia.org/fawikisource/20261101/dumpstatus.json), [fawikinews dump status](https://dumps.wikimedia.org/fawikinews/20261101/dumpstatus.json), [V1 sources.json](https://github.com/amirmhmdglstan-stack/AI-Persian-Dataset/blob/arena/f01a4e1e-ai-persian-dataset/sources.json), [V1 README.txt](https://github.com/amirmhmdglstan-stack/AI-Persian-Dataset/blob/arena/f01a4e1e-ai-persian-dataset/README.txt).

## 2026-11-03 correction to earlier estimates

The poetry and Common Voice fa entries previously listed in Profile A above are now **Profile C quarantine** pending per-work/per-sentence evidence; see `licensing_report.md`. Profile A volume and domain percentages above are historical *estimates* for a different source selection, not measurements of the current catalog. The poetry repository README explicitly identifies Ganjoor as the source. The pinned Common Voice README uses “majority” for CC0 `/server/data` text, not a guarantee for all 21 Persian files. No full downloads or content audit were performed.
