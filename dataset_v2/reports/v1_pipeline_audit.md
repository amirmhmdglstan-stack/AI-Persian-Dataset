# Persian Dataset V1 Pipeline Audit

**Audited commit:** `8e35d5e7e72597eda96e83db9c01a1335f235e19` on remote branch `arena/f01a4e1e-ai-persian-dataset`.  
**Current session branch:** `arena/b2bcd196-ai-persian-dataset` (based on `main`'s initial commit).  
**Method:** read-only GitHub API retrieval of V1 scripts, README, `sources.json`, `dataset_statistics.json`, reports and tokenizer. The 80.4 MB `train.txt` was **not** downloaded; V1's independently verified statistics were taken as given. V1 files were not modified.

Direct V1 file links: [fetch_sources.sh](https://github.com/amirmhmdglstan-stack/AI-Persian-Dataset/blob/arena/f01a4e1e-ai-persian-dataset/pipeline/fetch_sources.sh), [extract_sources.py](https://github.com/amirmhmdglstan-stack/AI-Persian-Dataset/blob/arena/f01a4e1e-ai-persian-dataset/pipeline/extract_sources.py), [build_dataset.py](https://github.com/amirmhmdglstan-stack/AI-Persian-Dataset/blob/arena/f01a4e1e-ai-persian-dataset/pipeline/build_dataset.py), [persian_norm.py](https://github.com/amirmhmdglstan-stack/AI-Persian-Dataset/blob/arena/f01a4e1e-ai-persian-dataset/pipeline/persian_norm.py), [verify_dataset.py](https://github.com/amirmhmdglstan-stack/AI-Persian-Dataset/blob/arena/f01a4e1e-ai-persian-dataset/pipeline/verify_dataset.py), [compute_statistics.py](https://github.com/amirmhmdglstan-stack/AI-Persian-Dataset/blob/arena/f01a4e1e-ai-persian-dataset/pipeline/compute_statistics.py), [make_tokenizer.py](https://github.com/amirmhmdglstan-stack/AI-Persian-Dataset/blob/arena/f01a4e1e-ai-persian-dataset/pipeline/make_tokenizer.py), [sources.json](https://github.com/amirmhmdglstan-stack/AI-Persian-Dataset/blob/arena/f01a4e1e-ai-persian-dataset/sources.json), [README.txt](https://github.com/amirmhmdglstan-stack/AI-Persian-Dataset/blob/arena/f01a4e1e-ai-persian-dataset/README.txt).

---

## 1. What V1 actually downloaded

`fetch_sources.sh` cloned seven public GitHub repositories:

| Repository | V1 source id(s) | Raw V1 docs / chars | Rights summary |
|---|---|---:|---|
| `hsmirzaie/fawiki` (2021-02-01 cleaned Wikipedia CSV) | `wikipedia_fa` | 864,889 / 621,656,144 | Wikipedia text CC BY-SA 4.0; mirror repo has no license file |
| `ganjoor/ganjoor-tex` | `ganjoor_classics` | 2,292 / 6,342,366 filtered | GPL-3.0 repository; classical works public domain |
| `aghasemi/ChronologicalPersianPoetryDataset` | `persian_poetry` | 14,348 / 41,305,942 filtered | CC BY-SA 4.0 compilation; classical poems PD |
| `persiannlp/parsinlu` | `parsinlu_qa`, `_mc`, `_entailment`, `_rc_passage` | ~14,309 raw | CC BY-NC-SA 4.0 annotations; RC passages from publisher news |
| `ZahraArshia/PeQa` | `peqa_dialogue` | 3,368 V1 blocks / 4,778,181 filtered chars | MIT repository; underlying Twitter posts unverified, privacy risk |
| `milad-4274/persian_news` | `persian_news` | 13,455 / 25,251,850 filtered | no repository license; publisher copyright |
| `hooshvare/pn-summary` | `pn_summary_news` | 11,185 extracted, 0 kept | MIT repository; publisher-copyright articles; upstream punctuation stripped |

V1 `sources.json` reports 925,316 raw documents / 736,630,689 raw characters, 274,860 documents / 538,849,962 characters after its quality gate, 317 exact + 4,521 near duplicates removed, and 36,592 shipped documents / 44,999,973 characters. The raw extracted files were **not committed**; only the scripts and small reports are in Git. They can be regenerated from the same public repositories, but the licenses above apply only to the text, not to the act of downloading.

## 2. Are the original extracted files still accessible?

Not as committed artifacts. `fetch_sources.sh` writes to `WORKDIR` (default `./work`), and `.gitignore` excludes `work/`, `build/`, `extracted/` and `*.jsonl`. On a fresh machine the pipeline re-clones the public GitHub repos and re-extracts. The Wikipedia mirror is a split 7z archive (~186 MB GitHub repo) requiring `py7zr` or `7z`; the other sources are CSV/TSV/JSONL/text trees. All seven repositories were reachable through the GitHub API during this audit. MirasText (a candidate, not a V1 shipped source) is password-controlled; V2 does not bypass that.

## 3. Which sources are worth expanding?

| Source | Worth expanding? | Reason |
|---|---|---|
| Wikipedia | **Yes, but cap it** | CC BY-SA/PD-compatible and large; V1 shipped only 35.3 M of 457.1 M filtered chars. Use the official 2026-10-01 dump rather than only the 2021 mirror. |
| Poetry / Ganjoor | **Moderately** | ~45 M filtered chars, but classical register; V1 shipped only ~2.25 M. V2 keeps natural units and caps share. |
| ParsiNLU | **Research-only** | clear CC BY-NC-SA, small, useful QA/educational register; must stay in PROFILE B. |
| Persian news / pn-summary | **No** | publisher copyright, no license; pn-summary also lacks sentence punctuation. |
| PeQa | **No** | Twitter-derived personal content; MIT repo does not license posts. |
| New official Wikimedia siblings | **Yes, small** | Wikibooks, Wikisource (PD-filtered) and Wikinews add educational, literary and news registers with documented rights. |
| Common Voice / Tatoeba | **Yes, small** | CC0 / CC BY 2.0 FR natural short sentences for dialogue register. |

## 4. Is V1's quality gate reliable enough?

Partly for layout, **no** as evidence of Persian purity or legal safety.

* `persian_norm.normalize()` replaces every character outside a Persian/Latin-free whitelist with a space, then `quality_check()` computes `latin_letters` and `max_latin_ratio`. Because all Latin letters were already deleted, the "100% Persian letters" statistic is an artifact of the whitelist, not a measurement of contamination. `verify_dataset.py`'s `low_non_persian_contamination` check is similarly tautological.
* `clean_lines()` drops lines with fewer than 3 Persian letters or shorter than 12 characters unless they end in punctuation. That deletes legitimate short dialogue turns and verse lines.
* `_RE_ADJACENT_DUP` collapses immediate repeated words in **all** sources, including verse/dialogue, where repetition can be natural. `_RE_PHRASE_DUP` also collapses short repeated phrases for prose sources. This can remove meaningful text.
* The gate's "language detection" is post-normalization script ratio, not a stopword/register check; English-only documents could be turned into short Persian-looking debris rather than explicitly classified.
* `require_sentence_end` and sentence-shape checks are reasonable for encyclopedic prose but were relaxed only for poetry; V1's QA/dialogue profiles still use synthetic wrappers and short blocks.
* V1 does not detect e-mails, phone numbers or national-ID-like strings as a privacy gate.
* `verify_dataset.py`'s `iter_documents()` yields only non-empty documents, so its `no_empty_documents` check cannot detect an empty document.
* Automated scores cannot establish semantic correctness, factual accuracy or absence of bias in any version.

## 5. Technical defects and scalability limits

1. **Linux-only fetch script.** `fetch_sources.sh` uses Bash, `cat fawiki.7z.00*`, `7z`/`py7zr` and `WORKDIR=/tmp/work`; it is not a Windows CMD/Python 3.12 pipeline.
2. **Unbounded Python memory.** `build_dataset.py` accumulates `all_rows` (one dict per kept document, ~274,860 in V1) and an in-memory `buckets` LSH dictionary. It also sorts the full row list. This does not scale to tens of millions of documents on 16 GB RAM.
3. **Near-dup LSH misses popular buckets.** `deduplicate()` skips a band entirely when `len(lst) > bucket_cap` (default 20,000), so the most crowded bands generate no candidates. V1 uses 16 CRC32 buckets, 4 bands and requires ≥12/16 equal buckets; this is a coarse approximation.
4. **Budget redistribution defeats the Wikipedia cap.** Source caps sum to 0.96, `pn_summary_news` contributes nothing, and the slack is given to the biggest provider. V1's `wikipedia_fa` cap was 0.65 but the shipped share was **78.44%**, with Wikipedia 35.3 M of 45 M characters. The README's proposed `--budget 450000000` would reproduce the same skew.
5. **Split leakage design.** V1 splits by SHA-1 of document id after dedup, but undetected near duplicates can still land on different sides. Its verification checks only 728 validation docs against train with MinHash LSH; there is no cluster-based split invariant and **no test split**.
6. **Artificial documents.** `extract_peqa` packs unrelated Twitter pairs into 1,400-character blocks and labels both turns `کاربر:`; `extract_ganjoor` merges unrelated poems into 3,200-character blocks; `extract_poetry` groups poems by poet; ParsiNLU records are wrapped in synthetic `پرسش:`/`پاسخ:`/`گزینه‌ها:`/`متن نخست:` templates. These are not naturally occurring documents and should not be reproduced in V2.
7. **Tokenizer counting defect.** `make_tokenizer.py::count_tokens()` loads all documents with `list(iter_corpus_n(...))` into memory (not streaming) and counts `chars += len(line) - 1` over **separator lines as well**. Hence `tokenizer/token_stats.json` reports 44,361,809 train characters while `dataset_statistics.json` reports 44,064,049 document-content characters. The token count itself is exact; the character count and memory behavior are not.
8. **No per-document provenance in final files.** `selected_manifest.tsv` lives in the uncommitted work dir. The shipped `train.txt`/`validation.txt` have no source/license metadata per document.
9. **No checksum of remote sources, no resumable downloads, no disk-space checks, no checkpointed processing.** `fetch_sources.sh` uses `git clone --depth 1` and concatenated 7z parts; a failed download restarts.
10. **No Windows-safe output guard.** V1 writes `train.txt`/`validation.txt` into the repository root by design; V2 must never do this.

## 6. Licensing defects in V1

V1's README correctly says the corpus is a mixed-license compilation and advises research/non-commercial use, but it still shipped 10% publisher-copyright news with no declared license, 2.5% Twitter-derived PeQa under an MIT repository license, and ParsiNLU RC passages from news sites under CC BY-NC-SA annotations. A "research use" label is not a license grant. Ganjoor's GPL-3.0 applies to its data files as well as software; the underlying classical texts are public domain but attribution/share-alike notices still matter. V2 separates these into A/B/C instead of mixing them in one `train.txt`.

## 7. How much additional useful text is realistically recoverable?

| Pool | Characters | Estimated tokens at V1's 3.991 chars/token | V2 status |
|---|---:|---:|---|
| All V1 unshipped filtered text | 493,849,989 | ≈123.7 M | **not** all usable |
| V1 core-like A unshipped (Wikipedia + poetry + Ganjoor) | 467,209,881 | ≈117.1 M | recoverable in principle, but 90% Wikipedia |
| V1 non-Wikipedia A unshipped | 45,398,708 | ≈11.4 M | small; caps still apply |
| ParsiNLU unshipped | 2,234,625 | ≈0.56 M | PROFILE B only |
| PeQa unshipped | 3,653,479 | ≈0.92 M | quarantine (privacy/rights) |
| Persian news unshipped | 20,752,004 | ≈5.20 M | quarantine (publisher rights) |

All token figures are **estimates**, not measured V2 results. After V2's stricter PII/quality gates, global disk-backed dedup, Wikipedia cap and fresh 2026 dumps, a *balanced* strict-A corpus is more plausibly **≈36–50 M measured tokens**; an upper-bound A corpus of **≈100–150 M** would again be mostly Wikipedia. The 2026 official Wikipedia dump (1,456,390,218 compressed bytes, ~1.10 M articles) adds modern text but its extracted character count has not been measured here.

## 8. Conclusion

V1 is a valid immutable baseline with a working 32K tokenizer and independently verified 11.35 M content tokens. Its claim that 538 M filtered characters can be regenerated is **technically plausible** but **legally and qualitatively misleading**: ~85% of that pool is Wikipedia, ~5% is unlicensed publisher news, ~1% is Twitter-derived personal content, and ~1% is NC task data. The V1 pipeline is not scalable, not Windows-native, not privacy-aware, not provenance-preserving, and its "pure Persian" verification is an artifact of deleting Latin characters. V2 replaces it with streaming Python, SQLite dedup, three profiles, three splits, cluster-consistent leakage prevention, measured Latin contamination, PII rejection, checksums and explicit estimated-vs-measured reporting.
