# Persian Dataset V2 — Known Limitations and Untested Operations

**Read this before training.** The V2 repository contains a designed, unit-tested and small-sample-tested pipeline. It does **not** contain a fully built or verified expanded corpus.

## 1. What was actually executed on the agent machine

* 32 unit tests: normalization, language detection, quality filters, exact/near deduplication, deterministic splitting, baseline tokenizer, and small synthetic parsers for MediaWiki XML bz2, TSV/tar.bz2, CSV, JSONL/gzip, plain text and Parquet. Result: **32/32 passed**.
* One small-sample end-to-end integration run on **54 synthetic test-fixture documents** (not production training data): `build_corpus.py` → `compute_statistics.py` → `verify_dataset.py` → `compare_tokenizers.py`. Result: **passed**; 30 final documents, 6,681 characters, 1,886 measured content tokens with the V1 32K tokenizer, 27 verification checks / 0 failures / 1 expected split-ratio warning.
* A separate tokenizer comparison on a **read-only 1.7 MB V1 validation sample** (728 documents / 935,924 characters), measuring the baseline 32K and from-scratch 8K/16K/requested-32K BPE vocabularies.
* Research: dataset cards, GitHub repository metadata/LICENSE files, Wikimedia dump indexes and `dumpstatus.json`, V1 scripts/metadata, and small generated fixtures. No complete large dataset was downloaded.

## 2. What was NOT executed

1. **No full V2 corpus was built, cleaned, deduplicated, balanced, split or tokenized.**
2. **No 1.456 GB official Wikipedia bz2 dump, 7z V1 mirror, Tatoeba weekly export, Wikisource/Wikibooks/Wikinews dump, Ganjoor tarball, poetry tarball or ParsiNLU tarball was downloaded or extracted at full size.** Only small synthetic equivalents were parsed in unit tests.
3. **No complete external dataset** (naab 137 GB, TLPC 98.6 GB, Shiraz 5.97 GB, MirasText, Hamshahri, OSCAR/CC-100/CulturaX/FineWeb-2 fa) was downloaded. These remain quarantined.
4. **No 7z extraction of the V1 Wikipedia mirror** was run; `py7zr` was installed in the test venv but the split 7z parts were not fetched.
5. **No Shiraz Parquet file** was downloaded; Parquet support was tested only on a tiny synthetic table with `pyarrow`.
6. **No full near-deduplication** over millions of documents. SQLite/MinHash/LSH was exercised on 54 documents and in unit tests; candidate-cap behavior at scale is unmeasured.
7. **No full-corpus tokenization** was run. The 1,886-token integration statistic and the 0.94 M-character tokenizer comparison are small-sample measurements only.
8. **No Windows 11 / Python 3.12 execution.** The agent machine is Linux / Python 3.11.2. All code is written to be Windows-compatible and uses no Linux-only commands, but the `.cmd` scripts, `D:\...` paths, `ctypes` Windows peak-RAM path and Python 3.12 behavior are untested here.
9. **No GPU operation and no model training** was performed.
10. **No full V1 integrity re-hash of `train.txt`** was performed locally; V1 Git blob SHA-1 values came from the remote V1 branch tree API. The verifier checks files only if present and size-compatible.

## 3. Licensing and privacy caveats

* This is engineering classification, **not legal advice**. CC BY-SA share-alike, GPL-3.0 data-file obligations, CC BY attribution, NC restrictions and jurisdiction-specific public-domain rules may affect redistribution of a derived corpus or model.
* PROFILE C sources remain unresolved. In particular, naab's MIT metadata/card ambiguity, Shiraz's ODC-BY-only compilation license, TLPC's named reader comments, publisher-copyright news, Twitter-derived PeQa, and Common-Crawl-derived text are **not** cleared for training.
* PII detection is conservative pattern matching, not a guarantee. Automated quality scores do not establish semantic correctness, factual accuracy, absence of bias or legal safety.
* Manual review of `reports/sample_audit.txt` (and the full-run audit file on D:) remains mandatory.

## 4. Reproducibility caveats

* Tatoeba's detailed export is weekly and not immutable; the pipeline records download date and SHA-256 but the exact snapshot depends on when the user runs it.
* Wikimedia `latest` symlinks move; V2 pins dated 2026-10-01 / 2026-11-01 dump URLs and SHA-1 values in `source_metadata.json`.
* GitHub sources are commit-pinned, but a repository can disappear; checksums and manifests will show exactly what was used.
* V1 root files are absent from this session's branch checkout; the current branch has its own 21-byte `README.md` from `main`'s initial commit. The verifier skips same-named files whose size differs from the V1 baseline. V1 remains untouched on `arena/f01a4e1e-ai-persian-dataset`.
* The user requested a separate `persian-dataset-v2-pipeline` branch. This Arena session is fixed to `arena/62203741-ai-persian-dataset`; creating or pushing another branch is not allowed in this session. All work is committed only to the fixed branch.

## 5. Numeric caveats

* All full-corpus token counts in the reports are **estimates** derived from V1's measured 3.991 characters/token and source metadata. They are labelled estimates and must be replaced by `compute_statistics.py` output after the user's full run.
* The 97/2/1 split is a deterministic target. The 30-document smoke test produced 28/1/1 documents and a verification **warning** because cluster-consistent minimum representation skews tiny samples. Full-run ratios still need verification.
* The from-scratch "32K" tokenizer reached only 15,958 entries on the 0.94 M-character sample; this is a real measured limitation, not a 32K result. Rerun on the final corpus.
* `pipeline_benchmark.json` peak RAM (24.16 MB) and throughput (771 docs/s) are **small-sample Linux measurements**, not predictions for a 16 GB Windows machine or a 100 M-document corpus.

## 6. Git / storage

* No expanded corpus, raw dump, SQLite index, normalized shard or final `train.txt` is committed to Git. `.gitignore` excludes them.
* No Git LFS, paid storage quota or >100 MB Git object was used. The only sizeable committed artifact is the 3.1 MB V1 baseline tokenizer JSON.
* GitHub's 100 MB per-object and 2 GB push limits are respected. Large outputs go to `D:\GPT\dataset\v2_final` and `D:\GPT\dataset\v2_work`.

## 7. Definition-of-done honesty

The handover satisfies: research report, licensing matrix, V1 audit, proposed distribution, scalable pipeline, unit tests, small integration test, measured small-sample statistics, reproducible config, checksums/version metadata, Windows CMD guide, honest untested list, storage/download estimate, and clearly labelled estimated V2 token count. It does **not** claim that the full expanded dataset is verified, that all licensing is clear, or that 100–150 M tokens were measured.
