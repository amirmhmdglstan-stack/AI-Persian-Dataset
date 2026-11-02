# Persian Dataset V2 — Known Limitations and Untested Operations

**Read this before training.** The V2 repository contains a designed, unit-tested and small-sample-tested pipeline. It does **not** contain a fully built or verified expanded corpus. The audit fixes were validated on **Linux / Python 3.11.2** with small synthetic fixtures; Windows 11 / Python 3.12 compatibility is **not fully verified**.

## 1. What was actually executed on the agent machine

* `python -m compileall -q dataset_v2/pipeline dataset_v2/tests dataset_v2/sources`: **passed**.
* `python -m unittest discover -s dataset_v2/tests -v` in a **core-only** venv with `tokenizers 0.23.3`: **72 tests, OK (1 skipped: optional `pyarrow` not installed)** (`reports/unit_test_output.txt`). A second system-Python run also passed 72 tests with 7 optional-dependency skips.
* New synthetic tests cover: repository-relative config paths and launcher safety; exact Wikisource licence-template allowlist/rejection reasons; `pes` Tatoeba catalog pins, bzip2 TSV parsing, MySQL OUTFILE escapes, attribution metadata and the real all-language `sentences_detailed.csv` member; download manifest URL+filename identity, SHA-256/size verification, corruption, `.part` Range resume, `--force` and mutable weekly snapshot skipping; full-run balancing caps below the 480 M target and unchanged `small_sample`; four Wikimedia page domains; core/optional dependency policy; and bounded tokenizer comparison.
* `extract_sources.py --only-sources tatoeba_fa_cc_by` on a synthetic `pes_sentences_detailed.tsv.bz2` through the real catalog/extractor mapping retained 1 `pes` document, rejected `eng`/`fas`, and kept provenance and CC BY 2.0 FR attribution in metadata only (`reports/tatoeba_extraction_integration_output.txt`).
* `pipeline/check_requirements.py`: default Profile A printed `tokenizers 0.23.3`, no optional packages required, V1 32K baseline OK, and exited 0. With `--include-optional` it reported optional `py7zr` missing and exited 1 (`reports/requirements_check_output.txt`). No optional package was auto-installed.
* `dataset_v2/tests/run_smoke_test.py`: **small-sample integration passed** on 54 synthetic fixtures; 30 final documents, 6,681 characters, 1,886 measured content tokens with the V1 32K tokenizer, 3.5424 chars/token, 6.6925 bytes/token, 1 exact + 6 near duplicates removed, train/validation/test = 28/1/1, 27 verification checks / 0 failures / 1 expected split-ratio warning (`reports/small_sample_statistics.json`, `reports/integration_test_output.txt`). `build_report.json` records `small_sample_caps_bypassed_for_synthetic_tests_only`; separate unit tests verify `full_run` caps are always applied.
* `compare_tokenizers.py` on a bounded 20,000-character fixture: 71 documents / 3,612 characters, V1 32K baseline 3.7276 chars/token and 100% round-trip, measured process peak RSS **51.64 MB** for this tiny capped Linux run (`reports/tokenizer_comparison.json`, `reports/tokenizer_comparison_run_output.txt`). A 40 M-character request is explicitly refused; the hard cap is 10 M characters and the launcher default is 8 M.
* Research: official Tatoeba downloads/terms, the official `per_language/pes/` index (filename, 695,149-byte size and 2026-10-31 date), Wikimedia dump indexes/licences, dataset cards, GitHub metadata, V1 scripts/metadata and small generated fixtures. No complete large dataset was downloaded.

## 2. What was NOT executed

1. **No full V2 corpus was built, cleaned, deduplicated, balanced, split or tokenized.**
2. **No Windows 11 / Python 3.12 execution.** The agent machine is Linux / Python 3.11.2. The `.cmd` launchers, CRLF behaviour, `D:\...` paths, Git dirty-tree/branch guards, `ctypes` Windows peak-RAM path and Python 3.12 behaviour are untested here. The launchers use `D:\GPT\dataset\repo_v2_review`; `D:\GPT\dataset\repo` is V1 and is refused.
3. **No full extraction of the 1.456 GB official Wikipedia bz2 dump, V1 7z mirror, Wikisource/Wikibooks/Wikinews dumps, Ganjoor tarball, poetry tarball or ParsiNLU tarball.** Only small synthetic equivalents were parsed.
4. **No real Tatoeba file was downloaded.** The official Persian-only filename/size/date and detailed field order were verified from official pages; the bzip2 TSV parser and fallback `sentences_detailed.csv` member were tested with synthetic archives. The 303 MB all-language fallback archive was not downloaded or parsed from real data, and no official direct tar-member listing was parsed.
5. **No download over the real network.** Resume/corruption/partial/force tests used mocked HTTP responses and temporary files. The mutable weekly Tatoeba snapshot date/checksum policy was tested synthetically, not across real weekly refreshes.
6. **No 7z extraction of the V1 Wikipedia mirror** was run; `py7zr` is optional and was **not installed** in the core-only audit venv.
7. **No Shiraz Parquet file** was downloaded; the optional Parquet test is skipped unless `pyarrow` is installed and was skipped in the core-only audit venv.
8. **No full near-deduplication** over millions of documents. SQLite/MinHash/LSH was exercised on 54 documents and in unit tests; candidate-cap behaviour at scale is unmeasured.
9. **No full-corpus tokenization and no full-corpus memory measurement.** The 1,886-token integration statistic, 20,000-character tokenizer comparison and 51.64 MB process peak are small-sample measurements only. BPE training input is bounded but `tokenizers.BpeTrainer` may internally materialise counts; no strict O(1)-memory claim is made.
10. **No GPU operation and no model training** was performed.
11. **No full V1 integrity re-hash of `train.txt`** was performed locally; V1 Git blob SHA-1 values came from the remote V1 branch tree API. The verifier checks files only if present and size-compatible.
12. **No automatic Git remote update, pull or branch switch** was performed by the launchers; they verify the fixed branch and a clean tracked tree instead. This safeguard itself was not executed on Windows.

## 3. Licensing and privacy caveats

* This is engineering classification, **not legal advice**. PROFILE A is **RIGHTS-CLEARED CORE (LICENSE CONDITIONS APPLY)**, not obligation-free: CC BY-SA / GFDL share-alike, GPL-3.0 data-file notices, Tatoeba CC BY 2.0 FR author attribution, Wikinews CC BY 2.5 attribution and other source-specific conditions still apply.
* Trained model weights are **not** assumed to be automatically free of upstream attribution, share-alike or redistribution obligations. Review per-document provenance and the applicable licence terms before distributing a model or derived corpus.
* PROFILE C sources remain unresolved. In particular, naab's MIT metadata/card ambiguity, Shiraz's ODC-BY-only compilation licence, TLPC's named reader comments, publisher-copyright news, Twitter-derived PeQa and Common-Crawl-derived text are **not** cleared for training.
* The Wikisource licence allowlist is conservative and exact: NC/ND/incompatible, generic `Copyright`/`مجوز`/`پروانه`, ambiguous and unverified licence-like templates are rejected with a reason. It was tested synthetically, not against every real Wikisource page.
* PII detection is conservative pattern matching, not a guarantee. Automated quality scores do not establish semantic correctness, factual accuracy, absence of bias or legal safety.
* Manual review of `reports/sample_audit.txt` (and the full-run audit file on D:) remains mandatory.

## 4. Reproducibility caveats

* Tatoeba's detailed export is weekly and not immutable. V2 prefers `pes_sentences_detailed.tsv.bz2` and records the observed HTTP snapshot date, actual SHA-256 and size in `download_manifest.json`; ordinary resume does not refresh a verified snapshot and `--force` is the explicit refresh. The catalog's observed date/size are from the official index on 2026-10-31.
* Wikimedia `latest` symlinks move; V2 pins dated 2026-10-01 / 2026-11-01 dump URLs and SHA-1 values in `source_metadata.json`. Per-project `page_base_url` values are configured for Wikipedia, Wikibooks, Wikisource and Wikinews.
* GitHub sources are commit-pinned, but a repository can disappear; checksums and manifests will show exactly what was used.
* V1 root files are absent from this session's branch checkout; the current branch has its own 21-byte `README.md` from `main`'s initial commit. The verifier skips same-named files whose size differs from the V1 baseline. V1 remains untouched on `arena/f01a4e1e-ai-persian-dataset`.
* The user requested a separate `persian-dataset-v2-pipeline` branch. This Arena session is fixed to `arena/62203741-ai-persian-dataset`; creating or pushing another branch is not allowed in this session. All work is committed only to the fixed branch.

## 5. Numeric caveats

* All full-corpus token counts in the reports are **estimates** derived from V1's measured 3.991 characters/token and source metadata. They are labelled estimates and must be replaced by `compute_statistics.py` output after the user's full run.
* The 97/2/1 split is a deterministic target. The 30-document smoke test produced 28/1/1 documents and a verification **warning** because cluster-consistent minimum representation skews tiny samples. Full-run ratios still need verification.
* The from-scratch alternatives in the 20,000-character comparison reached only 638 entries; this is a real measured limitation of the tiny fixture, not a 32K result. Rerun on a bounded sample of the final corpus.
* `pipeline_benchmark.json` peak RAM (**24.36 MB**) and throughput (**710.53 docs/s**) are **small-sample Linux measurements**, not predictions for a 16 GB Windows machine or a 100 M-document corpus.
* The measured 51.64 MB tokenizer-comparison peak RSS is process-wide for the tiny capped run, not a full-corpus memory measurement.

## 6. Git / storage

* No expanded corpus, raw dump, SQLite index, normalized shard or final `train.txt` is committed to Git. `.gitignore` excludes them.
* No Git LFS, paid storage quota or >100 MB Git object was used. The only sizeable committed artifact is the 3.1 MB V1 baseline tokenizer JSON.
* GitHub's 100 MB per-object and 2 GB push limits are respected. Large outputs go to `D:\GPT\dataset\v2_final` and `D:\GPT\dataset\v2_work`; the V2 repository is `D:\GPT\dataset\repo_v2_review`.

## 7. Definition-of-done honesty

The handover satisfies: research report, licensing matrix, V1 audit, proposed distribution, scalable pipeline, unit tests, small integration test, measured small-sample statistics, reproducible config, checksums/version metadata, Windows CMD guide, honest untested list, storage/download estimate, and clearly labelled estimated V2 token count. It does **not** claim that the full expanded dataset is verified, that Windows 11 / Python 3.12 was fully exercised, that full extraction/tokenization passed, that full-corpus memory was measured, that all licensing is clear, or that 100–150 M tokens were measured.
