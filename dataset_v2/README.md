# Persian Dataset V2 — Scalable, Licence-Aware Pretraining Pipeline

**Goal:** build the largest practical, diverse, legally documented **Persian-only** corpus for a GPT trained from random initialization. V2 fixes V1's Wikipedia dominance, adds three data profiles, three splits, global disk-backed deduplication, Persian-aware normalization, PII/quality gates, measured statistics, tokenizer comparison and a Windows 11 / Python 3.12 execution path.

**This repository does NOT contain the expanded corpus.** It contains the pipeline, configuration, source/licence metadata, tests and small verification reports. The user's PC performs the full download, processing, verification and tokenization.

> **Start here:** `reports/source_research.md`, `reports/licensing_report.md`, `reports/v1_pipeline_audit.md`, `reports/known_limitations.md`, then `windows/run_v2.cmd`.

---

## 1. Branch and V1 immutability

| Item | Value |
|---|---|
| Current session branch | `arena/62203741-ai-persian-dataset` |
| Base commit | `a770ebfc33b4d72c26b9655084b45ec13ef5dacc` (`main` initial commit) |
| V1 branch (untouched) | `arena/f01a4e1e-ai-persian-dataset` |
| V1 commit | `8e35d5e7e72597eda96e83db9c01a1335f235e19` |
| V1 status | immutable baseline; its `train.txt`, `validation.txt`, README, sources, reports and tokenizer remain on the remote V1 branch |

The user's requested branch name `persian-dataset-v2-pipeline` was **not** created: this Arena session is fixed to `arena/62203741-ai-persian-dataset` and must not create/switch/push another branch. V2 lives in a new `dataset_v2/` tree on the fixed branch. The only V1 artifact copied into V2 is `tokenizer/persian-bpe-32k.json`, byte-identical to the V1 blob (`sha256 8bdecc85dd05009b2abb2148d2cea6ac15ed4815d3d8064e6ee8b1d2500c53a5`), because it is the working baseline. `verify_dataset.py` checks V1 files that are present and size-compatible; `build_corpus.py` refuses to write into the repository root or over V1-named files.

---

## 2. Data profiles

| Profile | Meaning | Default | Output |
|---|---|---|---|
| **A** | PERMISSIVE CORE — verified underlying text rights | yes | `profile_A/` |
| **B** | RESEARCH ONLY — documented non-commercial sources | no | `profile_B/` |
| **C** | QUARANTINE — unknown/conflicting rights, provenance or privacy concerns | never trained | manifest only |

A source's MIT/ODC-BY/CC **compilation** label never counts as permission for the underlying text. `sources/source_catalog.csv`, `sources/licensing_matrix.csv` and `sources/source_metadata.json` record both. See `reports/licensing_report.md`.

Approved A sources include dated official Wikimedia dumps (Wikipedia 2026-10-01, Wikibooks/Wikisource/Wikinews 2026-11-01), Common Voice CC0 Persian sentence text, Tatoeba CC BY 2.0 FR sentences, a public-domain-only Ganjoor subset, CC BY-SA classical poetry, and the optional V1 Wikipedia mirror. PROFILE B contains ParsiNLU CC BY-NC-SA text. SLPL/naab, Shiraz, TLPC, MirasText, publisher news, PeQa and Common-Crawl-derived corpora are quarantined.

---

## 3. Architecture

```
discover_sources.py   metadata-only remote probes, regenerate CSVs
download_sources.py   resumable HTTP Range downloads, SHA-1/SHA-256, checkpoint manifest
extract_sources.py    streaming MediaWiki XML bz2, CSV/TSV/JSONL/txt/tar/Parquet/7z
normalize_persian.py  NFKC, Arabic→Persian, ZWNJ, Latin measurement, PII detection
filter_quality.py     explainable language/structure/repetition/boilerplate/quality gate
deduplicate.py        SQLite exact hash + 32-bin MinHash/8-band LSH, bounded candidates
split_dataset.py      deterministic 97/2/1, cluster-consistent, source-stratified
build_corpus.py       orchestrates all stages, sharded JSONL, checkpoints, audit samples
compute_statistics.py measured final-file stats, exact V1 32K token counts, checksums
verify_dataset.py     independent re-scan, split leakage, V1 integrity, checksums
compare_tokenizers.py 8K/16K/32K from-scratch BPE comparison, embedding cost
```

Scalability design: one document at a time; normalized text in 64 MiB JSONL shards; all dedup/audit/provenance metadata in SQLite (`PRAGMA cache_size=-32768`, WAL, `temp_store=FILE`); LSH candidate lists capped; streaming SQL cursors; atomic `.part` → `os.replace`; deterministic BLAKE2b hashes (never Python's randomized `hash()`); checkpoint every 5,000 docs; resumable `--resume`; `--force` rebuild; disk-space checks; logging to `workdir/pipeline.log`; no Linux-only commands; no GPU; standard library wherever practical.

Optional dependencies (`requirements.txt`): `tokenizers` (required for exact token counts), `pyarrow` (Parquet, quarantined sources only), `py7zr` (optional V1 7z mirror). No pandas/numpy/PyTorch/datasets library.

---

## 4. Output format

Each profile directory contains:

```
profile_A/train.txt
profile_A/validation.txt
profile_A/test.txt
profile_A/documents.jsonl       # per-document provenance
checksums.json                  # SHA-256 of final files + index
profile_C_quarantine_manifest_only/quarantine_manifest.json
reports/quality_report.json
reports/build_report.json
reports/pipeline_benchmark.json
reports/dataset_statistics.json
reports/sample_audit.txt
reports/verification_report_v2.json
reports/tokenizer_comparison.json
```

Documents are separated by a line containing exactly `<|endoftext|>` (V1-compatible). `compute_statistics.py` reports **content tokens excluding** document separators and **tokens including one document-ending token per document** separately. The V1 tokenizer's `<|endoftext|>` is ID 0 and a single token; it is not automatically replaced.

---

## 5. Reproducible configuration

* `configs/dataset_config.json`: seed `persian-dataset-v2-2026-11-02`, split ratios `0.97/0.02/0.01`, domain targets, source caps, normalization/quality/dedup thresholds, shard size, disk-space minimum, tokenizer baseline.
* `configs/source_policy.json`: profile enforcement, PII policy, no access-control bypass, no synthetic text, V1 protection.
* `sources/source_metadata.json`: exact dated dump URLs, SHA-1 checksums, pinned GitHub commit SHAs, license fields, privacy risk, rights verification, extractor options and references.
* `sources/v1_baseline.json`: V1 Git blob SHA-1 values and verified V1 statistics.

Domain targets are **investigation targets, not quotas**. Quality and licensing come first; unfilled capacity is reported, never filled by duplicating documents or admitting quarantined text.

---

## 6. Windows 11 / Python 3.12 execution

User paths:

| Purpose | Path |
|---|---|
| Python | `D:\GPT\venv\Scripts\python.exe` |
| Repository | `D:\GPT\dataset\repo` |
| Project | `D:\GPT\project` |
| Work / temp | `D:\GPT\dataset\v2_work` |
| Final corpus | `D:\GPT\dataset\v2_final` |

From a normal **CMD** window (not Git Bash / PowerShell):

```bat
cd /d D:\GPT\dataset\repo
D:\GPT\venv\Scripts\python.exe -m pip install -r dataset_v2\requirements.txt
dataset_v2\windows\run_v2_tests.cmd
dataset_v2\windows\run_v2.cmd
```

`run_v2.cmd` performs: dependency install → branch fetch/checkout/pull → `--probe` discovery → resumable PROFILE A downloads → streaming extraction → build with `--resume` → measured statistics → independent verification → full-corpus-sample tokenizer comparison → copy reports to `D:\GPT\project\v2_reports`.

For the separate research-only extension (A+B rebuilt together for global dedup):

```bat
dataset_v2\windows\run_v2_research.cmd
```

PROFILE B output is non-commercial and must never be merged into `profile_A`. Do not train if `verify_dataset.py` exits non-zero.

Exact single commands are also in the `.cmd` files; use `--force` instead of `--resume` for a clean rebuild, and add `--include-optional` to `download_sources.py` only if you want the 2021 V1 Wikipedia 7z mirror.

---

## 7. Tests actually executed on the agent machine

* `python -m unittest discover -s dataset_v2/tests -v`: **32 tests, all passed** (`reports/unit_test_output.txt`).
* `dataset_v2/tests/run_smoke_test.py`: **small-sample integration passed** on 54 synthetic fixtures; 30 final documents, 6,681 characters, 1,886 measured content tokens, 3.5424 chars/token, 6.6925 bytes/token, 1 exact + 6 near duplicates removed, train/val/test = 28/1/1, 27 verification checks / 0 failures / 1 expected split-ratio warning (`reports/small_sample_statistics.json`, `reports/integration_test_output.txt`).
* Tokenizer comparison on a 1.7 MB V1 validation sample: baseline 32K = 3.9643 chars/token and 100% round-trip; from-scratch 8K = 3.6354; from-scratch 16K and requested-32K reached 15,958 actual entries because the sample is too small (`reports/tokenizer_comparison.json`).

All of these are **small-sample** results. See `reports/known_limitations.md` for the complete list of untested operations.

---

## 8. Estimated full-corpus size (NOT measured)

V1's 538,849,962 filtered characters were 84.83% Wikipedia; only 504,758,077 characters were core-like A text, ≈126.5 M estimated tokens at V1's 3.991 chars/token. Because non-Wikipedia verified sources are small, a **balanced strict PROFILE A** V2 corpus is estimated at **≈36–50 M measured tokens**; an upper-bound A corpus of **≈100–150 M** would again be ~70–85% Wikipedia. PROFILE B could add volume but is non-commercial and privacy-constrained. These are estimates until `compute_statistics.py` runs on the user's PC. See `reports/source_research.md` and `reports/model_data_balance.md`.

---

## 9. Recommended next step

1. Run `run_v2_tests.cmd` on Windows to validate Python 3.12 and paths.
2. Run `run_v2.cmd` for PROFILE A only.
3. Manually review `sample_audit.txt` and `verification_report_v2.json`.
4. Use the **measured** token count to choose a small ~30 M-parameter GPT; do not start training on an unverified corpus.
5. Prioritize rights-cleared modern Persian prose, educational text and dialogue over simply increasing model size.

No model training is performed or recommended in this phase. The dataset must be finalized first.
