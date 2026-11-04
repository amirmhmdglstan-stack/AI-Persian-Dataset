# Persian Dataset V2 — Scalable, Licence-Aware Pretraining Pipeline

> Consolidation note: branch names and test results below describe the original repair session, not this integration. See the root [consolidation report](../CONSOLIDATION_REPORT.md) for current status and critical deduplication and licensing caveats. Windows launchers now accept the current Git branch without switching it.

**Goal:** build the largest practical, diverse, legally documented **Persian-only** corpus for a GPT trained from random initialization. V2 fixes V1's Wikipedia dominance, adds three data profiles, three splits, global disk-backed deduplication, Persian-aware normalization, PII/quality gates, measured statistics, a memory-bounded tokenizer comparison and a Windows 11 / Python 3.12 execution path. The agent machine is Linux; Windows 11 / Python 3.12 execution is NOT fully verified here.

**This repository does NOT contain the expanded corpus.** It contains the pipeline, configuration, source/licence metadata, tests and small verification reports. The user's PC performs the full download, processing, verification and a memory-bounded tokenizer comparison; full-corpus tokenization is not claimed here.

> **Start here:** `reports/source_research.md`, `reports/licensing_report.md`, `reports/v1_pipeline_audit.md`, `reports/known_limitations.md`, then `windows/run_v2.cmd`.

---

## 1. Branch and V1 immutability

| Item | Value |
|---|---|
| Current session branch | `arena/b2bcd196-ai-persian-dataset` |
| Base commit | `a770ebfc33b4d72c26b9655084b45ec13ef5dacc` (`main` initial commit) |
| V1 branch (untouched) | `arena/f01a4e1e-ai-persian-dataset` |
| V1 commit | `8e35d5e7e72597eda96e83db9c01a1335f235e19` |
| V1 status | immutable baseline; its `train.txt`, `validation.txt`, README, sources, reports and tokenizer remain on the remote V1 branch |

This repair session is fixed to `arena/b2bcd196-ai-persian-dataset`. Its parent history contains the V2 branch at `e9bbf532`; repairs are pushed **only** to this session branch, not to `arena/62203741-ai-persian-dataset` or V1. V2 lives in a new `dataset_v2/` tree on the fixed branch. The only V1 artifact copied into V2 is `tokenizer/persian-bpe-32k.json`, byte-identical to the V1 blob (`sha256 8bdecc85dd05009b2abb2148d2cea6ac15ed4815d3d8064e6ee8b1d2500c53a5`), because it is the working baseline. `verify_dataset.py` checks V1 files that are present and size-compatible; `build_corpus.py` refuses to write into the repository root or over V1-named files.

---

## 2. Data profiles

| Profile | Meaning | Default | Output |
|---|---|---|---|
| **A** | RIGHTS-CLEARED CORE (LICENSE CONDITIONS APPLY) — verified underlying text rights with attribution / share-alike / source-specific obligations | yes | `profile_A/` |
| **B** | RESEARCH ONLY — documented non-commercial sources | no | `profile_B/` |
| **C** | QUARANTINE — unknown/conflicting rights, provenance or privacy concerns | never trained | manifest only |

A source's MIT/ODC-BY/CC **compilation** label never counts as permission for the underlying text. `sources/source_catalog.csv`, `sources/licensing_matrix.csv` and `sources/source_metadata.json` record both. PROFILE A is **not obligation-free**: CC BY-SA / GFDL share-alike, Tatoeba CC BY 2.0 FR author attribution, Wikinews CC BY 2.5 attribution, GPL-3.0 data-file notices and other source-specific conditions still apply. Per-document provenance is retained in `documents.jsonl`. Trained model weights are **not** assumed to be automatically free of upstream licensing obligations. See `reports/licensing_report.md`.

Approved A sources include dated official Wikimedia dumps (Wikipedia 2026-10-01, Wikibooks/Wikisource/Wikinews 2026-11-01), Tatoeba CC BY 2.0 FR sentences, a public-domain-only Ganjoor subset, and the optional V1 Wikipedia mirror. PROFILE B contains ParsiNLU CC BY-NC-SA text. SLPL/naab, Shiraz, TLPC, MirasText, publisher news, PeQa and Common-Crawl-derived corpora are quarantined.

---

## 3. Architecture

```
discover_sources.py   metadata-only remote probes, regenerate CSVs
download_sources.py   resumable HTTP Range downloads, manifest SHA-256+size identity
extract_sources.py    MediaWiki XML bz2, Tatoeba pes .tsv.bz2, CSV/TSV/JSONL/txt/tar/Parquet/7z
normalize_persian.py  NFKC, Arabic→Persian, ZWNJ, Latin measurement, PII detection
filter_quality.py     explainable language/structure/repetition/boilerplate/quality gate
deduplicate.py        SQLite exact hash + 32-bin MinHash/8-band LSH, bounded candidates
split_dataset.py      deterministic 97/2/1, cluster-consistent, source-stratified
build_corpus.py       orchestrates all stages, sharded JSONL, checkpoints, audit samples
compute_statistics.py measured final-file stats, exact V1 32K token counts, checksums
verify_dataset.py     independent re-scan, split leakage, V1 integrity, checksums
compare_tokenizers.py capped 8M-char streaming BPE comparison, measured peak RSS
```

Scalability design: one document at a time; normalized text in 64 MiB JSONL shards; all dedup/audit/provenance metadata in SQLite (`PRAGMA cache_size=-32768`, WAL, `temp_store=FILE`); LSH candidate lists capped; streaming SQL cursors; atomic `.part` → `os.replace`; deterministic BLAKE2b hashes (never Python's randomized `hash()`); transactional SQLite ingestion checkpoint per document (JSON mirror every 5,000); resumable `--resume`; `--force` rebuild; disk-space checks; logging to `workdir/pipeline.log`; no Linux-only commands; no GPU; standard library wherever practical.

Default `requirements.txt` contains **only** `tokenizers>=0.20,<0.24`, which supports the user's installed `tokenizers 0.23.3` without downgrading it. `pyarrow` (Parquet) and `py7zr` (optional V1 7z mirror) are in separate optional requirement files and are **never auto-installed** by the default launchers. `pipeline/check_requirements.py` prints versions and missing optional packages only when corresponding sources are enabled. No PyTorch, NumPy, pandas, TensorFlow, `datasets` library, GPU packages or model weights.

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
* `configs/source_policy.json`: profile enforcement, exact configurable Wikisource licence-template allowlist / reject reasons, PII policy, no access-control bypass, no synthetic text, V1 protection.
* `sources/source_metadata.json`: exact dated dump URLs, SHA-1 checksums, pinned GitHub commit SHAs, the verified `pes_sentences_detailed.tsv.bz2` file/snapshot observation, per-project Wikimedia `page_base_url`, licence fields, privacy risk, rights verification, extractor options and references.
* `sources/v1_baseline.json`: V1 Git blob SHA-1 values and verified V1 statistics.

Domain targets are **investigation targets, not quotas**. Quality and licensing come first; for `scope=full_run` the configured source/domain caps are applied even when accepted text is below the 480 M-character target. The effective budget is `min(target_char_budget, actual_available_characters)`; unfilled capacity is reported in `build_report.json`, never filled by duplicating documents or admitting quarantined text. `scope=small_sample` may retain all accepted synthetic test documents, but production full runs never bypass caps.

---

## 6. Windows 11 / Python 3.12 execution

User paths:

| Purpose | Path |
|---|---|
| Python | `D:\GPT\venv\Scripts\python.exe` |
| **V2 repository** | `D:\GPT\dataset\repo_v2_review` |
| V1 repository (untouched) | `D:\GPT\dataset\repo` |
| Project | `D:\GPT\project` |
| Work / temp | `D:\GPT\dataset\v2_work` |
| Final corpus | `D:\GPT\dataset\v2_final` |

From a normal **CMD** window (not Git Bash / PowerShell):

```bat
cd /d D:\GPT\dataset\repo_v2_review
D:\GPT\venv\Scripts\python.exe -m pip install -r dataset_v2\requirements.txt
D:\GPT\venv\Scripts\python.exe dataset_v2\pipeline\check_requirements.py --config dataset_v2\configs\dataset_config.json --metadata dataset_v2\sources\source_metadata.json --tokenizer-baseline dataset_v2\tokenizer\persian-bpe-32k.json --profiles A
dataset_v2\windows\run_v2_tests.cmd
dataset_v2\windows\run_v2.cmd
```

`run_v2.cmd` verifies the V2 path, `dataset_v2` tree, Python executable, current Git checkout and clean tracked source/config tree; it prints the repository path and **never** performs an automatic Git remote update or branch switch. It installs core dependencies only, runs `check_requirements.py`, performs metadata-only probes (results in the workdir), manifest-verified resumable downloads, streaming extraction with exact Wikisource licence checks, a full-run build with caps always applied, measured statistics, independent verification, and a capped 8M-character streaming tokenizer comparison with measured process peak RSS. It copies small reports to `D:\GPT\project\v2_reports`. It is NOT full-corpus tokenization or a full-corpus memory measurement.

For the separate research-only extension (A+B rebuilt together for global dedup):

```bat
dataset_v2\windows\run_v2_research.cmd
```

Optional packages are explicit opt-ins only, e.g. before `--include-optional`:

```bat
D:\GPT\venv\Scripts\python.exe -m pip install -r dataset_v2\requirements-7z.txt
D:\GPT\venv\Scripts\python.exe -m pip install -r dataset_v2\requirements-parquet.txt
```

PROFILE B output is non-commercial and must never be merged into `profile_A`. Do not train if `verify_dataset.py` exits non-zero.

Exact single commands are also in the `.cmd` files; use `--force` instead of `--resume` for a clean rebuild, and add `--include-optional` to `download_sources.py` only if you want the 2021 V1 Wikipedia 7z mirror.

---

## 7. Tests actually executed on the agent machine

**Linux-only, Python 3.11.2.** These are small synthetic/unit results. They are **not** Windows 11 / Python 3.12 verification, full-corpus extraction, full-corpus tokenization or full-corpus memory measurements.

* `python -m compileall -q dataset_v2/pipeline dataset_v2/tests dataset_v2/sources`: **passed**.
* `python -m unittest discover -s dataset_v2/tests -v` in a **core-only** venv with `tokenizers 0.23.3`: **86 tests, OK (1 skipped: optional `pyarrow` not installed)** (`reports/unit_test_output.txt`). A second run on the system Python also passed 86 tests with 7 optional-dependency skips. New tests cover path resolution/launchers, `pes` Tatoeba catalog/parsing/fallback, Wikisource licence allowlist, download manifest resume/corruption/partial/force, full-run balancing, Wikimedia page domains, requirements policy and bounded tokenizer comparison.
* `pipeline/check_requirements.py`: default Profile A printed `tokenizers 0.23.3`, no optional packages required, V1 32K baseline OK, and exited 0. With `--include-optional` it correctly reported the optional `py7zr` missing and exited 1; no optional package was auto-installed (`reports/requirements_check_output.txt`).
* `dataset_v2/tests/run_smoke_test.py`: **small-sample integration passed** on 54 synthetic fixtures; 30 final documents, 6,681 characters, 1,886 measured content tokens, 3.5424 chars/token, 6.6925 bytes/token, 1 exact + 6 near duplicates removed, train/validation/test = 28/1/1, 27 verification checks / 0 failures / 1 expected split-ratio warning (`reports/small_sample_statistics.json`, `reports/integration_test_output.txt`). `build_report.json` records `small_sample_caps_bypassed_for_synthetic_tests_only`; unit tests separately verify that `full_run` applies caps below the 480 M target.
* `extract_sources.py --only-sources tatoeba_fa_cc_by` on a synthetic `pes_sentences_detailed.tsv.bz2` through the real catalog/extractor mapping: 1 Persian document retained, `eng` and `fas` rejected, provenance and CC BY 2.0 FR attribution in metadata only (`reports/tatoeba_extraction_integration_output.txt`).
* `compare_tokenizers.py` on a bounded 20,000-character fixture: 71 documents / 3,612 characters, V1 32K baseline 3.7276 chars/token and 100% round-trip, measured process peak RSS **51.64 MB** for this tiny capped Linux run. From-scratch alternatives reached only 638 entries because the fixture is tiny; this is not a vocabulary recommendation (`reports/tokenizer_comparison.json`, `reports/tokenizer_comparison_run_output.txt`). A 40 M-character request is explicitly refused.

All of these are **small-sample** results. See `reports/known_limitations.md` for the complete list of untested operations.

---

## 8. Estimated full-corpus size (NOT measured)

V1's 538,849,962 filtered characters were 84.83% Wikipedia; only 504,758,077 characters were core-like A text, ≈126.5 M estimated tokens at V1's 3.991 chars/token. Because non-Wikipedia verified sources are small, a **balanced strict PROFILE A** V2 corpus is estimated at **≈36–50 M measured tokens**; an upper-bound A corpus of **≈100–150 M** would again be ~70–85% Wikipedia. PROFILE B could add volume but is non-commercial and privacy-constrained. These are estimates until `compute_statistics.py` runs on the user's PC. See `reports/source_research.md` and `reports/model_data_balance.md`.

---

## 9. Recommended next step

1. Run `run_v2_tests.cmd` on Windows to validate Python 3.12 and paths (not verified on the agent machine).
2. Run `run_v2.cmd` for PROFILE A only.
3. Manually review `sample_audit.txt` and `verification_report_v2.json`.
4. Use the **measured** token count to choose a small ~30 M-parameter GPT; do not start training on an unverified corpus. Rerun the tokenizer comparison only on a bounded 8–10 M-character sample and review its measured peak RSS.
5. Prioritize rights-cleared modern Persian prose, educational text and dialogue over simply increasing model size.

No model training is performed or recommended in this phase. The dataset must be finalized first.

## Repair handover (2026-11-03)

**The source catalog changed:** Chronological Poetry and Common Voice Persian sentence files are now Profile C (quarantine; not downloaded or built). The former Profile A volume estimates and historical synthetic reports that predate this decision are NOT current full-corpus measurements. See `reports/licensing_report.md`. No full-scale V2 corpus has been built here.

Extraction now fingerprints raw files (size/SHA-256), source configuration and policy, writes only complete `.jsonl` files atomically, and leaves limited output as `.part` with `status: partial`. The builder rejects incomplete manifest entries, fingerprints inputs/configuration for resuming, and records ingestion cursor and statistics in SQLite in the same transaction as documents. The JSON checkpoint is an informational mirror. `--max-docs` writes `partial_build.json` and never final splits/reports. Downloads of mutable exports resume only with a matching validator, byte range and verified `.part` sidecar; a 404/410 Tatoeba primary triggers the all-language fallback.

**Existing V2 work directories must be rebuilt after this upgrade.** Old indexes lack transactional cursor/fingerprints. Use `--force` for the first build on the new branch; re-run extraction against existing raw files to refresh old manifests (it will hash raw inputs). Do not reuse an old final dataset or old generated reports. The launchers require a Git checkout and a clean tracked tree; they do not change Git branches automatically. Run `run_v2_tests.cmd` before `run_v2.cmd`. Windows 11 / Python 3.12 execution is not verified here.
