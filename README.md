# AI Persian Dataset — consolidated agent work

This repository combines **all four agent branches** into one maintained V2 pipeline, Persian BPE tokenizer, source/licensing research, tests, and an offline, content-deduplicated history bundle. No external datasets or research were added in this integration.

**Repository integration is complete; a production, rights-cleared Persian training dataset is not.** Historical V1 data has unresolved rights/privacy and missing document-level source attribution. It is preserved separately, never silently admitted to V2's training profiles. The existing tokenizer is a baseline trained on historical material, not a newly rights-cleared model. Do not interpret historical reports as current production certification.

## What to use

- [`dataset_v2/README.md`](dataset_v2/README.md): maintained extraction, Persian normalization, quality filtering, exact deduplication, approximate near-match review, splitting, statistics, verification, and tokenizer comparison/training tools.
- [`dataset_v2/tokenizer/persian-bpe-32k.json`](dataset_v2/tokenizer/persian-bpe-32k.json): the identical V1/V2 baseline, retained once.
- [`dataset_v2/sources/`](dataset_v2/sources/): combined source catalogs, rights matrix and historical metadata.
- [`CONSOLIDATION_REPORT.md`](CONSOLIDATION_REPORT.md): branch decisions, measured results and remaining limitations.
- [`provenance/README.md`](provenance/README.md): **every original file version**, recoverable offline, including V1 scripts, vocabulary, reports, and both corpus revisions. Superseded implementations are archived, not competing default pipelines.
- [`historical_corpus/summary.json`](historical_corpus/summary.json): measured union of the two original corpus revisions. Its 37,455 unique documents are reproducible locally, not a training release.

## Validate locally

Python 3.9+ (tested here on 3.11); no GPU or heavyweight ML stack required:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r dataset_v2/requirements.txt
python tools/consolidate_history.py verify
python -m unittest discover -s tests -v
python -m unittest discover -s dataset_v2/tests -v
python dataset_v2/pipeline/check_requirements.py --profiles A
```

The only install fetches software dependencies, not data. Bundle verification and historical reconstruction use the repository alone, with no Git history or network required:

```sh
python tools/consolidate_history.py reconcile
```

This produces ignored `historical_corpus/documents.jsonl.gz` with exact-document identities, original Git locations, and unresolved-rights labels. It is **not** connected to the production pipeline. See the provenance guide to recover original files byte-for-byte.

CI validates the bundle, unit tests and synthetic integration. The synthetic smoke test deletes its work directory and rewrites reports: run it only in a disposable copy (as CI does). Windows launchers have machine-specific paths; review them before use. No full-source build or Windows execution was performed here.
