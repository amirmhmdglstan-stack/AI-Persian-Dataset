# AI Persian Dataset

This checkout consolidates the repository's V2 pipeline and source research, but **does not yet contain a completed, rights-cleared merged corpus**. The existing V1 corpus mixes sources with unresolved redistribution rights and has no per-document provenance; it has not been copied here. V2 has not been built on full sources. Do not treat the historical reports as certification of a new dataset.

Start with [CONSOLIDATION_REPORT.md](CONSOLIDATION_REPORT.md) for branch inventory, measured V1 counts, decisions, and blocking limitations. See [dataset_v2/README.md](dataset_v2/README.md) for the pipeline, [source research](dataset_v2/reports/source_research.md), [licensing](dataset_v2/reports/licensing_report.md), and [known limitations](dataset_v2/reports/known_limitations.md). Historical V1 documentation and metadata are in `dataset_v2/reports/*_historical.*` and `dataset_v2/sources/v1_sources_historical.json`; these are evidence, not current licensing advice. The original data and V1 implementation remain available at the pinned V1 commit linked in the report.

## Local checks

Python 3.9+ and a virtual environment are recommended:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r dataset_v2/requirements.txt
python -m compileall -q dataset_v2/pipeline dataset_v2/tests dataset_v2/sources
python -m unittest discover -s dataset_v2/tests -v
python dataset_v2/pipeline/check_requirements.py --profiles A
```

The Windows launchers have machine-specific paths that must be reviewed and edited first; they have not been tested on Windows during consolidation. The synthetic smoke test deletes its `--workdir` recursively and overwrites files in `dataset_v2/reports`; run it only in a disposable copy of the checkout and with a disposable work directory. Never run a full build or distribute output without reviewing the deduplication and rights issues in the consolidation report.
