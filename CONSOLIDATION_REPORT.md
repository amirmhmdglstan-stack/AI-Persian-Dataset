# Consolidation report — work in progress

Investigation date: 2026-11-04. **This is not a completed dataset merge.** The integration branch contains the repaired pipeline and research; no new production corpus was built or claimed. The problems below prevent safely declaring it complete.

## Branches and history examined

The complete remote heads list from `git ls-remote --heads origin` and GitHub's branches API comprised:

| Branch / head | Contributions and decision |
|---|---|
| `main` / `a770ebfc33b4d72c26b9655084b45ec13ef5dacc` | Initial README only; this integration is based on this commit, which was still the latest main on fetch. |
| `arena/f01a4e1e-ai-persian-dataset` / `8e35d5e7e72597eda96e83db9c01a1335f235e19` | V1 corpus, source catalog, cleaning/build scripts, reports and BPE tokenizer. Its earlier commit `31d7e8f` produced an older corpus; the head repairs newline-eating regexes and rebuilds it (36,586 to 36,592 reported documents). Historical source metadata, statistics, verification, tokenizer statistics and long README have been copied here; the tokenizer is retained once. The head corpus and scripts remain at [the pinned V1 commit](https://github.com/amirmhmdglstan-stack/AI-Persian-Dataset/tree/8e35d5e7e72597eda96e83db9c01a1335f235e19). |
| `arena/62203741-ai-persian-dataset` / `e9bbf5327d4eecf8bcb04c8f26d99766b38136ff` | V2 pipeline, research and license policy, then Windows path/rights/Tatoeba/resume/balancing/tokenizer repairs. Its ancestor `62de4f6` introduced V2. All of this history is an ancestor of the repair head below. |
| `arena/b2bcd196-ai-persian-dataset` / `dda8eaf51a72d5fe1113828993b0feecc430014b` | Descendant of the preceding branch: extraction, transactional ingestion, download fallback and global split repairs, tests and updated rights research. Its full `dataset_v2` tree is the starting implementation here. |

No original branch was modified, merged, deleted or rewritten. Session policy fixes the head name to `arena/40ead0b9-ai-persian-dataset`; a separate integration branch cannot be created in this session. The research branch catalog and policy are engineering assessments, not legal advice; recheck the actual sources and terms before use.

## Reconciliation decisions and provenance

- The V2 descendant subsumes the earlier V2 branch and is the single pipeline implementation imported. V1 scripts are superseded for new builds due to the substantive defects documented in [V1 audit](dataset_v2/reports/v1_pipeline_audit.md), but remain retrievable at the immutable commit above for historical reproducibility. They were not blindly merged with V2: their semantics and output formats differ.
- V1's historical metadata and documentation were retained as evidence, together with V2's research, catalog, licensing matrix, policy, repair tests and small-sample reports. The historical documents include claims later qualified by the V2 audit; the audit and the warnings in this report take precedence. Earlier estimates and small-run reports are not full-run measurements.
- V1 and V2 contain the **identical** baseline tokenizer JSON (SHA-256 `8bdecc85dd05009b2abb2148d2cea6ac15ed4815d3d8064e6ee8b1d2500c53a5`); only the V2 copy is included. The V1 vocab text is a derived listing and is not duplicated.
- The committed V1 corpus includes publisher-copyright news without a declared license, Twitter-derived text with unresolved rights/privacy, and other mixed-license sources; its final text lacks per-document source attribution. Merely labeling it research-only does not grant redistribution rights. The corpus is not recopied or silently mixed into Profile A or B. Its contents remain intact in the original branch and commit. V2 quarantines several disputed sources and separates A/B/C, but those classifications still require source-specific legal review and attribution. The earlier V2 audit's Common Voice and poetry suggestions are superseded by the later quarantine decision.
- The Windows launchers previously hardcoded the repair branch and would reject this branch or future `main`; they now inspect the current checkout and retain the clean-tree guard without switching branches. Their machine-specific path and V1 protection remain. This change was statically tested on Linux, not executed under Windows.

## Dataset comparison and duplicates

`main` has no corpus. V1 head has 35,864 train and 728 validation documents (36,592 total); 44,999,973 document characters and 572,953 ZWNJs. These were independently counted in this session by reading each UTF-8 file, splitting on `<|endoftext|>`, stripping outer whitespace and hashing each resulting document with SHA-256. That found 36,592 unique exact strings, zero exact duplicates and zero exact train/validation overlaps. This does **not** check near duplicates or validate every datum. Bytes and SHA-256:

- `train.txt`: 80,400,899 bytes; `de69c84d4ec8f7ca7f4ebd050da0fbc5aeef34440c187497458475fe36313eef`.
- `validation.txt`: 1,706,558 bytes; `77104b7f1b369c11490e1535ca1e4f9455fd82341ea250dc6f56cef661f3d6c4`.

The same exact-string comparison against the earlier V1 commit `31d7e8f` found 36,586 unique documents there, 35,723 exact strings shared with the head, 863 old-only and 869 head-only. These differences have not been classified as corrected variants versus unique material, and must not be discarded as duplicates without review. The V1 reported 11,350,801 tokens was not independently recounted here. No production dataset was imported or merged, so there is no new dataset size or before/after record-removal count. The prior V2 synthetic report describes 54 inputs, 30 outputs, 1 exact and 6 near duplicates removed; those numbers are not V1/V2 production merge statistics. No records were removed during this integration.

## Validation performed

On Linux/Python 3.11.2 in an isolated virtual environment with `tokenizers 0.23.3`:

- `python -m compileall -q dataset_v2/pipeline dataset_v2/tests dataset_v2/sources`: passed.
- `python -m unittest discover -s dataset_v2/tests -v`: 86 tests passed, one skipped (optional pyarrow).
- `python dataset_v2/pipeline/check_requirements.py --config dataset_v2/configs/dataset_config.json --metadata dataset_v2/sources/source_metadata.json --tokenizer-baseline dataset_v2/tokenizer/persian-bpe-32k.json --profiles A`: passed.
- `run_smoke_test.py` in a disposable copy with a disposable workdir: passed, 54 synthetic inputs, 30 final documents, 1,886 content tokens, zero verification failures. This is not a production dataset run. The script deletes its workdir and writes reports into its checkout.
- V1 UTF-8 parsing, split counts, SHA-256 exact-duplicate and ZWNJ counts as above; tokenizer JSON SHA-256 comparison passed.
- After the conservative deduplication change: 88 unit tests passed with one optional pyarrow skip; the disposable synthetic smoke run passed with 54 inputs, 36 outputs, 9,161 characters, 2,576 content tokens, one exact and zero near matches removed, and zero verification failures. A separate tampering experiment changed a duplicate source ID; verification exited 1 and marked `duplicate_provenance_matches:A` as failed. These are synthetic tests only.

No full upstream sources were downloaded, no production build or cross-version record-level merge was performed, and no Windows, large-scale, licensing, semantic, complete near-duplicate or exhaustive reference audit was performed. Historical report output files record earlier runs, not current tests.

## Blocking limitations / remaining work

1. V1's missing per-document source lineage prevents safely separating its mixed rights and attribution. Re-extract original sources with stable source IDs and inspect rights and privacy before incorporating records; rights may prevent redistribution or require preserving source-specific conditions. Avoid assuming an MIT repository license applies to underlying text.
2. The inherited V2 exact key stripped punctuation, whitespace and ZWNJ and its MinHash near-deduplicator discarded similar texts. The integration now deduplicates identical extracted text only, separately per profile, and retains near matches; likely matches receive an audit entry and an approximate common split cluster. `near_match_review.jsonl` exports candidate IDs, source, similarity and selection/split status without raw text or excerpts. Distinct raw texts can still normalize to identical output, which the verifier flags. Clustering can miss similarities and is not a proof of cross-split isolation; overlapping candidate clusters are not unioned. Normalization and quality filtering can change or reject meaningful text. Raw text is referenced by extracted file, line and SHA-256, not included in final output; retain restricted/PII-containing work files securely. Old indexes cannot be resumed and require `--force`. These are improvements, not completion of the requested preservation audit.
3. The new `duplicate_sources.jsonl` provides source/license metadata for exact duplicates whose representative was selected, and verification compares it against the work index. Legal compatibility of all contributor licenses and obligations has not been established; discarded-by-cap documents and quality rejections remain in the work index rather than the final provenance files. The actual V1 and future V2 corpus have not been reconciled record by record.
4. Source access in this environment is limited, and large full-source downloads, per-work licensing, source availability/checksums, full-scale performance and full quality validation remain unverified. The inherited Windows paths/configuration require local review and Windows testing. Historical audit and test outputs must not be mistaken for a current complete dataset.
5. Only some V1 evidence was copied; its full reports, original scripts, and both historical corpus versions remain accessible through the pinned Git history. Their omission here does not imply all unique historical records were reconciled. Review the older/newer record differences against originals before declaring no information lost.

These limitations mean the requested final dataset and a review-ready completed consolidation are not yet delivered.
