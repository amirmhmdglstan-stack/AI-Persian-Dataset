# Final repository consolidation report

Date: 2026-11-04. Scope: consolidate every usable repository contribution, preserve superseded information, reconcile the available historical corpora, and propose the result to main. No external source data or research was fetched. Git/GitHub were used only for this repository and PR delivery; PyPI supplied the existing software dependency.

## Branch audit and integration decisions

All available remote heads were fetched. The complete pinned file/commit inventory is [provenance/history_manifest.json](provenance/history_manifest.json).

| Original branch | Head | Decision |
|---|---|---|
| main | a770ebf | Original README only; PR base. |
| arena/f01a4e1e-ai-persian-dataset | 8e35d5e | Preserve both V1 corpus revisions and all scripts, reports, vocab and metadata in the offline bundle. Reconcile both revisions document-by-document. Reuse its tokenizer once. Do not adopt its older pipeline as the default. |
| arena/62203741-ai-persian-dataset | e9bbf53 | V2 pipeline, licensing research, Windows, Tatoeba, download-resume, balancing and tokenizer-comparison improvements. Incorporated through its descendants; every older file revision retained in the bundle. |
| arena/b2bcd196-ai-persian-dataset | dda8eaf | Extraction, transactional ingestion, download fallback and global-split fixes. Incorporated through the consolidation descendant. |
| arena/40ead0b9-ai-persian-dataset | 84179ce | Most complete maintained V2 tree: conservative exact identity, near-match retention/review, duplicate provenance verification, normalization fix, extra tests, branch-independent Windows guards, and V1 evidence. Imported as the active implementation. Its incomplete historical consolidation is completed here with a self-contained history bundle and actual cross-version document reconciliation. |

Earlier V2 branches are ancestors of the repair branch; the prior consolidation copied the repair implementation and added targeted changes. No original branch was modified. All work and the PR remain on `arena/715b0aea-ai-persian-dataset`.

## Preservation: nothing silently discarded

Nine original reachable commits contain 446 file revisions and 183 distinct file contents. All are recoverable **without network or Git history**:

- 71 unique contents map to identical active `dataset_v2/` files.
- 112 other unique contents are stored once in the 44 MiB content-addressed ZIP.
- Manifest entries retain original path, commit, file mode, Git object ID, SHA-256 and byte count.
- Restoration checks hashes and refuses existing targets. Tests cover restoration integrity/refusal, corrupt content detection, exact deduplication, variant preservation and validation precedence.

Thus even superseded research, old statistics, original build/tokenizer scripts, derived vocabulary and corpus variants remain accessible. They are not blindly mixed into the maintained code. The historical bundle includes the earlier consolidation report and its cautions as originally written. Historical reports are measurements/claims of older runs, not fresh production results.

The tokenizer is byte-identical across V1 and V2: SHA-256 `8bdecc85dd05009b2abb2148d2cea6ac15ed4815d3d8064e6ee8b1d2500c53a5`. The maintained JSON is the only stored instance of that content; archived manifest references resolve to it. The V1 derived vocabulary is also recoverable.

## Actual corpus reconciliation

Only V1 contains committed production-like text; V2 contains code, research and synthetic fixtures/reports, not another full corpus. Both V1 commits were reconciled offline:

| Measurement | Result |
|---|---:|
| Earlier V1 documents | 36,586 |
| Latest V1 documents | 36,592 |
| Input occurrences across both revisions | 73,178 |
| Shared exact documents / repeated occurrences consolidated | 35,723 |
| Earlier-only documents retained | 863 |
| Latest-only documents retained | 869 |
| Distinct documents in historical union | 37,455 |
| Historical train / validation assignments | 36,713 / 742 |

Identity is SHA-256 of UTF-8 document text after delimiter splitting and outer whitespace removal. No internal whitespace, punctuation, digit, ZWNJ or spelling normalization is applied. Near matches are retained because changed texts cannot safely be classified as duplicates without original-source review. Every exact repeated occurrence remains represented in record origins. A document ever appearing in validation is excluded from historical train, so exact identity cannot leak across these assignments. This does not establish near-match separation or suitable evaluation data.

`python tools/consolidate_history.py reconcile` reproduces the union from the committed bundle. It writes an ignored gzip with text, stable ID, all original Git-file locations and unresolved-rights labels. Its committed summary records measurements and output checksum. The generated union is not duplicated in Git: original raw bytes already ship in the archive. `restore` recovers every original file byte-for-byte, including outer whitespace omitted from document identity.

## Safety and rights boundaries

V1 contains mixed-license news, Twitter-derived and other text with unresolved rights/privacy, and lacks per-document upstream attribution. Git origins cannot repair this. Preservation of existing repository material does not establish permission to distribute or train on it. The historical archive/union is **not** approved as Profile A/B/C training input; do not publish a model from it without rights/privacy review. No blanket license was added. The baseline tokenizer also retains this provenance caveat.

The maintained V2 pipeline keeps source policy, profile separation, licensing matrix, quality checks, exact duplicate provenance and approximate near-match review. Its active behavior is intentionally conservative: exact extracted-text duplicates are removed within each profile, near matches retained. It does not prove legal compatibility or perfect semantic preservation.

## Validation performed in this integration

Linux, Python 3.11.2, tokenizers 0.23.3:

- All 446 historical file revisions / 183 unique contents verified against SHA-256, byte length and Git blob identity.
- Historical corpus union generated successfully: counts above and deterministic gzip checksum in its summary.
- New consolidation tests: **3 passed**, covering per-revision metadata/Git-object tampering, unsafe content locations, exact deduplication, variants/ZWNJ retention, origin retention, validation precedence, deterministic repeated output, tampering detection, head coverage, restoration bytes and overwrite refusal.
- Maintained V2 suite: **88 tests, passed; one optional pyarrow test skipped** (87 executed).
- Requirements check: passed; tokenizer has 32,000 entries and single-token document delimiter ID 0.
- Disposable synthetic integration run: **passed**, 54 inputs, 36 outputs, 2,576 content tokens, zero verification failures. No historical report files were overwritten.
- Compile checks and whitespace checks passed.
- Added GitHub Actions workflow for integrity checks, both test suites, actual historical corpus reconstruction/count comparison, requirements and disposable synthetic integration. Remote CI outcome is reported by GitHub, not assumed here.

## Remaining production limitations

This completes repository preservation/integration, not full production dataset creation. No upstream data was downloaded and no new full-source corpus or tokenizer was trained. Rights clearance, upstream document attribution, privacy review, human language-quality audit, source availability/checksums, full-scale performance and Windows execution remain outstanding. The inherited Windows launchers contain machine-specific paths and must be reviewed locally.

Approximate LSH may miss near matches; overlapping candidate clusters are not fully unioned. Distinct raw texts may normalize to equal output (the verifier flags this), and normalization/quality filtering can alter or reject meaningful text. Restricted work indexes must be retained securely for audit. No claim of perfect deduplication, zero semantic loss, complete source licensing or model readiness is made. See the maintained V2 known-limitations and licensing reports before a production run.
