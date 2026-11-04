# Offline preservation and provenance

`history_manifest.json` pins main and all four pre-existing agent heads and lists every file in all nine reachable commits: **446 file revisions, 183 distinct contents**. Each entry records its commit, path, mode, Git blob ID, SHA-256, byte length and storage location.

- 71 distinct contents already present byte-for-byte under `dataset_v2/` are referenced there rather than duplicated.
- The other 112 distinct contents are stored once under `blobs/<sha256>` in `historical-blobs.zip` (about 44 MiB). This includes all omitted/superseded scripts and reports and exact original bytes of both V1 corpus revisions.
- Historical code is inert, not imported or run by the maintained pipeline. Archive filenames are content hashes; the manifest supplies original names.
- This bundle preserves existing repository content, **not a new permission grant**. In particular it contains mixed-rights/possibly personal historical text. Existing repository access does not establish rights to train on or redistribute it. Keep recovered data under appropriate access controls. No blanket corpus license is asserted.

## Verify without Git or network

```sh
python tools/consolidate_history.py verify
```

This checks SHA-256 and length of all 183 distinct contents against the manifest. Checksums establish integrity, not licensing or semantic correctness.

## Recover any original snapshot

Use a full commit from the manifest and a new output directory **outside the maintained checkout**. For example, recover the final V1 revision:

```sh
python tools/consolidate_history.py restore \
  --commit 8e35d5e7e72597eda96e83db9c01a1335f235e19 \
  --output ../historical-v1-review
```

All original files, including train/validation, V1-only scripts and vocabulary, are reconstructed byte-for-byte with original file modes. Existing targets are refused; historical code is not executed. The older V1 snapshot and all earlier V2 reports/configurations can be restored similarly. Treat historical scripts as unreviewed code; the V2 audit explains known V1 defects.

## Deduplicated historical document union

```sh
python tools/consolidate_history.py reconcile
```

This reads the four original corpus files (train/validation at two commits), splits on their original `<|endoftext|>` delimiter, strips only outer whitespace and deduplicates identical UTF-8 documents. It does **not** normalize letters, digits, punctuation or ZWNJs, or delete near matches. The output retains every occurrence's commit, file and zero-based delimiter segment index, and labels every record `unresolved_do_not_train_or_redistribute`.

Measured: 73,178 occurrences → 37,455 distinct documents; 35,723 repeated occurrences consolidated. Old-only documents are preserved rather than assumed obsolete. If a document appeared in validation in either version, its historical assignment is validation (742 documents); the other 36,713 are historically assigned train. These labels only prevent **exact** cross-split overlap, not near-duplicate leakage. They are not recommended model evaluation splits. Git-file origins cannot reconstruct missing upstream attribution.

Output gzip is deterministic for the same Python/zlib implementation and input order. Its measured SHA-256 is recorded in `historical_corpus/summary.json`; JSON content identity is independent of gzip implementation. The ~49 MiB generated union is ignored to avoid committing another copy of recoverable text. Original corpus bytes already ship in the preservation bundle.

## Regenerate the preservation bundle

`python tools/consolidate_history.py build` requires the fetched original remote refs and full Git history. It inventories all reachable commits of main and pre-existing agent branches, excluding this integration branch. The committed manifest is the authoritative pinned audit; regenerating after remote changes intentionally produces a new inventory. Building is a maintainer operation, not necessary for users or CI.
