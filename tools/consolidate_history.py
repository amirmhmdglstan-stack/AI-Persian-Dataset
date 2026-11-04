#!/usr/bin/env python3
"""Preserve all file versions reachable from pinned agent heads, once per blob.

No network access. Regeneration requires fetched Git history. Verification and
historical corpus reconciliation work from the committed bundle alone.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PROV = ROOT / 'provenance'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def build():
    heads = {}
    for ref in git('for-each-ref', '--format=%(refname:short)', 'refs/remotes/origin').decode().splitlines():
        if ref == 'origin/main' or (ref.startswith('origin/arena/') and '715b0aea' not in ref):
            heads[ref] = git('rev-parse', ref).decode().strip()
    commits = sorted(set(git('rev-list', *heads.values()).decode().splitlines()))
    current = {}
    for p in (ROOT / 'dataset_v2').rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:
            current[sha(p.read_bytes())] = p.relative_to(ROOT).as_posix()
    entries, stored = [], set()
    PROV.mkdir(exist_ok=True)
    with zipfile.ZipFile(PROV / 'historical-blobs.zip', 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for commit in commits:
            for record in git('ls-tree', '-r', '-z', commit).split(b'\0'):
                if not record:
                    continue
                meta, name = record.split(b'\t', 1)
                mode, kind, oid = meta.decode().split()
                if kind != 'blob':
                    raise ValueError('Unsupported Git entry: ' + kind)
                data = git('cat-file', 'blob', oid)
                digest = sha(data)
                active = current.get(digest)
                location = active or 'historical-blobs.zip!blobs/' + digest
                if not active and digest not in stored:
                    info = zipfile.ZipInfo('blobs/' + digest, date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(info, data)
                    stored.add(digest)
                entries.append(dict(commit=commit, path=name.decode(), mode=mode, git_blob=oid,
                                    sha256=digest, bytes=len(data), location=location))
    manifest = dict(schema_version=1, heads=heads, commits=commits, files=entries)
    (PROV / 'history_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Preserved {len(entries)} file revisions; {len(stored)} archived unique blobs')


def read_entry(entry, archive):
    loc = entry['location']
    if loc.startswith('historical-blobs.zip!'):
        data = archive.read(loc.split('!', 1)[1])
    else:
        target = (ROOT / loc).resolve()
        if not target.is_relative_to(ROOT.resolve()):
            raise ValueError('Unsafe content location')
        data = target.read_bytes()
    validate_entry(entry, data)
    return data


def validate_entry(entry, data):
    if len(data) != entry['bytes'] or sha(data) != entry['sha256']:
        raise ValueError('Content mismatch: ' + entry['path'])
    if 'git_blob' in entry:
        header = f'blob {len(data)}'.encode('ascii') + bytes([0])
        if hashlib.sha1(header + data).hexdigest() != entry['git_blob']:
            raise ValueError('Git object mismatch: ' + entry['path'])


def verify(manifest, archive):
    # Check every revision's metadata, not just the first entry for each digest.
    # Process one unique content at a time to avoid holding the corpus in RAM.
    groups = {}
    for entry in manifest['files']:
        groups.setdefault(entry['location'], []).append(entry)
    for entries in groups.values():
        data = read_entry(entries[0], archive)
        for entry in entries[1:]:
            validate_entry(entry, data)
    print(f'Verified {len(manifest["files"])} file revisions / '
          f'{len({e["sha256"] for e in manifest["files"]})} unique contents')


def reconcile(manifest, archive, output):
    # Historical/restricted material only: never presented as rights-cleared.
    docs, occurrences = {}, 0
    corpus_entries = [e for e in manifest['files'] if e['path'] in ('train.txt', 'validation.txt')]
    for entry in corpus_entries:
        text = read_entry(entry, archive).decode('utf-8')
        for index, doc in enumerate(text.split('<|endoftext|>')):
            doc = doc.strip()
            if not doc:
                continue
            digest = sha(doc.encode('utf-8'))
            record = docs.setdefault(digest, dict(id=digest, text=doc, origins=[]))
            if record['text'] != doc:
                raise ValueError('SHA256 collision')
            record['origins'].append(dict(commit=entry['commit'], path=entry['path'], document_index=index))
            occurrences += 1
    output.mkdir(parents=True, exist_ok=True)
    counts = {'train': 0, 'validation': 0}
    # If ever in validation, never assign to training in this historical union.
    with (output / 'documents.jsonl.gz').open('wb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as compressed:
            for digest in sorted(docs):
                record = docs[digest]
                split = 'validation' if any(o['path'] == 'validation.txt' for o in record['origins']) else 'train'
                record.update(historical_split=split, rights_status='unresolved_do_not_train_or_redistribute')
                counts[split] += 1
                compressed.write((json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n').encode())
    summary = dict(input_document_occurrences=occurrences, unique_documents=len(docs),
                   exact_repeated_occurrences=occurrences-len(docs), historical_splits=counts,
                   corpus_file_revisions=len(corpus_entries), sha256=sha((output / 'documents.jsonl.gz').read_bytes()),
                   identity='SHA256 of UTF-8 document after outer whitespace stripping; no normalization or near-match deletion',
                   warning='Historical mixed-rights corpus, NOT a training release. Origins identify Git files, not original sources. Near variants retained.')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


def restore(manifest, archive, commit, output):
    """Restore a pinned snapshot, including exact original corpus bytes, offline."""
    entries = [e for e in manifest['files'] if e['commit'] == commit]
    if not entries:
        raise ValueError('Use a full commit from history_manifest.json')
    output = output.resolve()
    targets = []
    for entry in entries:
        target = (output / entry['path']).resolve()
        if not target.is_relative_to(output) or target == output:
            raise ValueError('Unsafe snapshot path')
        if target.exists():
            raise FileExistsError(target)
        targets.append((entry, target))
    for entry, target in targets:
        data = read_entry(entry, archive)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as handle:
            handle.write(data)
        target.chmod(int(entry['mode'], 8) & 0o777)
    print(f'Restored {len(targets)} files to {output}; historical code is NOT executed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build', 'verify', 'reconcile', 'restore'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--commit', help='Full pinned commit for restore')
    args = parser.parse_args()
    if args.action == 'build':
        build()
        return
    manifest = json.loads((PROV / 'history_manifest.json').read_text(encoding='utf-8'))
    with zipfile.ZipFile(PROV / 'historical-blobs.zip') as archive:
        if args.action == 'verify':
            verify(manifest, archive)
        elif args.action == 'restore':
            if not args.commit or args.output is None:
                parser.error('restore requires --commit and --output (a new directory)')
            restore(manifest, archive, args.commit, args.output)
        else:
            reconcile(manifest, archive, args.output or ROOT / 'historical_corpus')


if __name__ == '__main__':
    main()
