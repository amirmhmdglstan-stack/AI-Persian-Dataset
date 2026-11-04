import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

SPEC = importlib.util.spec_from_file_location('history', Path(__file__).resolve().parents[1] / 'tools/consolidate_history.py')
history = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(history)


class HistoryTests(unittest.TestCase):
    def test_reconcile_preserves_variants_origins_and_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive_path = root / 'blobs.zip'
            entries = []
            with zipfile.ZipFile(archive_path, 'w') as archive:
                for commit, path, text in [('old', 'train.txt', 'می\u200cرود<|endoftext|>سلام'),
                                           ('new', 'validation.txt', 'می\u200cرود<|endoftext|>میرود')]:
                    data = text.encode()
                    digest = history.sha(data)
                    archive.writestr('blobs/' + digest, data)
                    entries.append(dict(commit=commit, path=path, bytes=len(data), sha256=digest,
                                        location='historical-blobs.zip!blobs/' + digest))
            with zipfile.ZipFile(archive_path) as archive:
                for entry in entries:
                    entry['mode'] = '100644'
                history.restore({'files': entries}, archive, 'old', root / 'restored')
                self.assertEqual((root / 'restored/train.txt').read_bytes(), 'می\u200cرود<|endoftext|>سلام'.encode())
                with self.assertRaises(FileExistsError):
                    history.restore({'files': entries}, archive, 'old', root / 'restored')
                history.reconcile({'files': entries}, archive, root / 'out')
                first = (root / 'out/documents.jsonl.gz').read_bytes()
                history.reconcile({'files': entries}, archive, root / 'out2')
                self.assertEqual(first, (root / 'out2/documents.jsonl.gz').read_bytes())
                broken = dict(entries[0], bytes=0)
                with self.assertRaises(ValueError):
                    history.read_entry(broken, archive)
            rows = [json.loads(line) for line in gzip.decompress(first).decode().splitlines()]
            self.assertEqual(len(rows), 3)
            repeated = next(r for r in rows if r['text'] == 'می\u200cرود')
            self.assertEqual(len(repeated['origins']), 2)
            self.assertEqual(repeated['historical_split'], 'validation')
            self.assertTrue(all(r['rights_status'].startswith('unresolved') for r in rows))

    def test_every_revision_metadata_is_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'blobs.zip'
            data = b'original file'
            digest = history.sha(data)
            entry = dict(path='a', bytes=len(data), sha256=digest,
                         location='historical-blobs.zip!blobs/' + digest)
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('blobs/' + digest, data)
            with zipfile.ZipFile(path) as archive:
                history.verify({'files': [entry, dict(entry)]}, archive)
                with self.assertRaises(ValueError):
                    history.verify({'files': [entry, dict(entry, bytes=0)]}, archive)
                with self.assertRaises(ValueError):
                    history.read_entry(dict(entry, git_blob='0' * 40), archive)
                with self.assertRaises(ValueError):
                    history.read_entry(dict(entry, location='../outside-repository'), archive)

    def test_manifest_covers_every_pinned_head(self):
        manifest = json.loads((history.PROV / 'history_manifest.json').read_text())
        commits = {e['commit'] for e in manifest['files']}
        self.assertTrue(set(manifest['heads'].values()).issubset(commits))
        self.assertEqual(len(manifest['heads']), 5)


if __name__ == '__main__':
    unittest.main()
