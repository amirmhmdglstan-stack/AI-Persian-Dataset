"""Range validator and primary/fallback routing tests (mocked HTTP)."""
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'dataset_v2/pipeline'))
import download_sources as DS
from dataset_v2.tests.test_download_resume import FakeResponse, URL, LAST_MODIFIED


class DownloadSafety(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dest = Path(self.tmp.name) / 'file.bz2'
        self.part = self.dest.with_name(self.dest.name + '.part')
        self.sidecar = self.part.with_name(self.part.name + '.json')

    def tearDown(self):
        self.tmp.cleanup()

    def partial(self, data=b'abcd', *, etag='"v1"', last_modified=LAST_MODIFIED, digest=None):
        self.part.write_bytes(data)
        self.sidecar.write_text(json.dumps({'url': URL, 'size': len(data),
            'sha256': digest or hashlib.sha256(data).hexdigest(), 'etag': etag,
            'last_modified': last_modified}))

    def run_case(self, responses, **kwargs):
        requests = []
        def urlopen(req, timeout=None):
            requests.append(req)
            return responses.pop(0)
        with patch('download_sources.urllib.request.urlopen', urlopen):
            rec = DS.download_one(URL, self.dest, mutable_weekly_snapshot=True, **kwargs)
        return rec, requests

    def test_matching_etag_and_range(self):
        self.partial()
        rec, requests = self.run_case([FakeResponse(b'efgh', {'Content-Range': 'bytes 4-7/8',
            'ETag': '"v1"'}, 206)])
        self.assertEqual(requests[0].get_header('If-range'), '"v1"')
        self.assertEqual(self.dest.read_bytes(), b'abcdefgh')
        self.assertTrue(rec['resumed'])
        self.assertFalse(self.sidecar.exists())

    def test_200_restarts_instead_of_appending(self):
        self.partial()
        rec, _ = self.run_case([FakeResponse(b'new', {'ETag': '"v2"'}, 200)])
        self.assertEqual(self.dest.read_bytes(), b'new')
        self.assertFalse(rec['resumed'])

    def test_wrong_range_or_changed_validator_restarts(self):
        for headers in ({'Content-Range': 'bytes 3-7/8', 'ETag': '"v1"'},
                        {'Content-Range': 'bytes 4-7/8', 'ETag': '"v2"'}):
            with self.subTest(headers=headers):
                self.dest.unlink(missing_ok=True)
                self.partial()
                rec, reqs = self.run_case([FakeResponse(b'XXXX', headers, 206),
                    FakeResponse(b'new snapshot', {'ETag': '"v2"'}, 200)])
                self.assertEqual(len(reqs), 2)
                self.assertIsNone(reqs[1].get_header('Range'))
                self.assertEqual(self.dest.read_bytes(), b'new snapshot')

    def test_changed_last_modified_restarts(self):
        self.partial(etag='')
        rec, reqs = self.run_case([FakeResponse(b'efgh', {'Content-Range': 'bytes 4-7/8',
            'Last-Modified': 'Sun, 01 Nov 2026 06:39:00 GMT'}, 206),
            FakeResponse(b'new', {}, 200)])
        self.assertEqual(self.dest.read_bytes(), b'new')

    def test_missing_validator_and_corrupt_partial_restart(self):
        for options in ({'etag': '', 'last_modified': ''}, {'digest': 'incorrect'}):
            with self.subTest(options=options):
                self.dest.unlink(missing_ok=True)
                self.partial(**options)
                rec, reqs = self.run_case([FakeResponse(b'fresh', {}, 200)])
                self.assertIsNone(reqs[0].get_header('Range'))
                self.assertEqual(self.dest.read_bytes(), b'fresh')

    def test_force_clears_stale_partial_and_sidecar(self):
        self.partial()
        rec, reqs = self.run_case([FakeResponse(b'fresh', {}, 200)], force=True)
        self.assertIsNone(reqs[0].get_header('Range'))
        self.assertFalse(self.sidecar.exists())
        self.assertEqual(self.dest.read_bytes(), b'fresh')

    def test_fallback_only_on_permanent_http_error(self):
        meta = json.loads((ROOT / 'dataset_v2/sources/source_metadata.json').read_text())
        src = next(s for s in meta['sources'] if s['id'] == 'tatoeba_fa_cc_by')
        metadata = Path(self.tmp.name) / 'metadata.json'
        metadata.write_text(json.dumps({'sources': [src]}))
        config = ROOT / 'dataset_v2/configs/dataset_config.json'
        def execute(fail):
            calls = []
            def download(url, dest, **kwargs):
                calls.append(dest.name)
                if len(calls) == 1 and fail:
                    raise RuntimeError(fail)
                dest.write_bytes(b'ok')
                return {'url': url, 'filename': dest.name, 'size': 2,
                        'sha256': hashlib.sha256(b'ok').hexdigest()}
            args = ['download', '--config', str(config), '--metadata', str(metadata),
                    '--workdir', str(Path(self.tmp.name) / 'work'), '--min-free-gb', '0']
            with patch.object(sys, 'argv', args), patch.object(DS, 'download_one', download):
                if fail and 'Network' in fail:
                    with self.assertRaisesRegex(RuntimeError, 'Network'):
                        DS.main()
                else:
                    DS.main()
            return calls
        self.assertEqual(len(execute(None)), 1)
        self.assertEqual(len(execute('HTTP 404 while downloading')), 2)
        self.assertEqual(len(execute('HTTP 410 while downloading')), 2)
        self.assertEqual(len(execute('Network error temporary')), 1)

if __name__ == '__main__':
    unittest.main()
