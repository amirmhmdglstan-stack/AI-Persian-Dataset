"""Synthetic on-disk regression tests exercising production entry points and archive streams."""
import bz2
import io
import os
import tracemalloc
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'dataset_v2' / 'pipeline'))
import extract_sources as EX
import build_corpus as BC
from common import load_config, sha256_file
from dataset_v2.tests.test_extraction import context, add_tar_member, source
from dataset_v2.tests.run_smoke_test import make_record, paragraph

CONFIG = ROOT / 'dataset_v2/configs/dataset_config.json'


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.work = self.root / 'work'
        self.out = self.root / 'out'
        self.reports = self.root / 'reports'

    def tearDown(self):
        self.tmp.cleanup()

    def test_real_tar_gz_and_bz2_poems_streamed_with_provenance(self):
        for mode, ext in [('w:gz', '.tar.gz'), ('w:bz2', '.tar.bz2')]:
            with self.subTest(ext=ext):
                path = self.root / ('poems' + ext)
                data = 'poet\tcentury\tpoem\nحافظ\t۸\tشعر فارسی اول\nسعدی\t۷\tشعر فارسی دوم\n'.encode()
                with tarfile.open(path, mode) as tf:
                    add_tar_member(tf, 'archive/poems.tsv', data)
                    add_tar_member(tf, 'archive/other.tsv', b'poet\tcentury\tpoem\nno\t1\twrong\n')
                ctx = context(self.root, [path], {
                    'has_header': True, 'columns': ['poet', 'century', 'poem'],
                    'text_column': 'poem', 'member_pattern': '*poems.tsv',
                    'delimiter': '\t'}, source(id='persian_poetry_ccbysa',
                    url='https://github.com/aghasemi/ChronologicalPersianPoetryDataset',
                    license_compilation='CC BY-SA 4.0'))
                docs = list(EX.extract_tsv(ctx))
                self.assertEqual([d['text'] for d in docs], ['شعر فارسی اول', 'شعر فارسی دوم'])
                self.assertEqual([d['meta']['poet'] for d in docs], ['حافظ', 'سعدی'])
                self.assertEqual(docs[0]['meta']['century'], '۸')
                self.assertEqual(docs[0]['meta']['member'], 'archive/poems.tsv')
                self.assertEqual(docs[0]['meta']['line'], 1)
                self.assertEqual(docs[0]['meta']['compilation_license'], 'CC BY-SA 4.0')
                self.assertNotEqual(docs[0]['doc_id'], docs[1]['doc_id'])
                self.assertEqual(ctx.stats['raw_records'], 2)

    def test_large_unrelated_tar_member_not_buffered(self):
        path = self.root / 'large.tar.gz'
        with tarfile.open(path, 'w:gz') as tf:
            add_tar_member(tf, 'poems.tsv',
                           'poet\tcentury\tpoem\nالف\t۱\tشعر آزمایشی\n'.encode())
            add_tar_member(tf, 'unrelated.bin', os.urandom(6 * 1024 * 1024))
        ctx = context(self.root, [path], {'columns': ['poet', 'century', 'poem'],
            'has_header': True, 'text_column': 'poem', 'member_pattern': '*poems.tsv'})
        tracemalloc.start()
        try:
            docs = list(EX.extract_tsv(ctx))
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertEqual(len(docs), 1)
        self.assertLess(peak, 3 * 1024 * 1024)

    def test_tatoeba_already_downloaded_fallback_is_selected_without_primary(self):
        meta = json.loads((ROOT / 'dataset_v2/sources/source_metadata.json').read_text())
        src = next(x for x in meta['sources'] if x['id'] == 'tatoeba_fa_cc_by')
        folder = self.work / 'raw' / src['id']
        folder.mkdir(parents=True)
        fallback = folder / 'sentences_detailed.tar.bz2'
        with tarfile.open(fallback, 'w:bz2') as tf:
            add_tar_member(tf, 'sentences_detailed.csv',
                '1\tpes\tسلام به جهان\tali\t2026-10-31\t2026-11-01\n'
                '2\teng\tHello\tbob\t2026-10-31\t2026-11-01\n'.encode())
        paths, _, ex = EX.source_raw_paths(src, self.work / 'raw', False, context(
            self.root, [], {}).logger)
        self.assertEqual(paths, [fallback])
        docs = list(EX.extract_tatoeba_detailed(context(self.root, paths,
            src['files'][1]['options'], src=src)))
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]['meta']['source_filename'], fallback.name)
        self.assertEqual(docs[0]['meta']['tatoeba_username_for_attribution_only'], 'ali')
        primary = folder / src['files'][0]['filename']
        with bz2.open(primary, 'wt') as f:
            f.write('3\tpes\tیک جمله فارسی\tsara\t2026-10-31\t2026-11-01\n')
        self.assertEqual(EX.source_raw_paths(src, self.work / 'raw', False, context(
            self.root, [], {}).logger)[0], [primary])

    def extraction(self, raw_text, *, limit=None, options=None, policy=None):
        src = source(id='fixture', download=True, files=[{'filename': 'poems.tsv', 'extractor': 'tsv',
            'required': True, 'format': 'tsv', 'options': options or {
            'columns': ['poet', 'century', 'poem'], 'has_header': True,
            'text_column': 'poem', 'delimiter': '\t'}}])
        raw = self.work / 'raw/fixture/poems.tsv'
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text(raw_text, encoding='utf-8')
        meta = self.root / 'metadata.json'
        meta.write_text(json.dumps({'sources': [src]}), encoding='utf-8')
        config = json.loads(CONFIG.read_text(encoding='utf-8'))
        if policy:
            pp = self.root / 'policy.json'
            original = json.loads((ROOT / config['source_policy']).read_text(encoding='utf-8'))
            original['test_change'] = policy
            pp.write_text(json.dumps(original), encoding='utf-8')
            config['source_policy'] = str(pp)
        cfg = self.root / 'config.json'
        cfg.write_text(json.dumps(config), encoding='utf-8')
        argv = ['extract', '--config', str(cfg), '--metadata', str(meta),
                '--workdir', str(self.work), '--min-free-gb', '0']
        if limit:
            argv += ['--max-docs', str(limit)]
        with patch.object(sys, 'argv', argv):
            EX.main()
        return json.loads((self.work / 'extract_manifest.json').read_text())['sources']['fixture']

    def test_partial_recovery_and_input_invalidation(self):
        raw = 'poet\tcentury\tpoem\nالف\t۱\tشعر اول\nب\t۲\tشعر دوم\n'
        part = self.extraction(raw, limit=1)
        final = self.work / 'extracted/A/fixture.jsonl'
        self.assertEqual(part['status'], 'partial')
        self.assertFalse(final.exists())
        (final.with_name(final.name + '.part')).write_text('crash debris', encoding='utf-8')
        done = self.extraction(raw)
        self.assertEqual(done['status'], 'complete')
        self.assertEqual(len(final.read_text().splitlines()), 2)
        self.assertEqual(done['sha256'], sha256_file(final))
        self.assertEqual(self.extraction(raw)['extracted_at_utc'], done['extracted_at_utc'])
        new = self.extraction(raw.replace('شعر دوم', 'شعر تازه'))
        self.assertNotEqual(done['sha256'], new['sha256'])
        changed_options = {'columns': ['poet', 'century', 'poem'], 'has_header': True,
                           'text_column': 'poem', 'delimiter': '\t', 'meta_columns': ['poet']}
        self.assertNotEqual(new['input_identity'], self.extraction(raw.replace('شعر دوم', 'شعر تازه'),
                             options=changed_options)['input_identity'])
        self.assertNotEqual(new['input_identity'], self.extraction(raw.replace('شعر دوم', 'شعر تازه'),
                             policy='changed')['input_identity'])

    def test_extractor_exception_marks_failed_and_builder_rejects_partial(self):
        raw = 'poet\tcentury\tpoem\nالف\t۱\tشعر اول\nب\t۲\tشعر دوم\n'
        def interrupted(ctx):
            yield next(EX.extract_tsv(ctx))
            raise OSError('simulated interruption')
        with patch.dict(EX.EXTRACTORS, {'tsv': interrupted}):
            with self.assertRaisesRegex(OSError, 'simulated interruption'):
                self.extraction(raw)
        manifest = json.loads((self.work / 'extract_manifest.json').read_text())
        self.assertEqual(manifest['sources']['fixture']['status'], 'failed')
        self.assertFalse((self.work / 'extracted/A/fixture.jsonl').exists())
        self.extraction(raw, limit=1)
        # A manually left-over normal file must not override a partial manifest.
        (self.work / 'extracted/A/fixture.jsonl').write_text('{}\n')
        argv = ['build', '--config', str(CONFIG), '--workdir', str(self.work),
                '--outdir', str(self.out), '--reports-dir', str(self.reports),
                '--min-free-gb', '0', '--force']
        with patch.object(sys, 'argv', argv):
            with self.assertRaisesRegex(RuntimeError, 'not verified complete'):
                BC.main()

    def build(self, extra=(), cfg=None):
        argv = ['build', '--config', str(cfg or CONFIG), '--workdir', str(self.work),
                '--extracted', str(self.root / 'extracted'), '--outdir', str(self.out),
                '--reports-dir', str(self.reports), '--scope', 'small_sample',
                '--min-free-gb', '0', *extra]
        with patch.object(sys, 'argv', argv):
            BC.main()

    def test_exact_and_near_candidates_preserve_variants_and_duplicate_provenance(self):
        import sqlite3
        extracted = self.root / 'extracted/A'
        extracted.mkdir(parents=True)
        text = paragraph(0, 9)
        # Subtle punctuation and wording differences must not be discarded.
        variant = text.replace('،', '؛', 1) if '،' in text else text + ' جمله دیگری است.'
        variant2 = text + ' این جمله تازه نیز برای آزمایش اضافه شده است.'
        docs = [make_record('smoke_news', 0, text, 'news'),
                make_record('smoke_news', 1, variant, 'news'),
                make_record('smoke_news', 2, variant2, 'news')]
        (extracted / 'smoke_news.jsonl').write_text(
            ''.join(json.dumps(d, ensure_ascii=False) + '\n' for d in docs), encoding='utf-8')
        duplicate = make_record('smoke_other', 0, text, 'news')
        duplicate['meta']['attribution'] = 'different contributor'
        (extracted / 'smoke_other.jsonl').write_text(
            json.dumps(duplicate, ensure_ascii=False) + '\n', encoding='utf-8')
        self.build(['--force'])
        manifest = [json.loads(s) for s in (self.out / 'profile_A/documents.jsonl').read_text(encoding='utf-8').splitlines()]
        duplicates = [json.loads(s) for s in (self.out / 'profile_A/duplicate_sources.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(manifest), 3)
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(duplicates[0]['source_id'], 'smoke_other')
        self.assertEqual(duplicates[0]['meta']['attribution'], 'different contributor')
        self.assertEqual(duplicates[0]['meta']['extracted_record']['line'], 1)
        self.assertIn(duplicates[0]['representative_doc_id'], {r['doc_id'] for r in manifest})
        candidates = [json.loads(s) for s in (self.out / 'near_match_review.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertGreater(len(candidates), 0)
        self.assertTrue(all('similarity' in candidate and 'text' not in candidate and
                            'excerpt' not in candidate for candidate in candidates))
        with sqlite3.connect(self.work / 'corpus_index.sqlite3') as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM docs WHERE status='near_duplicate'").fetchone()[0], 0)
            self.assertGreater(conn.execute("SELECT COUNT(*) FROM audit WHERE reason='near_match_retained'").fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT COUNT(DISTINCT split) FROM docs WHERE status='kept'").fetchone()[0], 1)
        import verify_dataset as verify
        verify_out = self.root / 'verify.json'
        argv = ['verify', '--config', str(CONFIG), '--workdir', str(self.work),
                '--outdir', str(self.out), '--reports-dir', str(self.reports),
                '--scope', 'small_sample', '--out', str(verify_out)]
        with patch.object(sys, 'argv', argv):
            verify.main()
        checks = json.loads(verify_out.read_text())['checks']
        self.assertEqual(next(c for c in checks if c['check'] == 'duplicate_provenance_matches:A')['status'], 'PASS')
        duplicates[0]['source_id'] = 'tampered'
        (self.out / 'profile_A/duplicate_sources.jsonl').write_text(
            json.dumps(duplicates[0], ensure_ascii=False) + '\n', encoding='utf-8')
        with patch.object(sys, 'argv', argv), self.assertRaises(SystemExit):
            verify.main()
        checks = json.loads(verify_out.read_text())['checks']
        self.assertEqual(next(c for c in checks if c['check'] == 'duplicate_provenance_matches:A')['status'], 'FAIL')

    def test_transactional_resume_after_stale_external_checkpoint(self):
        extracted = self.root / 'extracted/A'
        extracted.mkdir(parents=True)
        docs = [make_record('smoke_news', i, paragraph(i * 10, 8), 'news') for i in range(4)]
        (extracted / 'smoke_news.jsonl').write_text(''.join(json.dumps(d, ensure_ascii=False) + '\n'
                                                      for d in docs), encoding='utf-8')
        self.build(['--force', '--max-docs', '2'])
        self.assertTrue((self.reports / 'partial_build.json').exists())
        self.assertFalse((self.reports / 'build_report.json').exists())
        self.assertFalse((self.out / 'profile_A/train.txt').exists())
        import sqlite3
        conn = sqlite3.connect(self.work / 'corpus_index.sqlite3')
        state = json.loads(conn.execute("SELECT v FROM ingestion WHERE k='state'").fetchone()[0])
        self.assertEqual((state['last_source'], state['last_line']), ('smoke_news', 2))
        conn.close()
        # Simulate DB commit ahead of external checkpoint plus uncommitted shard tail.
        (self.work / 'checkpoint.json').write_text('{"processed_docs": 0}')
        with Path(state['shard_state']['current_shard']).open('ab') as f:
            f.write(b'UNCOMMITTED TAIL')
        self.build(['--resume'])
        report = json.loads((self.reports / 'build_report.json').read_text())
        self.assertEqual(report['statistics']['documents_seen'], 4)
        self.assertEqual(report['statistics']['documents_accepted'] +
                         report['statistics']['documents_rejected'] +
                         report['statistics']['exact_duplicates'] +
                         report['statistics']['near_duplicates'], 4)
        conn = sqlite3.connect(self.work / 'corpus_index.sqlite3')
        self.assertEqual(conn.execute('SELECT COUNT(*) FROM docs').fetchone()[0], 4)
        self.assertEqual(json.loads((self.out / "profile_A/documents.jsonl").read_text().splitlines()[0])["meta"]["attribution"], "synthetic unit-test fixture")
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM docs WHERE status='kept' AND byte_len>0").fetchone()[0],
                         report['statistics']['documents_accepted'])
        conn.close()
        modified = json.loads(CONFIG.read_text(encoding='utf-8'))
        modified['seed'] = 'different seed'
        path = self.root / 'modified.json'
        path.write_text(json.dumps(modified), encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError, 'Incompatible build'):
            self.build(['--resume'], cfg=path)
        self.build(['--resume'])  # identical run is compatible

if __name__ == '__main__':
    unittest.main()
