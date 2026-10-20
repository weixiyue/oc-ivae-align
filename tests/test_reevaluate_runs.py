"""Batch discovery and result exports with a mocked evaluator; no model inference."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.reevaluate_runs import discover_runs, method_name, reevaluate_runs, TABLE_COLUMNS
from utils.metrics import evaluate_all


def make_run(root, name, files=('config.yaml', 'best.pt')):
    path = root / name
    path.mkdir(parents=True)
    for filename in files:
        (path / filename).write_text('original artifact', encoding='utf-8')
    return path


class ReevaluateRuns(unittest.TestCase):
    def test_discovery_prunes_history_and_records_missing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_run(root, 'group/complete')
            make_run(root, 'history/old')
            make_run(root, 'group/history/deeper/old')
            make_run(root, 'config_only', ('config.yaml',))
            make_run(root, 'weights_only', ('best.pt',))
            make_run(root, 'empty', ())
            make_run(root, 'group/complete/.matplotlib', ('fontlist.json',))
            records = {record['run']: record for record in discover_runs(root)}
            self.assertEqual(set(records), {'group/complete', 'config_only', 'weights_only', 'empty'})
            self.assertEqual(records['group/complete']['status'], 'pending')
            self.assertEqual(records['config_only']['reason'], 'missing best.pt')
            self.assertEqual(records['weights_only']['reason'], 'missing config.yaml')
            self.assertEqual(records['empty']['reason'], 'missing config.yaml, best.pt')
            self.assertEqual(discover_runs(root / 'history'), [])

    def test_dry_run_neither_loads_checkpoints_nor_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'runs'
            make_run(root, 'complete')
            make_run(root, 'incomplete', ('config.yaml',))
            output = Path(directory) / 'results'
            with patch('experiments.reevaluate_runs.run_evaluation') as evaluate, patch('torch.load') as load:
                summary = reevaluate_runs(str(root), str(output), 'cuda', dry_run=True)
            evaluate.assert_not_called()
            load.assert_not_called()
            self.assertFalse(output.exists())
            self.assertEqual(summary['counts'], dict(pending=1, success=0, skipped=1, failed=0))

    def test_batch_continues_and_exports_authoritative_pooled_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'runs'
            make_run(root, 'a_bad')
            make_run(root, 'group_a/same')
            make_run(root, 'group_b/same')
            make_run(root, 'incomplete', ('config.yaml',))
            make_run(root, 'history/ignored')
            originals = {path: path.read_bytes() for path in root.rglob('*') if path.is_file()}
            output = root / 'new_results'
            frame = pd.DataFrame(dict(
                unit=[1] * 4 + [2] * 8,
                cycle=list(range(1, 5)) + list(range(1, 9)),
                hi=[4., 3., 2., 1.] + list(np.arange(11., 3., -1.)),
                hi_gt=[4., 3., 2., 1.] + list(np.arange(8., 0., -1.)),
                rul=[3., 2., 1., 0.] + list(np.arange(7., -1., -1.)),
            ))
            metrics = evaluate_all(frame)

            def fake_evaluate(*, config_path, checkpoint_path, output_dir, device):
                source = Path(config_path).parent
                self.assertEqual(Path(checkpoint_path), source / 'best.pt')
                self.assertEqual(device, 'cpu')
                if source.name == 'a_bad':
                    raise ValueError('checkpoint/config mismatch')
                destination = Path(output_dir)
                destination.mkdir(parents=True, exist_ok=False)
                # Name must come from resolved evaluator metadata, not path/YAML.
                config = dict(model_type='kgld', experiment=dict(baseline_type='vae'))
                values = dict(metrics)
                if source.parent.name == 'group_b':
                    values.pop('mi')  # missing metric must stay missing
                    values['mae'] = None
                (destination / 'metrics.json').write_text(json.dumps(dict(
                    metadata=dict(config=config), dev=metrics, test=values)), encoding='utf-8')
                return destination

            with patch('experiments.reevaluate_runs.run_evaluation', side_effect=fake_evaluate) as evaluate:
                summary = reevaluate_runs(str(root), str(output), 'cpu')
            self.assertEqual(evaluate.call_count, 3)
            self.assertEqual(summary['counts'], dict(pending=0, success=2, skipped=1, failed=1))
            self.assertIn('checkpoint/config mismatch', summary['runs'][0]['reason'])
            self.assertIn('ValueError', summary['runs'][0]['traceback'])
            self.assertEqual(json.loads((output / 'summary.json').read_text()), summary)
            with (output / 'main_results.csv').open(encoding='utf-8', newline='') as file:
                reader = csv.DictReader(file)
                rows = list(reader)
                self.assertEqual(reader.fieldnames, TABLE_COLUMNS)
            self.assertEqual([row['运行名称'] for row in rows], ['group_a/same', 'group_b/same'])
            self.assertEqual([row['方法'] for row in rows], ['VAE', 'VAE'])
            self.assertAlmostEqual(float(rows[0]['RMSE↓']), np.sqrt(6.))
            self.assertNotEqual(float(rows[0]['RMSE↓']), 1.5)
            self.assertEqual(rows[1]['MAE↓'], '')
            self.assertEqual(rows[1]['MI↑'], '')
            self.assertIn('—', (output / 'main_results.md').read_text())
            for name in ('group_a', 'group_b'):
                detailed = json.loads((output / 'evaluations' / name / 'same/metrics.json').read_text())
                self.assertIn('mape', detailed['test'])
                self.assertIn('dev', detailed)
            for path, contents in originals.items():
                self.assertEqual(path.read_bytes(), contents)

    def test_output_guards_and_empty_table(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'runs'
            source = make_run(root, 'complete')
            with self.assertRaises(FileExistsError):
                reevaluate_runs(str(root), str(source), dry_run=True)
            with self.assertRaisesRegex(ValueError, 'inside a source run'):
                reevaluate_runs(str(root), str(source / 'nested'))
            self.assertFalse((source / 'nested').exists())
            empty = Path(directory) / 'empty'
            empty.mkdir()
            output = Path(directory) / 'results'
            with patch('experiments.reevaluate_runs.run_evaluation') as evaluate:
                summary = reevaluate_runs(str(empty), str(output))
            evaluate.assert_not_called()
            self.assertEqual(sum(summary['counts'].values()), 0)
            self.assertEqual(len((output / 'main_results.csv').read_text().splitlines()), 1)
            self.assertEqual(len((output / 'main_results.md').read_text().splitlines()), 2)

    def test_method_names_follow_model_configuration(self):
        cases = [
            ({}, 'KGLD'),
            (dict(model_type='oc_ivae'), 'KGLD'),
            (dict(experiment=dict(baseline_type='vae')), 'VAE'),
            (dict(experiment=dict(baseline_type='cvae')), 'cVAE'),
            (dict(model_type='proposal_cnn'), 'Proposal-C'),
            (dict(model_type='ocald', model=dict(decoder_use_w=False), training=dict(pipeline='two_stage')),
             'OCALD (two_stage, decoder W=False)'),
            (dict(model_type='ocald', model=dict(decoder_use_w=True), training=dict(pipeline='joint')),
             'OCALD (joint, decoder W=True)'),
            (dict(model_type='residual', model=dict(variant='A'), data={}), 'Residual-A (flight)'),
            (dict(model_type='residual', model=dict(variant='B'), data=dict(input_mode='short_window')),
             'Residual-B (short_window)'),
        ]
        for config, expected in cases:
            with self.subTest(config=config):
                self.assertEqual(method_name(config), expected)


if __name__ == '__main__':
    unittest.main()
