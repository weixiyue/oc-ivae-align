"""Loss-ablation guards and an actual two-epoch CPU smoke on temporary synthetic data."""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import torch
import yaml

from dataset import NCMAPSSDataset
from evaluate import load_evaluation_model, predict_cycles, read_evaluation_checkpoint
from experiments import run_loss_ablation as experiment
from models.kgld import KGLD
import train as training
from utils.losses import compute_total_loss


def baseline_config():
    return yaml.safe_load(experiment.BASE_CONFIG.read_text())


class LossAblation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_strict_baseline_and_only_designated_weight_changes(self):
        baseline = baseline_config()
        baseline['run_metadata'] = dict(old_run=True)
        baseline['run_dir'] = 'old_run'
        baseline['experiment'] = dict(name='old_name', description='old_description', seed=8)
        original = copy.deepcopy(baseline)
        configs = experiment.variant_configs(baseline, Path('/tmp/loss_ablation_test_only'), 17)
        self.assertEqual(list(configs), ['no_hsic', 'no_pairwise', 'no_spearman'])
        for variant, config in configs.items():
            differences = [key for key, value in config['loss_weights'].items()
                           if value != baseline['loss_weights'][key]]
            self.assertEqual(differences, [experiment.VARIANTS[variant]])
            self.assertEqual(config['training'], baseline['training'])
            self.assertEqual(config['model'], baseline['model'])
            self.assertEqual(config['data'], baseline['data'])
            self.assertEqual(config['experiment']['seed'], 17)
            self.assertNotIn('old_run', config['run_metadata'])
            self.assertEqual(config['run_metadata']['sampler_seed'], 42)
            experiment.check_variant_parameters(baseline, config, variant)
        configs['no_hsic']['training']['lr'] *= 2
        with self.assertRaisesRegex(ValueError, 'beyond the designated'):
            experiment.check_variant_parameters(baseline, configs['no_hsic'], 'no_hsic')
        self.assertEqual(configs['no_pairwise']['training']['lr'], baseline['training']['lr'])
        self.assertEqual(baseline, original)
        for weights in ({}, dict(experiment.EXPECTED_LOSS_WEIGHTS, beta_d=.5),
                        dict(experiment.EXPECTED_LOSS_WEIGHTS, extra=1),
                        dict(experiment.EXPECTED_LOSS_WEIGHTS, lambda_hsic=True)):
            with self.subTest(weights=weights), self.assertRaisesRegex(ValueError, 'loss_weights mismatch'):
                experiment.variant_configs(dict(baseline, loss_weights=weights), Path('/tmp/unused'), 42)

    def test_dry_run_has_no_data_training_or_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'output'
            stream = io.StringIO()
            with patch.object(experiment, 'NCMAPSSDataset') as dataset, \
                    patch.object(experiment.training, 'train_model') as train, \
                    patch.object(experiment, 'run_evaluation') as evaluate, \
                    patch.object(experiment, 'KGLD') as model, contextlib.redirect_stdout(stream):
                experiment.run(output, 17, 'cuda', dry_run=True)
            for mocked in (dataset, train, evaluate, model):
                mocked.assert_not_called()
            self.assertFalse(output.exists())
            self.assertIn('"seed": 17', stream.getvalue())
            self.assertIn('"before": 10', stream.getvalue())
            for variant in experiment.VARIANTS:
                self.assertIn(str(output / variant), stream.getvalue())
            output.mkdir()
            with self.assertRaises(FileExistsError):
                experiment.run(output, 17, 'cpu')

    def test_checkpoint_phase_numbering_and_training_vs_selection(self):
        settings = dict(warmup_end_epoch=10, transition_end_epoch=25)
        for epoch, expected in ((10, 'reconstruction'), (11, 'kl_hsic_transition'),
                                (25, 'kl_hsic_transition'), (26, 'temporal_constraints')):
            report = experiment.checkpoint_stage(epoch, settings, 30)
            self.assertEqual(report['phase'], expected)
            self.assertEqual(report['best_checkpoint_in_temporal_stage'], epoch >= 26)
            self.assertEqual(report['first_temporal_checkpoint_epoch'], 26)
            self.assertTrue(report['training_reached_temporal_stage'])
        self.assertFalse(experiment.checkpoint_stage(2, settings, 2)['training_reached_temporal_stage'])

    def test_weight_changes_remove_only_the_specified_objective_term(self):
        baseline = baseline_config()
        baseline['model'].update(in_features=14, c_hidden=4, n_backbone_layers=1, n_decoder_layers=1)
        model = KGLD(baseline['model']).eval()
        x, w, u = torch.randn(8, 8, 14), torch.randn(8, 8, 4), torch.randn(8, 11)
        mask = torch.ones(8, 8, dtype=torch.bool)
        out = model(x, mask, u, w)
        units, cycles = torch.ones(8, dtype=torch.long), torch.arange(1, 9)
        settings = {**baseline['training'], **baseline['loss_weights']}
        epoch = settings['transition_end_epoch']
        full, terms = compute_total_loss(out, x, mask, cycles, units, epoch, settings)
        term_names = dict(no_hsic='hsic', no_pairwise='mono', no_spearman='spearman')
        configs = experiment.variant_configs(baseline, Path('/tmp/unused'), 42)
        for variant, config in configs.items():
            reduced, reduced_terms = compute_total_loss(out, x, mask, cycles, units, epoch,
                {**config['training'], **config['loss_weights']})
            removed = baseline['loss_weights'][experiment.VARIANTS[variant]] * terms[term_names[variant]]
            self.assertAlmostEqual((full - reduced).item(), removed, places=5)
            for key in ('kl_d', 'kl_c', 'recon', 'hsic', 'mono', 'spearman'):
                self.assertEqual(terms[key], reduced_terms[key])

    def test_table_keeps_missing_metrics_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            values = {key: None for _, key in experiment.METRICS}
            experiment.write_results(output, dict(no_hsic=dict(test=values)))
            with (output / 'loss_ablation.csv').open() as file:
                row = next(csv.DictReader(file))
            self.assertTrue(all(row[label] == '' for label, _ in experiment.METRICS))
            self.assertEqual((output / 'loss_ablation.md').read_text().count('—'), 6)

    def test_synthetic_cpu_training_reload_and_public_evaluation(self):
        baseline = baseline_config()
        with tempfile.TemporaryDirectory(prefix='loss_ablation_smoke_') as directory:
            root = Path(directory)
            rng = np.random.default_rng(71)
            raw = root / 'synthetic.h5'
            with h5py.File(raw, 'w') as source:
                for split, units in (('dev', range(1, 7)), ('test', range(7, 9))):
                    a = np.asarray([[u, c, u % 3 + 1, 1] for u in units for c in range(1, 9)
                                    for _ in range(19 + c)], dtype=float)
                    source[f'A_{split}'] = a
                    source[f'X_s_{split}'] = rng.normal(size=(len(a), 14))
                    source[f'W_{split}'] = rng.normal(size=(len(a), 4))
                    source[f'Y_{split}'] = (9 - a[:, 1])[:, None]
                    pd.DataFrame([dict(unit=u, cycle=c, HI=1 - c / 9) for u in units
                                  for c in range(1, 9)]).to_hdf(root / f'cycle_{split}.h5', key='df')
            # These reductions apply only to this temporary synthetic fixture, never the real YAML.
            baseline['data'].update(h5_path=str(raw), downsample_factor=2,
                cycle_hi_path=str(root / 'cycle_dev.h5'), test_cycle_hi_path=str(root / 'cycle_test.h5'),
                processed_dir=str(root / 'processed'))
            baseline['model'].update(c_hidden=4, n_backbone_layers=1, n_decoder_layers=1)
            baseline['training']['total_epochs'] = 2
            config_path = root / 'kgld.yaml'
            config_path.write_text(yaml.safe_dump(baseline))
            output = root / 'results'
            selected, calls = {}, []
            ordinary_train = training.train_model
            def record_selected(*args, **kwargs):
                calls.append(args[3]['experiment']['loss_ablation'])
                result = ordinary_train(*args, **kwargs)
                cp = torch.load(result['best_checkpoint'], map_location='cpu', weights_only=False)
                selected[calls[-1]] = (training.state_digest(cp['model_state_dict']), cp['epoch'], cp['best_val_recon'])
                return result
            with patch.object(experiment, 'BASE_CONFIG', config_path), \
                    patch.object(experiment.training, 'train_model', side_effect=record_selected), \
                    contextlib.redirect_stdout(io.StringIO()):
                experiment.run(output, 42, 'cpu')
            self.assertEqual(calls, list(experiment.VARIANTS))
            initial_hashes, cache_paths, split_keys = set(), set(), set()
            for variant in experiment.VARIANTS:
                run = output / variant
                config, flat, cp, _, device = read_evaluation_checkpoint(
                    str(run / 'config.yaml'), str(run / 'best.pt'), 'cpu')
                self.assertEqual((training.state_digest(cp['model_state_dict']), cp['epoch'], cp['best_val_recon']),
                                 selected[variant])
                experiment.check_variant_parameters(baseline, config, variant)
                self.assertEqual(cp['seed'], 42)
                self.assertEqual(cp['variant'], variant)
                initial_hashes.add(cp['run_metadata']['initial_state_sha256'])
                cache_paths.add(cp['run_metadata']['processed_data'])
                split_keys.add(tuple(cp['run_metadata']['train_units']))
                report = json.loads((run / 'checkpoint_report.json').read_text())
                self.assertFalse(report['best_checkpoint_in_temporal_stage'])
                self.assertEqual(report['phase'], 'reconstruction')
                self.assertEqual(report['epochs_completed'], 2)
                self.assertEqual(len(pd.read_csv(run / 'history.csv')), 2)
                metrics = json.loads((run / 'metrics.json').read_text())
                self.assertEqual(metrics['metadata']['scaling']['n_samples_seen_'], 48)
                self.assertEqual(metrics['test']['n_cycles'], 16)
                test = NCMAPSSDataset(str(raw), None, config['data'], split='test')
                model = load_evaluation_model(config, flat, cp, test)
                predicted = predict_cycles(model, test, device, batch_size=64)
                frame = pd.read_csv(run / 'scores.csv', float_precision='round_trip')
                np.testing.assert_array_equal(frame[frame.split == 'test'].score, predicted.score)
                expected_rmse = np.sqrt(np.mean((frame[frame.split == 'test'].hi - frame[frame.split == 'test'].hi_gt) ** 2))
                self.assertEqual(metrics['test']['rmse'], expected_rmse)
                self.assertTrue((run / 'hi_curves_dev.png').is_file())
                self.assertTrue((run / 'hi_curves_test.png').is_file())
            self.assertEqual((len(initial_hashes), len(cache_paths), len(split_keys)), (1, 1, 1))
            reports = json.loads((output / 'checkpoint_reports.json').read_text())
            self.assertEqual(set(reports), set(experiment.VARIANTS))
            with (output / 'loss_ablation.csv').open() as file:
                reader = csv.DictReader(file)
                rows = list(reader)
                self.assertEqual(reader.fieldnames, ['Variant', 'MAE↓', 'RMSE↓', 'Mon↑', 'Tren↑', 'FC↑', 'MI↑'])
            self.assertEqual([row['Variant'] for row in rows], list(experiment.VARIANTS))
            self.assertTrue((output / 'loss_ablation.md').is_file())
            self.assertFalse((output / 'full').exists())
            self.assertEqual(yaml.safe_load(config_path.read_text()), baseline)


if __name__ == '__main__':
    unittest.main()
