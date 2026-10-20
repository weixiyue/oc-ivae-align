"""Frozen batch probes on synthetic checkpoints, without any network training."""
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
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
import yaml

from dataset import NCMAPSSDataset
from experiments import evaluate_linear_probe as single
from experiments import run_linear_probe_comparison as batch
from experiments.run_knowledge_ablation import load_ablation_checkpoint
from models import build_model
from models.vae import build_knowledge_ablation, StandardNormalPrior, PassthroughZcBranch
from models.kgld import KGLD, PriorNetwork
from train import state_digest

ROOT = Path(__file__).resolve().parents[1]


def fixtures(root):
    """Prepare one tiny shared cache and eight untrained model checkpoints."""
    raw = root / 'data.h5'
    rng = np.random.default_rng(33)
    with h5py.File(raw, 'w') as file:
        for split, units in [('dev', range(1, 5)), ('test', range(5, 7))]:
            a = np.array([[u, c, 1, 1] for u in units for c in range(1, 7) for _ in range(10 + c)])
            file[f'A_{split}'] = a
            file[f'X_s_{split}'] = rng.normal(size=(len(a), 14))
            file[f'W_{split}'] = rng.normal(size=(len(a), 4))
            file[f'Y_{split}'] = (7 - a[:, 1])[:, None]
            reference = pd.DataFrame([dict(unit=u, cycle=c, HI=1 - c / 7)
                                      for u in units for c in range(1, 7)])
            reference.to_hdf(root / f'hi_{split}.h5', key='df')
    baseline = yaml.safe_load((ROOT / 'config/kgld.yaml').read_text())
    baseline['data'].update(h5_path=str(raw), processed_dir=str(root / 'processed'),
        downsample_factor=2, val_split=.25, cycle_hi_path=str(root / 'hi_dev.h5'),
        test_cycle_hi_path=str(root / 'hi_test.h5'))
    baseline['model'].update(in_features=14, c_hidden=4, n_backbone_layers=1, n_decoder_layers=1)
    dev = NCMAPSSDataset(str(raw), config=baseline['data'], val_fraction=0)
    split = NCMAPSSDataset._split_units([1, 2, 3, 4], .25, 42)
    runs = []
    for variant in ('full', *batch.STRUCTURAL_VARIANTS, 'no_hsic', 'no_pairwise', 'no_spearman', 'proposal_cnn_c'):
        config = copy.deepcopy(baseline)
        # Two different sources intentionally share a basename.
        source = root / 'runs' / variant / ('same' if variant in ('full', 'proposal_cnn_c') else variant)
        source.mkdir(parents=True)
        config['run_dir'] = str(source)
        config['run_metadata'] = dict(seed=42, split_seed=42, processed_data=str(dev.processed_path),
                                     train_units=sorted(split['train']), val_units=sorted(split['val']))
        cp = dict(model_type='kgld', seed=42, epoch=1, decoder_use_W_summary=True)
        torch.manual_seed(42)
        if variant in batch.STRUCTURAL_VARIANTS:
            config['experiment'] = dict(knowledge_ablation=variant, seed=42)
            config['run_metadata']['variant'] = variant
            cp.update(model_type=batch.CHECKPOINT_TYPE, variant=variant)
            model = build_knowledge_ablation(config['model'], variant)
        elif variant == 'proposal_cnn_c':
            proposal = yaml.safe_load((ROOT / 'config/proposal_cnn_c.yaml').read_text())
            config.update(model_type='proposal_cnn', source=proposal['source'],
                          model=proposal['model'], training=proposal['training'])
            config['model'].update(in_features=14, window_size=dev.L_max)
            cp.update(model_type='proposal_cnn', processed_data=str(dev.processed_path),
                      selection='last_epoch', stage='proposal_cnn_c')
            model = build_model(config, config['model'])
        else:
            if variant != 'full':
                weight = dict(no_hsic='lambda_hsic', no_pairwise='lambda_mono', no_spearman='lambda_spearman')[variant]
                config['loss_weights'][weight] = 0
                config['experiment'] = dict(loss_ablation=variant, seed=42)
                config['run_metadata']['variant'] = variant
                cp['variant'] = variant
            model = KGLD(config['model'])
        cp.update(config=config, model_state_dict=model.state_dict(), run_metadata=config['run_metadata'])
        torch.save(cp, source / 'best.pt')
        (source / 'config.yaml').write_text(yaml.safe_dump(config))
        runs.append(source)
    for kind in ('ocald', 'residual'):
        source = root / kind
        source.mkdir()
        (source / 'config.yaml').write_text(yaml.safe_dump(dict(model_type=kind)))
        runs.append(source)
    return runs


class LinearProbeComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_dry_run_checks_splits_skips_and_failures_without_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = fixtures(root)
            missing = root / 'missing'
            output = root / 'comparison'
            with patch.object(batch.probe, 'run') as run_probe, \
                    patch.object(batch, 'load_ablation_checkpoint') as load_model, \
                    contextlib.redirect_stdout(io.StringIO()):
                summary = batch.run([str(p) for p in runs + [missing]], str(output), 'cuda', dry_run=True)
            run_probe.assert_not_called()
            load_model.assert_not_called()
            self.assertFalse(output.exists())
            self.assertEqual(summary['counts'], dict(pending=8, success=0, skipped=2, failed=1))
            for record in summary['runs'][:8]:
                self.assertEqual(len(record['fit_units']), 3)
                self.assertEqual(len(record['val_units']), 1)
                self.assertEqual(record['test_units'], [5, 6])
                self.assertEqual(record['actual_seed'], 42)
            self.assertEqual(len({record['output'] for record in summary['runs']}), 11)
            with self.assertRaisesRegex(ValueError, 'inside a source run'):
                batch.run([str(runs[0])], str(runs[0] / 'comparison'), dry_run=True)
            output.mkdir()
            with self.assertRaises(FileExistsError):
                batch.run([str(runs[0])], str(output), dry_run=True)

    def test_missing_split_and_conflicting_identity_fail_without_guessing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = fixtures(root)
            source = runs[0]
            cp = torch.load(source / 'best.pt', weights_only=False)
            for metadata in (cp['run_metadata'], cp['config']['run_metadata']):
                for key in ('train_units', 'val_units', 'split_seed'):
                    metadata.pop(key, None)
            torch.save(cp, source / 'best.pt')
            (source / 'config.yaml').write_text(yaml.safe_dump(cp['config']))
            with self.assertRaisesRegex(ValueError, 'provenance missing'):
                batch.inspect_run(source)
            source = runs[4]  # no_hsic
            cp = torch.load(source / 'best.pt', weights_only=False)
            cp['variant'] = 'no_pairwise'
            torch.save(cp, source / 'best.pt')
            with self.assertRaisesRegex(ValueError, 'conflicting saved'):
                batch.inspect_run(source)

    def test_all_supported_probes_match_single_runs_and_freeze_models(self):
        with tempfile.TemporaryDirectory(prefix='batch_probe_synthetic_') as directory:
            root = Path(directory)
            runs = fixtures(root)
            originals = {p: p.read_bytes() for source in runs for p in source.iterdir() if p.is_file()}
            output = root / 'comparison'
            loaded = []
            def structural_loader(path, device):
                model, config, checkpoint = load_ablation_checkpoint(path, device)
                variant = checkpoint['variant']
                if variant != 'no_film':
                    self.assertIsInstance(model.prior, StandardNormalPrior)
                else:
                    self.assertIsInstance(model.prior, PriorNetwork)
                if variant != 'no_prior':
                    self.assertIsInstance(model.zc_branch, PassthroughZcBranch)
                loaded.append((model, state_digest(model.state_dict())))
                return model, config, checkpoint
            original_loader = single.load_evaluation_model
            def ordinary_loader(*args):
                model = original_loader(*args)
                loaded.append((model, state_digest(model.state_dict())))
                return model
            reparameterize = KGLD.reparameterize
            def no_sampling(model, mu, log_var):
                self.assertFalse(model.training)
                self.assertFalse(torch.is_grad_enabled())
                return reparameterize(model, mu, log_var)
            original_reader = single.read_evaluation_checkpoint
            with patch.object(batch, 'load_ablation_checkpoint', side_effect=structural_loader), \
                    patch.object(single, 'load_evaluation_model', side_effect=ordinary_loader), \
                    patch.object(KGLD, 'reparameterize', no_sampling), \
                    contextlib.redirect_stdout(io.StringIO()):
                summary = batch.run([str(p) for p in runs + [root / 'missing']], str(output), 'cpu')
            self.assertIs(single.read_evaluation_checkpoint, original_reader)
            self.assertIs(single.load_evaluation_model, original_loader)
            self.assertEqual(summary['counts'], dict(pending=0, success=8, skipped=2, failed=1))
            self.assertEqual(len(loaded), 8)
            for model, digest in loaded:
                self.assertEqual(state_digest(model.state_dict()), digest)
                self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
            with (output / 'linear_probe_comparison.csv').open() as file:
                reader = csv.DictReader(file)
                rows = list(reader)
                self.assertEqual(reader.fieldnames, batch.COLUMNS)
            self.assertEqual(len(rows), 8)
            for record, row in zip(summary['runs'][:8], rows):
                destination = Path(record['output'])
                metrics = json.loads((destination / 'probe_metrics.json').read_text())
                parameters = json.loads((destination / 'probe_parameters.json').read_text())
                features = pd.read_csv(destination / 'probe_features.csv', float_precision='round_trip')
                predictions = pd.read_csv(destination / 'probe_predictions.csv', float_precision='round_trip')
                fit = features.probe_split.eq('fit')
                columns = parameters['representation']['feature_columns']
                x = features.loc[fit, columns].values
                y = predictions.loc[fit, 'hi_gt'].values
                expected = make_pipeline(StandardScaler(), LinearRegression(fit_intercept=True)).fit(x, y)
                np.testing.assert_allclose(parameters['representation']['scaler_mean'], x.mean(axis=0), atol=1e-12)
                np.testing.assert_allclose(parameters['representation']['coefficient'], expected[-1].coef_, atol=1e-12)
                self.assertEqual(parameters['representation']['n_fit'], 18)
                self.assertEqual(metrics['representation']['test']['n_samples'], 12)
                self.assertEqual(metrics['metadata']['comparison_run']['source_run'], record['source_run'])
                for prefix, name in [('Representation', 'representation'), ('HI-probe', 'hi')]:
                    for label, key in [('R²↑', 'r2'), ('MAE↓', 'mae'), ('RMSE↓', 'rmse')]:
                        self.assertEqual(float(row[f'{prefix} {label}']), metrics[name]['test'][key])
                self.assertEqual(int(row['Feature dim']), 1 if record['method'] == 'Proposal-C' else 2)
                self.assertEqual(metrics['metadata']['public_hi_scaling']['n_samples_seen_'], 24)
                self.assertTrue((destination / 'config.yaml').is_file())
            # The unadapted single-run entry must give exactly the same metrics and parameters.
            for index in (0, 7):
                source = runs[index]
                with contextlib.redirect_stdout(io.StringIO()):
                    direct = single.run(str(source / 'config.yaml'), str(source / 'best.pt'),
                                        str(root / f'single_{index}'), 'cpu')
                actual = json.loads((Path(summary['runs'][index]['output']) / 'probe_metrics.json').read_text())
                expected = json.loads((direct / 'probe_metrics.json').read_text())
                for name in ('representation', 'hi', 'unprobed_hi'):
                    self.assertEqual(actual[name], expected[name])
            for path, contents in originals.items():
                self.assertEqual(path.read_bytes(), contents)
            self.assertIn('may be equivalent', (output / 'linear_probe_comparison.md').read_text())
            self.assertTrue((output / 'comparison_summary.json').is_file())

    def test_missing_metrics_are_not_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            row = {column: None for column in batch.COLUMNS}
            row.update(Method='KGLD', Variant='full', Run='some/run', **{'Feature dim': 2})
            batch.write_comparison(output, [row])
            with (output / 'linear_probe_comparison.csv').open() as file:
                written = next(csv.DictReader(file))
            self.assertEqual(written['Representation R²↑'], '')
            self.assertEqual((output / 'linear_probe_comparison.md').read_text().count('—'), 6)


if __name__ == '__main__':
    unittest.main()
