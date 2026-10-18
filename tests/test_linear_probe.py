"""Keyed, deterministic, train-device-only linear probe math and CLI integration."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
import yaml

from dataset import NCMAPSSDataset
from evaluate import predict_cycles, run_evaluation
from models import build_model
from models.ocald import to_valid_prefix
from utils.linear_probe import (extract_representations, fit_linear_probes, join_probe_features,
                                regression_metrics, resolve_probe_split)

ROOT = Path(__file__).resolve().parents[1]


class LinearProbes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_split_provenance_and_failures(self):
        config = dict(data=dict(val_split=.25), training=dict(split_seed=42))
        split = resolve_probe_split(config, {}, [1, 2, 3, 4], [5, 6])
        metadata = dict(train_units=split['fit_units'], val_units=split['val_units'])
        self.assertEqual(resolve_probe_split(config, dict(run_metadata=metadata), [1, 2, 3, 4], [5, 6])['fit_units'], split['fit_units'])
        with self.assertRaisesRegex(ValueError, 'provenance missing'):
            resolve_probe_split(dict(data={}), {}, [1, 2], [3])
        provided = dict(fit_units=[1], val_units=[2])
        self.assertEqual(resolve_probe_split(dict(data={}), {}, [1, 2], [3], provided)['fit_units'], [1])
        self.assertEqual(resolve_probe_split(dict(data={}), dict(run_metadata=dict(train_units=[1])),
            [1, 2], [3], provided)['fit_units'], [1])
        with self.assertRaisesRegex(ValueError, 'contradicts'):
            resolve_probe_split(config, dict(run_metadata=metadata), [1, 2, 3, 4], [5],
                                dict(fit_units=[1], val_units=[2, 3, 4]))
        for invalid in (dict(fit_units=[1, 1], val_units=[2]), dict(fit_units=[1], val_units=[1, 2]),
                        dict(fit_units=[1], val_units=[])):
            with self.assertRaises(ValueError):
                resolve_probe_split(dict(data={}), {}, [1, 2], [3], invalid)
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            resolve_probe_split(dict(data={}), {}, [1, 2], [2], provided)

    def test_pipeline_fit_only_and_rank_and_metrics(self):
        rng = np.random.default_rng(31)
        x = rng.normal(size=(12, 2)); x[8:] += 100
        y = 3 * x[:, 0] - 2 * x[:, 1] + 5
        dev = pd.DataFrame(dict(unit=np.repeat([1, 2, 3], 4), cycle=list(range(1, 5)) * 3,
            representation_0=x[:, 0], representation_1=x[:, 1], hi=x[:, 0] * 2 + 1,
            hi_gt=y, split='dev'))
        test = dev.iloc[:4].copy().assign(unit=[4, 4, 4, 5], split='test', representation_0=[-200., 0, 200, 400], hi_gt=[0., 1, 4, 9])
        split = dict(fit_units=[1, 2], val_units=[3], test_units=[4, 5])
        frame, metrics, parameters = fit_linear_probes(dev, test, split)
        fit = dev.unit.isin([1, 2])
        oracle = make_pipeline(StandardScaler(), LinearRegression(fit_intercept=True)).fit(x[fit], y[fit])
        features = frame[['representation_0', 'representation_1']].to_numpy()
        np.testing.assert_allclose(frame.predicted_hi_representation, oracle.predict(features), rtol=0, atol=0)
        np.testing.assert_allclose(parameters['representation']['scaler_mean'], x[:8].mean(0), rtol=0, atol=0)
        pooled = frame.probe_split.eq('test')
        self.assertEqual(metrics['representation']['test']['r2'], r2_score(frame.hi_gt[pooled], frame.predicted_hi_representation[pooled]))
        self.assertIsNone(metrics['representation']['test']['per_unit']['5']['r2'])
        changed = test.copy(); changed.hi_gt += 5000; changed.representation_0 *= 100
        changed_dev = dev.copy(); changed_dev.loc[~fit, 'hi_gt'] += 1000
        _, _, unchanged = fit_linear_probes(changed_dev, changed, split)
        self.assertEqual(parameters, unchanged)
        self.assertGreater(frame.predicted_hi_representation.max(), 1)
        self.assertLess(frame.predicted_hi_representation.min(), 0)
        for name in ('representation', 'hi'):
            p = parameters[name]
            restored = ((frame[p['feature_columns']].values - p['scaler_mean']) / p['scaler_scale']) @ p['coefficient'] + p['intercept']
            np.testing.assert_allclose(restored, frame[f'predicted_hi_{name}'], rtol=0, atol=0)
        self.assertEqual(regression_metrics(np.array([2, 2]), np.array([1, 2]))['r2_undefined_reason'], 'constant_target')
        self.assertIsNone(regression_metrics(np.array([1]), np.array([2]))['r2'])
        self.assertIsNone(regression_metrics(np.array([]), np.array([]))['r2'])
        tiny = dev.iloc[:1].copy().assign(representation_1=lambda f: f.representation_0)
        _, _, rank = fit_linear_probes(tiny, test, dict(fit_units=[1], val_units=[], test_units=[4, 5]))
        self.assertTrue(rank['representation']['underdetermined'])
        self.assertTrue(rank['representation']['rank_deficient'])

    def test_feature_adapters_means_freezing_and_no_sampling(self):
        torch.manual_seed(17)
        x, w = torch.randn(3, 6, 14), torch.randn(3, 6, 4)
        mask = torch.tensor([[0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1]], dtype=torch.bool)
        u = torch.randn(3, 11)
        fixture = [(x[i], mask[i], u[i], w[i], torch.tensor(1), torch.tensor(i + 1), torch.tensor(float('nan'))) for i in range(3)]
        for name in ('ds03_default', 'ocald_joint', 'proposal_cnn_c'):
            with self.subTest(name=name):
                config = yaml.safe_load((ROOT / f'config/{name}.yaml').read_text())
                config['model']['in_features'] = 14
                if name == 'proposal_cnn_c': config['model']['window_size'] = 6
                model = build_model(config, config['model']).eval()
                before = copy.deepcopy(model.state_dict())
                with torch.no_grad():
                    if name == 'ds03_default': expected = model(x, mask, u, w)['mu_d']
                    elif name == 'ocald_joint':
                        a, _, m = to_valid_prefix(x, w, mask)
                        expected = model.encode(a, m)[0]
                    else: expected = model.encode(x)[:, None]
                rng = torch.get_rng_state().clone()
                with patch('torch.randn_like', side_effect=AssertionError('posterior sampled')):
                    frame = extract_representations(model, fixture, 'cpu', 3)
                torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
                np.testing.assert_array_equal(frame.filter(like='representation_').values, expected.numpy())
                self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
                for key in before: torch.testing.assert_close(before[key], model.state_dict()[key], rtol=0, atol=0)
                with self.assertRaisesRegex(ValueError, 'duplicate'):
                    extract_representations(model, fixture + fixture[:1], 'cpu', 4)

    def test_key_join_rejects_mismatch_duplicates_and_missing_labels(self):
        reps = pd.DataFrame(dict(unit=[1, 1], cycle=[1, 2], representation_0=[3., 4.]))
        hi = pd.DataFrame(dict(unit=[1, 1], cycle=[2, 1], hi=[.4, .8], hi_gt=[.3, .9]))
        joined = join_probe_features(reps, hi)
        np.testing.assert_array_equal(joined.hi_gt, [.9, .3])
        for bad in (hi.iloc[:1], pd.concat([hi, hi.iloc[:1]]), hi.assign(hi_gt=np.nan)):
            with self.assertRaises(ValueError): join_probe_features(reps, bad)

    def test_three_cli_entries_and_public_hi_equivalence(self):
        with tempfile.TemporaryDirectory(prefix='linear_probe_cli_') as directory:
            root = Path(directory); raw = root / 'data.h5'; rng = np.random.default_rng(3)
            with h5py.File(raw, 'w') as h5:
                for split, units in [('dev', range(1, 5)), ('test', range(5, 7))]:
                    a = np.array([[u, c, 1, 1] for u in units for c in range(1, 7) for _ in range(10 + c)])
                    h5[f'A_{split}'] = a; h5[f'X_s_{split}'] = rng.normal(size=(len(a), 14))
                    h5[f'W_{split}'] = rng.normal(size=(len(a), 4)); h5[f'Y_{split}'] = (7 - a[:, 1])[:, None]
                    reference = pd.DataFrame([dict(unit=u, cycle=c, HI=1 - c / 7) for u in units for c in range(1, 7)])
                    if split == 'dev': reference = reference.drop(columns='cycle')  # legacy ordinal protocol
                    else: reference = reference.iloc[::-1]  # explicit cycle keys, no positional matching
                    reference.to_hdf(root / f'hi_{split}.h5', key='df')
            for name in ('ds03_default', 'ocald_joint', 'proposal_cnn_c'):
                with self.subTest(name=name):
                    cfg = yaml.safe_load((ROOT / f'config/{name}.yaml').read_text())
                    cfg['data'].update(h5_path=str(raw), downsample_factor=2, val_split=.25,
                        processed_dir=str(root / 'processed'), cycle_hi_path=str(root / 'hi_dev.h5'),
                        test_cycle_hi_path=str(root / 'hi_test.h5'))
                    cfg['run_dir'] = str(root / name)
                    ds = NCMAPSSDataset(str(raw), config=cfg['data'])
                    val = NCMAPSSDataset(str(raw), config=cfg['data'], split='val', val_fraction=.25)
                    cfg['model']['in_features'] = 14
                    if name == 'proposal_cnn_c': cfg['model']['window_size'] = ds.L_max
                    if name == 'ds03_default':
                        cfg['model'].update(c_hidden=4, n_backbone_layers=1, n_decoder_layers=1)
                        cp = dict(run_metadata=dict(split_seed=42))  # existing OC-iVAE format
                    else:
                        cp = dict(config=cfg, model_type=cfg['model_type'], stage='joint', selection='last_epoch',
                            processed_data=str(ds.processed_path), run_metadata=dict(
                                train_units=sorted({u for u, _ in ds.samples}),
                                val_units=sorted({u for u, _ in val.samples}), split_seed=42))
                    model = build_model(cfg, cfg['model'])
                    cp['model_state_dict'] = model.state_dict()
                    source = root / f'{name}.yaml'; source.write_text(yaml.safe_dump(cfg))
                    checkpoint = root / f'{name}.pt'; torch.save(cp, checkpoint)
                    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                    baseline = run_evaluation(str(source), str(checkpoint), str(root / f'{name}_public'), 'cpu', save_plots=False)
                    old_metrics = (baseline / 'metrics.json').read_bytes()
                    output = root / f'{name}_probe'
                    command = [sys.executable, '-m', 'experiments.evaluate_linear_probe', '--config', str(source),
                        '--checkpoint', str(checkpoint), '--output-dir', str(output), '--device', 'cpu', '--threads', '2']
                    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    report = json.loads((output / 'probe_metrics.json').read_text())
                    params = json.loads((output / 'probe_parameters.json').read_text())
                    pred = pd.read_csv(output / 'probe_predictions.csv', float_precision='round_trip')
                    original = pd.read_csv(baseline / 'scores.csv', float_precision='round_trip')
                    np.testing.assert_array_equal(pred.hi, original.hi)
                    np.testing.assert_array_equal(pred.hi_gt, original.hi_gt)
                    features = pd.read_csv(output / 'probe_features.csv', float_precision='round_trip')
                    rep_columns = params['representation']['feature_columns']
                    np.testing.assert_allclose(features.loc[features.probe_split.eq('fit'), rep_columns].mean().values,
                        params['representation']['scaler_mean'], rtol=1e-12, atol=1e-12)
                    fit_rows = pred.probe_split.eq('fit')
                    self.assertEqual(fit_rows.sum(), 18)
                    self.assertEqual(params['representation']['n_fit'], 18)
                    self.assertEqual(report['representation']['test']['n_samples'], 12)
                    self.assertEqual(report['representation']['val']['n_samples'], 6)
                    self.assertIn('legacy', report['metadata']['reference_HI']['dev']['key_provenance'])
                    self.assertEqual(report['metadata']['public_hi_scaling']['n_samples_seen_'], 24)
                    self.assertFalse((output / 'metrics.json').exists())
                    self.assertEqual(old_metrics, (baseline / 'metrics.json').read_bytes())
                    self.assertEqual(checksum, hashlib.sha256(checkpoint.read_bytes()).hexdigest())
                    if name == 'proposal_cnn_c':
                        np.testing.assert_allclose(pred.predicted_hi_representation, pred.predicted_hi_hi, atol=1e-12)
                    repeat = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
                    self.assertNotEqual(repeat.returncode, 0)
                    self.assertIn('already exists', repeat.stderr)


if __name__ == '__main__': unittest.main()
