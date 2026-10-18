"""Short-window data, numerical ordering, reload and CPU CLI smoke checks."""
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
from sklearn.decomposition import PCA
import torch
import yaml

from dataset import NCMAPSSDataset, ShortWindowDataset, healthy_training_indices
from evaluate import predict_cycles, run_evaluation, scale_dev_test
from models.residual import ResidualModel, fit_residual_pca, window_residual_matrix
from utils.metrics import evaluate_all
from utils.preprocess import prepare_data, group_by_unit_cycle, load_h5, normalize_cycle, process_cycle

ROOT = Path(__file__).resolve().parents[1]


class ResidualWindows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='residual_window_test_')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.raw = self.root / 'synthetic.h5'
        rng = np.random.default_rng(71)
        with h5py.File(self.raw, 'w') as h5:
            for split, units in [('dev', range(1, 7)), ('test', range(7, 9))]:
                a = np.asarray([[u, c, u % 3 + 1, 1] for u in units for c in range(1, 23)
                                for _ in range(2 * (50 + c + (200 if c == 22 else 0)))], dtype=float)
                x = rng.normal(size=(len(a), 14)); x[:, 0] = 2
                h5[f'A_{split}'], h5[f'X_s_{split}'] = a, x
                h5[f'W_{split}'] = rng.normal(size=(len(a), 4))
                h5[f'Y_{split}'] = (23 - a[:, 1])[:, None]
                pd.DataFrame([dict(unit=u, cycle=c, HI=1 - c / 23) for u in units
                              for c in range(1, 23)]).to_hdf(self.root / f'hi_{split}.h5', key='df')
        self.config = yaml.safe_load((ROOT / 'config/residual_a_short_window.yaml').read_text())
        self.config['data'].update(h5_path=str(self.raw), downsample_factor=2,
            processed_dir=str(self.root / 'processed'), cycle_hi_path=str(self.root / 'hi_dev.h5'),
            test_cycle_hi_path=str(self.root / 'hi_test.h5'))

    def test_full_sequence_windows_shared_stats_and_cache_identity(self):
        data = self.config['data']
        base = prepare_data(str(self.raw), dict(data, input_mode='flight'))
        original_hash = hashlib.sha256(base.read_bytes()).hexdigest()
        with patch('utils.preprocess.compute_unit_stats', side_effect=AssertionError('recomputed X stats')), \
             patch('utils.preprocess.compute_W_stats', side_effect=AssertionError('recomputed W stats')):
            dataset = ShortWindowDataset(str(self.raw), config=data, val_fraction=0)
            stride = ShortWindowDataset(str(self.raw), config=dict(data, stride=3), val_fraction=0)
        self.assertNotEqual(base, dataset.processed_path)
        self.assertNotEqual(dataset.processed_path, stride.processed_path)
        arrays = load_h5(str(self.raw))
        cycles = group_by_unit_cycle(arrays['A'], arrays['W'], arrays['X_s'], factor=2)
        with h5py.File(base, 'r') as common:
            stats = common['dev/sensor_stats']
            units = stats['units'][()].tolist()
            for i, key in enumerate(dataset.flight_keys):
                j = units.index(key[0])
                x, w = normalize_cycle(cycles[key], stats['mean'][j], stats['std'][j],
                    stats['valid_mask'][j], common['W_mean_global'][()], common['W_std_global'][()])
                np.testing.assert_array_equal(dataset.flights_x[i], x)
                np.testing.assert_array_equal(dataset.flights_w[i], w)
                fixed_x, fixed_w, mask = process_cycle(cycles[key], dataset.L_max, stats['mean'][j],
                    stats['std'][j], stats['valid_mask'][j], common['W_mean_global'][()], common['W_std_global'][()])
                torch.testing.assert_close(fixed_x, torch.from_numpy(common['dev/x'][i]), rtol=0, atol=0)
                count = (len(x) - 50) // 1 + 1
                selected = np.flatnonzero(dataset.index[:, 0] == i)
                self.assertEqual(len(selected), count)
                self.assertEqual(sum(stride.index[:, 0] == i), (len(x) - 50) // 3 + 1)
                for index in (selected[0], selected[-1]):
                    a, valid, _, b, unit, cycle, start = dataset[index]
                    self.assertEqual((unit, cycle), key)
                    self.assertTrue(valid.all())
                    np.testing.assert_array_equal(a, x[start:start + 50])
                    np.testing.assert_array_equal(b, w[start:start + 50])
                if key[1] == 22:
                    self.assertGreater(len(x), dataset.L_max)
                    self.assertFalse(np.array_equal(x[:50], fixed_x[:50]))
        self.assertEqual(hashlib.sha256(base.read_bytes()).hexdigest(), original_hash)
        train = ShortWindowDataset(str(self.raw), config=data)
        val = ShortWindowDataset(str(self.raw), config=data, split='val')
        eligible = healthy_training_indices(train)
        self.assertTrue((train.samples[eligible, 1] <= 20).all())
        self.assertTrue(set(train.samples[:, 0]).isdisjoint(val.samples[:, 0]))
        with self.assertRaisesRegex(ValueError, r'shorter.*\(1, 1\)'):
            ShortWindowDataset(str(self.raw), config=dict(data, window_length=1000))
        self.assertFalse(list(base.parent.glob('*.tmp')))

    def test_pca_direction_window_mean_scaling_and_batch_size(self):
        # Real model residuals; unequal window counts distinguish window and flight weighting.
        data = dict(self.config['data'], stride=7)
        dev = ShortWindowDataset(str(self.raw), config=data, val_fraction=0)
        test = ShortWindowDataset(str(self.raw), config=data, split='test')
        torch.manual_seed(5)
        model = ResidualModel(self.config['model']).eval()
        with window_residual_matrix(model, dev, 'cpu', 31) as matrix:
            temporary_matrix = Path(matrix.filename)
            self.assertIsInstance(matrix, np.memmap)
            pca = PCA(n_components=1, random_state=42).fit(matrix)
            z = pca.transform(matrix).ravel()
            state = fit_residual_pca(matrix, dev.samples, 42)
            np.testing.assert_allclose(state['components'], pca.components_, rtol=0, atol=0)
            np.testing.assert_allclose(state['mean'], pca.mean_, rtol=0, atol=0)
        self.assertFalse(temporary_matrix.exists())
        differences = []
        for unit in sorted(set(dev.samples[:, 0])):
            indices = np.flatnonzero(dev.samples[:, 0] == unit)
            order = np.lexsort((dev.samples[indices, 2], dev.samples[indices, 1]))
            values = z[indices[order]]
            differences.append(np.mean(values[:10] - values[-10:]))
        self.assertEqual(state['direction'], -1 if np.mean(differences) < 0 else 1)
        state.update(window_length=50, stride=7)
        expected = pd.DataFrame(dev.samples[:, :2], columns=['unit', 'cycle']).assign(
            score=z.astype(float) * state['direction']).groupby(['unit', 'cycle'], as_index=False).score.mean()
        actual = predict_cycles(model, dev, batch_size=23, residual_state=state)
        np.testing.assert_allclose(actual.score, expected.score, rtol=2e-5, atol=2e-6)
        changed_batch = predict_cycles(model, dev, batch_size=127, residual_state=state)
        np.testing.assert_allclose(actual.score, changed_batch.score, rtol=2e-5, atol=2e-6)
        test_pred = predict_cycles(model, test, batch_size=31, residual_state=state)
        before = copy.deepcopy(state)
        _, _, scaler = scale_dev_test(actual, test_pred)
        self.assertEqual(scaler['n_samples_seen_'], len(dev.flight_keys))
        self.assertEqual(scaler['fit_keys'], [list(k) for k in dev.flight_keys])
        np.testing.assert_array_equal(state['mean'], before['mean'])
        self.assertEqual(state['direction'], before['direction'])
        self.assertEqual(actual[['unit', 'cycle']].values.tolist(), [list(k) for k in dev.flight_keys])
        with self.assertRaisesRegex(ValueError, 'settings'):
            predict_cycles(model, test, residual_state=dict(state, stride=1))

    def test_short_window_cli_smokes_and_reload(self):
        for variant in ('a', 'b'):
            with self.subTest(variant=variant):
                config = yaml.safe_load((ROOT / f'config/residual_{variant}_short_window.yaml').read_text())
                config['data'] = dict(self.config['data'])
                config['training']['batch_size'] = 7  # synthetic smoke only; formal configs retain 512
                config['evaluation']['batch_size'] = 127
                source, output = self.root / f'{variant}.yaml', self.root / variant
                source.write_text(yaml.safe_dump(config))
                command = [sys.executable, '-m', 'experiments.train_residual', '--config', str(source),
                           '--run-dir', str(output), '--device', 'cpu', '--threads', '2', '--smoke']
                result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                        env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                saved = yaml.safe_load((output / 'config.yaml').read_text())
                cp = torch.load(output / 'best.pt', weights_only=False, map_location='cpu')
                self.assertEqual(cp['config'], saved)
                self.assertEqual(cp['selection'], 'last_epoch')
                self.assertEqual(cp['epoch'], 1)
                dev = ShortWindowDataset(str(self.raw), config=config['data'], val_fraction=0)
                train = ShortWindowDataset(str(self.raw), config=config['data'])
                state = cp['residual_state']
                self.assertEqual(state['n_samples_seen'], len(dev))
                self.assertEqual(state['fit_keys'], [list(k) for k in dev.flight_keys])
                self.assertEqual(state['fit_window_key_sha256'], hashlib.sha256(dev.samples).hexdigest())
                history = pd.read_csv(output / 'history.csv')
                self.assertEqual(history.optimizer_steps.iloc[0], 1)
                self.assertEqual(history.samples_drawn.iloc[0], 7)
                eligible = healthy_training_indices(train)
                self.assertEqual(history.eligible_windows.iloc[0], len(eligible))
                self.assertEqual(history.healthy_flights.iloc[0], len(set(map(tuple, train.samples[eligible, :2]))))
                for u, c, start in saved['run_metadata']['last_epoch_training_keys']:
                    self.assertIn(u, saved['run_metadata']['train_units'])
                    self.assertLessEqual(c, 20)
                    self.assertTrue(np.any(np.all(train.samples == [u, c, start], axis=1)))
                scores = pd.read_csv(output / 'scores.csv', float_precision='round_trip')
                metrics = json.loads((output / 'metrics.json').read_text())
                self.assertEqual(metrics['metadata']['n_cycles'], dict(dev=132, test=44))
                self.assertEqual(metrics['metadata']['scaling']['n_samples_seen_'], 132)
                for split in ('dev', 'test'):
                    frame = scores[scores.split == split]
                    for key, value in evaluate_all(frame, random_state=42).items():
                        self.assertEqual(metrics[split][key], value)
                    self.assertTrue((output / f'hi_curves_{split}.png').is_file())
                with patch('models.residual.PCA.fit', side_effect=AssertionError('refit on reload')):
                    reeval = run_evaluation(str(output / 'config.yaml'), str(output / 'best.pt'),
                                            str(self.root / f'{variant}_reeval'), 'cpu', save_plots=False)
                self.assertEqual((reeval / 'scores.csv').read_bytes(), (output / 'scores.csv').read_bytes())
                # Legacy flight inference is explicitly rejected for a window checkpoint.
                fixed = NCMAPSSDataset(str(self.raw), config=dict(config['data'], input_mode='flight'), val_fraction=0)
                with self.assertRaisesRegex(ValueError, 'legacy flight'):
                    predict_cycles(ResidualModel(config['model']), fixed, residual_state=state)


if __name__ == '__main__':
    unittest.main()
