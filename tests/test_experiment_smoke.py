"""CPU-only end-to-end smokes; all fixtures and generated weights auto-clean."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np
import pandas as pd
import torch
import yaml

from dataset import NCMAPSSDataset
from evaluate import predict_cycles, run_evaluation
from models import build_model
from utils.metrics import evaluate_all

ROOT = Path(__file__).resolve().parents[1]


class ExperimentSmoke(unittest.TestCase):
    def test_all_new_entries_and_full_dev_evaluation(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(prefix='oc_ivae_migration_smoke_') as directory:
            root = Path(directory)
            rng = np.random.default_rng(25)
            raw = root / 'synthetic.h5'
            with h5py.File(raw, 'w') as source:
                for split, units in [('dev', range(1, 7)), ('test', range(7, 9))]:
                    a = np.asarray([[u, c, u % 3 + 1, 1] for u in units for c in range(1, 9)
                                    for _ in range(9 + c)], dtype=float)
                    x = rng.normal(size=(len(a), 14))
                    x[:, 0] = 2  # dead sensor follows shared preprocessing
                    source[f'A_{split}'] = a
                    source[f'X_s_{split}'] = x
                    source[f'W_{split}'] = rng.normal(size=(len(a), 4))
                    source[f'Y_{split}'] = (9 - a[:, 1])[:, None]
                    pd.DataFrame([dict(unit=u, cycle=c, HI=1 - c / 9) for u in units
                                  for c in range(1, 9)]).to_hdf(root / f'cycle_{split}.h5', key='df')
            reference = None
            for name in ('ocald_original', 'ocald_joint', 'ocald_joint_decoder_w', 'proposal_cnn_c'):
                with self.subTest(name=name):
                    config = yaml.safe_load((ROOT / f'config/{name}.yaml').read_text())
                    config['data'].update(h5_path=str(raw), downsample_factor=2,
                        cycle_hi_path=str(root / 'cycle_dev.h5'), test_cycle_hi_path=str(root / 'cycle_test.h5'),
                        processed_dir=str(root / 'processed'))
                    source = root / f'{name}.yaml'
                    source.write_text(yaml.safe_dump(config))
                    output = root / name
                    entry = 'experiments.train_proposal_cnn' if name == 'proposal_cnn_c' else 'experiments.train_ocald'
                    command = [sys.executable, '-m', entry, '--config', str(source), '--run-dir', str(output),
                               '--device', 'cpu', '--threads', '2', '--smoke']
                    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                        env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    saved = yaml.safe_load((output / 'config.yaml').read_text())
                    cp = torch.load(output / 'best.pt', map_location='cpu', weights_only=False)
                    self.assertEqual(cp['config'], saved)
                    self.assertEqual(cp['selection'], 'last_epoch')
                    self.assertEqual(cp['seed'], 42)
                    self.assertEqual(cp['epoch'], 1)
                    train = NCMAPSSDataset(str(raw), None, saved['data'])
                    val = NCMAPSSDataset(str(raw), None, saved['data'], split='val')
                    dev = NCMAPSSDataset(str(raw), None, saved['data'], val_fraction=0)
                    keys = (str(train.processed_path), train.samples, val.samples, dev.samples)
                    if reference is None: reference = keys
                    self.assertEqual(reference, keys)
                    metrics = json.loads((output / 'metrics.json').read_text())
                    self.assertEqual(metrics['metadata']['n_cycles'], dict(dev=48, test=16))
                    self.assertEqual(metrics['metadata']['scaling']['n_samples_seen_'], 48)
                    self.assertEqual(metrics['metadata']['original_unit_split']['train'], saved['run_metadata']['train_units'])
                    frame = pd.read_csv(output / 'scores.csv', float_precision='round_trip')
                    model = build_model(saved, saved['model']).eval()
                    model.load_state_dict(cp['model_state_dict'])
                    pred = predict_cycles(model, dev, batch_size=saved['evaluation']['batch_size'])
                    np.testing.assert_array_equal(pred.score, frame[frame.split == 'dev'].score)
                    for split in ('dev', 'test'):
                        recomputed = evaluate_all(frame[frame.split == split], random_state=42)
                        for key in ('mae', 'mape', 'monotonicity', 'trendability', 'fc', 'mi'):
                            self.assertEqual(recomputed[key], metrics[split][key])
                        self.assertTrue((output / f'hi_curves_{split}.png').is_file())
                    history = pd.read_csv(output / 'history.csv')
                    self.assertTrue((history.optimizer_steps == 1).all())
                    self.assertTrue((history.seconds > 0).all())
                    if name == 'proposal_cnn_c':
                        self.assertNotIn('kl_i', history)
                        self.assertNotIn('hsic', history)
                    if name == 'ocald_original':
                        rep = torch.load(output / 'representation.pt', map_location='cpu', weights_only=False)
                        for key, value in rep['model_state_dict'].items():
                            if not key.startswith('hi_head.'):
                                torch.testing.assert_close(value, cp['model_state_dict'][key], rtol=0, atol=0)
                        self.assertTrue(any(not torch.equal(value, cp['model_state_dict'][key])
                            for key, value in rep['model_state_dict'].items() if key.startswith('hi_head.')))
                    # External model or preprocessing modifications must fail before inference.
                    for section, key, value in [('model', 'in_features', 13), ('data', 'downsample_factor', 3)]:
                        altered = copy.deepcopy(saved)
                        altered[section][key] = value
                        bad = root / 'bad.yaml'
                        bad.write_text(yaml.safe_dump(altered))
                        with self.assertRaises(ValueError):
                            run_evaluation(str(bad), str(output / 'best.pt'), str(root / 'invalid'), 'cpu')
                    with self.assertRaises(FileExistsError):
                        run_evaluation(str(output / 'config.yaml'), str(output / 'best.pt'), str(output), 'cpu')


if __name__ == '__main__':
    unittest.main()
