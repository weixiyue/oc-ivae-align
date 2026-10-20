"""Knowledge ablation math, checkpoint guards and synthetic orchestration; no training."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
import yaml

from experiments import run_knowledge_ablation as experiment
from evaluate import predict_cycles, read_evaluation_checkpoint
from models.kgld import KGLD, PriorNetwork, ZcBranch
from models.vae import build_knowledge_ablation, PassthroughZcBranch, StandardNormalPrior
from train import save_checkpoint
from utils.losses import compute_total_loss


def small_baseline():
    config = yaml.safe_load(experiment.BASE_CONFIG.read_text())
    config['model'].update(in_features=14, c_hidden=4, n_backbone_layers=1, n_decoder_layers=1)
    return config


def synthetic_flights(units):
    generator = torch.Generator().manual_seed(901)
    n, length = len(units) * 4, 8
    x = torch.randn(n, length, 14, generator=generator)
    w = torch.randn(n, length, 4, generator=generator)
    mask = torch.ones(n, length, dtype=torch.bool)
    mask[:, :2] = False
    x[:, :2] = 0
    w[:, :2] = 0
    u = torch.randn(n, 11, generator=generator)
    unit = torch.tensor(units).repeat_interleave(4)
    cycles = torch.arange(1, 5).repeat(len(units))
    dataset = TensorDataset(x, mask, u, w, unit, cycles, torch.full((n,), float('nan')))
    dataset.samples = list(zip(unit.tolist(), cycles.tolist()))
    dataset.rul = pd.DataFrame(dict(unit=unit.numpy(), cycle=cycles.numpy(), rul=4 - cycles.numpy()))
    dataset.processed_path = Path('synthetic_shared_cache.h5')
    return dataset


class KnowledgeAblation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_config_parameters_and_dry_run(self):
        baseline = small_baseline()
        original = copy.deepcopy(baseline)
        configs = experiment.variant_configs(baseline, Path('/tmp/knowledge_test_only'), 17)
        self.assertEqual(list(configs), list(experiment.VARIANTS))
        for variant, config in configs.items():
            self.assertEqual(config['training'], baseline['training'])
            self.assertEqual(config['loss_weights'], baseline['loss_weights'])
            self.assertEqual(config['model'], baseline['model'])
            self.assertEqual(config['data'], baseline['data'])
            self.assertEqual(config['run_metadata']['sampler_seed'], 42)
            self.assertEqual(config['experiment']['seed'], 17)
            experiment.check_experiment_parameters(baseline, config)
        configs['no_prior']['loss_weights']['beta_c'] += 1
        with self.assertRaisesRegex(ValueError, 'parameters differ'):
            experiment.check_experiment_parameters(baseline, configs['no_prior'])
        self.assertEqual(baseline, original)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'kgld.yaml'
            source.write_text(yaml.safe_dump(baseline))
            output = Path(directory) / 'results'
            with patch.object(experiment, 'BASE_CONFIG', source), \
                    patch.object(experiment.training, 'train_model') as train, \
                    patch.object(experiment, 'NCMAPSSDataset') as dataset, \
                    patch.object(experiment, 'build_knowledge_ablation') as build:
                experiment.run(output, 42, 'cpu', dry_run=True)
            train.assert_not_called()
            dataset.assert_not_called()
            build.assert_not_called()
            self.assertFalse(output.exists())
            output.mkdir()
            with self.assertRaises(FileExistsError):
                experiment.run(output, 42, 'cpu', dry_run=True)

    def test_standard_prior_and_shared_initialization(self):
        config = small_baseline()['model']
        experiment.seed_model(42)
        full = KGLD(config)
        full_rng = torch.get_rng_state().clone()
        for variant in experiment.VARIANTS:
            experiment.seed_model(42)
            model = build_knowledge_ablation(config, variant)
            self.assertTrue(torch.equal(torch.get_rng_state(), full_rng))
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, full.state_dict()[name], rtol=0, atol=0)
            if variant != 'no_film':
                self.assertIsInstance(model.prior, StandardNormalPrior)
                u = torch.randn(3, 11, dtype=torch.float64)
                outputs = model.prior(u)
                self.assertEqual([tuple(v.shape) for v in outputs], [(3, 2), (3, 2), (3, 8), (3, 8)])
                for value in outputs:
                    self.assertEqual(value.dtype, u.dtype)
                    self.assertEqual(value.device, u.device)
                    self.assertEqual(torch.count_nonzero(value).item(), 0)
            if variant == 'no_prior':
                self.assertIsInstance(model.zc_branch, ZcBranch)

    def test_no_film_ignores_u_and_keeps_conditional_prior(self):
        config = small_baseline()['model']
        model = build_knowledge_ablation(config, 'no_film').eval()
        self.assertIsInstance(model.prior, PriorNetwork)
        self.assertIsInstance(model.zc_branch, PassthroughZcBranch)
        self.assertFalse(hasattr(model.zc_branch, 'film_generator'))
        h = torch.randn(3, 8, 4)
        mask = torch.ones(3, 8, dtype=torch.bool)
        mask[:, :2] = False
        u = torch.randn(3, 11, requires_grad=True)
        mu, log_var = model.zc_branch(h, mask, u)
        other = model.zc_branch(h, mask, u + 7)
        torch.testing.assert_close(mu, other[0], rtol=0, atol=0)
        torch.testing.assert_close(log_var, other[1], rtol=0, atol=0)
        self.assertIsNone(torch.autograd.grad(mu.sum() + log_var.sum(), u, allow_unused=True)[0])
        prior = model.prior(u)
        changed = model.prior(u + 7)
        # The degradation prior head starts at zero by source design; C still depends on u.
        self.assertFalse(torch.equal(prior[2], changed[2]))
        prior[2].sum().backward()
        self.assertGreater(model.prior.zd_head.weight.numel(), 0)
        self.assertGreater(u.grad.abs().sum().item(), 0)

    def test_checkpoint_reload_and_public_loader_rejection(self):
        baseline = small_baseline()
        dataset = synthetic_flights([1, 2])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            configs = experiment.variant_configs(baseline, output, 42)
            for variant, config in configs.items():
                root = output / variant
                root.mkdir()
                experiment.seed_model(42)
                model = build_knowledge_ablation(config['model'], variant).eval()
                expected = predict_cycles(model, dataset)
                (root / 'config.yaml').write_text(yaml.safe_dump(config))
                optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'])
                path = root / 'best.pt'
                save_checkpoint(path, model, optimizer, config, 3, .25)
                experiment.mark_ablation_checkpoint(path, baseline)
                loaded, resolved, checkpoint = experiment.load_ablation_checkpoint(path)
                pd.testing.assert_frame_equal(predict_cycles(loaded, dataset), expected)
                self.assertEqual(checkpoint['variant'], variant)
                self.assertEqual(checkpoint['epoch'], 3)
                self.assertEqual(resolved['loss_weights'], baseline['loss_weights'])
                with self.assertRaisesRegex(ValueError, 'external model_type differs'):
                    read_evaluation_checkpoint(str(root / 'config.yaml'), str(path), 'cpu')
                checkpoint['variant'] = 'no_film' if variant != 'no_film' else 'no_prior'
                torch.save(checkpoint, path)
                with self.assertRaisesRegex(ValueError, 'inconsistent'):
                    experiment.load_ablation_checkpoint(path)

    def test_existing_losses_keep_kl_weights_and_gradients(self):
        baseline = small_baseline()
        settings = {**baseline['training'], **baseline['loss_weights']}
        x, mask, u, w, units, cycles, _ = synthetic_flights([1, 2]).tensors
        for variant in experiment.VARIANTS:
            model = build_knowledge_ablation(baseline['model'], variant).eval()
            out = model(x, mask, u, w)
            loss, terms = compute_total_loss(out, x, mask, cycles, units,
                                             settings['transition_end_epoch'], settings)
            if variant != 'no_film':
                for block in ('d', 'c'):
                    mu, log_var = out[f'mu_{block}'], out[f'log_var_{block}']
                    expected = .5 * (log_var.exp() + mu.square() - 1 - log_var).mean()
                    self.assertAlmostEqual(terms[f'kl_{block}'], expected.item(), places=6)
            expected_total = (terms['recon'] + settings['beta_d'] * terms['kl_d']
                + settings['beta_c'] * terms['kl_c'] + settings['lambda_hsic'] * terms['hsic']
                + settings['lambda_mono'] * terms['mono'] + settings['lambda_spearman'] * terms['spearman'])
            self.assertAlmostEqual(loss.item(), expected_total, places=5)
            loss.backward()
            for branch in (model.zd_branch, model.zc_branch):
                self.assertTrue(torch.isfinite(branch.head.weight.grad).all())
                self.assertGreater(branch.head.weight.grad.abs().sum().item(), 0)
            if variant == 'no_film':
                self.assertGreater(model.prior.zd_head.weight.grad.abs().sum().item(), 0)
                self.assertGreater(model.prior.zc_head.weight.grad.abs().sum().item(), 0)

    def test_sequential_workflow_with_no_updates_and_public_evaluation(self):
        baseline = small_baseline()
        calls = []
        datasets = {key: synthetic_flights(units) for key, units in (
            ('train', [1, 2]), ('val', [3]), ('dev', [1, 2, 3]), ('test', [4]))}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'kgld.yaml'
            raw = root / 'raw.h5'
            raw.touch()
            baseline['data']['h5_path'] = str(raw)
            for split in ('dev', 'test'):
                labels = datasets[split].rul[['unit', 'cycle']].copy()
                labels['HI'] = 1 - (labels.cycle - 1) / 4
                path = root / f'cycle_{split}.h5'
                labels.to_hdf(path, key='df')
                baseline['data']['cycle_hi_path' if split == 'dev' else 'test_cycle_hi_path'] = str(path)
            source.write_text(yaml.safe_dump(baseline))
            def dataset(h5, hi, config, split='train', val_fraction=.2):
                return datasets['dev' if split == 'train' and val_fraction == 0 else split]
            def save_without_training(model, train_loader, val_loader, config, device, *, run_dir):
                variant = config['experiment']['knowledge_ablation']
                calls.append(variant)
                self.assertEqual(config['training'], baseline['training'])
                self.assertEqual(train_loader.batch_sampler.random_seed, 42)
                self.assertEqual(val_loader.dataset, datasets['val'])
                run_dir.mkdir()
                (run_dir / 'config.yaml').write_text(yaml.safe_dump(config))
                optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'])
                save_checkpoint(run_dir / 'best.pt', model, optimizer, config, 0, 0.)
                return dict(best_checkpoint=str(run_dir / 'best.pt'))
            output = root / 'results'
            with patch.object(experiment, 'BASE_CONFIG', source), \
                    patch.object(experiment, 'NCMAPSSDataset', side_effect=dataset), \
                    patch.object(experiment.training, 'train_model', side_effect=save_without_training):
                experiment.run(output, 42, 'cpu')
            self.assertEqual(calls, list(experiment.VARIANTS))
            for variant in experiment.VARIANTS:
                run = output / variant
                metrics = json.loads((run / 'metrics.json').read_text())
                self.assertEqual(metrics['metadata']['scaling']['n_samples_seen_'], 12)
                self.assertEqual(metrics['test']['n_cycles'], 4)
                self.assertIsNotNone(metrics['test']['rmse'])
                self.assertTrue((run / 'hi_curves_dev.png').is_file())
                self.assertTrue((run / 'hi_curves_test.png').is_file())
                frame = pd.read_csv(run / 'scores.csv', float_precision='round_trip')
                loaded, _, _ = experiment.load_ablation_checkpoint(run / 'best.pt')
                predicted = predict_cycles(loaded, datasets['test'])
                np.testing.assert_array_equal(frame[frame.split == 'test'].score, predicted.score)
            with (output / 'comparison.csv').open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 3)
            self.assertEqual(list(rows[0]), ['变体', 'MAE↓', 'RMSE↓', 'Mon↑', 'Tren↑', 'FC↑', 'MI↑'])
            self.assertTrue((output / 'comparison.md').is_file())
            self.assertEqual(yaml.safe_load(source.read_text()), baseline)


if __name__ == '__main__':
    unittest.main()
