"""Focused mathematical and protocol checks for migrated models.

Run: PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
"""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import yaml

from dataset import ShuffledSequentialBatches, ocald_joint_schedule
from evaluate import scale_dev_test, predict_residuals
from utils.optimizers import KerasAdam
from models import build_model
from models.ocald import OCALD, freeze_representation, time_positions, to_valid_prefix, training_loss
from models.proposal_cnn import ProposalCNN, proposal_loss
from models.residual import ResidualModel, fit_residual_pca, residual_scores
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]


def config(name):
    path = ROOT / f'config/history/{name}.yaml'
    if name == 'ds03_default':
        path = ROOT / 'config/kgld.yaml'
    elif name == 'proposal_cnn_c':
        path = ROOT / 'config/proposal_cnn_c.yaml'
    return yaml.safe_load(path.read_text())


def reference_conv(layer, x):
    """Independent channels-last SAME convolution using only contributing taps."""
    width = layer.kernel_size[0]
    rows = []
    for t in range(x.shape[1]):
        terms = [x[:, j] @ layer.weight[:, :, j - t + width // 2].T
                 for j in range(x.shape[1]) if 0 <= j - t + width // 2 < width]
        rows.append(sum(terms) + layer.bias)
    return torch.stack(rows, dim=1)


class MigratedModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_kgld_loads_saved_legacy_model_type(self):
        from evaluate import read_evaluation_checkpoint, load_evaluation_model
        from models.kgld import KGLD
        raw = config('ds03_default')
        raw['model']['in_features'] = 14
        raw['model'].update(c_hidden=4, n_backbone_layers=1, n_decoder_layers=1)
        source = KGLD(raw['model']).eval()
        legacy = copy.deepcopy(raw)
        legacy['model_type'] = 'oc_ivae'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'legacy.pt'
            torch.save(dict(config=legacy, model_type='oc_ivae', model_state_dict=source.state_dict()), checkpoint)
            requested = root / 'kgld.yaml'
            requested.write_text(yaml.safe_dump(raw))
            resolved, flat, saved, _, _ = read_evaluation_checkpoint(str(requested), str(checkpoint), 'cpu')
            x, w = torch.randn(2, 6, 14), torch.randn(2, 6, 4)
            mask, u = torch.ones(2, 6, dtype=torch.bool), torch.zeros(2, 11)
            loaded = load_evaluation_model(resolved, flat, saved, [(x[0],)]).eval()
            self.assertIsInstance(loaded, KGLD)
            before, after = source(x, mask, u, w), loaded(x, mask, u, w)
            for key in before:
                torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)

    def test_prefix_preserves_valid_values_and_time(self):
        x = torch.arange(2 * 5 * 14, dtype=torch.float64).reshape(2, 5, 14).requires_grad_()
        w = torch.arange(40, dtype=torch.float64).reshape(2, 5, 4)
        mask = torch.tensor([[0, 0, 1, 1, 1], [0, 0, 0, 0, 1]], dtype=torch.bool)
        a, b, prefix = to_valid_prefix(x, w, mask)
        for i in range(2):
            torch.testing.assert_close(a[i, prefix[i]], x[i, mask[i]], rtol=0, atol=0)
            torch.testing.assert_close(b[i, prefix[i]], w[i, mask[i]], rtol=0, atol=0)
        torch.testing.assert_close(time_positions(prefix).double(), torch.tensor([[0, .5, 1, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.float64))
        self.assertEqual(torch.count_nonzero(a[~prefix]).item(), 0)
        a.sum().backward()
        torch.testing.assert_close(x.grad, mask[:, :, None].expand_as(x).double(), rtol=0, atol=0)

    def test_ocald_losses_and_freezing(self):
        torch.manual_seed(3)
        model = OCALD(config('ocald_joint')['model']).double()
        x, w = torch.randn(4, 5, 14, dtype=torch.double), torch.randn(4, 5, 4, dtype=torch.double)
        mask = torch.tensor([[1, 1, 1, 0, 0]] * 4, dtype=torch.bool)
        out = model(x, w, mask)
        r = model.score(out['mu_i'])
        weights = dict(reconstruction=1., kl_i=1., kl_c=1., order=10., l2=1e-4)
        terms = training_loss(model.hi_head, weights, x=x, mask=mask, output=out,
                              r_early=r[:2], r_late=r[2:])
        recon = ((x - out['x_hat']).square() * mask[:, :, None]).sum((1, 2)) / 2
        kl_i = .5 * (out['logvar_i'].exp() + out['mu_i'].square() - 1 - out['logvar_i']).sum(-1)
        kl_c = .5 * (out['prior_logvar_c'] - out['logvar_c']
                    + out['logvar_c'].exp() / out['prior_logvar_c'].exp()
                    + (out['mu_c'] - out['prior_mu_c']).square() / out['prior_logvar_c'].exp() - 1).sum(-1)
        expected = (recon + kl_i + kl_c).mean() + 10 * F.softplus(r[:2] - r[2:]).mean()
        expected = expected + 1e-4 * sum(p.square().sum() for p in model.hi_head.parameters())
        torch.testing.assert_close(terms['loss'], expected)
        terms['loss'].backward()
        for module in (model.sensor_encoder, model.conditional_prior, model.decoder, model.hi_head):
            self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None), 0)
        model.zero_grad(set_to_none=True)
        freeze_representation(model)
        loss = F.softplus(model.score(model.encode(x, mask)[0])[:2] - model.score(model.encode(x, mask)[0])[2:]).mean()
        loss.backward()
        self.assertTrue(all(p.grad is None for name, p in model.named_parameters() if not name.startswith('hi_head.')))
        self.assertTrue(any(p.grad is not None for p in model.hi_head.parameters()))

    def test_cnn_forward_and_gradient_channels_last_reference(self):
        torch.manual_seed(2)
        settings = config('proposal_cnn_c')['model']
        settings['window_size'] = 3
        model = ProposalCNN(settings).double()
        reference = copy.deepcopy(model)
        x = torch.randn(3, 3, 14, dtype=torch.double)
        w = torch.randn(3, 3, 4, dtype=torch.double)
        out = model(x, w)
        h = x
        for layer in reference.encoder:
            h = reference_conv(layer, h).relu() if isinstance(layer, torch.nn.Conv1d) else h
        z = F.linear(h.reshape(3, -1), reference.z.weight, reference.z.bias).squeeze(-1)
        h = torch.cat((z[:, None, None].expand(-1, 3, -1), w), -1)
        for layer in reference.decoder:
            h = reference_conv(layer, h).relu() if isinstance(layer, torch.nn.Conv1d) else h
        expected = dict(z=z, x_hat=F.linear(h, reference.output.weight, reference.output.bias))
        torch.testing.assert_close(out['z'], z, rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(out['x_hat'], expected['x_hat'], rtol=1e-10, atol=1e-12)
        cycles = torch.tensor([1, 2, 8])
        a, b = [proposal_loss(x, value, cycles) for value in (out, expected)]
        a['loss'].backward(); b['loss'].backward()
        for p, q in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-8, atol=1e-10)

    def test_correlation_and_unmasked_mae(self):
        x = torch.zeros(3, 4, 14, dtype=torch.double)
        prediction = torch.ones_like(x)
        z = torch.tensor([3., 1., -2.], dtype=torch.double, requires_grad=True)
        t = torch.tensor([1., 3., 4.], dtype=torch.double)
        terms = proposal_loss(x, dict(z=z, x_hat=prediction), t)
        self.assertEqual(terms['mae'].item(), 1.)
        self.assertAlmostEqual(terms['correlation'].item(), np.corrcoef(z.detach().numpy(), t.numpy())[0, 1])
        self.assertTrue(torch.autograd.gradcheck(lambda a: proposal_loss(x, dict(z=a, x_hat=prediction), t)['loss'], (z,)))
        with self.assertRaises(FloatingPointError):
            proposal_loss(x, dict(z=torch.ones(3), x_hat=prediction), t)
        with self.assertRaises(FloatingPointError):
            proposal_loss(x, dict(z=z, x_hat=prediction), torch.ones(3))

    def test_keras_adam_epsilon_hat(self):
        p = torch.nn.Parameter(torch.tensor([1., -1.], dtype=torch.double))
        optimizer = KerasAdam([p], lr=1e-4, betas=(.9, .999), eps=1e-7)
        expected, m, v = p.detach().numpy().copy(), np.zeros(2), np.zeros(2)
        for step, g in enumerate(([1e-6, -.3], [.5, .2]), 1):
            g = np.array(g)
            p.grad = torch.tensor(g)
            optimizer.step()
            m += (g - m) * .1; v += (g * g - v) * .001
            expected -= 1e-4 * np.sqrt(1 - .999 ** step) / (1 - .9 ** step) * m / (np.sqrt(v) + 1e-7)
            np.testing.assert_allclose(p.detach().numpy(), expected, rtol=0, atol=1e-15)

    def test_samplers_and_global_scaling(self):
        sampler = ShuffledSequentialBatches(43, 20, 42)
        batches = list(sampler)
        self.assertEqual(sorted(batches, key=lambda b: b[0]), [list(range(20)), list(range(20, 40)), [40, 41, 42]])
        samples = [(u, c) for u in range(1, 6) for c in range(1, 9)]
        schedule = ocald_joint_schedule(samples, 2, 2, 42)
        for indices, pairs in zip(schedule['flights'].reshape(-1, 16), schedule['pairs'].reshape(-1, 16, 2)):
            keys = torch.tensor([samples[i] for i in indices])
            self.assertTrue(torch.equal(keys[pairs[:, 0], 0], keys[pairs[:, 1], 0]))
            self.assertTrue((keys[pairs[:, 1], 1] - keys[pairs[:, 0], 1] >= 5).all())
        dev = pd.DataFrame(dict(unit=[1, 1, 2], cycle=[1, 2, 1], score=[2., 4., 6.]))
        test = pd.DataFrame(dict(unit=[3, 3], cycle=[1, 2], score=[0., 10.]))
        a, b, meta = scale_dev_test(dev, test)
        np.testing.assert_array_equal(a.hi, [0, .5, 1])
        np.testing.assert_array_equal(b.hi, [-.5, 2.])
        self.assertEqual(meta['fit_split'], 'dev')

    def test_residual_paths_mae_and_gradients(self):
        torch.manual_seed(19)
        for variant in ('a', 'b'):
            with self.subTest(variant=variant):
                model = ResidualModel(config(f'residual_{variant}')['model']).double()
                for layer in model.modules():
                    if isinstance(layer, (torch.nn.Conv1d, torch.nn.Linear)):
                        self.assertEqual(layer.bias.count_nonzero(), 0)
                        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(layer.weight)
                        self.assertLessEqual(layer.weight.abs().max(), (6 / (fan_in + fan_out)) ** .5)
                reference = copy.deepcopy(model)
                x = torch.randn(2, 5, 14, dtype=torch.double, requires_grad=True)
                w = torch.randn(2, 5, 4, dtype=torch.double, requires_grad=True)
                x.data[:, :2] = 0  # common left padding still contributes to MAE
                output = model(x, w)
                h = torch.cat((x, w), -1) if variant == 'a' else w
                for layer in reference.convolutions:
                    if isinstance(layer, torch.nn.Conv1d):
                        h = reference_conv(layer, h)
                    else:
                        h = h.tanh() if variant == 'a' else h.relu()
                expected = (F.linear(h, reference.output.weight, reference.output.bias)
                            if variant == 'a' else reference_conv(reference.output, h))
                self.assertEqual(output.shape, x.shape)
                torch.testing.assert_close(output, expected, rtol=1e-10, atol=1e-12)
                if variant == 'b':
                    torch.testing.assert_close(output, model(x.detach() + 99, w), rtol=0, atol=0)
                    self.assertIsNone(torch.autograd.grad(output.sum(), x, allow_unused=True)[0])
                else:
                    self.assertGreater(torch.autograd.grad(output.sum(), x, retain_graph=True)[0].abs().sum(), 0)
                self.assertGreater(torch.autograd.grad(output.sum(), w, retain_graph=True)[0].abs().sum(), 0)
                loss = (output - x).abs().mean()
                expected_loss = (expected - x).abs().sum() / x.numel()
                torch.testing.assert_close(loss, expected_loss)
                self.assertGreater((output[:, :2] - x[:, :2]).abs().sum(), 0)
                loss.backward(); expected_loss.backward()
                for p, q in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(p.grad, q.grad, rtol=1e-8, atol=1e-10)

    def test_residual_flatten_pca_and_direction(self):
        # A minimal seven-field fixture checks actual inference, not a second flatten helper.
        x = torch.arange(2 * 3 * 14, dtype=torch.float32).reshape(2, 3, 14) / 50
        w = torch.ones(2, 3, 4)
        fixture = [(x[i], torch.ones(3, dtype=torch.bool), torch.zeros(11), w[i],
                    torch.tensor(1), torch.tensor(i + 1), torch.tensor(float('nan'))) for i in range(2)]
        model = ResidualModel(config('residual_a')['model']).eval()
        expected = (model(x, w) - x).abs().detach().numpy().reshape(2, -1)
        np.testing.assert_array_equal(predict_residuals(model, fixture, 'cpu', 2), expected)
        rng = np.random.default_rng(12)
        keys = [(u, c) for u in (2, 1) for c in range(30, 0, -1)]
        residuals = rng.uniform(size=(60, 18)).astype(np.float32)
        residuals[:, 0] += np.asarray([c / 3 for _, c in keys])
        state = fit_residual_pca(residuals, keys, 42)
        pca = PCA(n_components=1, random_state=42).fit(residuals)
        z = pca.transform(residuals).ravel()
        by_key = dict(zip(keys, z))
        delta = np.mean([np.mean([by_key[u, c] for c in range(1, 11)])
                         - np.mean([by_key[u, c] for c in range(21, 31)]) for u in (1, 2)])
        self.assertEqual(state['direction'], -1 if delta < 0 else 1)
        np.testing.assert_allclose(residual_scores(residuals, state), z * state['direction'], rtol=0, atol=0)
        test = rng.uniform(20, 30, size=(5, 18)).astype(np.float32)
        np.testing.assert_allclose(residual_scores(test, state),
                                   pca.transform(test).ravel() * state['direction'], rtol=0, atol=0)
        self.assertEqual(state['fit_keys'], [list(k) for k in keys])
        wide = rng.uniform(size=(60, 600)).astype(np.float32)
        wide_state = fit_residual_pca(wide, keys, 42)
        self.assertEqual(wide_state['solver'], 'randomized')
        wide_pca = PCA(n_components=1, random_state=42).fit(wide)
        np.testing.assert_array_equal(residual_scores(wide, wide_state),
                                      wide_pca.transform(wide).ravel() * wide_state['direction'])
        # Reverse chronology produces the other sign; test has no role in either decision.
        reverse_keys = [(u, 31 - c) for u, c in keys]
        reverse = fit_residual_pca(residuals, reverse_keys, 42)
        self.assertEqual(reverse['direction'], -state['direction'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pca.pt'
            torch.save(state, path)
            restored = torch.load(path, weights_only=False)
            np.testing.assert_array_equal(residual_scores(test, restored), residual_scores(test, state))

    def test_score_direction_and_reload(self):
        torch.manual_seed(4)
        x, w = torch.randn(2, 6, 14), torch.randn(2, 6, 4)
        mask = torch.tensor([[0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1]], dtype=torch.bool)
        for name in ('ocald_original', 'ocald_joint_decoder_w', 'proposal_cnn_c'):
            raw = config(name)
            if name == 'proposal_cnn_c': raw['model']['window_size'] = 6
            model = build_model(raw, raw['model']).eval()
            if isinstance(model, OCALD):
                a, _, m = to_valid_prefix(x, w, mask)
                expected = -model.score(model.encode(a, m)[0])
                score = model.health_score(x, w, mask)
            else:
                expected = score = model.encode(x)
            torch.testing.assert_close(score, expected, rtol=0, atol=0)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'weights.pt'
                torch.save(model.state_dict(), path)
                restored = build_model(raw, raw['model']).eval()
                restored.load_state_dict(torch.load(path, weights_only=True))
            reloaded = restored.health_score(x, w, mask) if isinstance(restored, OCALD) else restored.encode(x)
            torch.testing.assert_close(score, reloaded, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
