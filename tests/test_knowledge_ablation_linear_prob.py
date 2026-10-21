"""Dedicated knowledge probe report checks on untrained synthetic checkpoints."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml

from experiments import run_knowledge_ablation_linear_prob as knowledge
from experiments import run_linear_probe_comparison as batch
from experiments import evaluate_linear_probe as single
from models.kgld import PriorNetwork
from models.vae import StandardNormalPrior, PassthroughZcBranch
from train import state_digest
from test_linear_probe_comparison import fixtures


class KnowledgeLinearProb(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_dry_run_checks_reference_hi_without_probing_or_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = fixtures(root)[:4]
            output = root / 'knowledge_probe'
            with patch.object(knowledge, 'run_structural_probe') as structural, \
                    patch.object(knowledge.probe, 'run') as ordinary, contextlib.redirect_stdout(io.StringIO()):
                summary = knowledge.run([str(p) for p in runs], str(output), 'cuda', dry_run=True)
            structural.assert_not_called()
            ordinary.assert_not_called()
            self.assertFalse(output.exists())
            self.assertEqual(summary['counts'], dict(pending=4, success=0, skipped=0, failed=0))
            self.assertEqual(summary['protocol_reference_run'], str(runs[0]))
            self.assertEqual([r['method'] for r in summary['runs']], list(knowledge.LABELS.values()))
            for record in summary['runs']:
                self.assertEqual(record['protocol_comparison']['differences'], [])
                self.assertEqual(record['reference_HI']['dev']['n_cycles'], 24)
            output.mkdir()
            with self.assertRaises(FileExistsError):
                knowledge.run([str(runs[0])], str(output), dry_run=True)

    def test_partition_and_reference_differences_are_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = fixtures(root)[:4]
            source = runs[2]  # no_film
            checkpoint = torch.load(source / 'best.pt', weights_only=False)
            config = checkpoint['config']
            original = pd.read_hdf(config['data']['cycle_hi_path'], key='df')
            changed = original.copy()
            changed['HI'] += .1
            changed_path = root / 'changed_dev_hi.h5'
            changed.to_hdf(changed_path, key='df')
            config['data']['cycle_hi_path'] = str(changed_path)
            metadata = copy.deepcopy(checkpoint['run_metadata'])
            metadata['train_units'], metadata['val_units'] = sorted(
                metadata['train_units'][1:] + metadata['val_units']), metadata['train_units'][:1]
            metadata.pop('split_seed')
            checkpoint['run_metadata'] = metadata
            config['run_metadata'] = metadata
            torch.save(checkpoint, source / 'best.pt')
            (source / 'config.yaml').write_text(yaml.safe_dump(config))
            with contextlib.redirect_stdout(io.StringIO()):
                summary = knowledge.run([str(p) for p in runs], str(root / 'output'), dry_run=True)
            self.assertEqual(summary['counts']['failed'], 0)
            differences = summary['runs'][2]['protocol_comparison']['differences']
            self.assertEqual(differences, ['fit_units', 'val_units', 'dev_reference_HI'])
            self.assertTrue(summary['runs'][2]['protocol_comparison']['matches']['test_reference_HI'])

    def test_strict_reloads_freeze_and_match_single_probe_results(self):
        with tempfile.TemporaryDirectory(prefix='knowledge_probe_synthetic_') as directory:
            root = Path(directory)
            runs = fixtures(root)[:4]
            originals = {p: p.read_bytes() for source in runs for p in source.iterdir() if p.is_file()}
            loaded = []
            ordinary_loader = single.load_evaluation_model
            structural_loader = batch.load_ablation_checkpoint
            def capture_ordinary(*args):
                model = ordinary_loader(*args)
                loaded.append((model, state_digest(model.state_dict())))
                return model
            def capture_structural(*args):
                model, config, checkpoint = structural_loader(*args)
                variant = checkpoint['variant']
                self.assertIsInstance(model.prior, PriorNetwork if variant == 'no_film' else StandardNormalPrior)
                if variant != 'no_prior':
                    self.assertIsInstance(model.zc_branch, PassthroughZcBranch)
                else:
                    self.assertTrue(hasattr(model.zc_branch, 'film_generator'))
                loaded.append((model, state_digest(model.state_dict())))
                return model, config, checkpoint
            output = root / 'dedicated'
            with patch.object(single, 'load_evaluation_model', side_effect=capture_ordinary), \
                    patch.object(batch, 'load_ablation_checkpoint', side_effect=capture_structural), \
                    contextlib.redirect_stdout(io.StringIO()):
                summary = knowledge.run([str(p) for p in runs], str(output), 'cpu')
            self.assertEqual(summary['counts'], dict(pending=0, success=4, skipped=0, failed=0))
            self.assertEqual(len(loaded), 4)
            for model, digest in loaded:
                self.assertEqual(state_digest(model.state_dict()), digest)
                self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
            for source, record in zip(runs, summary['runs']):
                path = Path(record['output'])
                features = pd.read_csv(path / 'probe_features.csv', float_precision='round_trip')
                parameters = json.loads((path / 'probe_parameters.json').read_text())
                fit = features.probe_split.eq('fit')
                columns = parameters['representation']['feature_columns']
                self.assertEqual(len(columns), 2)
                self.assertEqual(parameters['representation']['n_fit'], 18)
                np.testing.assert_allclose(features.loc[fit, columns].mean().values,
                                           parameters['representation']['scaler_mean'], atol=1e-12)
                direct = root / f'direct_{record["variant"]}'
                context = batch.inspect_run(source)
                with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
                    if context['structural']:
                        batch.run_structural_probe(source, direct, context, 'cpu')
                    else:
                        single.run(str(source / 'config.yaml'), str(source / 'best.pt'), str(direct), 'cpu')
                expected = json.loads((direct / 'probe_metrics.json').read_text())
                actual = json.loads((path / 'probe_metrics.json').read_text())
                for name in ('representation', 'hi', 'unprobed_hi'):
                    self.assertEqual(actual[name], expected[name])
                self.assertEqual(actual['metadata']['knowledge_ablation_comparison']['method'], record['method'])
            report = (output / knowledge.REPORT_NAME).read_text()
            for label in knowledge.LABELS.values():
                self.assertIn(label, report)
            self.assertIn('Seed', report)
            self.assertIn('all device partitions and reference HI agree', report)
            self.assertEqual([p.name for p in output.iterdir() if p.is_file()], [knowledge.REPORT_NAME])
            for path, contents in originals.items():
                self.assertEqual(path.read_bytes(), contents)

    def test_corrupt_structure_fails_strictly_and_continues(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = fixtures(root)[:4]
            source = runs[1]
            checkpoint = torch.load(source / 'best.pt', weights_only=False)
            checkpoint['model_state_dict'].pop('zc_branch.head.weight')
            torch.save(checkpoint, source / 'best.pt')
            with contextlib.redirect_stdout(io.StringIO()):
                summary = knowledge.run([str(source), str(runs[2])], str(root / 'output'), 'cpu')
            self.assertEqual(summary['counts']['failed'], 1)
            self.assertEqual(summary['counts']['success'], 1)
            self.assertIn('Missing key(s) in state_dict', summary['runs'][0]['reason'])


if __name__ == '__main__':
    unittest.main()
