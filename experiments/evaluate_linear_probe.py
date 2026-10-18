"""Labelled frozen-representation HI evaluation, separate from unsupervised metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import sklearn
import torch
import yaml

from dataset import NCMAPSSDataset
from evaluate import (PROTOCOL_VERSION, _resolve_cycle_hi_path, _resolve_main_h5_path,
                      associate, load_evaluation_model, predict_cycles, read_evaluation_checkpoint,
                      read_reference, scale_dev_test)
from utils.linear_probe import (extract_representations, fit_linear_probes, join_probe_features,
                                resolve_probe_split)

HCDE_COMMIT = 'b5cfa1e069bd824ad65fc0ad427dd0e828ebf936'


def run(config_path: str, checkpoint_path: str | None, output_dir: str, device: str | None = None,
        batch_size: int | None = None, split_config: str | None = None) -> Path:
    """Freeze a saved model; fit each probe on train devices and evaluate held-out devices."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f'linear probe directory already exists: {output}')
    config, flat, checkpoint, checkpoint_file, device = read_evaluation_checkpoint(
        config_path, checkpoint_path, device)
    model_type = config.get('model_type', 'oc_ivae')
    if model_type not in ('oc_ivae', 'ocald', 'proposal_cnn'):
        raise ValueError('linear probes support OC-iVAE, OCALD and Proposal-C; Residual is excluded')
    if config['data'].get('input_mode', 'flight') != 'flight':
        raise ValueError('current probe adapters require one complete model input per flight')
    batch_size = batch_size if batch_size is not None else config.get('evaluation', {}).get('batch_size', 64)
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    h5_path = _resolve_main_h5_path(config['data'])
    dev = NCMAPSSDataset(h5_path, None, flat, val_fraction=0)
    test = NCMAPSSDataset(h5_path, None, flat, split='test')
    explicit = None
    if split_config is not None:
        with open(split_config, encoding='utf-8') as file:
            explicit = yaml.safe_load(file)
        if not isinstance(explicit, dict) or set(explicit) != {'fit_units', 'val_units'}:
            raise ValueError('split YAML must contain exactly fit_units and val_units')
    split = resolve_probe_split(config, checkpoint, sorted({u for u, _ in dev.samples}),
                                 sorted({u for u, _ in test.samples}), explicit)
    recorded_cache = checkpoint.get('processed_data', checkpoint.get('run_metadata', {}).get('processed_data'))
    if recorded_cache is not None and str(dev.processed_path) != recorded_cache:
        raise ValueError('probe data differs from recorded checkpoint preprocessing source')
    model = load_evaluation_model(config, flat, checkpoint, dev)
    model.requires_grad_(False).to(device).eval()
    checkpoint_hash = hashlib.sha256(checkpoint_file.read_bytes()).hexdigest()
    labelled, representations, reference_metadata = {}, {}, {}
    for name, dataset in [('dev', dev), ('test', test)]:
        path = _resolve_cycle_hi_path(flat, h5_path, name)
        if path is None:
            raise ValueError(f'{name}: labelled linear probes require DS03 reference HI')
        reference, provenance = read_reference(path, dataset.rul)
        labelled[name] = associate(predict_cycles(model, dataset, device, batch_size), dataset.rul, reference)
        representations[name] = extract_representations(model, dataset, device, batch_size)
        reference_metadata[name] = dict(path=str(Path(path).resolve()), key_provenance=provenance,
                                       n_reference=len(reference), n_matched=len(labelled[name]))
    # This scaler belongs to the established HI protocol, not to the probe.
    # It deliberately fits ALL dev scores without labels, including val devices.
    dev_hi, test_hi, hi_scaling = scale_dev_test(labelled['dev'], labelled['test'])
    dev_features = join_probe_features(representations['dev'], dev_hi)
    test_features = join_probe_features(representations['test'], test_hi)
    predictions, metrics, parameters = fit_linear_probes(dev_features, test_features, split)
    descriptions = dict(oc_ivae='mu_d, complete posterior mean used by HIHead',
                        ocald='mu_I, complete internal-block posterior mean after valid-prefix adaptation',
                        proposal_cnn='one-dimensional Proposal-C Z')
    metadata = dict(
        protocol='frozen-cycle-linear-probe-v1', supervised_representation_evaluation=True,
        target='DS03 reference HI, associated by unit/cycle', model_type=model_type,
        baseline_type=config.get('experiment', {}).get('baseline_type'),
        representation=descriptions[model_type], representation_PCA=False,
        feature_aggregation='none: current models return one vector per flight; duplicate keys rejected',
        proposal_equivalence_note=('Z and final HI differ only by dev-global affine scaling; '
                                   'the two probes may produce identical results'),
        reference_method=dict(repository='https://github.com/EPFL-IMOS/hcde', commit=HCDE_COMMIT,
            files=['src/train/task.py::DegradInferenceTask.get_latent_space',
                   'src/model/hcde.py::HCDE.forward', 'src/train/train.py::MappingTrainer.compute_loss'],
            adaptation='frozen complete health representation -> labelled OLS; DS03 HI and shared OC-iVAE inputs'),
        **split, split_config=None if split_config is None else str(Path(split_config).resolve()),
        fit_keys=predictions.loc[predictions.probe_split.eq('fit'), ['unit', 'cycle']].values.tolist(),
        cycle_weighting='pooled test cycles equally weighted; per-unit diagnostics do not replace pooled R2',
        probe='sklearn Pipeline(StandardScaler(), LinearRegression(fit_intercept=True))',
        scaler_and_regression_fit='fit_units ONLY; no val/test labels or features used for fitting',
        prediction_postprocessing='none: no sigmoid, output scaling or clipping',
        public_hi_protocol=PROTOCOL_VERSION, public_hi_scaling=hi_scaling,
        reference_HI=reference_metadata, processed_data=str(dev.processed_path),
        checkpoint=dict(path=str(checkpoint_file.resolve()), sha256=checkpoint_hash,
                        epoch=checkpoint.get('epoch'), stage=checkpoint.get('stage')),
        training_run_metadata=checkpoint.get('run_metadata', {}),
        config_path=str(Path(config_path).resolve()), device=device, batch_size=batch_size,
        torch_num_threads=torch.get_num_threads(),
        versions=dict(torch=str(torch.__version__), numpy=np.__version__, sklearn=sklearn.__version__),
        original_six_metrics='unchanged; not recomputed or overwritten by this labelled evaluation',
        diagnostics='fit scores are training diagnostics; val scores are held-out diagnostics, not used to select probes',
    )
    if hashlib.sha256(checkpoint_file.read_bytes()).hexdigest() != checkpoint_hash:
        raise ValueError('checkpoint changed during probe evaluation')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    (output / 'probe_metrics.json').write_text(json.dumps(dict(metadata=metadata, **metrics),
        indent=2, allow_nan=False), encoding='utf-8')
    (output / 'probe_parameters.json').write_text(json.dumps(parameters, indent=2, allow_nan=False), encoding='utf-8')
    feature_columns = [c for c in predictions if c.startswith('representation_')]
    predictions[['split', 'probe_split', 'unit', 'cycle', *feature_columns]].to_csv(
        output / 'probe_features.csv', index=False)
    predictions[['split', 'probe_split', 'unit', 'cycle', 'hi_gt', 'score', 'hi',
                 'predicted_hi_representation', 'predicted_hi_hi']].to_csv(output / 'probe_predictions.csv', index=False)
    for name in ('representation', 'hi'):
        pooled = {key: value for key, value in metrics[name]['test'].items() if key != 'per_unit'}
        print(name, 'test', pooled, 'rank', parameters[name]['rank'],
              'rank_deficient', parameters[name]['rank_deficient'],
              'underdetermined', parameters[name]['underdetermined'], flush=True)
    print(f'Labelled linear probe saved separately to: {output}', flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='training run/config.yaml')
    parser.add_argument('--checkpoint', help='defaults to saved run/best.pt')
    parser.add_argument('--output-dir', required=True, help='new independent probe directory')
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--split-config', help='explicit fit_units/val_units YAML if provenance is missing')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    run(args.config, args.checkpoint, args.output_dir, args.device, args.batch_size, args.split_config)


if __name__ == '__main__':
    main()
