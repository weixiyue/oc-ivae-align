"""Train Residual-A/B on train-unit healthy cycles, then fit PCA on complete dev."""
from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader, Subset

from dataset import NCMAPSSDataset, ShortWindowDataset, healthy_training_indices
from evaluate import predict_residuals, run_evaluation
from models.residual import ResidualModel, fit_residual_pca, window_residual_matrix
from utils.experiment import append_history, save_config, save_experiment_checkpoint, start_run
from utils.optimizers import KerasAdam


def run(config_path: str, run_dir: str | None, device: str, smoke: bool = False) -> str:
    """Train unmasked mean MAE, select last epoch, then evaluate without refitting PCA."""
    config, output, train, _ = start_run(config_path, run_dir, 'residual', device, smoke)
    settings = config['training']
    window_mode = isinstance(train, ShortWindowDataset)
    reduction = ('mean_absolute_error_all_batch_time_features' if window_mode
                 else 'mean_absolute_error_all_batch_time_features_including_padding')
    if (settings['optimizer'], settings['healthy_cycle_threshold'], settings['sampling'],
            settings['loss_reduction'], settings['epsilon_convention']) != (
                'keras_adam', 20, 'shuffle_samples_then_batch',
                reduction,
                'epsilon_hat_after_uncorrected_sqrt_v'):
        raise ValueError('residual training requires the reference healthy-prefix/MAE/Adam protocol')
    evaluation = config['evaluation']
    if (evaluation['pca_fit'], evaluation['pca_components'], evaluation['pca_solver']) != (
            'complete_dev_windows' if window_mode else 'complete_dev', 1, 'auto'):
        raise ValueError('residual PCA requires one component fitted on complete dev with sklearn auto solver')
    indices = healthy_training_indices(train, settings['healthy_cycle_threshold'])
    settings['max_batches_per_epoch'] = config['smoke']['max_batches_per_epoch'] if smoke else None
    if smoke:
        settings['epochs'] = config['smoke']['epochs']
    healthy_keys = sorted({tuple(map(int, train.samples[i][:2])) for i in indices})
    config['run_metadata'].update(
        healthy_network_prior='train units cycle<=20; no HI/RUL/hs filtering',
        eligible_training_keys=[list(key) for key in healthy_keys],
        pca_fit_split='complete dev (train union val), independent of healthy network subset',
        input_protocol='shared KGLD cached normalization/left padding; differs from source notebook',
    )
    healthy_flights = len(healthy_keys)
    config['run_metadata'].update(healthy_flights=healthy_flights,
        eligible_training_samples=len(indices), input_mode='short_window' if window_mode else 'flight')
    if window_mode:
        config['run_metadata'].update(
            input_protocol='KGLD normalization; full effective sequences before truncation/padding',
            pca_fit_split='all dev windows, including validation units; no network updates on validation',
            training_key_record='first 16 updated windows of last epoch (unit,cycle,start)')
    save_config(output, config)
    model = ResidualModel(config['model']).to(device)
    optimizer = KerasAdam(model.parameters(), settings['learning_rate'], tuple(settings['betas']), settings['eps'])
    loader = DataLoader(Subset(train, indices), batch_size=settings['batch_size'], shuffle=True,
                        generator=torch.Generator().manual_seed(settings['seed']))
    steps = 0
    for epoch in range(1, settings['epochs'] + 1):
        model.train()
        if device == 'cuda':
            torch.cuda.synchronize()
        started = time.perf_counter()
        loss_sum = samples = epoch_steps = 0
        drawn_keys = []
        for x, _, _, w, units, cycles, starts in loader:
            if settings['max_batches_per_epoch'] is not None and epoch_steps >= settings['max_batches_per_epoch']:
                break
            x, w = x.to(device), w.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x, w) - x).abs().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite residual MAE')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('nonfinite residual gradients')
            optimizer.step()
            loss_sum += loss.detach().item() * len(x)
            samples += len(x)
            epoch_steps += 1
            steps += 1
            if window_mode:
                drawn_keys.extend([list(key) for key in zip(units.tolist(), cycles.tolist(), starts.tolist(),
                                  strict=True)][:max(0, 16 - len(drawn_keys))])
            else:
                drawn_keys.extend([list(pair) for pair in zip(units.tolist(), cycles.tolist(), strict=True)])
        if not samples:
            raise ValueError('residual model produced no training updates')
        if device == 'cuda':
            torch.cuda.synchronize()
        append_history(output, dict(stage=f"residual_{config['model']['variant'].lower()}", epoch=epoch,
            mae=loss_sum / samples, healthy_flights=healthy_flights,
            eligible_windows=len(indices) if window_mode else 0,
            optimizer_steps=epoch_steps, cumulative_steps=steps,
            samples_drawn=samples, seconds=time.perf_counter() - started), first=epoch == 1)
    config['run_metadata']['last_epoch_training_keys'] = drawn_keys
    save_config(output, config)
    dataset_class = ShortWindowDataset if window_mode else NCMAPSSDataset
    dev = dataset_class(config['data']['h5_path'], None, config['data'], val_fraction=0)
    started = time.perf_counter()
    if window_mode:
        with window_residual_matrix(model, dev, device, evaluation['batch_size']) as residuals:
            state = fit_residual_pca(residuals, dev.samples, evaluation['pca_random_state'])
        state.update(window_length=dev.window_length, stride=dev.stride,
                     preprocessing_source=str(dev.processed_path))
    else:
        residuals = predict_residuals(model, dev, device, evaluation['batch_size'])
        state = fit_residual_pca(residuals, dev.samples, evaluation['pca_random_state'])
    print(f"PCA fit on {len(dev)} complete dev inputs in {time.perf_counter() - started:.3f}s", flush=True)
    save_experiment_checkpoint(output / 'best.pt', model, optimizer, config, 'residual', epoch,
                               residual_state=state)
    run_evaluation(str(output / 'config.yaml'), str(output / 'best.pt'), str(output), device)
    return str(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-dir')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    run(args.config, args.run_dir, args.device, args.smoke)


if __name__ == '__main__':
    main()
