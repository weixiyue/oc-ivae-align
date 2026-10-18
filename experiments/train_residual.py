"""Train Residual-A/B on train-unit healthy cycles, then fit PCA on complete dev."""
from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader, Subset

from dataset import NCMAPSSDataset, healthy_training_indices
from evaluate import predict_residuals, run_evaluation
from models.residual import ResidualModel, fit_residual_pca
from utils.experiment import append_history, save_config, save_experiment_checkpoint, start_run
from utils.optimizers import KerasAdam


def run(config_path: str, run_dir: str | None, device: str, smoke: bool = False) -> str:
    """Train unmasked mean MAE, select last epoch, then evaluate without refitting PCA."""
    config, output, train, _ = start_run(config_path, run_dir, 'residual', device, smoke)
    settings = config['training']
    if (settings['optimizer'], settings['healthy_cycle_threshold'], settings['sampling'],
            settings['loss_reduction'], settings['epsilon_convention']) != (
                'keras_adam', 20, 'shuffle_samples_then_batch',
                'mean_absolute_error_all_batch_time_features_including_padding',
                'epsilon_hat_after_uncorrected_sqrt_v'):
        raise ValueError('residual training requires the reference healthy-prefix/MAE/Adam protocol')
    evaluation = config['evaluation']
    if (evaluation['pca_fit'], evaluation['pca_components'], evaluation['pca_solver']) != (
            'complete_dev', 1, 'auto'):
        raise ValueError('residual PCA requires one component fitted on complete dev with sklearn auto solver')
    indices = healthy_training_indices(train, settings['healthy_cycle_threshold'])
    settings['max_batches_per_epoch'] = config['smoke']['max_batches_per_epoch'] if smoke else None
    if smoke:
        settings['epochs'] = config['smoke']['epochs']
    config['run_metadata'].update(
        healthy_network_prior='train units cycle<=20; no HI/RUL/hs filtering',
        eligible_training_keys=[list(train.samples[i]) for i in indices],
        pca_fit_split='complete dev (train union val), independent of healthy network subset',
        input_protocol='shared OC-iVAE cached normalization/left padding; differs from source notebook',
    )
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
        for x, _, _, w, units, cycles, _ in loader:
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
            drawn_keys.extend([list(pair) for pair in zip(units.tolist(), cycles.tolist(), strict=True)])
        if not samples:
            raise ValueError('residual model produced no training updates')
        if device == 'cuda':
            torch.cuda.synchronize()
        append_history(output, dict(stage=f"residual_{config['model']['variant'].lower()}", epoch=epoch,
            mae=loss_sum / samples, optimizer_steps=epoch_steps, cumulative_steps=steps,
            samples_drawn=samples, seconds=time.perf_counter() - started), first=epoch == 1)
    config['run_metadata']['last_epoch_training_keys'] = drawn_keys
    save_config(output, config)
    dev = NCMAPSSDataset(config['data']['h5_path'], None, config['data'], val_fraction=0)
    started = time.perf_counter()
    residuals = predict_residuals(model, dev, device, config['evaluation']['batch_size'])
    state = fit_residual_pca(residuals, dev.samples, evaluation['pca_random_state'])
    print(f"PCA fit on {len(dev)} complete dev flights in {time.perf_counter() - started:.3f}s", flush=True)
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
