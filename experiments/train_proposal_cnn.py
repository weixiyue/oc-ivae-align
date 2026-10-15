"""Train Proposal-C with sequential-then-shuffled batches and the shared evaluator."""
from __future__ import annotations

import argparse
import math
import time

import torch
from torch.utils.data import DataLoader

from dataset import ShuffledSequentialBatches
from evaluate import run_evaluation
from models.proposal_cnn import ProposalCNN, proposal_loss
from utils.experiment import append_history, save_config, save_experiment_checkpoint, start_run


class KerasAdam(torch.optim.Optimizer):
    """The source Keras Adam epsilon-hat convention, without weight decay/AMSGrad.

    alpha = lr*sqrt(1-beta2**t)/(1-beta1**t); denominator = sqrt(v)+epsilon.
    This differs from moving epsilon outside the bias-corrected second moment.
    """
    def __init__(self, parameters, lr: float, betas: tuple[float, float], eps: float):
        super().__init__(parameters, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                state = self.state[parameter]
                if not state:
                    state.update(step=0, exp_avg=torch.zeros_like(parameter),
                                 exp_avg_sq=torch.zeros_like(parameter))
                state['step'] += 1
                m, v = state['exp_avg'], state['exp_avg_sq']
                m.add_((gradient - m) * (1 - beta1))
                v.add_((gradient.square() - v) * (1 - beta2))
                alpha = group['lr'] * math.sqrt(1 - beta2 ** state['step']) / (1 - beta1 ** state['step'])
                parameter.addcdiv_(m, v.sqrt().add_(group['eps']), value=-alpha)


def run(config_path: str, run_dir: str | None, device: str, smoke: bool = False) -> str:
    """Train on shared train units, select last epoch, evaluate full dev/test."""
    config, output, dataset, _ = start_run(config_path, run_dir, 'proposal_cnn', device, smoke)
    settings = config['training']
    if (settings['optimizer'] != 'keras_adam' or settings['correlation_weight'] != 1
            or settings['mae_weight'] != 1):
        raise ValueError('Proposal-C requires Keras Adam and C weight 1')
    if settings['weight_decay'] != 0 or settings['amsgrad'] or settings['gradient_clip_norm'] is not None:
        raise ValueError('source Proposal-C uses no weight decay, AMSGrad or gradient clipping')
    config['model']['window_size'] = dataset.L_max
    settings['max_batches_per_epoch'] = config['smoke']['max_batches_per_epoch'] if smoke else None
    if smoke:
        settings['epochs'] = config['smoke']['epochs']
    save_config(output, config)
    model = ProposalCNN(config['model']).to(device)
    optimizer = KerasAdam(model.parameters(), settings['learning_rate'], tuple(settings['betas']), settings['eps'])
    loader = DataLoader(dataset, batch_sampler=ShuffledSequentialBatches(
        len(dataset), settings['batch_size'], settings['seed']))
    steps = 0
    for epoch in range(1, settings['epochs'] + 1):
        model.train()
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        started = time.perf_counter()
        totals = dict.fromkeys(('loss', 'mae', 'correlation'), 0.)
        samples = epoch_steps = 0
        for x, _, _, w, _, cycles, _ in loader:
            if settings['max_batches_per_epoch'] is not None and epoch_steps >= settings['max_batches_per_epoch']:
                break
            x, w, cycles = x.to(device), w.to(device), cycles.to(device)
            optimizer.zero_grad(set_to_none=True)
            terms = proposal_loss(x, model(x, w), cycles, settings['correlation_weight'])
            terms['loss'].backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('nonfinite Proposal-C gradients')
            optimizer.step()
            for key in totals:
                totals[key] += terms[key].detach().item() * len(x)
            samples += len(x)
            epoch_steps += 1
            steps += 1
        if not samples:
            raise ValueError('Proposal-C produced no training updates')
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        append_history(output, dict(stage='proposal_cnn_c', epoch=epoch,
            **{k: v / samples for k, v in totals.items()}, mae_weight=settings['mae_weight'],
            correlation_weight=settings['correlation_weight'],
            optimizer_steps=epoch_steps, cumulative_steps=steps, samples_drawn=samples,
            seconds=time.perf_counter() - started), first=epoch == 1)
    save_experiment_checkpoint(output / 'best.pt', model, optimizer, config, 'proposal_cnn_c', epoch)
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
