"""Train the original/global OCALD under the shared OC-iVAE data protocol."""
from __future__ import annotations

import argparse
import math
import time

import torch
from torch.utils.data import DataLoader, Subset, default_collate

from dataset import ocald_joint_schedule, sample_pairs, valid_pair_pools
from evaluate import run_evaluation
from models.ocald import OCALD, REPRESENTATION_MODULES, freeze_representation, to_valid_prefix, training_loss
from utils.experiment import append_history, save_config, save_experiment_checkpoint, seed_run, start_run


def adam(parameters, settings: dict) -> torch.optim.Adam:
    """OCALD source Adam numerical defaults, made explicit in each YAML."""
    if settings['optimizer'] != 'Adam':
        raise ValueError('OCALD source optimizer is Adam')
    return torch.optim.Adam(parameters, lr=settings['learning_rate'],
                            betas=tuple(settings['betas']), eps=settings['eps'],
                            weight_decay=settings['weight_decay'], amsgrad=False)


def update(terms: dict, optimizer: torch.optim.Optimizer, parameters: list,
           clip: float | None) -> None:
    """Fail on nonfinite objectives/gradients; preserve stage-specific clipping."""
    if not all(torch.isfinite(value) for value in terms.values()):
        raise FloatingPointError('nonfinite OCALD loss')
    optimizer.zero_grad(set_to_none=True)
    terms['loss'].backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
        raise FloatingPointError('nonfinite OCALD gradient')
    if clip is not None:
        torch.nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
    optimizer.step()


def run(config_path: str, run_dir: str | None, device: str, smoke: bool = False) -> str:
    """Run two-stage or joint training; always evaluate complete dev and test."""
    config, output, dataset, _ = start_run(config_path, run_dir, 'ocald', device, smoke)
    training = config['training']
    pipeline = training['pipeline']
    if pipeline not in ('two_stage', 'joint'):
        raise ValueError('OCALD pipeline must be two_stage or joint')
    stages = ['representation', 'hi'] if pipeline == 'two_stage' else ['joint']
    if smoke:
        for stage in stages:
            training[stage]['epochs'] = config['smoke']['epochs']
        # Keep valid same-unit cycle gaps for ranking; only training is reduced.
        indices = []
        for unit in sorted({u for u, _ in dataset.samples}):
            unit_indices = [i for i, (u, _) in enumerate(dataset.samples) if u == unit]
            indices.extend(unit_indices[:config['smoke']['flights_per_unit']])
        training['max_batches_per_epoch'] = config['smoke']['max_batches_per_epoch']
    else:
        indices = list(range(len(dataset)))
        training['max_batches_per_epoch'] = None
    train = Subset(dataset, indices)
    samples = [dataset.samples[i] for i in indices]
    config['run_metadata']['training_flights'] = [list(key) for key in samples]
    model = OCALD(config['model']).to(device)
    save_config(output, config)
    steps = 0
    first = True
    for stage in stages:
        settings = training[stage]
        if stage == 'hi':
            # Source Stage 2 starts from seed-initialized HIHead and saved representation.
            seed_run(training['seed'])
            representation = {k: v.detach().clone() for k, v in model.state_dict().items()
                              if k.split('.')[0] in REPRESENTATION_MODULES}
            model = OCALD(config['model']).to(device)
            model.load_state_dict(representation, strict=False)
            freeze_representation(model)
            means, units, cycles = [], [], []
            with torch.no_grad():
                for x, mask, _, w, unit, cycle, _ in DataLoader(train, batch_size=16, shuffle=False):
                    x, w, mask = to_valid_prefix(x.to(device), w.to(device), mask.to(device))
                    means.append(model.encode(x, mask)[0])
                    units.append(unit.to(device))
                    cycles.append(cycle.to(device))
            means = torch.cat(means)
            pools = valid_pair_pools(torch.cat(units), torch.cat(cycles), settings['min_cycle_gap'])
            parameters = list(model.hi_head.parameters())
            frozen = {k: v.detach().clone() for k, v in model.state_dict().items()
                      if k.split('.')[0] in REPRESENTATION_MODULES}
        else:
            for p in model.hi_head.parameters():
                p.requires_grad_(stage == 'joint')
            parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = adam(parameters, settings)
        weights = settings['loss_weights']
        if stage == 'representation':
            loader = DataLoader(train, batch_size=settings['batch_size'], shuffle=True,
                                generator=torch.Generator().manual_seed(training['seed']))
        elif stage == 'joint':
            if (settings['batch_size'], settings['units_per_batch'], settings['flights_per_unit'],
                    settings['pairs_per_batch'], settings['min_cycle_gap']) != (16, 4, 4, 16, 5):
                raise ValueError('joint training preserves source 4x4 grouping, 16 pairs, gap 5')
            batches = math.ceil(len(train) / settings['batch_size'])
            if smoke:
                batches = min(batches, training['max_batches_per_epoch'])
            schedule = ocald_joint_schedule(samples, settings['epochs'], batches, training['seed'])
        for epoch in range(1, settings['epochs'] + 1):
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            started = time.perf_counter()
            totals = dict.fromkeys(('loss', 'reconstruction', 'kl_i', 'kl_c', 'order', 'l2'), 0.)
            count = pairs_drawn = epoch_steps = 0
            if stage == 'hi':
                model.eval()
                model.hi_head.train()
                pairs = sample_pairs(pools, len(train))
                batches_iter = pairs.split(settings['batch_size_pairs'])
            elif stage == 'joint':
                model.train()
                batches_iter = zip(schedule['flights'][epoch - 1], schedule['pairs'][epoch - 1])
            else:
                model.train()
                batches_iter = loader
            for batch in batches_iter:
                limit = training['max_batches_per_epoch']
                if limit is not None and epoch_steps >= limit:
                    break
                if stage == 'hi':
                    pair = batch
                    terms = training_loss(model.hi_head, weights,
                        r_early=model.score(means[pair[:, 0]]), r_late=model.score(means[pair[:, 1]]),
                        kappa=settings['kappa'])
                    size = len(pair)
                    pairs_drawn += len(pair)
                else:
                    if stage == 'joint':
                        selected, pair = batch
                        batch = default_collate([train[i] for i in selected.tolist()])
                        pair = pair.to(device)
                    x, mask, _, w, units, cycles, _ = batch
                    x, w, mask = to_valid_prefix(x.to(device), w.to(device), mask.to(device))
                    units, cycles = units.to(device), cycles.to(device)
                    out = model(x, w, mask)
                    if stage == 'representation':
                        pair = torch.nonzero((units[:, None] == units[None, :])
                            & (cycles[None, :] - cycles[:, None] >= settings['min_cycle_gap']))
                    scores = model.score(out['mu_i'])
                    terms = training_loss(model.hi_head, weights, x=x, mask=mask, output=out,
                        r_early=scores[pair[:, 0]], r_late=scores[pair[:, 1]],
                        sigma_x_squared=settings['sigma_x_squared'], kappa=settings['kappa'])
                    pairs_drawn += len(pair)
                    size = len(x)
                update(terms, optimizer, parameters, settings['gradient_clip_norm'])
                # Source joint logs equal batch averages; two-stage logs flight/pair averages.
                weight = 1 if stage == 'joint' else size
                for key in totals:
                    totals[key] += terms[key].detach().item() * weight
                count += size
                epoch_steps += 1
                steps += 1
            divisor = epoch_steps if stage == 'joint' else count
            if not divisor:
                raise ValueError('OCALD training produced no updates')
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            row = dict(stage=stage, epoch=epoch, **{k: v / divisor for k, v in totals.items()},
                **{f'weight_{k}': v for k, v in weights.items()},
                optimizer_steps=epoch_steps, cumulative_steps=steps,
                samples_drawn=count * 2 if stage == 'hi' else count, pairs_drawn=pairs_drawn,
                seconds=time.perf_counter() - started)
            append_history(output, row, first=first)
            first = False
        if stage == 'representation':
            save_experiment_checkpoint(output / 'representation.pt', model, optimizer, config, stage, epoch)
        elif stage == 'hi':
            for key, value in frozen.items():
                if not torch.equal(value, model.state_dict()[key]):
                    raise RuntimeError('frozen OCALD representation changed')
    save_experiment_checkpoint(output / 'best.pt', model, optimizer, config, stage, epoch)
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
