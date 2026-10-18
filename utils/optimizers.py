"""Keras numerical optimizer semantics shared by migrated CNN experiments."""
import math

import torch


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
