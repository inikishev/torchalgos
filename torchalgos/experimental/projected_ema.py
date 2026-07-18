from collections.abc import Sequence
import torch
from torch.optim import Optimizer
from torchalgos.experimental import orthopulse
from typing import Literal

class ProjectedEMA(Optimizer):
    """
    Super simple idea. Project fast EMA onto a slow EMA.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.99),
        weight_decay=0.0,
        cautious: bool = False,
        preprocess: Literal["none", "normalize", "sign", "rmsprop"] = "normalize",
        rmsprop_beta: float = 0.99,

    ):
        defaults = dict(lr=lr, betas=betas, cautious=cautious, weight_decay=weight_decay, preprocess=preprocess, rmsprop_beta=rmsprop_beta)
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None): # pyright:ignore[reportIncompatibleMethodOverride]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                g = p.grad
                eps = torch.finfo(p.dtype).eps

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg_fast'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_slow'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                g = orthopulse.preprocess_grad(g, group, state, eps)

                step = state['step'] + 1
                state['step'] = step

                exp_avg_fast = state['exp_avg_fast']
                exp_avg_slow = state['exp_avg_slow']

                exp_avg_fast.lerp_(g, 1-beta1)
                exp_avg_slow.lerp_(g, 1-beta2)

                denom = torch.linalg.vector_norm(exp_avg_slow) # pylint:disable=not-callable
                scalar_proj = (exp_avg_fast * exp_avg_slow).sum() / denom.clip(min=eps)
                lr = group["lr"] * scalar_proj

                if group["cautious"]:
                    exp_avg_slow *= (exp_avg_slow * p.grad).gt_(0)

                if group['weight_decay'] != 0:
                    exp_avg_slow = exp_avg_slow.add(p, alpha=group['weight_decay'])

                p.sub_(exp_avg_slow, alpha=lr)

        return loss


class AlternatingEMAs(Optimizer):
    """
    Alternate EMAs.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas: Sequence[float] = (0.9, 0.95, 0.99, 0.999),
        weight_decay=0.0,
        cautious: bool = False,
        preprocess: Literal["none", "normalize", "sign", "rmsprop"] = "normalize",
        rmsprop_beta: float = 0.99,

    ):
        defaults = dict(lr=lr, betas=betas, cautious=cautious, weight_decay=weight_decay, preprocess=preprocess, rmsprop_beta=rmsprop_beta)
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None): # pyright:ignore[reportIncompatibleMethodOverride]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                g = p.grad
                eps = torch.finfo(p.dtype).eps

                if len(state) == 0:
                    state['step'] = 0
                    for i, beta in enumerate(group["betas"]):
                        state[f"exp_avg_{i}"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                g = orthopulse.preprocess_grad(g, group, state, eps)

                step = state['step'] + 1
                state['step'] = step

                for i, beta in enumerate(group["betas"]):
                    state[f"exp_avg_{i}"].lerp_(g, 1-beta)

                index = step % len(group["betas"])
                update = state[f"exp_avg_{index}"]

                if group["cautious"]:
                    update *= (update * p.grad).gt_(0)

                if group['weight_decay'] != 0:
                    update = update.add(p, alpha=group['weight_decay'])

                p.sub_(update, alpha=group["lr"])

        return loss

class OrthoProjectedEMA(Optimizer):

    def __init__(
        self,
        params,
        lr=1e-3,
        betas: tuple[float,float,float] = (0.9, 0.9, 0.99),
        weight_decay=0.0,
        cautious: bool = False,
        preprocess: Literal["none", "normalize", "sign", "rmsprop"] = "normalize",
        rmsprop_beta: float = 0.99,
        cossim_threshold: float = 1e-8,
        weights = (0.5, 0.1, 0.01),

    ):
        defaults = dict(lr=lr, betas=betas, cautious=cautious, weight_decay=weight_decay, preprocess=preprocess, rmsprop_beta=rmsprop_beta, weights=weights, cossim_threshold=cossim_threshold)
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None): # pyright:ignore[reportIncompatibleMethodOverride]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                g = p.grad
                eps = torch.finfo(p.dtype).eps

                if len(state) == 0:
                    state['step'] = 0
                    for i, beta in enumerate(group["betas"]):
                        state[f"exp_avg_{i}"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                g = orthopulse.preprocess_grad(g, group, state, eps)

                step = state['step'] + 1
                state['step'] = step

                for i, beta in enumerate(group["betas"]):
                    state[f"exp_avg_{i}"].lerp_(g, 1-beta)

                exp_avg_fast = state["exp_avg_0"]
                exp_avg_medium = state["exp_avg_1"]
                exp_avg_slow = state["exp_avg_2"]

                dot_product = (exp_avg_medium * exp_avg_slow).sum()
                denom = exp_avg_slow.square().sum().clip(min=eps)
                medium_onto_slow = (dot_product / denom) * exp_avg_slow
                medium_ortho = exp_avg_medium - medium_onto_slow

                # graft to fast
                fast_norm = torch.linalg.vector_norm(exp_avg_fast) # pylint:disable=not-callable
                medium_ortho_norm = torch.linalg.vector_norm(medium_ortho) # pylint:disable=not-callable

                denom = fast_norm * medium_ortho_norm
                cossim = (exp_avg_fast * medium_ortho).sum() / denom.clip(min=eps)

                if cossim.abs() < group["cossim_threshold"]:
                    exp_avg_medium.lerp_(g, 1-group["betas"][0])
                    exp_avg_slow.lerp_(g, 1-group["betas"][0])

                scale = fast_norm / medium_ortho_norm.clip(min=eps)
                if cossim < 0: scale = -scale
                medium_ortho *= scale

                # mix in
                w_fast, w_med, w_slow = group['weights']
                update = medium_ortho.add_(exp_avg_fast, alpha=w_fast).add_(exp_avg_medium, alpha=w_med).add_(exp_avg_slow, alpha=w_slow)

                if group["cautious"]:
                    update *= (update * p.grad).gt_(0)

                if group['weight_decay'] != 0:
                    update = update.add(p, alpha=group['weight_decay'])

                p.sub_(update, alpha=group["lr"])

        return loss

