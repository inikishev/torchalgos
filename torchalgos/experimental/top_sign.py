from collections.abc import Iterable

import torch


def vec_to_tensors(vec: torch.Tensor, reference: Iterable[torch.Tensor]) -> list[torch.Tensor]:
    tensors = []
    cur = 0
    for r in reference:
        numel = r.numel()
        tensors.append(vec[cur:cur+numel].reshape_as(r))
        cur += numel
    return tensors


class TopSign(torch.optim.Optimizer):
    """Quantile 0.75 means update magnitudes under 0.75th quantile become zero"""
    def __init__(self, params, lr: float = 1e-3, beta:float=0.9, quantile: float = 0.75, use_ema_quantile: bool = True):
        defaults = dict(lr=lr, beta=beta, quantile=quantile, use_ema_quantile=use_ema_quantile)
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None): # pyright:ignore[reportIncompatibleMethodOverride]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = [p for p in group['params'] if p.grad is not None]
            grads = [p.grad for p in params]

            emas = []
            for p in params:
                state = self.state[p]
                if "exp_avg" not in state: state["exp_avg"] = torch.zeros_like(p)
                emas.append(state['exp_avg'])

            torch._foreach_lerp_(emas, grads, 1-group['beta'])

            emas_flat = torch.cat([t.ravel() for t in emas])
            update_flat = emas_flat.sign()
            if group['use_ema_quantile']:
                ema_abs = emas_flat.abs()
                mask = ema_abs < torch.quantile(ema_abs, group['quantile'])

            else:
                g_flat = torch.cat([t.ravel() for t in grads])
                g_abs = g_flat.abs()
                mask = g_abs < torch.quantile(g_abs, group['quantile'])

            update_flat[mask] = 0
            update = vec_to_tensors(update_flat, params)
            torch._foreach_sub_(params, update, alpha=group['lr'])

        return loss

