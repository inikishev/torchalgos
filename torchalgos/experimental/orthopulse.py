"""The idea is that when fast EMA starts pointing in the opposite direction to slow EMA, we want to go orthogonally to slow EMA."""
import torch
from typing import Literal

def preprocess_grad(g: torch.Tensor, group: dict, state: dict, eps: float):
    if group["preprocess"] == "normalize":
        g = g / torch.linalg.vector_norm(g).clip(min=eps) # pylint:disable=not-callable
    elif group["preprocess"] == "sign":
        g = g.sign()
    elif group["preprocess"] == "rmsprop":
        if "exp_avg_sq" not in state:
            state["exp_avg_sq"] = g.square()
        exp_avg_sq = state["exp_avg_sq"]
        exp_avg_sq.mul_(group["rmsprop_beta"]).addcmul_(g, g, value=1-group["rmsprop_beta"])
        g = g / exp_avg_sq.sqrt().clip(min=eps)

    return g

class OrthoPulseG(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas: tuple[float,float] = (0.9, 0.99),
        cossim_threshold: float = 0,
        eps: float | None = None,
        preprocess: Literal["none", "normalize", "sign", "rmsprop"] = "normalize",
        rmsprop_beta=0.99,
        slow_reset_mode: Literal["ortho", "zeros"] = "zeros",
        cautious: bool = False,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            cossim_threshold=cossim_threshold,
            eps = eps,
            preprocess=preprocess,
            rmsprop_beta=rmsprop_beta,
            slow_reset_mode=slow_reset_mode,
            cautious=cautious,
        )
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None): # pyright:ignore[reportIncompatibleMethodOverride]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None: continue
                g = p.grad
                state = self.state[p]

                eps = group["eps"]
                if eps is None: eps = torch.finfo(p.dtype).eps
                g = preprocess_grad(g, group, state, eps)

                # init state
                if "exp_avg_fast" not in state:
                    state["exp_avg_fast"] = g.clone()
                    state["exp_avg_slow"] = g.clone()

                exp_avg_fast: torch.Tensor = state["exp_avg_fast"]
                exp_avg_slow: torch.Tensor = state["exp_avg_slow"]
                beta1, beta2 = group["betas"]
                exp_avg_fast.lerp_(g, 1-beta1)
                exp_avg_slow.lerp_(g, 1-beta2)

                fast_norm = torch.linalg.vector_norm(exp_avg_fast) # pylint:disable=not-callable
                slow_norm = torch.linalg.vector_norm(exp_avg_slow) # pylint:disable=not-callable
                denom = fast_norm * slow_norm
                cossim = (exp_avg_fast * exp_avg_slow).sum() / denom.clip(min=eps)

                if cossim < group["cossim_threshold"]:
                    # remove slow EMA component from fast EMA
                    dot_product = (exp_avg_fast * exp_avg_slow).sum()
                    denom = exp_avg_slow.square().sum().clip(min=eps)
                    fast_onto_slow = (dot_product / denom) * exp_avg_slow
                    fast_ortho = exp_avg_fast - fast_onto_slow

                    # set magnitude to one of slow EMA
                    fast_ortho *= slow_norm / torch.linalg.vector_norm(fast_ortho).clip(min=eps) # pylint:disable=not-callable

                    # set both fast and slow EMAs to new vec
                    exp_avg_fast.copy_(fast_ortho)

                    if group["slow_reset_mode"] == "ortho": exp_avg_slow.copy_(fast_ortho)
                    else: exp_avg_slow.zero_()

                if group["cautious"]:
                    exp_avg_fast *= (exp_avg_fast * p.grad).gt_(0)

                p.sub_(exp_avg_fast, alpha=group["lr"])

        return loss
