from typing import Literal
import torch
from torchalgos import opt_utils


def make_muon_param_groups(params):
    """makes param groups for `pytorch_optimizer.Muon`"""

    if isinstance(params, torch.nn.Module):
        adam_params = []
        muon_params = []
        seen = set()

        for module in params.modules():
            for p in module.parameters(recurse=False):
                if id(p) not in seen:
                    seen.add(id(p))
                    if isinstance(module, (torch.nn.Embedding, torch.nn.EmbeddingBag)):
                        adam_params.append(p)
                    else:
                        if p.ndim <= 1:
                            adam_params.append(p)
                        else:
                            muon_params.append(p)

        params_groups = [
            {"params": muon_params, "use_muon": True},
            {"params": adam_params, "use_muon": False},
        ]
        return [g for g in params_groups if len(g["params"]) > 0]


    params = list(params)
    if isinstance(params[0], dict):
        return params

    params_groups = [
        {"params": [p for p in params if p.ndim >= 2], "use_muon": True},
        {"params": [p for p in params if p.ndim < 2], "use_muon": False},
    ]
    return [g for g in params_groups if len(g["params"]) > 0]



# zeropower_via_newtonschulz5 from:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
# and
# https://github.com/HomebrewML/HeavyBall/blob/main/heavyball/utils.py#L452
_NS_COEFFS = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012)
)

@torch.compile
def zeropower_muon(G: torch.Tensor, coeffs=_NS_COEFFS) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng

    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True).clip(min=torch.finfo(X.dtype).tiny * 2))

    # Perform the NS iterations
    for a,b,c in coeffs:
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=3e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        adam_lr_mul=1 / 66,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        adam_eps: float = 1e-7,
        use_muon: bool | Literal["auto"] = "auto",
        matrix_func=zeropower_muon,
        ema_rate: float = 0,
    ):

        if isinstance(params, torch.nn.Module): params = make_muon_param_groups(params)

        defaults = dict(
            lr=lr,
            adam_lr_mul=adam_lr_mul,
            adam_betas=adam_betas,
            adam_eps=adam_eps,
            beta=momentum,
            nesterov=nesterov,
            use_muon=use_muon,
            matrix_func=matrix_func,
            ema_rate=ema_rate,
        )
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None): # pyright:ignore[reportIncompatibleMethodOverride]

        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:

            beta1, beta2 = group['adam_betas']

            muon_params = []
            muon_grads = []
            muon_exp_avgs = []

            adam_params = []
            adam_grads = []
            adam_exp_avgs = []
            adam_exp_avg_sqs = []
            adam_lrs = []

            for param in group['params']:
                if param.grad is None: continue
                state = self.state[param]

                use_muon = group["use_muon"]
                if use_muon == 'auto':
                    use_muon = sum(1 for d in param.shape if d > 1) >= 2

                if use_muon:
                    if param.ndim != 2: raise RuntimeError(f"Can't use muon on param of shape {param.shape}")
                    muon_params.append(param)
                    muon_grads.append(param.grad)

                    if "muon_exp_avg" not in state:
                        state["muon_exp_avg"] = torch.zeros_like(param)

                    muon_exp_avgs.append(state['muon_exp_avg'])

                else:
                    adam_params.append(param)
                    adam_grads.append(param.grad)
                    if "adam_exp_avg" not in state:
                        state["adam_exp_avg"] = torch.zeros_like(param)
                        state["adam_exp_avg_sq"] = torch.zeros_like(param)
                        state["step"] = 0

                    adam_exp_avgs.append(state['adam_exp_avg'])
                    adam_exp_avg_sqs.append(state['adam_exp_avg_sq'])

                    state["step"] += 1
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    adam_lr = group["lr"] * group['adam_lr_mul'] * (bias_correction2 ** 0.5) / bias_correction1
                    adam_lrs.append(adam_lr)

            # muon
            if len(muon_params) > 0:
                torch._foreach_lerp_(muon_exp_avgs, muon_grads, weight=1-group['beta'])

                if group['nesterov']:
                    muon_update = torch._foreach_lerp(muon_grads, muon_exp_avgs, weight=group['beta'])
                else:
                    muon_update = muon_exp_avgs

                muon_update = [group['matrix_func'](t) for t in muon_update]
                torch._foreach_sub_(muon_params, muon_update, alpha=group['lr'])

            # adam
            if len(adam_params) > 0:
                # v1 = v1 * beta + g * (1-beta)
                torch._foreach_lerp_(adam_exp_avgs, adam_grads, weight=(1 - beta1))
                # v2 = v2 * beta + g² * (1-beta)
                torch._foreach_mul_(adam_exp_avg_sqs, beta2)
                torch._foreach_addcmul_(adam_exp_avg_sqs, adam_grads, adam_grads, value=(1 - beta2))
                # u = v1 / (sqrt(v2) + eps)
                denom = torch._foreach_sqrt(adam_exp_avg_sqs)
                torch._foreach_clamp_min_(denom, group["adam_eps"])
                adam_update = torch._foreach_div(adam_exp_avgs, denom)
                torch._foreach_mul_(adam_update, adam_lrs)
                torch._foreach_sub_(adam_params, adam_update)

                if group["ema_rate"] != 0:
                    opt_utils.update_parameter_ema(self, group=group)

        return loss

    @torch.no_grad
    def train(self):
        opt_utils.optimizer_train(self)

    @torch.no_grad
    def eval(self):
        opt_utils.optimizer_eval(self)