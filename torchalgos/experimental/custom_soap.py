from collections import deque
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Literal

import torch
from torch.optim import Optimizer

from torchalgos import kron_utils, soap, opt_utils
from typing import Protocol

def update_accumulators_op_(
    vec1: torch.Tensor,
    vec2: torch.Tensor | None,
    accumulators_: list[torch.Tensor | None],
    shampoo_beta: float,
    operation: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    reduce: Callable[[torch.Tensor, int], torch.Tensor],
    update_fn: Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor],
    copysign: bool,
):
    for i, acc in enumerate(accumulators_):
        if acc is None:
            continue

        c1 = vec1.movedim(i, 0).reshape(vec1.shape[i], -1) # (shape[i], batch)
        c2 = c1 if vec2 is None else vec2.movedim(i, 0).reshape(vec2.shape[i], -1)

        if operation is torch.mul and reduce is torch.sum:
            update = c1 @ c2.T
        else:
            update = reduce(operation(c1.unsqueeze(0), c2.unsqueeze(1)), -1)

        if copysign: update.copysign_(c1 @ c2.T)
        acc.copy_(update_fn(acc, update, 1-shampoo_beta))

class CorrectionProtocol(Protocol):
    def __call__(self, grad: torch.Tensor, update: torch.Tensor, param: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        ...

def _call_symmetric(fn: CorrectionProtocol, grad: torch.Tensor, update: torch.Tensor, param: torch.Tensor) -> torch.Tensor | None:
    vec1, vec2 = fn(grad=grad, update=update, param=param)
    assert vec2 is None
    return vec1

def _maybe_call_symmetric(fn: CorrectionProtocol | int | float, grad: torch.Tensor, update: torch.Tensor, param: torch.Tensor) -> float | torch.Tensor | None:
    if isinstance(fn, (int,float)): return fn
    return _call_symmetric(fn, grad=grad, update=update, param=param)

def get_grad(grad: torch.Tensor, update: torch.Tensor, param: torch.Tensor):
    return grad, None

def get_param(grad: torch.Tensor, update: torch.Tensor, param: torch.Tensor):
    return param, None

def get_update(grad: torch.Tensor, update: torch.Tensor, param: torch.Tensor):
    return update, None

def asymmetric_mix(corr1: CorrectionProtocol, corr2: CorrectionProtocol) -> CorrectionProtocol:
    def wrapped(grad: torch.Tensor, update: torch.Tensor, param: torch.Tensor):
        vec1 = _call_symmetric(corr1, grad=grad, update=update, param=param)
        vec2 = _call_symmetric(corr2, grad=grad, update=update, param=param)
        return vec1, vec2
    return wrapped

class CorrectionFunc(ABC):
    @abstractmethod
    def __call__(self, grad: torch.Tensor, update: torch.Tensor, param: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        ...

    def __add__(self, other: CorrectionProtocol | float):
        return Sum(self, other)

    def __sub__(self, other: CorrectionProtocol | float):
        return Sub(self, other)

    def __rsub__(self, other: CorrectionProtocol | float):
        return Sub(other, self)

    def __mul__(self, other: CorrectionProtocol | float):
        return Prod(self, other)

    def __div__(self, other: CorrectionProtocol | float):
        return Div(self, other)

    def __rdiv__(self, other: CorrectionProtocol | float):
        return Div(other, self)

    def asym_mix(self, other: CorrectionProtocol):
        return FromFunction(asymmetric_mix(self, other))

    def rmix(self, other: CorrectionProtocol):
        return FromFunction(asymmetric_mix(other, self))

    def ema(self, beta: float = 0.9, update_fn: Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor] = torch.lerp):
        return EMA(self, beta=beta, update_fn=update_fn)

    def sym_lambda(self, lambda_: Callable[[torch.Tensor], torch.Tensor | None]):
        return SymLambda(self, lambda_)

    def asym_lambda(self, lambda_: Callable[[torch.Tensor | None, torch.Tensor | None], tuple[torch.Tensor | None, torch.Tensor | None]]):
        return AsymLambda(self, lambda_)

    def abs(self):
        return self.sym_lambda(torch.abs)

    def normalize(self, metric: float | Literal["mad","rms"] = 2, eps: float = 1e-8):
        return self.sym_lambda(lambda t: t / max(opt_utils._compute_metric([t], metric)[0], eps))

    def clip(self, min: CorrectionProtocol | float | None = None, max: CorrectionProtocol | float | None = None):
        f = self
        if min is not None: f = Max(f, min)
        if max is not None: f = Min(f, max)
        return f

    def previous(self, n_back: int):
        return Previous(self, n_back=n_back)

class FromFunction(CorrectionFunc):
    def __init__(self, fn: CorrectionProtocol):
        self.fn = fn
    def __call__(self, grad, update, param):
        return self.fn(grad=grad, update=update, param=param)

class SymLambda(CorrectionFunc):
    def __init__(self, fn: CorrectionProtocol, lambda_: Callable[[torch.Tensor], torch.Tensor | None]):
        self.fn = fn
        self.lambda_ = lambda_
    def __call__(self, grad, update, param):
        res = _call_symmetric(self.fn, grad=grad, update=update, param=param)
        if res is None: return None, None
        return self.lambda_(res), None

class AsymLambda(CorrectionFunc):
    def __init__(self, fn: CorrectionProtocol, lambda_: Callable[[torch.Tensor | None, torch.Tensor | None], tuple[torch.Tensor | None, torch.Tensor | None]]):
        self.fn = fn
        self.lambda_ = lambda_
    def __call__(self, grad, update, param):
        vec1, vec2 = self.fn(grad=grad, update=update, param=param)
        return self.lambda_(vec1, vec2)

class Sum(CorrectionFunc):
    def __init__(self, *funcs: CorrectionProtocol | float):
        self.funcs = funcs
    def __call__(self, grad, update, param):
        res = None
        for f in self.funcs:
            f_out = _maybe_call_symmetric(f, grad=grad, update=update, param=param)
            if res is None: res = f_out
            elif f_out is not None: res = res + f_out

        if isinstance(res, (int,float)): res = None
        return res, None

class Prod(CorrectionFunc):
    def __init__(self, *funcs: CorrectionProtocol | float):
        self.funcs = funcs
    def __call__(self, grad, update, param):
        res = None
        for f in self.funcs:
            f_out = _maybe_call_symmetric(f, grad=grad, update=update, param=param)
            if res is None: res = f_out
            elif f_out is not None: res = res * f_out

        if isinstance(res, (int,float)): res = None
        return res, None

class Sub(CorrectionFunc):
    def __init__(self, f1: CorrectionProtocol | float, f2: CorrectionProtocol | float):
        self.f1 = f1
        self.f2 = f2
    def __call__(self, grad, update, param):
        f1 = _maybe_call_symmetric(self.f1, grad=grad, update=update, param=param)
        f2 = _maybe_call_symmetric(self.f2, grad=grad, update=update, param=param)
        if f1 is None or f2 is None: return None, None
        res = f1 - f2
        if isinstance(res, (int,float)): return None, None
        return res, None


class Div(CorrectionFunc):
    def __init__(self, num: CorrectionProtocol | float, denom: CorrectionProtocol | float):
        self.num = num
        self.denom = denom
    def __call__(self, grad, update, param):
        num = _maybe_call_symmetric(self.num, grad=grad, update=update, param=param)
        denom = _maybe_call_symmetric(self.denom, grad=grad, update=update, param=param)
        if num is None or denom is None: return None, None
        res = num / denom
        if isinstance(res, (int,float)): return None, None
        return res, None

class EMA(CorrectionFunc):
    def __init__(self, func: CorrectionProtocol, beta: float = 0.9, update_fn: Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor] = torch.lerp):
        self.beta = beta
        self.func = func
        self.buf = None
        self.update_fn = update_fn

    def __call__(self, grad: torch.Tensor, update: torch.Tensor, param: torch.Tensor):
        f_out = _call_symmetric(self.func, grad=grad, update=update, param=param)
        if self.buf is None: self.buf = f_out
        elif f_out is not None:
            self.buf.copy_(self.update_fn(self.buf, f_out, 1-self.beta))
        return self.buf, None

class Max(CorrectionFunc):
    def __init__(self, *funcs: CorrectionProtocol | float):
        self.funcs = funcs
    def __call__(self, grad, update, param):
        res = None
        for f in self.funcs:
            f_out = _maybe_call_symmetric(f, grad=grad, update=update, param=param)
            if res is None: res = f_out
            elif f_out is not None:
                if isinstance(res, (int,float)):
                    if isinstance(f_out, (int,float)): res = max(res, f_out)
                    else: res = f_out.clip(min=res)
                elif isinstance(f_out, (int,float)): res = res.clip(min=f_out)
                else: res = torch.maximum(res, f_out)

        if isinstance(res, (int,float)): res = None
        return res, None

class Min(CorrectionFunc):
    def __init__(self, *funcs: CorrectionProtocol | float):
        self.funcs = funcs
    def __call__(self, grad, update, param):
        res = None
        for f in self.funcs:
            f_out = _maybe_call_symmetric(f, grad=grad, update=update, param=param)
            if res is None: res = f_out
            elif f_out is not None:
                if isinstance(res, (int,float)):
                    if isinstance(f_out, (int,float)): res = min(res, f_out)
                    else: res = f_out.clip(max=res)
                elif isinstance(f_out, (int,float)): res = res.clip(max=f_out)
                else: res = torch.minimum(res, f_out)

        if isinstance(res, (int,float)): res = None
        return res, None


class Previous(CorrectionFunc):
    def __init__(self, fn: CorrectionProtocol, n_back: int):
        self.fn = fn
        self.history = deque(maxlen=n_back + 1)

    def __call__(self, grad, update, param):
        self.history.append(self.fn(grad=grad, update=update, param=param))
        return self.history[0]

class CustomSOAP(Optimizer):
    """SOAP but with custom accumulator update operations (e.g. you can update with outer maximum of gradients, etc).

    Args:
        operation: outer operation applied to `G.unsqueeze(0), G.unsqueeze(1)`, default is `torch.mul` (outer product).
        reduce: this is reduction operation which comes from kronecker structure and changing this will probably have weird effects.
        copysign: whether to copy sign from G G^T after applying operation to `G.unsqueeze(0), G.unsqueeze(1)`.
        update_fn: update function for the accumulator with signature `fn(accumulator, update, 1-beta)`.
        corrections_fn: function that takes in gradient, update and parameter, and returns the correction as two tensors that are added as $C_1 C_2^T$ (if second tensor is None, $C_1 C_1^T$).
    """

    GRAD = FromFunction(get_grad)
    PARAM = FromFunction(get_param)
    UPDATE = FromFunction(get_update)

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        operation: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = torch.mul,
        reduce: Callable[[torch.Tensor, int], torch.Tensor] = torch.sum,
        copysign: bool = False,
        update_fn: Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor] = torch.lerp,
        corrections_fn: CorrectionProtocol = get_grad,
        symmetrize: bool = False,
        betas = (0.95, 0.95),
        shampoo_beta: float = -1,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        ema_rate: float = 0.0,
        precond_freq: float = 10,
        solver: Literal["subspace", "eigh", 'pogo'] = "pogo",
        power_iters: int = 2,
        pogo_lr: float = 1.0,
        pogo_steps: int = 1,
        max_dim: int = 4096,
        merge_dims: bool = False,
        merge_whitelist: int | list[int] | None = None,
        merge_blacklist: int | list[int] | None = 0,
        precond_dims: int | list[int] | None | Literal['all'] = 'all',
        precondition_1d: bool = True,
        normalize: bool = False,
        gtue_mode: Literal["disabled", "clip", "normalize"] = "normalize",
        gtue_metric: float | Literal["mad"] = 2,
        gtue_beta: float = 0.99,
        gtue_max_metric_growth: float | None = 1.5,
        gtue_min_metric: float = 1e-5,
        gtue_max_metric: float | None = None,
        cautious: bool = False,
        mars_scale: float = 0,
        max_grad_norm: float | None = None,
        max_grad_norm_type: float | Literal["mad"] = 2,
    ):
        defaults = dict(
            lr=lr,
            operation = operation,
            reduce = reduce,
            copysign = copysign,
            update_fn = update_fn,
            corrections_fn = corrections_fn,
            symmetrize = symmetrize,
            betas = betas,
            shampoo_beta = shampoo_beta,
            eps = eps,
            weight_decay = weight_decay,
            ema_rate = ema_rate,
            precond_freq = precond_freq,
            power_iters = power_iters,
            pogo_lr = pogo_lr,
            pogo_steps = pogo_steps,
            max_dim = max_dim,
            merge_dims = merge_dims,
            merge_whitelist = merge_whitelist,
            merge_blacklist = merge_blacklist,
            precond_dims = precond_dims,
            precondition_1d = precondition_1d,
            normalize = normalize,
            solver = solver,
            gtue_mode = gtue_mode,
            gtue_metric = gtue_metric,
            gtue_beta = gtue_beta,
            gtue_max_metric_growth = gtue_max_metric_growth,
            gtue_min_metric = gtue_min_metric,
            gtue_max_metric = gtue_max_metric,
            cautious = cautious,
            mars_scale = mars_scale,
            max_grad_norm = max_grad_norm,
            max_grad_norm_type = max_grad_norm_type,
        )

        if isinstance(params, torch.nn.Module):
            params = kron_utils.make_kron_param_groups_for_emb(params)

        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None):  # pyright:ignore[reportIncompatibleMethodOverride]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            # collect all buffers for foreach operations
            grads_merged = []
            grads_proj = []
            exp_avgs = []
            exp_avg_sqs = []
            params_with_grad = []
            grads_prev = []
            params_merged = []

            beta1, beta2 = group["betas"]
            shampoo_beta = group["shampoo_beta"]
            if shampoo_beta < 0: shampoo_beta = beta2

            for param in group["params"]:
                if param.grad is None: continue
                params_with_grad.append(param)

                grad = param.grad
                state = self.state[param]

                param_merged = param
                if group["merge_dims"]: # merge small dims to get correct shapes
                    (grad, param_merged), *state["merge_state"] = kron_utils.merge_small_dims(
                        (grad, param),
                        max_dim=group["max_dim"],
                        whitelist=group["merge_whitelist"],
                        blacklist=group["merge_blacklist"],
                    )

                grads_merged.append(grad)
                params_merged.append(param_merged)

                if "accumulators" not in state:

                    # ----------------------- Initialize state on 1st step ----------------------- #
                    state["step"] = 0

                    state["accumulators"] = soap.initialize_accumulators(
                        grad, precond_dims=group["precond_dims"], precondition_1d=group["precondition_1d"], max_dim=group["max_dim"]
                    )

                    vec1, vec2 = group["corrections_fn"](grad=grad, update=grad.sign(), param=param_merged)
                    if vec1 is not None:
                        update_accumulators_op_(vec1, vec2, state["accumulators"], shampoo_beta, operation=group["operation"],
                                                reduce=group["reduce"], copysign=group["copysign"], update_fn=group["update_fn"])

                    state["Qs"] = soap.initialize_eigenbasis(state["accumulators"])

                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                    if group["mars_scale"] != 0: state["g_prev"] = torch.zeros_like(grad)

                if state["step"] == 0:
                    # first step is skipped so that we never use the current gradients in the projection.
                    state["step"] += 1
                    continue

                # ---------------------------------- Project --------------------------------- #
                (grad_proj,) = soap.project((grad,), state["Qs"])

                grads_proj.append(grad_proj)
                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                if group["mars_scale"] != 0:
                    if state["step"] == 1: state["g_prev"] = grad_proj.clone()
                    grads_prev.append(state["g_prev"])

            if len(exp_avgs) == 0:  # skip 1st step
                continue

            # --------------------------------- run adam --------------------------------- #
            if group["mars_scale"]  != 0:
                grads_proj = opt_utils.mars_correction_(grads_proj, grads_prev, group["betas"][0], group["mars_scale"])

            if group["max_grad_norm"] is not None:
                opt_utils.clip_norm_(grads_proj, max_norm=group["max_grad_norm"], metric=group["max_grad_norm_type"])

            # v1 = v1 * beta + g * (1-beta)
            torch._foreach_lerp_(exp_avgs, grads_proj, weight=(1 - beta1))
            # v2 = v2 * beta + g² * (1-beta)
            torch._foreach_mul_(exp_avg_sqs, beta2)
            torch._foreach_addcmul_(exp_avg_sqs, grads_proj, grads_proj, value=(1 - beta2))
            # u = v1 / (sqrt(v2) + eps)
            denom = torch._foreach_sqrt(exp_avg_sqs)
            torch._foreach_clamp_min_(denom, group["eps"])
            dirs_proj = torch._foreach_div(exp_avgs, denom)

            if group["cautious"]:
                torch._foreach_mul_(dirs_proj, [t.gt_(0) for t in torch._foreach_mul(dirs_proj, grads_proj)])

            updates = []
            lrs = []

            for param, grad, dir_proj, param_merged in zip(params_with_grad, grads_merged, dirs_proj, params_merged, strict=True):

                state = self.state[param]

                # ------------------------------- project back ------------------------------- #
                (dir_merged,) = soap.project_back((dir_proj,), state["Qs"])
                dir = dir_merged
                if group["merge_dims"]:
                    dir = kron_utils.unmerge_small_dims(dir_merged, *state["merge_state"])

                if group["normalize"]:
                    # no debiasing because update is normalized
                    lr = group["lr"] / dir.square().mean().sqrt().clip(min=group["eps"])

                else:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    lr = group["lr"] * (bias_correction2 ** 0.5) / bias_correction1

                updates.append(dir)
                lrs.append(lr)


                # ---------------------------- update accumulators --------------------------- #
                vec1, vec2 = group["corrections_fn"](grad=grad, update=dir_merged, param=param_merged)
                if vec1 is not None:
                    update_accumulators_op_(vec1, vec2, state["accumulators"], shampoo_beta, operation=group["operation"],
                                            reduce=group["reduce"], copysign=group["copysign"], update_fn=group["update_fn"])

                # ------------------------------- update basis ------------------------------- #
                if state["step"] % group["precond_freq"] == 0:

                    accumulators = state["accumulators"]
                    if group["symmetrize"]:
                        accumulators = [(acc + acc.T) / 2 for acc in accumulators]

                    state["Qs"], (state["exp_avg"],), (state["exp_avg_sq"],) = soap.update_eigenbasis(
                        power_iters = group["power_iters"],
                        accumulators = accumulators,
                        Qs = state["Qs"],
                        grads = (state["exp_avg"],),
                        diags = (state["exp_avg_sq"],),
                        solver = group["solver"],
                        pogo_lr = group["pogo_lr"],
                        pogo_steps = group["pogo_steps"],
                    )

                state["step"] += 1


            # graft to update EMA
            if group["gtue_mode"] != "disabled":
                opt_utils.graft_to_update_ema_(
                    self = self,
                    params_with_grad = params_with_grad,
                    updates_ = updates,
                    metric = group["gtue_metric"],
                    beta = group["gtue_beta"],
                    max_metric_growth=group["gtue_max_metric_growth"],
                    min_metric=group["gtue_min_metric"],
                    max_metric=group['gtue_max_metric'],
                    eps=group["eps"],
                    mode=group["gtue_mode"],
                )

            # ----------------------------- update parameters ---------------------------- #
            if group["weight_decay"] > 0.0:
                torch._foreach_add_(
                    updates, torch._foreach_mul(params_with_grad, group["weight_decay"])
                )

            torch._foreach_mul_(updates, lrs)
            torch._foreach_sub_(params_with_grad, updates)

            if group["ema_rate"] != 0:
                opt_utils.update_parameter_ema(self, group=group)

        return loss

    @torch.no_grad
    def train(self):
        opt_utils.optimizer_train(self)

    @torch.no_grad
    def eval(self):
        opt_utils.optimizer_eval(self)