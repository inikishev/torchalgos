"""SOAP"""
import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Literal

import torch
from torch.optim import Optimizer

from torchalgos import kron_utils, opt_utils, pogo

EIGH_FN = torch.linalg.eigh # can swap to something from https://www.gpumode.com/leaderboard/775?tab=rankings
QR_FN = torch.linalg.qr

def update_accumulators_(
    grad: torch.Tensor,
    accumulators_: Sequence[torch.Tensor | None],
    shampoo_beta: float,
):
    for ind, acc in enumerate(accumulators_):
        if acc is None: continue
        g = grad.movedim(ind, 0).reshape(grad.shape[ind], -1) # (shape[i], batch)
        acc.lerp_(g @ g.T, 1-shampoo_beta)


def project(tensors: Sequence[torch.Tensor], Qs: Sequence[torch.Tensor | None]):
    assert not isinstance(tensors, torch.Tensor)
    if len(tensors) == 0: return []

    x = tensors[0]

    for Q in Qs:

        if Q is None:
            permute_order = list(range(1, x.ndim)) + [0]
            tensors = [t.permute(permute_order) for t in tensors]
        else:
            tensors = [torch.tensordot(t, Q, dims=[[0], [0]]) for t in tensors] # pyright:ignore[reportArgumentType]

    return tensors


def project_back(tensors: Sequence[torch.Tensor], Qs: Sequence[torch.Tensor | None]):
    assert not isinstance(tensors, torch.Tensor)
    if len(tensors) == 0: return []

    x = tensors[0]

    for Q in Qs:
        if Q is None:
            permute_order = list(range(1, x.ndim)) + [0]
            tensors = [t.permute(permute_order) for t in tensors]
        else:
            tensors = [torch.tensordot(t, Q, dims=[[0], [1]]) for t in tensors] # pyright:ignore[reportArgumentType]

    return tensors


def initialize_accumulators(tensor: torch.Tensor, precond_dims, precondition_1d:bool, max_dim: int):
    """``tensor`` is only used for shape, device and dtype."""
    if precond_dims is None: precond_dims = []
    elif isinstance(precond_dims, int): precond_dims = [precond_dims]
    elif precond_dims == 'all': precond_dims = list(range(tensor.ndim))
    # always skip 0d parameters, and 1d based on setting
    if (tensor.ndim == 0) or ((precondition_1d is False) and (tensor.ndim == 1)):
        precond_dims = []

    accumulators = []

    for dim, size in enumerate(tensor.shape):
        if (dim not in precond_dims) or (size > max_dim):
            accumulators.append(None)

        else:
            accumulators.append(torch.zeros((size, size), device=tensor.device, dtype=tensor.dtype))

    return accumulators

def initialize_eigenbasis(accumulators: Sequence[torch.Tensor | None]):
    Qs = []
    for acc in accumulators:

        if acc is None:
            Qs.append(None)

        else:
            A = torch.randn(acc.shape, device=acc.device, dtype=torch.float64)
            I = torch.eye(acc.shape[0], device=acc.device, dtype=torch.float64)
            _, Q = EIGH_FN(A @ A.T + I * torch.finfo(acc.dtype).eps) # pylint:disable=not-callable
            Qs.append(Q.flip(1).to(acc.dtype))

    return Qs

# pylint:disable=not-callable

def update_eigenbasis(
    power_iters: int,
    accumulators: Sequence[torch.Tensor | None],
    Qs: Sequence[torch.Tensor | None],
    grads: Sequence[torch.Tensor],
    diags: Sequence[torch.Tensor],
    solver: Literal["subspace", "eigh", "pogo"],
    pogo_lr: float = 0.1,
    pogo_steps: int = 1,
):
    assert not isinstance(grads, torch.Tensor)
    assert not isinstance(diags, torch.Tensor)

    # Unproject buffers with current Q to reproject to new Q
    grads = project_back(grads, Qs)

    for i, (acc, Q) in enumerate(zip(accumulators, Qs, strict=True)):
        if acc is None:
            if len(grads) > 0 and len(diags) > 0:
                ndim = grads[0].ndim if len(grads) > 0 else diags[0].ndim
                permute_order = list(range(1, ndim)) + [0]
                grads = [g.permute(permute_order) for g in grads]
                diags = [d.permute(permute_order) for d in diags]
            continue

        assert Q is not None
        dtype = Q.dtype

        if solver == "subspace":
            power_iter = acc
            for _ in range(power_iters):
                power_iter = power_iter @ Q
            if torch.finfo(dtype).eps > 1e-7:
                power_iter = power_iter.to(torch.float32)
            try:
                Q_new, _ = QR_FN(power_iter)
            except torch.linalg.LinAlgError:
                Q_new, _ = QR_FN(torch.randn_like(Q))
            Q_new = Q_new.to(dtype=dtype)

        elif solver == "eigh":
            try:
                _, Q_new = EIGH_FN(acc)
                Q_new = Q_new.flip(1)
            except torch.linalg.LinAlgError:
                Q_new, _ = QR_FN(torch.randn_like(Q))
            Q_new = Q_new.to(dtype=dtype)

        elif solver == "pogo":
            n = acc.shape[0]
            D = torch.linspace(1.0, 0.1, n, device=acc.device, dtype=acc.dtype).unsqueeze(0)

            Q_curr = Q.clone()
            norm_factor = acc.norm().clamp_min(1e-8)
            acc_normalized = acc / norm_factor

            for _ in range(pogo_steps):
                G = -2.0 * (acc_normalized @ (Q_curr * D))
                Q_curr = pogo.pogo_step(Q_curr, G, lr=pogo_lr)

            Q_new = Q_curr.to(dtype=dtype)

        else:
            raise ValueError(f"Unknown solver: {solver}")

        # Reproject buffers
        grads = [torch.tensordot(g, Q_new, dims=[[0], [0]]) for g in grads] # pyright:ignore[reportArgumentType]

        if len(diags) > 0:
            C_sq = (Q_new.T @ Q) ** 2
            diags = [torch.tensordot(d, C_sq, dims=[[0], [0]]) for d in diags] # pyright:ignore[reportArgumentType]

        Qs[i].set_(Q_new) # pyright:ignore[reportOptionalMemberAccess,reportArgumentType]

    return Qs, grads, diags

class SOAP(Optimizer):
    """
    [SOAP: Improving and Stabilizing Shampoo using Adam](https://arxiv.org/abs/2409.11321).

    Changes compared to the paper:
    - basis is initialized to random orthogonal matrix (I found that it works much better and way more stable)
    - 2 power iters by default before QR (thats more efficient in my benchmarks)
    - has graft to update EMA

    Args:
        params: Iterable of tensors to optimize, or pass the model itself
            which automatically uses diagonal preconditioner for embedding parameters.
        lr: Learning rate. Defaults to 3e-3.
        betas: Betas for Adam in the eigenbasis and for parameter where kronecker preconditioner isn't used,
            first is for momentum and second is for squared gradients. Defaults to (0.95, 0.95).
        shampoo_beta: beta for kronecker-factored accumulators, -1 means `betas[1]`. Defaults to -1.
        eps: clips Adam's denominator below this value. Defaults to 1e-8.
        weight_decay: decoupled weight decay, NOT decoupled from learning rate. Defaults to 0.01.
        precond_freq: frequency of updating eigenbasis, default 10.
        solver: how to update eigenbasis, eigendecomposition, subspace iteration, or POGO.
        power_iters: performs this many power iterations before reorthogonalizing via QR per eigenbasis update. default 2.
        pogo_lr: learning rate for POGO for basis updates. defaut 1.0.
        pogo_steps: number of POGO steps per basis updates. default 1.
        max_dim: won't precondition dims larger than this. Defaults to 4096.
        merge_dims: whether to merge small dimensions, default False.
        merge_whitelist: if specified, only specified dimensions can be merged. Defaults to None.
        merge_blacklist: prevents specified dimensions from being merged,
            you should set this to 0 or 1 for channel-first weights like Linear and Conv,
            and -1 or -2 for channel-last weights. Applies after whitelist. Defaults to 0.
        precond_dims: dimensions to use kronecker preconditioner for,
            note that if `merge_dims=True`, dimensions are specified after merging.
            Set to None to use diagonal preconditioner, "all" to precondition all dimensions smaller than `max_dim`. Setting this to 1 for channel-first weights or -2 for channel-last weights
            may be much faster than "all" but still have good performance.
            You should set this to None on embeddings, or pass model as `params` to do that automatically.
            Defaults to 'all'.
        precondition_1d: whether to precondition 1d parameters, unlike the official implementation the default is True.
        normalize: whether to normalize the updates. Defaults to False.
    """
    def __init__(
        self,
        params,
        lr: float = 3e-3,
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
        cautious: bool = False,
        mars_scale: float = 0,
        max_norm: float | None = None,
        max_norm_type: float | Literal["mad"] = 2,
    ):
        defaults = dict(
            lr = lr,
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
            solver = solver,
            normalize = normalize,
            gtue_mode = gtue_mode,
            gtue_metric = gtue_metric,
            gtue_beta = gtue_beta,
            gtue_max_metric_growth = gtue_max_metric_growth,
            gtue_min_metric = gtue_min_metric,
            cautious = cautious,
            mars_scale = mars_scale,
            max_norm = max_norm,
            max_norm_type = max_norm_type,

        )

        if isinstance(params, torch.nn.Module):
            params = kron_utils.make_kron_param_groups_for_emb(params)

        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None): # pyright:ignore[reportIncompatibleMethodOverride]
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

            beta1, beta2 = group["betas"]
            shampoo_beta = group["shampoo_beta"]
            if shampoo_beta < 0: shampoo_beta = beta2

            for param in group["params"]:
                if param.grad is None: continue
                params_with_grad.append(param)

                grad = param.grad
                state = self.state[param]

                if group["merge_dims"]: # merge small dims to get correct shapes
                    (grad, ), *state["merge_state"] = kron_utils.merge_small_dims(
                        (grad, ),
                        max_dim=group["max_dim"],
                        whitelist=group["merge_whitelist"],
                        blacklist=group["merge_blacklist"],
                    )

                grads_merged.append(grad)

                if "accumulators" not in state:

                    # ----------------------- Initialize state on 1st step ----------------------- #
                    state["step"] = 0

                    state["accumulators"] = initialize_accumulators(
                        grad, precond_dims=group["precond_dims"], precondition_1d=group["precondition_1d"], max_dim=group["max_dim"])

                    update_accumulators_(grad, state["accumulators"], shampoo_beta)

                    state["Qs"] = initialize_eigenbasis(state["accumulators"])

                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                    if group["mars_scale"] != 0: state["g_prev"] = torch.zeros_like(grad)

                if state["step"] == 0:
                    # first step is skipped so that we never use the current gradients in the projection.
                    state["step"] += 1
                    continue

                # ---------------------------------- Project --------------------------------- #
                (grad_proj, ) = project((grad, ), state["Qs"])

                grads_proj.append(grad_proj)
                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                if group["mars_scale"] != 0:
                    if state["step"] == 1: state["g_prev"] = grad_proj.clone()
                    grads_prev.append(state["g_prev"])


            if len(exp_avgs) == 0: # skip 1st step
                continue

            # --------------------------------- run adam --------------------------------- #
            if group["mars_scale"]  != 0:
                grads_proj = opt_utils.mars_correction_(grads_proj, grads_prev, group["betas"][0], group["mars_scale"])

            if group["max_norm"] is not None:
                opt_utils.clip_norm_(grads_proj, max_norm=group["max_norm"], metric=group["max_norm_type"])

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

            for param, grad, dir_proj in zip(params_with_grad, grads_merged, dirs_proj, strict=True):

                state = self.state[param]

                # ------------------------------- project back ------------------------------- #
                (dir, ) = project_back((dir_proj, ), state["Qs"])
                if group["merge_dims"]:
                    dir = kron_utils.unmerge_small_dims(dir, *state["merge_state"])

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
                # Update is done after the gradient step to avoid using current gradients in the projection.
                update_accumulators_(grad, state["accumulators"], shampoo_beta=shampoo_beta)


                # ------------------------------- update basis ------------------------------- #
                if state["step"] % group["precond_freq"] == 0:
                    state["Qs"], (state["exp_avg"], ), (state["exp_avg_sq"], ) = update_eigenbasis(
                        power_iters = group["power_iters"],
                        accumulators = state["accumulators"],
                        Qs = state["Qs"],
                        grads = (state["exp_avg"], ),
                        diags = (state["exp_avg_sq"], ),
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
                    max_metric_growth = group["gtue_max_metric_growth"],
                    min_metric = group["gtue_min_metric"],
                    eps = group["eps"],
                    mode = group["gtue_mode"],
                )

            # ----------------------------- update parameters ---------------------------- #
            if group["weight_decay"] > 0.0:
                torch._foreach_add_(
                    updates,
                    torch._foreach_mul(params_with_grad, group["weight_decay"])
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