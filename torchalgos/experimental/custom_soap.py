from collections.abc import Callable
from typing import Literal

import torch
from torch.optim import Optimizer

from torchalgos import kron_utils, soap, opt_utils


def update_accumulators_op_(
    grad: torch.Tensor,
    accumulators_: list[torch.Tensor | None],
    shampoo_beta: float,
    operation: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    reduce: Callable[[torch.Tensor, int], torch.Tensor],
    update_fn: Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor],
    copysign: bool
):
    for i, acc in enumerate(accumulators_):
        if acc is None:
            continue

        g = grad.movedim(i, 0).reshape(grad.shape[i], -1) # (shape[i], batch)
        update = reduce(operation(g.unsqueeze(0), g.unsqueeze(1)), -1)
        if copysign: update.copysign_(g @ g.T)
        acc.copy_(update_fn(acc, update, 1-shampoo_beta))


class CustomSOAP(Optimizer):
    """SOAP but with custom accumulator update operations (e.g. you can update with outer maximum of gradients, etc).

    Args:
        operation: outer operation applied to `G.unsqueeze(0), G.unsqueeze(1)`, default is `torch.mul` (outer product).
        reduce: this is reduction operation which comes from kronecker structure and changing this will probably have weird effects.
        copysign: whether to copy sign from G G^T after applying operation to `G.unsqueeze(0), G.unsqueeze(1)`.
        update_fn: update function for the accumulator with signature `fn(accumulator, update, 1-beta)`.
    """
    def __init__(
        self,
        params,
        lr: float = 3e-3,
        operation: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = torch.mul,
        reduce: Callable[[torch.Tensor, int], torch.Tensor] = torch.sum,
        copysign: bool = False,
        update_fn: Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor] = torch.lerp,
        symmetrize: bool = False,
        betas = (0.95, 0.95),
        shampoo_beta: float = -1,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        ema_rate: float = 0.0,
        precond_freq: float = 10,
        power_iters: int = 1,
        max_dim: int = 4096,
        merge_dims: bool = False,
        merge_whitelist: int | list[int] | None = None,
        merge_blacklist: int | list[int] | None = 0,
        precond_dims: int | list[int] | None | Literal['all'] = 'all',
        precondition_1d: bool = True,
        solver: Literal["subspace", "eigh"] = "subspace",
        normalize: bool = False,
        gtue_mode: Literal["disabled", "clip", "normalize"] = "disabled",
        gtue_metric: float | Literal["mad"] = 2,
        gtue_beta: float = 0.99,
        gtue_max_metric_growth: float | None = 1.5,
        gtue_min_metric: float = 1e-5,
    ):
        defaults = dict(
            lr=lr,
            operation = operation,
            reduce = reduce,
            copysign = copysign,
            update_fn = update_fn,
            symmetrize = symmetrize,
            betas = betas,
            shampoo_beta = shampoo_beta,
            eps = eps,
            weight_decay = weight_decay,
            ema_rate = ema_rate,
            precond_freq = precond_freq,
            power_iters = power_iters,
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

            beta1, beta2 = group["betas"]
            shampoo_beta = group["shampoo_beta"]
            if shampoo_beta < 0:
                shampoo_beta = beta2

            for param in group["params"]:
                if param.grad is None:
                    continue
                params_with_grad.append(param)

                grad = param.grad
                state = self.state[param]

                if group["merge_dims"]:  # merge small dims to get correct shapes
                    (grad,), *state["merge_state"] = kron_utils.merge_small_dims(
                        (grad,),
                        max_dim=group["max_dim"],
                        whitelist=group["merge_whitelist"],
                        blacklist=group["merge_blacklist"],
                    )

                grads_merged.append(grad)

                if "accumulators" not in state:

                    # ----------------------- Initialize state on 1st step ----------------------- #
                    state["step"] = 0

                    state["accumulators"] = soap.initialize_accumulators(
                        grad, precond_dims=group["precond_dims"], precondition_1d=group["precondition_1d"], max_dim=group["max_dim"]
                    )

                    update_accumulators_op_(grad, state["accumulators"], shampoo_beta, operation=group["operation"],
                                            reduce=group["reduce"], copysign=group["copysign"], update_fn=group["update_fn"])

                    state["Qs"] = soap.initialize_eigenbasis(state["accumulators"])

                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                if state["step"] == 0:
                    # first step is skipped so that we never use the current gradients in the projection.
                    state["step"] += 1
                    continue

                # ---------------------------------- Project --------------------------------- #
                (grad_proj,) = soap.project((grad,), state["Qs"])

                grads_proj.append(grad_proj)
                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])

            if len(exp_avgs) == 0:  # skip 1st step
                continue

            # --------------------------------- run adam --------------------------------- #
            # v1 = v1 * beta + g * (1-beta)
            torch._foreach_lerp_(exp_avgs, grads_proj, weight=(1 - beta1))
            # v2 = v2 * beta + g² * (1-beta)
            torch._foreach_mul_(exp_avg_sqs, beta2)
            torch._foreach_addcmul_(exp_avg_sqs, grads_proj, grads_proj, value=(1 - beta2))
            # u = v1 / (sqrt(v2) + eps)
            denom = torch._foreach_sqrt(exp_avg_sqs)
            torch._foreach_clamp_min_(denom, group["eps"])
            dirs_proj = torch._foreach_div(exp_avgs, denom)

            updates = []
            lrs = []

            for param, grad, dir_proj in zip(params_with_grad, grads_merged, dirs_proj):

                state = self.state[param]

                # ------------------------------- project back ------------------------------- #
                (dir,) = soap.project_back((dir_proj,), state["Qs"])
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
                update_accumulators_op_(grad, state["accumulators"], shampoo_beta, operation=group["operation"],
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
                    eps=group["eps"],
                    mode=group["gtue_mode"],
                )

            # ----------------------------- update parameters ---------------------------- #
            if group["weight_decay"] > 0.0:
                torch._foreach_add_(
                    updates, torch._foreach_mul(group["params"], group["weight_decay"])
                )

            torch._foreach_mul_(updates, lrs)
            torch._foreach_sub_(group["params"], updates)

            if group["ema_rate"] != 0:
                opt_utils.update_parameter_ema(self, group=group)

        return loss

    @torch.no_grad
    def train(self):
        opt_utils.optimizer_train(self)

    @torch.no_grad
    def eval(self):
        opt_utils.optimizer_eval(self)