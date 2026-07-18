from collections.abc import Callable
from typing import Literal

import torch
from torch.optim import Optimizer

from torchalgos import kron_utils, opt_utils, soap
from torchalgos.experimental import custom_soap


class CustomSPlus(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-1,
        operation: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = torch.mul,
        reduce: Callable[[torch.Tensor, int], torch.Tensor] = torch.sum,
        copysign: bool = False,
        update_fn: Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor] = torch.lerp,
        beta_unproj: float = 0.9,
        beta_proj: float = 0,
        beta_update: float = 0,
        shampoo_beta: float = 0.999,
        ema_rate: float = 0.999,
        eps: float = 1e-8,
        nonstandard_constant: float = 0.001,
        weight_decay: float = 1e-2,
        precond_freq: float = 100,
        solver: Literal["subspace", "eigh"] = "eigh",
        power_iters: int = 1,
        max_dim: int = 4096,
        merge_dims: bool = False,
        merge_whitelist: int | list[int] | None = None,
        merge_blacklist: int | list[int] | None = 0,
        precond_dims: int | list[int] | None | Literal['all'] = 'all',
        precondition_1d: bool = True,
        normalize: bool = False,
    ):
        defaults = dict(
            lr = lr,
            operation = operation,
            reduce = reduce,
            copysign = copysign,
            update_fn = update_fn,
            beta_unproj = beta_unproj,
            beta_proj = beta_proj,
            beta_update = beta_update,
            shampoo_beta = shampoo_beta,
            ema_rate = ema_rate,
            eps = eps,
            nonstandard_constant = nonstandard_constant,
            weight_decay = weight_decay,
            precond_freq = precond_freq,
            power_iters = power_iters,
            max_dim = max_dim,
            merge_dims = merge_dims,
            merge_whitelist = merge_whitelist,
            merge_blacklist = merge_blacklist,
            precond_dims = precond_dims,
            precondition_1d = precondition_1d,
            solver = solver,
            normalize = normalize,
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
            params_with_grad = []

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

                    state["accumulators"] = soap.initialize_accumulators(
                        grad, precond_dims=group["precond_dims"], precondition_1d=group["precondition_1d"], max_dim=group["max_dim"])

                    custom_soap.update_accumulators_op_(grad, state["accumulators"], group["shampoo_beta"],
                                                        operation=group["operation"], reduce=group["reduce"],
                                                        copysign=group["copysign"], update_fn=group["update_fn"])

                    state["Qs"] = soap.initialize_eigenbasis(state["accumulators"])

                    if group["beta_proj"] != 0:
                        state["exp_avg"] = torch.zeros_like(grad)


                if state["step"] == 0:
                    # first step is skipped so that we never use the current gradients in the projection.
                    state["step"] += 1
                    continue

                # Unprojected exponential moving average
                beta_unproj = group["beta_unproj"]
                if beta_unproj != 0:
                    if "exp_avg_unproj" not in state:
                        state["exp_avg_unproj"] = torch.zeros_like(grad)

                    state["exp_avg_unproj"].lerp_(grad, 1-beta_unproj)
                    grad = state["exp_avg_unproj"]

                # ---------------------------------- Project --------------------------------- #
                (grad_proj, ) = soap.project((grad, ), state["Qs"])

                grads_proj.append(grad_proj)

                if group["beta_proj"] != 0:
                    exp_avgs.append(state["exp_avg"])


            if len(grads_proj) == 0: # skip 1st step
                continue

            # ------------------------------- projected EMA ------------------------------ #
            if group["beta_proj"] != 0:
                torch._foreach_lerp_(exp_avgs, grads_proj, weight=(1 - group["beta_proj"]))
                dirs_proj = torch._foreach_sign(exp_avgs)

            else:
                dirs_proj = torch._foreach_sign(grads_proj)

            updates = []
            lrs = []
            update_emas = []

            for param, grad, dir_proj in zip(params_with_grad, grads_merged, dirs_proj):

                state = self.state[param]

                # ------------------------------- project back ------------------------------- #
                (dir, ) = soap.project_back((dir_proj, ), state["Qs"])
                if group["merge_dims"]:
                    dir = kron_utils.unmerge_small_dims(dir, *state["merge_state"])

                if group["normalize"]:
                    # no step size scaling because update is normalized
                    lr = group["lr"] / dir.square().mean().sqrt().clip(min=group["eps"])

                else:
                    if param.ndim >= 1:
                        sum_precond = sum(acc.shape[0] for acc in state["accumulators"] if acc is not None)
                        if sum_precond == 0:
                            lr = group["lr"] * group["nonstandard_constant"]
                        else:
                            lr = group["lr"] * 2 / sum_precond
                    else:
                        lr = group["lr"] * group["nonstandard_constant"]

                updates.append(dir)
                lrs.append(lr)
                if group["beta_update"] != 0:
                    if "u_exp_avg" not in state:
                        state["u_exp_avg"] = torch.zeros_like(dir)
                    update_emas.append(state["u_exp_avg"])

                # ---------------------------- update accumulators --------------------------- #
                # Update is done after the gradient step to avoid using current gradients in the projection.
                custom_soap.update_accumulators_op_(grad, state["accumulators"], group["shampoo_beta"],
                                                    operation=group["operation"], reduce=group["reduce"],
                                                    copysign=group["copysign"], update_fn=group["update_fn"])

                # ------------------------------- update basis ------------------------------- #
                if state["step"] % group["precond_freq"] == 0:
                    state["Qs"], diags, _ = soap.update_eigenbasis(
                        power_iters = group["power_iters"],
                        accumulators = state["accumulators"],
                        Qs = state["Qs"],
                        grads = (state["exp_avg"], ) if "exp_avg" in state else (),
                        diags =  (state["exp_avg_sq"], ) if "exp_avg_sq" in state else (),
                        solver = group["solver"],
                    )
                    if "exp_avg" in state:
                        state["exp_avg"] = diags[0]

                state["step"] += 1

            # ema on update
            if group["beta_update"] != 0:
                torch._foreach_lerp_(update_emas, updates, 1-group["beta_update"])
                torch._foreach_copy_(updates, update_emas)

            # ----------------------------- update parameters ---------------------------- #
            if group["weight_decay"] != 0.0:
                torch._foreach_add_(
                    updates,
                    torch._foreach_mul(group["params"], group["weight_decay"])
                )

            torch._foreach_mul_(updates, lrs)
            torch._foreach_sub_(group["params"], updates)

            # --------------------------- update parameter EMA --------------------------- #
            if group["ema_rate"] != 0:
                opt_utils.update_parameter_ema(self, group=group)

        return loss

    @torch.no_grad
    def train(self):
        opt_utils.optimizer_train(self)

    @torch.no_grad
    def eval(self):
        opt_utils.optimizer_eval(self)