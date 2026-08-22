from typing import Literal

import torch
from torch.optim import Optimizer

from torchalgos import kron_utils, soap, opt_utils


class RePlus(Optimizer):
    """SPlus but uses reciprocal of shifted and clipped gradient instead of sign.

    Recommended hyperparams:
    - shampoo_beta=0.95 for MLP
    - shampoo_beta=0 for RNN and ConvNet. If its acting weird set to 0.95

    Args:
        clip_min: Clips magnitude below before taking reciprocal.
        clip_max: Clips magnitude above before taking reciprocal.
        shift: Adds to magnitude before taking reciprocal, applies after clipping.
        power: Power applied to magnitudes before clipping and shifting.

    Other args are from SPlus.
    """
    def __init__(
        self,
        params,
        lr: float = 1e-1,
        clip_min: float = 0.01,
        clip_max: float | None = 10,
        shift: float = 0.0,
        power: float = 1,
        beta_unproj: float = 0,
        beta_proj: float = 0.9,
        beta_update: float = 0,
        shampoo_beta: float = 0.95,
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
        gtue_mode: Literal["disabled", "clip", "normalize"] = "normalize",
        gtue_metric: float | Literal["mad"] = 2,
        gtue_beta: float = 0.99,
        gtue_max_metric_growth: float | None = 1.5,
        gtue_min_metric: float = 1e-5,
        cautious: bool = True,
    ):
        defaults = dict(
            lr = lr,
            clip_min = clip_min,
            clip_max = clip_max,
            shift = shift,
            power = power,
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
            gtue_mode = gtue_mode,
            gtue_metric = gtue_metric,
            gtue_beta = gtue_beta,
            gtue_max_metric_growth = gtue_max_metric_growth,
            gtue_min_metric = gtue_min_metric,
            cautious = cautious,
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

                    soap.update_accumulators_(grad, state["accumulators"], group["shampoo_beta"])

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
                inputs_proj = exp_avgs

            else:
                inputs_proj = grads_proj

            # compute and stabilize magnitudes
            magnitudes = torch._foreach_abs(inputs_proj)
            if group["power"] != 1:
                torch._foreach_pow_(magnitudes, group['power'])
            if group["clip_min"] != 0:
                torch._foreach_clamp_min_(magnitudes, group["clip_min"])
            if group["clip_max"] is not None:
                torch._foreach_clamp_max_(magnitudes, group["clip_max"])
            if group["shift"] != 0:
                torch._foreach_add_(magnitudes, group["shift"])

            # compute reciprocal magnitudes
            torch._foreach_reciprocal_(magnitudes)

            # compute the update
            dirs_proj = torch._foreach_sign(inputs_proj)
            torch._foreach_mul_(dirs_proj, magnitudes)

            if group["cautious"]:
                torch._foreach_mul_(dirs_proj, [t.gt_(0) for t in torch._foreach_mul(dirs_proj, grads_proj)])

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
                soap.update_accumulators_(grad, state["accumulators"], shampoo_beta=group["shampoo_beta"])

                # ------------------------------- update basis ------------------------------- #
                if state["step"] % group["precond_freq"] == 0:

                    grad_buffers = []
                    if "exp_avg" in state: grad_buffers.append(state["exp_avg"])
                    if "reciprocal_exp_avg" in state: grad_buffers.append(state["reciprocal_exp_avg"])

                    state["Qs"], diags, _ = soap.update_eigenbasis(
                        power_iters = group["power_iters"],
                        accumulators = state["accumulators"],
                        Qs = state["Qs"],
                        grads = grad_buffers,
                        diags = (),
                        solver = group["solver"],
                    )
                    if "exp_avg" in state:
                        state["exp_avg"] = diags[0]

                state["step"] += 1

            # ema on update
            if group["beta_update"] != 0:
                torch._foreach_lerp_(update_emas, updates, 1-group["beta_update"])
                torch._foreach_copy_(updates, update_emas)

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
            if group["weight_decay"] != 0.0:
                torch._foreach_add_(
                    updates,
                    torch._foreach_mul(params_with_grad, group["weight_decay"])
                )

            torch._foreach_mul_(updates, lrs)
            torch._foreach_sub_(params_with_grad, updates)

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