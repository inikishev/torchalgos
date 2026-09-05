"""SOAP"""
import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Literal

import torch
from torch.optim import Optimizer

from torchalgos import kron_utils, opt_utils, soap

class SOAPCMA(Optimizer):
    """CMA like SOAP"""
    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas = (0.95, 0.95),
        cma_beta: float = 0.95,
        update_means_if_worse: Literal["update", "downdate", "skip"] = "skip",
        update_cov_if_worse: Literal["update", "downdate", "skip"] = "update",
        correction: Literal['grad', 'diff'] = 'grad',
        pullback_beta: float = 1,
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
        max_norm: float | None = None,
        max_norm_type: float | Literal["mad"] = 2,
    ):
        defaults = dict(
            lr = lr,
            betas = betas,
            cma_beta = cma_beta,
            update_means_if_worse = update_means_if_worse,
            update_cov_if_worse = update_cov_if_worse,
            correction = correction,
            pullback_beta = pullback_beta,
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
            gtue_max_metric = gtue_max_metric,
            cautious = cautious,
            mars_scale = mars_scale,
            max_norm = max_norm,
            max_norm_type = max_norm_type,

        )

        if isinstance(params, torch.nn.Module):
            params = kron_utils.make_kron_param_groups_for_emb(params)

        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure): # pyright:ignore[reportIncompatibleMethodOverride]

        # Compute loss at means
        for group in self.param_groups:
            for param in group["params"]:
                if not param.requires_grad: continue
                state = self.state[param]
                if "cma_p_mean" not in state: state["cma_p_mean"] = param.clone()
                state["cma_p_train"] = param.clone()
                param.copy_(state["cma_p_mean"])

        loss_means = closure()

        # Go back to train params
        for group in self.param_groups:
            for param in group["params"]:
                if not param.requires_grad: continue
                state = self.state[param]
                param.copy_(state["cma_p_train"])

        # Compute loss and grad at current params
        with torch.enable_grad():
            loss = closure()

        is_better = loss <= loss_means

        for group in self.param_groups:

            # collect all buffers for foreach operations
            grads_merged = []
            grads_proj = []
            exp_avgs = []
            exp_avg_sqs = []
            params_with_grad = []
            grads_prev = []

            beta1, beta2 = group["betas"]

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

                    state["Qs"] = soap.initialize_eigenbasis(state["accumulators"])

                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                    if group["mars_scale"] != 0: state["g_prev"] = torch.zeros_like(grad)

                if state["step"] == 0:
                    # first step is skipped so that we never use the current gradients in the projection.
                    state["step"] += 1
                    continue

                # ---------------------------------- Project --------------------------------- #
                (grad_proj, ) = soap.project((grad, ), state["Qs"])

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
                (dir, ) = soap.project_back((dir_proj, ), state["Qs"])
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
                # update param means and compute correction
                cma_beta = group["cma_beta"]
                pullback_beta = group["pullback_beta"]
                shampoo_beta = group["shampoo_beta"]
                if shampoo_beta < 0: shampoo_beta = beta2

                if group["correction"] == "diff": correction = state["cma_p_mean"] - param
                else: correction = grad

                if is_better:
                    state['cma_p_mean'].lerp_(state["cma_p_train"], weight=(1-cma_beta))

                else:
                    # pull back to better parameter means
                    if pullback_beta != 0:
                        param.lerp_(state["cma_p_mean"], weight=(1-pullback_beta))

                    # update means
                    if group["update_means_if_worse"] == "update":
                        state['cma_p_mean'].lerp_(state["cma_p_train"], weight=(1-cma_beta))

                    elif group["update_means_if_worse"] == "downdate":
                        state['cma_p_mean'].lerp_(state["cma_p_train"], weight=-(1-cma_beta))

                    elif group["update_means_if_worse"] == "skip":
                        pass

                    else:
                        raise ValueError(group["update_means_if_worse"])

                    # covariance correction
                    if group["update_cov_if_worse"] == "update":
                        pass

                    elif group["update_cov_if_worse"] == "downdate":
                        shampoo_beta = -shampoo_beta

                    elif group["update_cov_if_worse"] == "skip":
                        correction = None

                    else:
                        raise ValueError(group["update_means_if_worse"])

                if correction is not None:
                    soap.update_accumulators_(correction, state["accumulators"], shampoo_beta=shampoo_beta)

                # ------------------------------- update basis ------------------------------- #
                if state["step"] % group["precond_freq"] == 0:
                    state["Qs"], (state["exp_avg"], ), (state["exp_avg_sq"], ) = soap.update_eigenbasis(
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
                    max_metric=group["gtue_max_metric"],
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