"""SOAP"""
import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Literal

import torch
from torch.optim import Optimizer

from torchalgos import kron_utils, opt_utils, soap

class PGSOAP(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas = (0.95, 0.95),
        shampoo_beta: float = -1,
        eps: float = 1e-8,
        symmetrize: bool = False,
        from_ema: bool = False,
        cov_ema_beta: float = 0.999,
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
        gtue_mode: Literal["disabled", "clip", "normalize"] = "disabled",
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
            lr = lr,
            betas = betas,
            shampoo_beta = shampoo_beta,
            eps = eps,
            symmetrize = symmetrize,
            from_ema = from_ema,
            cov_ema_beta = cov_ema_beta,
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
            max_grad_norm = max_grad_norm,
            max_grad_norm_type = max_grad_norm_type,
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
            params_merged = []
            grads_prev = []

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
                        grad, precond_dims=group["precond_dims"], precondition_1d=group["precondition_1d"], max_dim=group["max_dim"])

                    if not group["from_ema"]:
                        if group["symmetrize"]:
                            correction1 = (grad + param_merged) / 2
                            correction2 = None
                        else:
                            correction1 = grad
                            correction2 = param_merged
                        soap.update_accumulators_(correction1, state["accumulators"], shampoo_beta, vec2=correction2)

                    state["Qs"] = soap.initialize_eigenbasis(state["accumulators"])

                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                    if group["mars_scale"] != 0: state["g_prev"] = torch.zeros_like(grad)

                    if group["from_ema"]:
                        state["cov_p_ema"] = param_merged.clone()

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

                # ---------------------- update param ema for covariance --------------------- #
                if group["from_ema"] and group["cov_ema_beta"] != 0:
                    state["cov_p_ema"].lerp_(param_merged, weight=1-group['cov_ema_beta'])


            if len(exp_avgs) == 0: # skip 1st step
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
                if group["from_ema"]: vec2 = state["cov_p_ema"] - param
                else: vec2 = param_merged

                if group["symmetrize"]:
                    correction1 = (grad + vec2) / 2
                    correction2 = None
                else:
                    correction1 = grad
                    correction2 = vec2

                soap.update_accumulators_(correction1, state["accumulators"], shampoo_beta=shampoo_beta, vec2=correction2)


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
                    max_metric = group["gtue_min_metric"],
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