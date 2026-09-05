import math
from typing import Literal

import torch
from torch.optim import Optimizer


def _compute_metric(tensors: list[torch.Tensor], metric: float | Literal["mad","rms"]):
    if isinstance(metric, (int, float)):
        return torch._foreach_norm(tensors, ord=metric)
    if metric == "mad":
        return [t.mean() for t in torch._foreach_abs(tensors)]
    if metric == "rms":
        return [
            t.norm() / math.sqrt(max(1, t.numel()))
            for t in tensors
        ]
    raise ValueError(metric)

def graft_to_update_ema_(
    self: Optimizer,
    params_with_grad: list[torch.Tensor],
    updates_: list[torch.Tensor],
    metric: float | Literal["mad", "rms"],
    beta: float,
    max_metric_growth: float | None,
    min_metric: float,
    max_metric: float | None,
    eps: float,
    mode: Literal["clip", "normalize"],
):
    u_exp_avg: list[torch.Tensor] = []

    # initialize / gather state
    for p in params_with_grad:
        state = self.state[p]
        if "u_exp_avg" not in state:
            state["u_exp_avg"] = torch.zeros_like(p)

        u_exp_avg.append(state["u_exp_avg"])

    # update state
    torch._foreach_lerp_(u_exp_avg, updates_, 1-beta)

    # compute new metric
    u_ema_metric_new = _compute_metric(u_exp_avg, metric)

    # clip norm growth
    if max_metric_growth is not None:

        # initialize / gather previous metrics
        u_ema_metric_prev: list[torch.Tensor] = []
        for p, m_new in zip(params_with_grad, u_ema_metric_new, strict=True):
            state = self.state[p]
            if "u_metric_prev" not in state:
                state["u_metric_prev"] = m_new
            u_ema_metric_prev.append(state["u_metric_prev"])

        # compute max allowed metric (prev metric * max growth)
        u_allowed_metric = torch._foreach_mul(u_ema_metric_prev, max_metric_growth)
        torch._foreach_clamp_min_(u_allowed_metric, min_metric)
        if max_metric is not None:
            torch._foreach_clamp_max_(u_allowed_metric, max_metric)

        # clip growth
        nums = []
        denoms = []
        for exp_avg, m_new, m_allowed, m_prev in zip(u_exp_avg, u_ema_metric_new, u_allowed_metric, u_ema_metric_prev, strict=True):
            if m_new > m_allowed:
                nums.append(exp_avg)
                denoms.append(m_new / m_allowed)
                m_new.copy_(m_allowed)

            m_prev.copy_(m_new)

        if len(nums) > 0:
            torch._foreach_div_(nums, denoms)

    # apply clipping or norm
    u_metric = _compute_metric(updates_, metric)
    torch._foreach_clamp_min_(u_ema_metric_new, eps)
    denom = torch._foreach_div(u_metric, u_ema_metric_new)

    if mode == "normalize": denom_min = eps
    elif mode == "clip": denom_min = 1
    else: raise ValueError(mode)

    torch._foreach_clamp_min_(denom, denom_min)
    torch._foreach_div_(updates_, denom)
    return updates_

def update_parameter_ema(self: "Optimizer", group: dict):
    if group["ema_rate"] != 0:
        params = []
        p_exp_avgs = []

        # gather exp avgs
        for p in group["params"]:
            state = self.state[p]

            if "p_exp_avg" not in state:
                state["p_exp_avg"] = p.clone()
                state["training"] = True

            params.append(p)
            p_exp_avgs.append(state["p_exp_avg"])

        # lerp
        torch._foreach_lerp_(p_exp_avgs, params, 1-group["ema_rate"])

@torch.no_grad
def optimizer_train(self: "Optimizer"):
    for group in self.param_groups:

        # skip if EMA is disabled
        if group["ema_rate"] == 0: continue

        for param in group["params"]:
            state = self.state[param]

            if "p_train" in state:
                if state["training"]: continue
                param.copy_(state["p_train"])

            state["training"] = True

@torch.no_grad
def optimizer_eval(self: "Optimizer"):
    for group in self.param_groups:

        # skip if EMA is disabled
        if group["ema_rate"] == 0: continue

        for param in group["params"]:
            state = self.state[param]

            if "p_exp_avg" in state:

                if not state["training"]: continue
                state["p_train"] = param.clone()
                param.copy_(state["p_exp_avg"])

            state["training"] = False

def clip_norm_(grads_: list[torch.Tensor], max_norm: float, metric: float | Literal["mad"]):
    norms = _compute_metric(grads_, metric)

    tensors = []
    scalars = []
    for g, norm in zip(grads_, norms, strict=True):
        if norm > max_norm:
            tensors.append(g)
            scalars.append(max_norm / norm)

    if tensors:
        torch._foreach_mul_(tensors, scalars)

def mars_correction_(
    g: list[torch.Tensor],
    g_prev_: list[torch.Tensor],
    beta: float,
    scaling: float,
) -> list[torch.Tensor]:
    dg = torch._foreach_sub(g, g_prev_)
    torch._foreach_mul_(dg, scaling * beta / (1-beta))
    torch._foreach_copy_(g_prev_, g)

    g_mars = dg
    torch._foreach_add_(g_mars, g)

    return list(g_mars)

