"""SOAP. https://github.com/inikishev/torchalgos"""
from typing import Literal

import torch
from torch.optim import Optimizer

from torchalgos import kron_utils, soap

def update_accumulators_asymmetric_(
    s: torch.Tensor,
    y: torch.Tensor,
    accumulators_: list[torch.Tensor | None],
    shampoo_beta: float,
):
    for i, acc in enumerate(accumulators_):
        if acc is None: continue

        axes = list(range(i)) + list(range(i + 1, s.ndim)) # this works fine with 1d params
        acc.lerp_(torch.tensordot(s, y, (axes, axes)), 1-shampoo_beta) # pyright:ignore[reportArgumentType]


class QSOAP(Optimizer):
    """SOAP which accumulates and uses hessian estimate, doesn't seem any better than normal SOAP while being more expensive."""
    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas = (0.95, 0.95),
        shampoo_beta: float = -1,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        update_freq: int = 4,
        precond_freq: int = 12,
        inner_opt: Literal["Adam", "AdaHessian", "SophiaH"] = "SophiaH",
        inner_clip: float = 1,
        solver: Literal["subspace", "eigh"] = "subspace",
        power_iters: int = 1,
        max_dim: int = 4096,
        merge_dims: bool = False,
        merge_whitelist: int | list[int] | None = None,
        merge_blacklist: int | list[int] | None = 0,
        precond_dims: int | list[int] | None | Literal['all'] = 'all',
        precondition_1d: bool = True,
        normalize: bool = False,
        reproject_v2: bool = True,
        warmup_steps: int = 10,
    ):
        defaults = dict(
            lr = lr,
            betas = betas,
            shampoo_beta = shampoo_beta,
            eps = eps,
            weight_decay = weight_decay,
            update_freq = update_freq,
            precond_freq = precond_freq,
            inner_opt = inner_opt,
            inner_clip = inner_clip,
            power_iters = power_iters,
            max_dim = max_dim,
            merge_dims = merge_dims,
            merge_whitelist = merge_whitelist,
            merge_blacklist = merge_blacklist,
            precond_dims = precond_dims,
            precondition_1d = precondition_1d,
            normalize = normalize,
            reproject_v2 = reproject_v2,
            warmup_steps = warmup_steps,
            solver = solver,
        )

        if isinstance(params, torch.nn.Module):
            params = kron_utils.make_kron_param_groups_for_emb(params)

        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure): # pyright:ignore[reportIncompatibleMethodOverride] # pylint:disable=signature-differs

        global_group = self.param_groups[0]
        global_state = self.state[global_group["params"][0]]
        if "global_step" not in global_state:
            global_state["global_step"] = 0
            global_state["num_accs"] = 0
            with torch.enable_grad():
                closure() # do extra evaluation on 1st step for initializing p_prev to p+sign(g)*lr
        else:
            global_state["global_step"] += 1

        should_update = (
            (global_state["global_step"] % global_group["update_freq"] == 0) or
            (global_state["global_step"] <= global_group["warmup_steps"])
        )
        if should_update:
            global_state["num_accs"] += 1

            # collect gradients at previous point
            for group in self.param_groups:

                for param in group["params"]:
                    state = self.state[param]
                    if "p_prev" not in state:
                        # for first step we don't have previous parameters
                        # we can initialize to params + sign(grad) if grad exists,
                        # if grad is None, it is assumed to be zeros (to support conditional params)
                        if param.grad is None: state["p_prev"] = param
                        else: state["p_prev"] = param + param.grad.sign() * group["lr"]

                    state["p_cur"] = param.clone()
                    param.copy_(state["p_prev"])

            with torch.enable_grad():
                closure()

            # Move back to current point
            for group in self.param_groups:
                for param in group["params"]:
                    state = self.state[param]
                    state["g_prev"] = param.grad.clone() if param.grad is not None else torch.zeros_like(param)
                    state["s"] = state["p_cur"] - state["p_prev"]
                    param.copy_(state["p_cur"])
                    state["p_prev"] = param.clone()

        with torch.enable_grad():
            loss = closure()


        for group in self.param_groups:

            # collect all buffers for foreach operations
            g_merged_list = []
            g_prev_merged_list = []
            s_merged_list = []
            g_proj_list = []
            g_prev_proj_list = []
            s_proj_list = []
            v1s = []
            v2s = []
            params_with_grad = []

            beta1, beta2 = group["betas"]
            shampoo_beta = group["shampoo_beta"]
            if shampoo_beta < 0: shampoo_beta = beta2

            for param in group["params"]:
                state = self.state[param]

                if param.grad is None: continue
                params_with_grad.append(param)

                g = param.grad
                s = state["s"]
                g_prev = state["g_prev"]

                if group["merge_dims"]: # merge small dims to get correct shapes
                    (g, g_prev, s), *state["merge_state"] = kron_utils.merge_small_dims(
                        [g, g_prev, s],
                        max_dim=group["max_dim"],
                        whitelist=group["merge_whitelist"],
                        blacklist=group["merge_blacklist"],
                    )

                g_merged_list.append(g)
                g_prev_merged_list.append(g_prev)
                s_merged_list.append(s)

                if "accumulators" not in state:
                    assert should_update

                    # ----------------------- Initialize state on 1st step ----------------------- #
                    state["step"] = 1

                    state["accumulators"] = soap.initialize_accumulators(
                        g, precond_dims=group["precond_dims"], precondition_1d=group["precondition_1d"], max_dim=group["max_dim"])

                    update_accumulators_asymmetric_(s, g - g_prev, state["accumulators"], shampoo_beta)

                    state["Qs"] = soap.initialize_eigenbasis(state["accumulators"])

                    state["v1"] = torch.zeros_like(g)
                    state["v2"] = torch.zeros_like(g)


                # ---------------------------------- Project --------------------------------- #
                g_prev_proj = s_proj = None
                (g_proj, ) = soap.project([g], state["Qs"])
                if should_update:
                    g_prev_proj, s_proj = soap.project([g_prev, s], state["Qs"])

                g_proj_list.append(g_proj)
                v1s.append(state["v1"])
                v2s.append(state["v2"])
                if should_update:
                    g_prev_proj_list.append(g_prev_proj)
                    s_proj_list.append(s_proj)


            # --------------------------------- run inner opt --------------------------------- #
            # v1 = v1 * beta + g * (1-beta)
            torch._foreach_lerp_(v1s, g_proj_list, weight=(1 - beta1))

            if group["inner_opt"] == "Adam":
                # v2 = v2 * beta + D² * (1-beta)
                torch._foreach_mul_(v2s, beta2)
                torch._foreach_addcmul_(v2s, g_proj_list, g_proj_list, value=(1 - beta2))
                # u = v1 / (sqrt(v2) + eps)
                denom = torch._foreach_sqrt(v2s)
                torch._foreach_clamp_min_(denom, group["eps"])

            elif group["inner_opt"] == "AdaHessian":
                if should_update:

                    # D = s * y
                    y = torch._foreach_sub(g_proj_list, g_prev_proj_list)
                    D = torch._foreach_mul(y, s_proj_list)

                    # v2 = v2 * beta + D² * (1-beta)
                    torch._foreach_mul_(v2s, beta2)
                    torch._foreach_addcmul_(v2s, D, D, value=(1 - beta2))

                    # u = v1 / (sqrt(v2) + eps)
                    denom = torch._foreach_sqrt(v2s)
                    torch._foreach_clamp_min_(denom, group["eps"])
                    for p, d in zip(params_with_grad, denom): self.state[p]["denom"] = d
                else:
                    denom = [self.state[p]["denom"] for p in params_with_grad]

            elif group["inner_opt"] == "SophiaH":
                if should_update:

                    # D = s * y
                    y = torch._foreach_sub(g_proj_list, g_prev_proj_list)
                    D = torch._foreach_mul(y, s_proj_list)

                    # v2 = v2 * beta + D * (1-beta)
                    torch._foreach_lerp_(v2s, D, weight=(1 - beta2))

                    # u = v1 / clip(D, min=eps)
                    denom = torch._foreach_clamp_min(v2s, group["eps"])
                    for p, d in zip(params_with_grad, denom): self.state[p]["denom"] = d
                else:
                    denom = [self.state[p]["denom"] for p in params_with_grad]


            else:
                raise ValueError(group["inner_opt"])

            dirs_proj = torch._foreach_div(v1s, denom)
            torch._foreach_clamp_min_(dirs_proj, -group["inner_clip"])
            torch._foreach_clamp_max_(dirs_proj, group["inner_clip"])

            updates = []
            lrs = []

            buffs = zip(params_with_grad, g_merged_list, g_prev_merged_list, s_merged_list, dirs_proj)
            for param, g, g_prev, s, dir_proj in buffs:

                state = self.state[param]

                # ------------------------------- project back ------------------------------- #
                (dir, ) = soap.project_back([dir_proj], state["Qs"])
                if group["merge_dims"]:
                    dir = kron_utils.unmerge_small_dims(dir, *state["merge_state"])

                if group["normalize"]:
                    # no debiasing because update is normalized
                    lr = group["lr"] / dir.square().mean().sqrt().clip(min=group["eps"])

                else:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    if group["inner_opt"] == "Adam":
                        bias_correction2 = 1.0 - beta2 ** state["step"]
                        lr = group["lr"] * (bias_correction2 ** 0.5) / bias_correction1
                    elif group["inner_opt"] == "AdaHessian":
                        bias_correction2 = 1.0 - beta2 ** global_state["num_accs"]
                        lr = group["lr"] * (bias_correction2 ** 0.5) / bias_correction1
                    elif group["inner_opt"] == "SophiaH":
                        bias_correction2 = 1.0 - beta2 ** global_state["num_accs"]
                        lr = group["lr"] * bias_correction2 / bias_correction1
                    else:
                        raise ValueError(group["inner_opt"])

                updates.append(dir)
                lrs.append(lr)

                if state["step"] == 1:
                    # Q and accumulators already updated on 1st step
                    state["step"] += 1
                    continue

                # ---------------------------- update accumulators --------------------------- #
                # Update is done after the gradient step to avoid using current gradients in the projection.
                if should_update:
                    update_accumulators_asymmetric_(s, g - g_prev, state["accumulators"], shampoo_beta=shampoo_beta)

                if (
                    (state['step'] % group['precond_freq'] == 0) or
                    (global_state["global_step"] <= global_group["warmup_steps"])
                ):

                    # ------------------------------- update basis ------------------------------- #
                    state["Qs"], (state["v1"], ), (state["v2"], ) = soap.update_eigenbasis(
                        power_iters = group["power_iters"],
                        accumulators = state["accumulators"],
                        Qs = state["Qs"],
                        grads = (state["v1"], ),
                        diags = (state["v2"], ),
                        solver = group["solver"],
                    )


                state["step"] += 1

            # ----------------------------- update parameters ---------------------------- #
            if group["weight_decay"] > 0.0:
                torch._foreach_add_(
                    updates,
                    torch._foreach_mul(group["params"], group["weight_decay"])
                )

            torch._foreach_mul_(updates, lrs)
            torch._foreach_sub_(group["params"], updates)

        return loss

