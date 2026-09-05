from typing import Literal

import torch
from torch.optim import Optimizer

from torchalgos import kron_utils, soap


class DSOAP(Optimizer):
    """SOAP with accumulator gradients replaced with an estimate which uses function values at two distant points, but it doesn't work well."""
    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas = (0.95, 0.95),
        shampoo_beta: float = -1,
        eps: float = 1e-8,
        damping: float = 1e-4,
        weight_decay: float = 0.01,
        update_freq: int = 4,
        precond_freq: int = 12,
        inner_cma: bool = False,
        inner_clip: float = 1e5,
        solver: Literal["subspace", "eigh", "pogo"] = "pogo",
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
            damping = damping,
            weight_decay = weight_decay,
            update_freq = update_freq,
            precond_freq = precond_freq,
            inner_cma = inner_cma,
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

        f_prev = None
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

            f_prev = closure(False) # no gradient

            # Move back to current point
            for group in self.param_groups:
                for param in group["params"]:
                    state = self.state[param]
                    state["s"] = state["p_cur"] - state["p_prev"]
                    param.copy_(state["p_cur"])
                    state["p_prev"] = param.clone()

        with torch.enable_grad():
            f = closure()

        for group in self.param_groups:


            # collect all buffers for foreach operations
            gs_merged = []
            ests_merged = []
            gs_proj = []
            ests_proj = []
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

                if should_update:
                    group["est"] = state["s"] * (abs(f - f_prev) + group["damping"])
                    del state["s"]

                est = group["est"]

                if group["merge_dims"]: # merge small dims to get correct shapes
                    (g, est), *state["merge_state"] = kron_utils.merge_small_dims(
                        [g, est],
                        max_dim=group["max_dim"],
                        whitelist=group["merge_whitelist"],
                        blacklist=group["merge_blacklist"],
                    )

                gs_merged.append(g)
                ests_merged.append(est)

                if "accumulators" not in state:
                    assert should_update

                    # ----------------------- Initialize state on 1st step ----------------------- #
                    state["step"] = 1

                    state["accumulators"] = soap.initialize_accumulators(
                        g, precond_dims=group["precond_dims"],
                        precondition_1d=group["precondition_1d"], max_dim=group["max_dim"])

                    soap.update_accumulators_(est, state["accumulators"], shampoo_beta)

                    state["Qs"] = soap.initialize_eigenbasis(state["accumulators"])

                    state["v1"] = torch.zeros_like(g)
                    state["v2"] = torch.zeros_like(g)


                # ---------------------------------- Project --------------------------------- #
                (g_proj, ) = soap.project([g], state["Qs"])
                gs_proj.append(g_proj)

                est_proj = None
                if should_update and group["inner_cma"]:
                    (est_proj, ) = soap.project([est], state["Qs"])
                    ests_proj.append(est_proj)

                v1s.append(state["v1"])
                v2s.append(state["v2"])


            # --------------------------------- run inner opt --------------------------------- #
            torch._foreach_lerp_(v1s, gs_proj, weight=(1 - beta1))
            if group["inner_cma"]:
                c = ests_proj if should_update else None
            else:
                c = gs_proj
            if c is not None:
                # v1 = v1 * beta + g * (1-beta)
                # v2 = v2 * beta + g² * (1-beta)
                torch._foreach_mul_(v2s, beta2)
                torch._foreach_addcmul_(v2s, c, c, value=(1 - beta2))
                # u = v1 / (sqrt(v2) + eps)
                denom = torch._foreach_sqrt(v2s)
                torch._foreach_clamp_min_(denom, group["eps"])
                for p, d in zip(params_with_grad, denom): self.state[p]["denom"] = d
                for p, d in zip(params_with_grad, denom): self.state[p]["denom"] = d
            else:
                denom = [self.state[p]["denom"] for p in params_with_grad]

            dirs_proj = torch._foreach_div(v1s, denom)
            torch._foreach_clamp_min_(dirs_proj, -group["inner_clip"])
            torch._foreach_clamp_max_(dirs_proj, group["inner_clip"])

            updates = []
            lrs = []

            buffs = zip(params_with_grad, gs_merged, ests_merged, dirs_proj)
            for param, g, est, dir_proj in buffs:

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
                    if group["inner_cma"]:
                        bias_correction2 = 1.0 - beta2 ** global_state["num_accs"]
                    else:
                        bias_correction2 = 1.0 - beta2 ** state["step"]
                    lr = group["lr"] * (bias_correction2 ** 0.5) / bias_correction1

                updates.append(dir)
                lrs.append(lr)

                if state["step"] == 1:
                    # Q and accumulators already updated on 1st step
                    state["step"] += 1
                    continue

                # ---------------------------- update accumulators --------------------------- #
                # Update is done after the gradient step to avoid using current gradients in the projection.
                if should_update:
                    soap.update_accumulators_(est, state["accumulators"], shampoo_beta=shampoo_beta)

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

        return f

