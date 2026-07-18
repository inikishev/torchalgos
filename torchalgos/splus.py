"""SPlus"""
from typing import Literal

import torch
from torch.optim import Optimizer

from torchalgos import kron_utils, soap, opt_utils


class SPlus(Optimizer):
    """[A Stable Whitening Optimizer for Efficient Neural Network Training](https://arxiv.org/abs/2506.07254).

    This one is slightly different in that it can maintain momentum in both original and rotated space, and by default uses rotated,
    and you can also change the solver to subspace one from SOAP.

    Args:
        params: Iterable of tensors to optimize, or pass the model itself
            which automatically uses diagonal preconditioner for embedding parameters.
        lr: Learning rate. Defaults to 3e-3.
        beta_unproj: Beta for momentum in unprojected space. Defaults to 0.9.
        beta_proj: Beta for momentum in projected space. Defaults to 0.
        beta_sign: Beta for momentum of the sign in projected space. Defaults to 0.
        shampoo_beta: Beta for kronecker factored covariance accumulators. Defaults to (0.9, 0.999).
        ema_rate: beta for exponential moving average of parameters.
            Calling optimizer.eval() sets parameters to the EMA. Set to 0 to disable.
        eps: clips Adam's denominator below this value. Defaults to 1e-8.
        nonstandard_constant: Step size scale for parameters that aren't preconditioned.
        weight_decay: decoupled weight decay, NOT decoupled from learning rate. Defaults to 0.01.
        precond_freq: frequency of updating eigenbasis, default 10.
        solver: how to update eigenbasis, eigendecomposition or subspace iteration.
        power_iters: number of subspace iterations per eigenbasis update, 1 is enough in most cases, only has effect for subspace iteration. default 1.
        max_dim: won't precondition dims larger than this. Defaults to 4096.
        merge_dims: whether to merge small dimensions, default True.
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
        lr: float = 1e-1,
        beta_unproj: float = 0.9,
        beta_proj: float = 0,
        beta_sign: float = 0,
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
        cautious: bool = False,
    ):
        defaults = dict(
            lr = lr,
            beta_unproj = beta_unproj,
            beta_proj = beta_proj,
            beta_sign = beta_sign,
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
            sign_exp_avgs = []
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

                    if group["beta_sign"] != 0:
                        state["sign_exp_avg"] = torch.zeros_like(grad)


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

                if group["beta_sign"] != 0:
                    sign_exp_avgs.append(state["sign_exp_avg"])


            if len(grads_proj) == 0: # skip 1st step
                continue

            # ------------------------------- projected EMA ------------------------------ #
            if group["beta_proj"] != 0:
                torch._foreach_lerp_(exp_avgs, grads_proj, weight=(1 - group["beta_proj"]))
                dirs_proj = torch._foreach_sign(exp_avgs)

            else:
                dirs_proj = torch._foreach_sign(grads_proj)

            if group["beta_sign"] != 0:
                torch._foreach_lerp_(sign_exp_avgs, dirs_proj, weight=(1 - group["beta_sign"]))
                dirs_proj = sign_exp_avgs

            if group["cautious"]:
                dirs_proj = torch._foreach_mul(dirs_proj, [t.gt_(0) for t in torch._foreach_mul(dirs_proj, grads_proj)])

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
                    if "sign_exp_avg" in state: grad_buffers.append(state["sign_exp_avg"])

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