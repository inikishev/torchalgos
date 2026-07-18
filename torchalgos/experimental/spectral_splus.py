"""I gave all of the torchalgos code to Gemini 3.5 pro saying this is the current SOTA, beat it, and it suggested this. But its not that good."""
from typing import Literal
import torch
from torch.optim import Optimizer
from torchalgos import kron_utils, soap, opt_utils


def get_eigenvalues(accumulators, Qs):
    lambdas_list = []
    for acc, Q in zip(accumulators, Qs):
        if acc is not None and Q is not None:
            # Efficiently compute the diagonal of Q^T @ acc @ Q
            lambdas = torch.sum(Q * (acc @ Q), dim=0)
            lambdas_list.append(lambdas.clamp(min=0.0))
        else:
            lambdas_list.append(None)
    return lambdas_list


class SpectralSPlus(Optimizer):
    """Spectral SPlus with smooth activation functions and analytical scale-awareness.
    Supports interpolating between SOAP and SPlus while saving considerable memory.
    """
    def __init__(
        self,
        params,
        lr: float = 1e-1,
        beta_unproj: float = 0.9,
        beta_proj: float = 0.0,
        beta_update: float = 0.0,
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
        activation: Literal["sign", "tanh", "soft_sign", "identity"] = "tanh",
        gamma: float = 1.0,
        exponent: float | None = None,
    ):
        defaults = dict(
            lr=lr,
            beta_unproj=beta_unproj,
            beta_proj=beta_proj,
            beta_update=beta_update,
            shampoo_beta=shampoo_beta,
            ema_rate=ema_rate,
            eps=eps,
            nonstandard_constant=nonstandard_constant,
            weight_decay=weight_decay,
            precond_freq=precond_freq,
            power_iters=power_iters,
            max_dim=max_dim,
            merge_dims=merge_dims,
            merge_whitelist=merge_whitelist,
            merge_blacklist=merge_blacklist,
            precond_dims=precond_dims,
            precondition_1d=precondition_1d,
            solver=solver,
            normalize=normalize,
            cautious=cautious,
            activation=activation,
            gamma=gamma,
            exponent=exponent,
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
            grads_merged = []
            dirs_proj_list = []
            params_with_grad = []

            for param in group["params"]:
                if param.grad is None: continue
                params_with_grad.append(param)

                grad = param.grad
                state = self.state[param]

                if group["merge_dims"]:
                    (grad, ), *state["merge_state"] = kron_utils.merge_small_dims(
                        (grad, ),
                        max_dim=group["max_dim"],
                        whitelist=group["merge_whitelist"],
                        blacklist=group["merge_blacklist"],
                    )

                grads_merged.append(grad)

                if "accumulators" not in state:
                    state["step"] = 0
                    state["accumulators"] = soap.initialize_accumulators(
                        grad, precond_dims=group["precond_dims"], precondition_1d=group["precondition_1d"], max_dim=group["max_dim"]
                    )
                    soap.update_accumulators_(grad, state["accumulators"], group["shampoo_beta"])
                    state["Qs"] = soap.initialize_eigenbasis(state["accumulators"])

                    if group["beta_proj"] != 0:
                        state["exp_avg"] = torch.zeros_like(grad)

                if state["step"] == 0:
                    state["step"] += 1
                    continue

                # Unprojected EMA
                beta_unproj = group["beta_unproj"]
                if beta_unproj != 0:
                    if "exp_avg_unproj" not in state:
                        state["exp_avg_unproj"] = torch.zeros_like(grad)
                    state["exp_avg_unproj"].lerp_(grad, 1 - beta_unproj)
                    grad = state["exp_avg_unproj"]

                # ------------------ Project ------------------ #
                (grad_proj, ) = soap.project((grad, ), state["Qs"])

                # Temporal momentum in projected space
                if group["beta_proj"] != 0:
                    state["exp_avg"].lerp_(grad_proj, 1 - group["beta_proj"])
                    bias_correction = 1.0 - group["beta_proj"] ** state["step"]
                    active_grad = state["exp_avg"] / bias_correction
                else:
                    active_grad = grad_proj

                # Compute mode-wise eigenvalues
                lambdas_list = get_eigenvalues(state["accumulators"], state["Qs"])
                k = sum(1 for lambdas in lambdas_list if lambdas is not None)

                # Standardise coordinate-wise scale
                for d, lambdas in enumerate(lambdas_list):
                    if lambdas is not None:
                        rms = torch.sqrt(lambdas).clamp(min=group["eps"])
                        shape = [1] * active_grad.ndim
                        shape[d] = -1
                        active_grad = active_grad / rms.view(shape)

                # Non-linear activation mapping
                act = group["activation"]
                if act == "sign":
                    dir_proj = torch.sign(active_grad)
                elif act == "tanh":
                    dir_proj = torch.tanh(active_grad / group["gamma"])
                elif act == "soft_sign":
                    dir_proj = (active_grad / group["gamma"]) / (1.0 + torch.abs(active_grad / group["gamma"]))
                else:  # "identity"
                    dir_proj = active_grad

                # Re-scale back to proper curvature space
                exponent = group["exponent"]
                if exponent is None:
                    exponent = 0.5 - 1.0 / (2.0 * k) if k > 0 else 0.0

                if exponent != 0.0:
                    for d, lambdas in enumerate(lambdas_list):
                        if lambdas is not None:
                            scale = torch.pow(lambdas, exponent)
                            shape = [1] * dir_proj.ndim
                            shape[d] = -1
                            dir_proj = dir_proj * scale.view(shape)

                if group["cautious"]:
                    dir_proj *= dir_proj * (dir_proj * grad_proj).gt_(0)

                dirs_proj_list.append(dir_proj)

            if len(dirs_proj_list) == 0:
                continue

            updates = []
            lrs = []
            update_emas = []

            for param, grad, dir_proj in zip(params_with_grad, grads_merged, dirs_proj_list):
                state = self.state[param]

                # ----------------- Project Back ----------------- #
                (dir, ) = soap.project_back((dir_proj, ), state["Qs"])
                if group["merge_dims"]:
                    dir = kron_utils.unmerge_small_dims(dir, *state["merge_state"])

                if group["normalize"]:
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

                # Update accumulators
                soap.update_accumulators_(grad, state["accumulators"], shampoo_beta=group["shampoo_beta"])

                # Update basis
                if state["step"] % group["precond_freq"] == 0:
                    state["Qs"], diags, _ = soap.update_eigenbasis(
                        power_iters=group["power_iters"],
                        accumulators=state["accumulators"],
                        Qs=state["Qs"],
                        grads=(state["exp_avg"], ) if "exp_avg" in state else (),
                        diags=(),
                        solver=group["solver"],
                    )
                    if "exp_avg" in state:
                        state["exp_avg"] = diags[0]

                state["step"] += 1

            if group["beta_update"] != 0:
                torch._foreach_lerp_(update_emas, updates, 1 - group["beta_update"])
                torch._foreach_copy_(updates, update_emas)

            if group["weight_decay"] != 0.0:
                torch._foreach_add_(
                    updates,
                    torch._foreach_mul(group["params"], group["weight_decay"])
                )

            torch._foreach_mul_(updates, lrs)
            torch._foreach_sub_(group["params"], updates)

            if group["ema_rate"] != 0:
                opt_utils.update_parameter_ema(self, group=group,)

        return loss

    @torch.no_grad
    def train(self):
        opt_utils.optimizer_train(self)

    @torch.no_grad
    def eval(self):
        opt_utils.optimizer_eval(self)