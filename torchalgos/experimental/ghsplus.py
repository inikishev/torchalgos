"""SPlus"""
from typing import Literal
import warnings
import torch
from torch.optim import Optimizer
from collections.abc import Sequence
from torchalgos import kron_utils, soap, opt_utils

def update_accumulators_avg_(
    grad: torch.Tensor,
    accumulators_: Sequence[torch.Tensor | None],
    t: int,
):
    for ind, acc in enumerate(accumulators_):
        if acc is None: continue
        g = grad.movedim(ind, 0).reshape(grad.shape[ind], -1) # (shape[i], batch)

        if t == 0:
            acc.copy_(g @ g.T)
        else:
            acc.add_((g @ g.T).div_(t))
            acc.mul_(t / (t + 1))



class GHSPlus(Optimizer):
    """Gaussian homotopy SPlus

    this implementation is really bad and is to be rewritten"""
    def __init__(
        self,
        params,
        lr: float = 1e-1,
        init_steps: int = 0,
        max_sigma_steps: int = 1000,
        decay_steps: int = 1000,
        sigma_init: float = 0.0,
        sigma_max: float = 1.0,
        sigma_final: float = 0.0,
        sigma_distribution: Literal["normal", "uniform", "rademacher", "sphere"] = "normal",
        sigma_magnitude: float | Literal["rms", "mad", "fixed", "multiply"] = 'mad',
        precond_freq_init: int = 100,
        precond_freq_max_sigma: int = 10,
        precond_freq_decay: int = 10,
        perturbations_precond_mode: Literal["disabled", "sq_avg", "sq_avg_normalize", "sign"] = "sq_avg_normalize",
        beta_unproj: float = 0.9,
        beta_proj: float = 0,
        beta_sign: float = 0,
        beta_update: float = 0,
        shampoo_beta: float = 0.999,
        ema_rate: float = 0.999,
        eps: float = 1e-8,
        nonstandard_constant: float = 0.001,
        weight_decay: float = 1e-2,
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
            init_steps = init_steps,
            max_sigma_steps = max_sigma_steps,
            decay_steps = decay_steps,
            sigma_max = sigma_max,
            sigma_init = sigma_init,
            sigma_final = sigma_final,
            sigma_distribution = sigma_distribution,
            sigma_magnitude = sigma_magnitude,
            precond_freq_init = precond_freq_init,
            precond_freq_max_sigma = precond_freq_max_sigma,
            precond_freq_decay = precond_freq_decay,
            perturbations_precond_mode = perturbations_precond_mode,
            beta_unproj = beta_unproj,
            beta_proj = beta_proj,
            beta_sign = beta_sign,
            beta_update = beta_update,
            shampoo_beta = shampoo_beta,
            ema_rate = ema_rate,
            eps = eps,
            nonstandard_constant = nonstandard_constant,
            weight_decay = weight_decay,
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
    def step(self, closure): # pyright:ignore[reportIncompatibleMethodOverride]


        # apply perturbation before closure
        for group in self.param_groups:

            params_to_perturb = []
            sigmas = []

            for param in group["params"]:
                if not param.requires_grad: continue

                state = self.state[param]

                if "step" not in state:
                    state["step"] = 0

                if state["step"] < group["init_steps"]:
                    # initial stage
                    sigma = group["sigma_init"]

                elif state["step"] < group["init_steps"] + group["max_sigma_steps"]:
                    # max sigma stage
                    sigma = group["sigma_max"]

                elif state["step"] < group["init_steps"] + group["max_sigma_steps"] + group["decay_steps"]:
                    # decay stage
                    decay_start_step = group["init_steps"] + group["max_sigma_steps"]
                    frac = (state["step"] - decay_start_step) / group["decay_steps"]
                    sigma = group["sigma_final"] * frac + group["sigma_max"] * (1 - frac)

                else:
                    # final stage
                    sigma = group["sigma_final"]

                if sigma == 0: continue

                state["param_init"] = param.clone()
                params_to_perturb.append(param)
                sigmas.append(sigma)

            if len(params_to_perturb) == 0:
                continue

            # Compute perturbations
            if group["sigma_distribution"] == "normal":
                perturbations = [torch.randn_like(p) for p in params_to_perturb]
            elif group["sigma_distribution"] == "uniform":
                perturbations = [torch.empty_like(p).uniform_(-1, 1) for p in params_to_perturb]
            elif group["sigma_distribution"] == "rademacher":
                perturbations = [torch.randint_like(p, 0, 2) for p in params_to_perturb]
                torch._foreach_mul_(perturbations, 2)
                torch._foreach_sub_(perturbations, 1)
            elif group["sigma_distribution"] == "sphere":
                perturbations = [torch.randn_like(p) for p in params_to_perturb]
                norms = torch._foreach_norm(perturbations)
                torch._foreach_div_(perturbations, norms)
            else:
                raise ValueError(group["sigma_distribution"])

            if group["perturbations_precond_mode"] != "disabled":
                # Precondition perturbations
                precond_perturbations = []
                for param, pert in zip(params_to_perturb, perturbations, strict=True):
                    state = self.state[param]
                    if "Qs" in state:
                        (pert_merged, ), *state["merge_state"] = kron_utils.merge_small_dims(
                            (pert, ),
                            max_dim=group["max_dim"],
                            whitelist=group["merge_whitelist"],
                            blacklist=group["merge_blacklist"],
                        )
                        (pert_proj, ) = soap.project((pert_merged, ), state["Qs"])

                        if group["perturbations_precond_mode"].startswith("sq_avg"):
                            if "avg_sq_proj" in state:
                                pert_proj.div_(state["avg_sq_proj"].sqrt().clip_(min=1e-7))
                            else:
                                pert_proj.zero_() # no perturbation when no gradient sq avg available

                        elif group["perturbations_precond_mode"] == "sign":
                            pert_proj.sign_()

                        else:
                            raise ValueError(group["perturbations_precond_mode"])

                        (pert_merged, ) = soap.project_back((pert_proj, ), state["Qs"])
                        (pert ) = kron_utils.unmerge_small_dims(pert_merged, *state["merge_state"])

                    else:
                        pert.zero_()

                    precond_perturbations.append(pert)

                perturbations = precond_perturbations

            # Scale perturbations with params and sigmas
            if group["sigma_magnitude"] != "fixed":
                if group["sigma_magnitude"] == "multiply":
                    sigmas = torch._foreach_mul(params_to_perturb, sigmas)
                else:
                    metrics = opt_utils._compute_metric(params_to_perturb, group["sigma_magnitude"])
                    sigmas = [s * m for s, m in zip(sigmas, metrics)]

            torch._foreach_mul_(perturbations, sigmas)

            # perturb parameters
            torch._foreach_add_(params_to_perturb, perturbations)

        # Compute closure at perturbed parameters
        with torch.enable_grad():
            loss = closure()

        for group in self.param_groups:

            # collect all buffers for foreach operations
            grads_merged = []
            grads_proj = []
            exp_avgs = []
            sign_exp_avgs = []
            avg_sqs_proj = []
            ts = []
            t_div_tplus1s = []
            params_with_grad = []

            for param in group["params"]:
                if param.grad is None: continue

                # undo perturbation
                state = self.state[param]
                if "param_init" in state:
                    param.copy_(state["param_init"])
                    del state["param_init"]

                params_with_grad.append(param)

                grad = param.grad

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
                    state["t"] = 0
                    state["t_increment"] = 1

                    state["accumulators"] = soap.initialize_accumulators(
                        grad, precond_dims=group["precond_dims"], precondition_1d=group["precondition_1d"], max_dim=group["max_dim"])

                    update_accumulators_avg_(grad, state["accumulators"], state["t"])

                    state["Qs"] = soap.initialize_eigenbasis(state["accumulators"])

                    if group["beta_proj"] != 0:
                        state["exp_avg"] = torch.zeros_like(grad)

                    if group["beta_sign"] != 0:
                        state["sign_exp_avg"] = torch.zeros_like(grad)

                    if group["perturbations_precond_mode"].startswith("sq_avg"):
                        state["avg_sq_proj"] = soap.project((grad, ), state["Qs"])[0].square() # for whitening perturbations


                if state["step"] == 0:
                    # first step is skipped so that we never use the current gradients in the projection.
                    state["step"] += 1
                    state["t"] += 1
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

                if group["perturbations_precond_mode"].startswith("sq_avg"):
                    avg_sqs_proj.append(state["avg_sq_proj"])
                    t_div_tplus1s.append(state["step"] / (state["step"] + 1))
                    ts.append(state["step"])


            if len(grads_proj) == 0: # skip 1st step
                continue

            # ------------------------------- projected EMA ------------------------------ #
            if len(avg_sqs_proj) > 0:
                if group["perturbations_precond_mode"] == 'sq_avg':
                    inputs = grads_proj
                else:
                    metrics = [max(m, 1e-7) for m in opt_utils._compute_metric(grads_proj, 'mad')]
                    inputs = torch._foreach_div(grads_proj, metrics)

                # update squared gradient average
                grads_proj_sq = torch._foreach_pow(inputs, 2)
                torch._foreach_add_(avg_sqs_proj, torch._foreach_div(grads_proj_sq, ts))
                torch._foreach_mul_(avg_sqs_proj, t_div_tplus1s)

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
                if state["step"] < group["init_steps"] + group["max_sigma_steps"] + group["decay_steps"]:
                    update_accumulators_avg_(grad, state["accumulators"], t=state["t"])

                # ------------------------------- update basis ------------------------------- #
                if state["step"] < group["init_steps"]:
                    precond_freq = group["precond_freq_init"]
                elif state["step"] < group['init_steps'] + group['max_sigma_steps']:
                    precond_freq = group["precond_freq_max_sigma"]
                elif state["step"] < group['init_steps'] + group['max_sigma_steps'] + group['decay_steps']:
                    precond_freq = group['precond_freq_decay']
                else:
                    precond_freq = None

                if precond_freq is not None and state["step"] % precond_freq == 0:

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
                state["t"] += state["t_increment"]
                if state["step"] > group["init_steps"] + group["max_sigma_steps"]:
                    state["t_increment"] += 1

            # ema on update
            if group["beta_update"] != 0:
                torch._foreach_lerp_(update_emas, updates, 1-group["beta_update"])
                torch._foreach_copy_(updates, update_emas)

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