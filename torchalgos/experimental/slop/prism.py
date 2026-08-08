"""GLM5.2 made this. Its an ok optimizer but SOAP and SPlus are better."""
import math
from typing import Literal

import torch
from torch.optim import Optimizer

from torchalgos import kron_utils, opt_utils, soap


class Prism(Optimizer):
    """
    PRISM: Power-law Rooted Iterative Spectral Method.

    A novel optimizer that generalizes SOAP and SPlus through a spectral
    exponent `alpha` controlling whitening strength in the eigenbasis of
    Kronecker-factored gradient covariance.

    The core update in projected space is:
        u = m / (v^alpha + eps)

    with generalized bias correction:
        lr_eff = lr * (1 - beta2^t)^alpha / (1 - beta1^t)

    Properties:
    - alpha=0.5 recovers SOAP (Adam in eigenbasis, standard whitening)
    - alpha>0.5 over-whitens (amplifies low-variance directions)
    - alpha<0.5 under-whitens (more conservative, for noisy gradients)
    - alpha=0.75 (default) provides moderate over-whitening

    SPlus uses sign(m) which is aggressive whitening via *instantaneous*
    variance. PRISM's over-whitening uses EMA variance, making it smoother
    and more stable while still accelerating convergence beyond SOAP.

    Optional alpha warmup anneals from 0.5 (conservative) to the target
    alpha over `alpha_warmup` steps, providing stability when Kronecker
    factors are poorly estimated early in training.

    Args:
        params: Iterable of tensors, or pass a model to auto-configure
            diagonal preconditioner for embeddings.
        lr: Learning rate. Default 2e-3.
        betas: (beta1, beta2) for momentum and second moment. Default (0.95, 0.95).
        shampoo_beta: EMA beta for Kronecker accumulators. -1 = beta2. Default -1.
        alpha: Spectral exponent controlling whitening strength.
            0.5 = Adam/SOAP, >0.5 = over-whitening, <0.5 = under-whitening.
            Default 0.75.
        alpha_warmup: Steps to anneal alpha from 0.5 to target. 0 = no warmup.
            Default 0.
        eps: Denominator floor. Default 1e-8.
        weight_decay: Decoupled weight decay. Default 0.01.
        ema_rate: Parameter EMA beta. 0 = disabled. Default 0.999.
        precond_freq: Eigenbasis update frequency. Default 10.
        solver: "subspace" (QR-based) or "eigh". Default "subspace".
        power_iters: Power iterations before QR. Default 2.
        max_dim: Max dimension to precondition. Default 4096.
        merge_dims: Merge small dimensions for efficiency. Default False.
        merge_whitelist: Only these dims can be merged. Default None.
        merge_blacklist: These dims won't be merged. Default 0.
        precond_dims: Dims to precondition. 'all' = all below max_dim.
            None = diagonal only. Default 'all'.
        precondition_1d: Precondition 1d parameters. Default True.
        normalize: Normalize updates (overrides bias correction). Default False.
        gtue_mode: Update EMA grafting: "disabled", "clip", or "normalize".
            Default "normalize".
        gtue_metric: Metric for grafting (float or "mad"). Default 2.
        gtue_beta: Grafting EMA beta. Default 0.99.
        gtue_max_metric_growth: Max growth ratio per step. Default 1.5.
        gtue_min_metric: Minimum metric floor. Default 1e-5.
        cautious: Only step where update aligns with gradient. Default False.
        mars_scale: MARS gradient correction scale. 0 = disabled. Default 0.
        max_norm: Max gradient norm. None = disabled. Default None.
        max_norm_type: Norm type for clipping (float or "mad"). Default 2.
    """

    def __init__(
        self,
        params,
        lr: float = 2e-3,
        betas=(0.95, 0.95),
        shampoo_beta: float = -1,
        alpha: float = 0.75,
        alpha_warmup: int = 0,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        ema_rate: float = 0.999,
        precond_freq: float = 10,
        solver: Literal["subspace", "eigh"] = "subspace",
        power_iters: int = 2,
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
        cautious: bool = False,
        mars_scale: float = 0,
        max_norm: float | None = None,
        max_norm_type: float | Literal["mad"] = 2,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            shampoo_beta=shampoo_beta,
            alpha=alpha,
            alpha_warmup=alpha_warmup,
            eps=eps,
            weight_decay=weight_decay,
            ema_rate=ema_rate,
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
            gtue_mode=gtue_mode,
            gtue_metric=gtue_metric,
            gtue_beta=gtue_beta,
            gtue_max_metric_growth=gtue_max_metric_growth,
            gtue_min_metric=gtue_min_metric,
            cautious=cautious,
            mars_scale=mars_scale,
            max_norm=max_norm,
            max_norm_type=max_norm_type,
        )

        if isinstance(params, torch.nn.Module):
            params = kron_utils.make_kron_param_groups_for_emb(params)

        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None):  # pyright:ignore[reportIncompatibleMethodOverride]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            # Collect all buffers for foreach operations
            grads_merged = []
            grads_proj = []
            exp_avgs = []
            exp_avg_sqs = []
            params_with_grad = []
            grads_prev = []

            beta1, beta2 = group["betas"]
            shampoo_beta = group["shampoo_beta"]
            if shampoo_beta < 0:
                shampoo_beta = beta2
            alpha_target = group["alpha"]
            alpha_warmup = group["alpha_warmup"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                params_with_grad.append(param)

                grad = param.grad
                state = self.state[param]

                if group["merge_dims"]:
                    (grad,), *state["merge_state"] = kron_utils.merge_small_dims(
                        (grad,),
                        max_dim=group["max_dim"],
                        whitelist=group["merge_whitelist"],
                        blacklist=group["merge_blacklist"],
                    )

                grads_merged.append(grad)

                if "accumulators" not in state:
                    # ----------------------- Initialize state on 1st step ----------------------- #
                    state["step"] = 0

                    state["accumulators"] = soap.initialize_accumulators(
                        grad,
                        precond_dims=group["precond_dims"],
                        precondition_1d=group["precondition_1d"],
                        max_dim=group["max_dim"],
                    )

                    soap.update_accumulators_(grad, state["accumulators"], shampoo_beta)

                    state["Qs"] = soap.initialize_eigenbasis(state["accumulators"])

                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                    if group["mars_scale"] != 0:
                        state["g_prev"] = torch.zeros_like(grad)

                if state["step"] == 0:
                    # First step is skipped so we never use current gradients in projection
                    state["step"] += 1
                    continue

                # ---------------------------------- Project --------------------------------- #
                (grad_proj,) = soap.project((grad,), state["Qs"])

                grads_proj.append(grad_proj)
                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                if group["mars_scale"] != 0:
                    if state["step"] == 1:
                        state["g_prev"] = grad_proj.clone()
                    grads_prev.append(state["g_prev"])

            if len(exp_avgs) == 0:  # skip 1st step
                continue

            # ------------------------------- MARS correction ------------------------------ #
            if group["mars_scale"] != 0:
                grads_proj = opt_utils.mars_correction_(
                    grads_proj, grads_prev, group["betas"][0], group["mars_scale"]
                )

            # ------------------------------ Gradient clipping ----------------------------- #
            if group["max_norm"] is not None:
                opt_utils.clip_norm_(
                    grads_proj,
                    max_norm=group["max_norm"],
                    metric=group["max_norm_type"],
                )

            # --------------------------- Update moments (Adam-style) ----------------------- #
            # m = beta1 * m + (1 - beta1) * g
            torch._foreach_lerp_(exp_avgs, grads_proj, weight=(1 - beta1))
            # v = beta2 * v + (1 - beta2) * g^2
            torch._foreach_mul_(exp_avg_sqs, beta2)
            torch._foreach_addcmul_(exp_avg_sqs, grads_proj, grads_proj, value=(1 - beta2))

            # ----------------- Compute per-parameter alpha (with optional warmup) ------------ #
            # During warmup, anneal from 0.5 (conservative, Adam-like) to target alpha.
            # This provides stability when Kronecker factors are poorly estimated.
            if alpha_warmup > 0:
                alphas = []
                for p in params_with_grad:
                    s = self.state[p]["step"]
                    a = 0.5 + (alpha_target - 0.5) * min(1.0, s / alpha_warmup)
                    alphas.append(a)
            else:
                alphas = [alpha_target] * len(params_with_grad)

            # ------------------------- PRISM core: spectral update ------------------------- #
            # u = m / (v^alpha + eps)
            #
            # alpha=0.5: standard whitening (Adam/SOAP) — m / sqrt(v)
            # alpha>0.5: over-whitening — amplifies low-variance directions
            # alpha<0.5: under-whitening — more conservative
            #
            # The over-whitening at alpha=0.75 is motivated by the observation that
            # after Kronecker projection, the gradient is already partially whitened
            # at the covariance level. Standard sqrt(v) whitening is then doubly
            # conservative; v^0.75 better matches the remaining spectral structure.
            dirs_proj = []
            for m, v, a in zip(exp_avgs, exp_avg_sqs, alphas, strict=True):
                denom = v.clamp_min(0).pow(a)
                denom.clamp_min_(group["eps"])
                dirs_proj.append(m / denom)

            # ------------------------------ Cautious updates ------------------------------- #
            # Only apply updates where the update direction aligns with the gradient.
            # This prevents the optimizer from "undoing" progress in directions where
            # the momentum disagrees with the current gradient.
            if group["cautious"]:
                for d, g in zip(dirs_proj, grads_proj, strict=True):
                    d.mul_((d * g).gt_(0))

            updates = []
            lrs = []

            for param, grad, dir_proj, alpha_t in zip(
                params_with_grad, grads_merged, dirs_proj, alphas, strict=True
            ):
                state = self.state[param]

                # ------------------------------- project back ------------------------------- #
                (dir,) = soap.project_back((dir_proj,), state["Qs"])
                if group["merge_dims"]:
                    dir = kron_utils.unmerge_small_dims(dir, *state["merge_state"])

                if group["normalize"]:
                    # Normalize update to unit RMS — scale-invariant like Muon
                    lr = group["lr"] / dir.square().mean().sqrt().clip(min=group["eps"])
                else:
                    # Generalized bias correction for arbitrary alpha:
                    #
                    # u_hat = m_hat / v_hat^alpha
                    #       = [m / (1-b1^t)] / [v / (1-b2^t)]^alpha
                    #       = m * (1-b2^t)^alpha / [(1-b1^t) * v^alpha]
                    #
                    # So: lr_eff = lr * (1-b2^t)^alpha / (1-b1^t)
                    #
                    # For alpha=0.5: lr * sqrt(1-b2^t) / (1-b1^t)  → standard Adam ✓
                    # For alpha=1.0: lr * (1-b2^t) / (1-b1^t)      → more aggressive early on
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    lr = group["lr"] * (bias_correction2 ** alpha_t) / bias_correction1

                updates.append(dir)
                lrs.append(lr)

                # ---------------------------- update accumulators --------------------------- #
                # Done after gradient step to avoid using current gradients in projection.
                soap.update_accumulators_(grad, state["accumulators"], shampoo_beta=shampoo_beta)

                # ------------------------------- update basis ------------------------------- #
                if state["step"] % group["precond_freq"] == 0:
                    state["Qs"], (state["exp_avg"],), (state["exp_avg_sq"],) = soap.update_eigenbasis(
                        power_iters=group["power_iters"],
                        accumulators=state["accumulators"],
                        Qs=state["Qs"],
                        grads=(state["exp_avg"],),
                        diags=(state["exp_avg_sq"],),
                        solver=group["solver"],
                    )

                state["step"] += 1

            # ----------------------- Graft to update EMA (stability) ----------------------- #
            # Normalizes/clips the update magnitude to match an EMA of past update magnitudes.
            # This is crucial for stability with alpha>0.5, as over-whitening can produce
            # large updates in low-variance directions.
            if group["gtue_mode"] != "disabled":
                opt_utils.graft_to_update_ema_(
                    self=self,
                    params_with_grad=params_with_grad,
                    updates_=updates,
                    metric=group["gtue_metric"],
                    beta=group["gtue_beta"],
                    max_metric_growth=group["gtue_max_metric_growth"],
                    min_metric=group["gtue_min_metric"],
                    eps=group["eps"],
                    mode=group["gtue_mode"],
                )

            # ----------------------------- update parameters ------------------------------ #
            if group["weight_decay"] > 0.0:
                torch._foreach_add_(
                    updates,
                    torch._foreach_mul(params_with_grad, group["weight_decay"]),
                )

            torch._foreach_mul_(updates, lrs)
            torch._foreach_sub_(params_with_grad, updates)

            # --------------------------- update parameter EMA ----------------------------- #
            if group["ema_rate"] != 0:
                opt_utils.update_parameter_ema(self, group=group)

        return loss

    @torch.no_grad
    def train(self):
        opt_utils.optimizer_train(self)

    @torch.no_grad
    def eval(self):
        opt_utils.optimizer_eval(self)