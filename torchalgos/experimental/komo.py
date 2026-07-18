"""Gemini 3.5 flash made this but its not that good"""
import math
from typing import Literal

import torch
from torch.optim import Optimizer

from .. import kron_utils, soap, opt_utils


class KOMO(Optimizer):
    """
    [KOMO: Kronecker-factored Orthonormalized Momentum Optimizer]

    KOMO bridges Kronecker-factored preconditioning (SOAP/SPlus) with
    matrix spectral orthogonalization (Muon). By projecting the unprojected
    momentum into the Kronecker eigenbasis, it performs a highly efficient
    "Damp-and-Sign" (DnS) operation followed by a quintic Newton-Schulz
    iteration. This yields stable, matrix-aware updates for arbitrary tensor
    dimensions with a single-state memory footprint.

    Args:
        params: Iterable of tensors to optimize, or the model itself.
        lr: Learning rate.
        beta_momentum: Beta for the unprojected momentum buffer. Defaults to 0.95.
        shampoo_beta: Beta for Kronecker-factored covariance accumulators. Defaults to 0.999.
        ema_rate: Beta for exponential moving average of parameters. Defaults to 0.999.
        eps: Small constant for numerical stability. Defaults to 1e-8.
        nonstandard_constant: Step size scale for non-preconditioned parameters (1D, embeddings).
        weight_decay: Decoupled weight decay. Defaults to 0.01.
        precond_freq: Frequency of updating the eigenbasis. Defaults to 10.
        solver: Solver for eigenbasis update ("subspace" or "eigh"). Defaults to "subspace".
        power_iters: Subspace iterations per update. Defaults to 2.
        ns_steps: Number of quintic Newton-Schulz steps. Defaults to 5.
        damping: Damping factor for off-diagonals in the lagging eigenbasis. Defaults to 0.5.
        max_dim: Won't precondition dimensions larger than this. Defaults to 4096.
        merge_dims: Merges small dimensions to enable preconditioning on high-D tensors.
        precond_dims: Dimensions to use Kronecker preconditioning for. Defaults to 'all'.
        precondition_1d: Whether to precondition 1D parameters. Defaults to True.
        cautious: If True, masks the update based on sign agreement with the gradient.
    """
    def __init__(
        self,
        params,
        lr: float = 1e-1,
        beta_momentum: float = 0.95,
        shampoo_beta: float = 0.999,
        ema_rate: float = 0.999,
        eps: float = 1e-8,
        nonstandard_constant: float = 0.001,
        weight_decay: float = 1e-2,
        precond_freq: float = 10,
        solver: Literal["subspace", "eigh"] = "subspace",
        power_iters: int = 2,
        ns_steps: int = 5,
        damping: float = 0.5,
        max_dim: int = 4096,
        merge_dims: bool = True,
        merge_whitelist: int | list[int] | None = None,
        merge_blacklist: int | list[int] | None = 0,
        precond_dims: int | list[int] | None | Literal['all'] = 'all',
        precondition_1d: bool = True,
        cautious: bool = False,
    ):
        defaults = dict(
            lr = lr,
            beta_momentum = beta_momentum,
            shampoo_beta = shampoo_beta,
            ema_rate = ema_rate,
            eps = eps,
            nonstandard_constant = nonstandard_constant,
            weight_decay = weight_decay,
            precond_freq = precond_freq,
            power_iters = power_iters,
            ns_steps = ns_steps,
            damping = damping,
            max_dim = max_dim,
            merge_dims = merge_dims,
            merge_whitelist = merge_whitelist,
            merge_blacklist = merge_blacklist,
            precond_dims = precond_dims,
            precondition_1d = precondition_1d,
            solver = solver,
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
            params_with_grad = []
            updates = []
            lrs = []

            for param in group["params"]:
                if param.grad is None: continue

                grad = param.grad
                state = self.state[param]

                # Merge small dimensions to prepare correct tensor shapes
                if group["merge_dims"]:
                    (grad, ), *state["merge_state"] = kron_utils.merge_small_dims(
                        (grad, ),
                        max_dim=group["max_dim"],
                        whitelist=group["merge_whitelist"],
                        blacklist=group["merge_blacklist"],
                    )

                # Initialize states on the first step
                if "accumulators" not in state:
                    state["step"] = 0
                    state["accumulators"] = soap.initialize_accumulators(
                        grad,
                        precond_dims=group["precond_dims"],
                        precondition_1d=group["precondition_1d"],
                        max_dim=group["max_dim"]
                    )
                    soap.update_accumulators_(grad, state["accumulators"], group["shampoo_beta"])
                    state["Qs"] = soap.initialize_eigenbasis(state["accumulators"])
                    state["momentum"] = torch.zeros_like(grad)

                if state["step"] == 0:
                    state["step"] += 1
                    continue

                # Check if this parameter is preconditioned
                is_preconditioned = any(acc is not None for acc in state["accumulators"])

                if is_preconditioned:
                    # 1. Update unprojected momentum buffer
                    state["momentum"].lerp_(grad, 1 - group["beta_momentum"])
                    M = state["momentum"]

                    # 2. Project momentum into the Kronecker eigenbasis
                    (M_proj, ) = soap.project((M, ), state["Qs"])

                    # 3. Handle arbitrary dimensions for orthogonalization (DnS + Quintic Newton-Schulz)
                    orig_shape = M_proj.shape
                    if M_proj.ndim < 2:
                        # 1D or 0D fallback: preconditioned coordinate-wise sign
                        Y = torch.sign(M_proj)
                    else:
                        # 2D or higher: flatten to 2D to apply matrix orthogonalization
                        if M_proj.ndim > 2:
                            M_proj_2d = M_proj.flatten(1)
                        else:
                            M_proj_2d = M_proj

                        d1, d2 = M_proj_2d.shape[0], M_proj_2d.shape[1]

                        # Normalize Frobenius norm to 1.0 for guaranteed Newton-Schulz stability (spectral norm <= 1.0)
                        X = M_proj_2d / (M_proj_2d.norm() + group["eps"])

                        # Damp lagging off-diagonal elements to eliminate noise (Damp-and-Sign operator)
                        if group["damping"] < 1.0:
                            diag_mask = torch.eye(d1, d2, device=X.device, dtype=torch.bool)
                            X_damp = torch.where(diag_mask, X, X * group["damping"])
                        else:
                            X_damp = X

                        # Quintic Newton-Schulz Orthogonalization (converges rapidly and stably)
                        a, b, c = (3.4445, -4.7750, 2.0315)
                        X_ns = X_damp.to(dtype=X_damp.dtype)

                        # Transpose if d1 > d2 to minimize compute (Gram matrix optimization)
                        if d1 > d2:
                            X_ns = X_ns.T

                        for _ in range(group["ns_steps"]):
                            A = X_ns @ X_ns.T
                            B = b * A + c * A @ A
                            X_ns = a * X_ns + B @ X_ns

                        if d1 > d2:
                            X_ns = X_ns.T

                        Y_2d = X_ns

                        # Scale back to RMS = 1.0 to match SPlus/SOAP scale
                        Y_2d = Y_2d * math.sqrt(max(d1, d2))

                        # Restore original high-dimensional shape
                        if M_proj.ndim > 2:
                            Y = Y_2d.view(orig_shape)
                        else:
                            Y = Y_2d

                    # 4. Apply cautious masking (if enabled)
                    if group["cautious"]:
                        (grad_proj, ) = soap.project((grad, ), state["Qs"])
                        Y = Y * (Y * grad_proj > 0).to(Y.dtype)

                    # 5. Project orthogonalized update back to original coordinate system
                    (dir, ) = soap.project_back((Y, ), state["Qs"])

                    # Compute scale based on SPlus-style dimensional scaling
                    sum_precond = sum(acc.shape[0] for acc in state["accumulators"] if acc is not None)
                    lr = group["lr"] * 2.0 / sum_precond

                else:
                    # Fallback path for non-preconditioned parameters (1D, embeddings)
                    state["momentum"].lerp_(grad, 1 - group["beta_momentum"])
                    M = state["momentum"]

                    # Robust coordinate-wise sign update
                    dir = torch.sign(M)

                    if group["cautious"]:
                        dir = dir * (dir * grad > 0).to(dir.dtype)

                    lr = group["lr"] * group["nonstandard_constant"]

                # Unmerge back to parameter space if dimensions were merged
                if group["merge_dims"]:
                    dir = kron_utils.unmerge_small_dims(dir, *state["merge_state"])

                # Collect update states for efficient batched execution
                params_with_grad.append(param)
                updates.append(dir)
                lrs.append(lr)

                # Update covariance accumulators with current gradient
                soap.update_accumulators_(grad, state["accumulators"], shampoo_beta=group["shampoo_beta"])

                # Periodically update the Kronecker eigenbasis
                if state["step"] % group["precond_freq"] == 0:
                    state["Qs"], *__ = soap.update_eigenbasis(
                        power_iters = group["power_iters"],
                        accumulators = state["accumulators"],
                        Qs = state["Qs"],
                        grads = (),
                        diags = (),
                        solver = group["solver"],
                    )

                state["step"] += 1

            if len(updates) == 0:
                continue

            # Apply decoupled weight decay
            if group["weight_decay"] != 0.0:
                torch._foreach_add_(
                    updates,
                    torch._foreach_mul(params_with_grad, group["weight_decay"])
                )

            # Perform parameter updates in batch
            torch._foreach_mul_(updates, lrs)
            torch._foreach_sub_(params_with_grad, updates)

            # Update parameter exponential moving average (EMA)
            if group["ema_rate"] != 0:
                opt_utils.update_parameter_ema(self, group=group)

        return loss

    @torch.no_grad
    def train(self):
        opt_utils.optimizer_train(self)

    @torch.no_grad
    def eval(self):
        opt_utils.optimizer_eval(self)