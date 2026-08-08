"""This is one of Gemini 3.5 flash or 3.1 pro, don't remember which one."""
import math
from typing import Literal

import torch
from torch.optim import Optimizer

# Assuming opt_utils is importable from your environment
from torchalgos import opt_utils


def apply_factors(tensor: torch.Tensor, factors: list[torch.Tensor | None], transpose: bool):
    """
    Applies the Kronecker-factored preconditioner to the tensor.
    If transpose is True, applies S^T. Otherwise, applies S.
    The tensor's shape is perfectly preserved through cyclic permutations.
    """
    if tensor.ndim == 0 or len(factors) == 0:
        return tensor

    for S in factors:
        if S is None:
            # Cycle dimensions to the left
            permute_order = list(range(1, tensor.ndim)) + [0]
            tensor = tensor.permute(permute_order)
        else:
            # S is of shape (d, d)
            # When transpose=True, contract tensor's dim 0 with S's dim 0 (which applies S^T)
            # When transpose=False, contract tensor's dim 0 with S's dim 1 (which applies S)
            dim_S = 0 if transpose else 1
            tensor = torch.tensordot(tensor, S, dims=[[0], [dim_S]])

    return tensor


class KronCBFGS(Optimizer):
    def __init__(
        self,
        params,
        lr=1.0,
        beta:float=0,
        eps=1e-8,
        max_dim=4096,
        precond_dims="all",
        gtue_mode: Literal["disabled", "clip", "normalize"] = "normalize",
        gtue_metric: float | Literal["mad", "rms"] = 2,
        gtue_beta: float = 0.99,
        gtue_max_metric_growth: float | None = 1.5,
        gtue_min_metric: float = 1e-5,
        cautious: bool = False,
        use_grad_diff: bool = True
    ):
        """
        Kronecker-factored Square Root Quasi-Newton Optimizer.

        Args:
            params: Iterable of parameters to optimize.
            lr: Learning rate.
            eps: Epsilon for numerical stability and BFGS update skipping.
            max_dim: Maximum dimension size to precondition.
            precond_dims: Dimensions to precondition. 'all' means all dims <= max_dim.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(
            lr=lr,
            beta=beta,
            eps=eps,
            max_dim=max_dim,
            precond_dims=precond_dims,
            gtue_mode=gtue_mode,
            gtue_metric=gtue_metric,
            gtue_beta=gtue_beta,
            gtue_max_metric_growth=gtue_max_metric_growth,
            gtue_min_metric=gtue_min_metric,
            cautious=cautious,
            use_grad_diff=use_grad_diff,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            eps = group['eps']
            max_dim = group['max_dim']
            precond_dims = group['precond_dims']

            params_with_grad = []
            updates = []

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                # Parse which dimensions to precondition
                p_dims = []
                if precond_dims == 'all':
                    p_dims = list(range(p.ndim))
                elif isinstance(precond_dims, int):
                    p_dims = [precond_dims]
                elif isinstance(precond_dims, list):
                    p_dims = precond_dims

                # Count the number of active factors for initialization scaling
                num_factors = sum(
                    1 for dim, size in enumerate(p.shape)
                    if dim in p_dims and size <= max_dim and size > 1
                )

                if len(state) == 0:
                    state['step'] = 0
                    state['S_factors'] = []

                    if num_factors > 0:
                        initial_scale = 1.0 / (p.grad.norm() + eps)
                        c = initial_scale ** (1.0 / (2 * num_factors))
                    else:
                        c = 1.0

                    for dim, size in enumerate(p.shape):
                        if dim not in p_dims or size > max_dim or size == 1:
                            state['S_factors'].append(None)
                        else:
                            state['S_factors'].append(
                                torch.eye(size, device=p.device, dtype=p.dtype) * c
                            )

                    state['prev_theta'] = p.clone()
                    state['prev_grad'] = p.grad.clone()

                if state['step'] > 0 and num_factors > 0:
                    if group["use_grad_diff"]:
                        x = p - state['prev_theta']
                        b = p.grad - state['prev_grad']

                    else:
                        x = torch.randn_like(p)
                        g_dot_x = torch.sum(p.grad * x)
                        b = g_dot_x * p.grad

                    # Global secant dot product
                    bx = torch.sum(b * x)

                    # BFGS skipping condition to maintain positive definiteness
                    if bx > eps:
                        sqrt_bx = torch.sqrt(bx)

                        for ind, S in enumerate(state['S_factors']):
                            if S is None:
                                continue

                            # Marginalize other dimensions to treat x and b as (size, N)
                            X = x.movedim(ind, 0).reshape(S.shape[0], -1)
                            B = b.movedim(ind, 0).reshape(S.shape[0], -1)

                            U = torch.mm(S.t(), B)
                            u_norm_sq = torch.sum(U * U)

                            # Local skipping condition for numerical stability
                            if u_norm_sq > eps:
                                u_norm = torch.sqrt(u_norm_sq)
                                Su = torch.mm(S, U)

                                # Compute rank-N update vector/matrix V
                                V = X.mul(1.0 / (u_norm * sqrt_bx)) - Su.mul(1.0 / u_norm_sq)

                                # In-place update: S_new = S + V @ U.T
                                S.addmm_(V, U.t(), alpha=1.0)

                # Compute preconditioned search direction d = S S^T g
                input = p.grad
                if group["beta"] != 0:
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(input)
                    input = state["exp_avg"]
                    input.lerp_(p.grad, 1-group["beta"])
                    if group["cautious"]:
                        input = input * (input * p.grad).gt_(0)

                if num_factors > 0:
                    # 1. Multiply by S^T across all valid dimensions
                    step_dir = apply_factors(input, state['S_factors'], transpose=True)
                    # 2. Multiply by S across all valid dimensions
                    step_dir = apply_factors(step_dir, state['S_factors'], transpose=False)
                else:
                    step_dir = input.clone()

                # Store evaluation points BEFORE stepping
                state['prev_theta'].copy_(p)
                state['prev_grad'].copy_(p.grad)

                # Parameter update
                params_with_grad.append(p)
                updates.append(step_dir)

                state['step'] += 1

            # graft to update EMA
            if group["gtue_mode"] != "disabled":
                opt_utils.graft_to_update_ema_(
                    self = self,
                    params_with_grad = params_with_grad,
                    updates_ = updates,
                    metric = group["gtue_metric"],
                    beta = group["gtue_beta"],
                    max_metric_growth=group["gtue_max_metric_growth"],
                    min_metric=group["gtue_min_metric"],
                    eps=group["eps"],
                    mode=group["gtue_mode"],
                )


            torch._foreach_mul_(updates, group["lr"])
            torch._foreach_sub_(params_with_grad, updates)

        return loss


class KFSRC(Optimizer):
    """Kronecker-factored square root covariance"""
    def __init__(
        self,
        params,
        lr=1.0,
        betas=(0.9, 0.99),
        eps=1e-8,
        max_dim=4096,
        precond_dims="all",
        gtue_mode: Literal["disabled", "clip", "normalize"] = "normalize",
        gtue_metric: float | Literal["mad"] = 2,
        gtue_beta: float = 0.99,
        gtue_max_metric_growth: float | None = 1.5,
        gtue_min_metric: float = 1e-5,
        cautious: bool = False,
    ):
        # ... (initialization stays mostly the same)
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            max_dim=max_dim,
            precond_dims=precond_dims,
            gtue_mode=gtue_mode,
            gtue_metric=gtue_metric,
            gtue_beta=gtue_beta,
            gtue_max_metric_growth=gtue_max_metric_growth,
            gtue_min_metric=gtue_min_metric,
            cautious=cautious,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            eps = group['eps']
            beta1, beta2 = group['betas']
            max_dim = group['max_dim']
            precond_dims = group['precond_dims']
            params_with_grad = []
            updates = []

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                # Parse which dimensions to precondition
                p_dims = []
                if precond_dims == 'all':
                    p_dims = list(range(p.ndim))
                elif isinstance(precond_dims, int):
                    p_dims = [precond_dims]
                elif isinstance(precond_dims, list):
                    p_dims = precond_dims

                # Count the number of active factors for initialization scaling
                num_factors = sum(
                    1 for dim, size in enumerate(p.shape)
                    if dim in p_dims and size <= max_dim and size > 1
                )

                if len(state) == 0:
                    state['step'] = 0
                    state['S_factors'] = []

                    if num_factors > 0:
                        initial_scale = 1.0 / (p.grad.norm() + eps)
                        c = initial_scale ** (1.0 / (2 * num_factors))
                    else:
                        c = 1.0

                    for dim, size in enumerate(p.shape):
                        if dim not in p_dims or size > max_dim or size == 1:
                            state['S_factors'].append(None)
                        else:
                            state['S_factors'].append(
                                torch.eye(size, device=p.device, dtype=p.dtype) * c
                            )

                    state['prev_theta'] = p.clone()
                    state['prev_grad'] = p.grad.clone()

                if state['step'] > 0 and num_factors > 0:
                    # Scaling factor to maintain EMA: C_new = beta * C_old + (1-beta) * gg^T
                    # This allows the preconditioner to act like Adam/RMSProp instead of Adagrad
                    grad_scale = math.sqrt((1.0 - beta2) / beta2)

                    for ind, S in enumerate(state['S_factors']):
                        if S is None:
                            continue

                        # Marginalize gradient to treat as (size, N)
                        G = p.grad.movedim(ind, 0).reshape(S.shape[0], -1) * grad_scale

                        # u = S^T G
                        U = torch.mm(S.t(), G)
                        u_norm_sq = torch.sum(U * U) # Trace norm for marginalized dimensions

                        if u_norm_sq > eps:
                            # Calculate alpha for the exact symmetric square root update
                            # We use a Taylor expansion for very small u_norm_sq to prevent precision loss
                            if u_norm_sq < 1e-4:
                                alpha = 0.5 - 0.375 * u_norm_sq
                            else:
                                alpha = (1.0 - 1.0 / torch.sqrt(1.0 + u_norm_sq)) / u_norm_sq

                            SU = torch.mm(S, U)

                            # Update formula: S = S - alpha * (S @ U) @ U^T
                            # This exactly updates S so that S @ S^T tracks the inverse covariance
                            S.sub_(torch.mm(SU, U.t()), alpha=alpha)

                        # Apply the EMA decay restoration to S
                        S.mul_(1.0 / math.sqrt(beta2))

                # Compute preconditioned search direction d = S S^T g
                input = p.grad
                if beta1 != 0:
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(input)
                    input = state["exp_avg"]
                    input.lerp_(p.grad, 1-beta1)
                    if group["cautious"]:
                        input = input * (input * p.grad).gt_(0)

                if num_factors > 0:
                    # 1. Multiply by S^T across all valid dimensions
                    step_dir = apply_factors(input, state['S_factors'], transpose=True)
                    # 2. Multiply by S across all valid dimensions
                    step_dir = apply_factors(step_dir, state['S_factors'], transpose=False)
                else:
                    step_dir = input.clone()

                # Store evaluation points BEFORE stepping
                state['prev_theta'].copy_(p)
                state['prev_grad'].copy_(p.grad)

                # Parameter update
                params_with_grad.append(p)
                updates.append(step_dir)

                state['step'] += 1

            # graft to update EMA
            if group["gtue_mode"] != "disabled":
                opt_utils.graft_to_update_ema_(
                    self = self,
                    params_with_grad = params_with_grad,
                    updates_ = updates,
                    metric = group["gtue_metric"],
                    beta = group["gtue_beta"],
                    max_metric_growth=group["gtue_max_metric_growth"],
                    min_metric=group["gtue_min_metric"],
                    eps=group["eps"],
                    mode=group["gtue_mode"],
                )


            torch._foreach_mul_(updates, group["lr"])
            torch._foreach_sub_(params_with_grad, updates)

        return loss



def apply_factors(tensor: torch.Tensor, factors: list[torch.Tensor | None], transpose: bool):
    """
    Applies the Kronecker-factored preconditioner to the tensor.
    If transpose is True, applies S^T. Otherwise, applies S.
    The tensor's shape is preserved through cyclic permutations.
    """
    if tensor.ndim == 0 or len(factors) == 0:
        return tensor

    for S in factors:
        if S is None:
            # Cycle dimensions to the left
            permute_order = list(range(1, tensor.ndim)) + [0]
            tensor = tensor.permute(permute_order)
        else:
            dim_S = 0 if transpose else 1
            tensor = torch.tensordot(tensor, S, dims=[[0], [dim_S]])

    return tensor


class KFSRCV2(Optimizer):
    """Kronecker-factored square root covariance with direct QN tracking"""
    def __init__(
        self,
        params,
        lr=1.0,
        betas=(0.9, 0.99),
        eps=1e-8,
        max_dim=4096,
        precond_dims="all",
        gtue_mode: Literal["disabled", "clip", "normalize"] = "normalize",
        gtue_metric: float | Literal["mad"] = 2,
        gtue_beta: float = 0.99,
        gtue_max_metric_growth: float | None = 1.5,
        gtue_min_metric: float = 1e-5,
        cautious: bool = False,
        scale_G: bool = False,
        use_adaptive_factors: bool = False,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            max_dim=max_dim,
            precond_dims=precond_dims,
            gtue_mode=gtue_mode,
            gtue_metric=gtue_metric,
            gtue_beta=gtue_beta,
            gtue_max_metric_growth=gtue_max_metric_growth,
            gtue_min_metric=gtue_min_metric,
            cautious=cautious,
            scale_G=scale_G,
            use_adaptive_factors=use_adaptive_factors
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            eps = group['eps']
            beta1, beta2 = group['betas']
            max_dim = group['max_dim']
            precond_dims = group['precond_dims']
            params_with_grad = []
            updates = []

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                # Parse which dimensions to precondition
                p_dims = []
                if precond_dims == 'all':
                    p_dims = list(range(p.ndim))
                elif isinstance(precond_dims, int):
                    p_dims = [precond_dims]
                elif isinstance(precond_dims, list):
                    p_dims = precond_dims

                # Count active factors
                num_factors = sum(
                    1 for dim, size in enumerate(p.shape)
                    if dim in p_dims and size <= max_dim and size > 1
                )

                if len(state) == 0:
                    state['step'] = 0
                    state['S_factors'] = []

                    if num_factors > 0:
                        initial_scale = 1.0 / (p.grad.norm() + eps)
                        c = initial_scale ** (1.0 / (2 * num_factors))
                    else:
                        c = 1.0

                    for dim, size in enumerate(p.shape):
                        if dim not in p_dims or size > max_dim or size == 1:
                            state['S_factors'].append(None)
                        else:
                            state['S_factors'].append(
                                torch.eye(size, device=p.device, dtype=p.dtype) * c
                            )

                    state['prev_theta'] = p.clone()
                    state['prev_grad'] = p.grad.clone()

                if state['step'] > 0 and num_factors > 0:
                    # Scaling factor to maintain EMA
                    if group["use_adaptive_factors"]:
                        active_factors = max(1, num_factors)

                        # Use this if you want the whole tensor to have effective beta2.
                        beta_eff = beta2 ** (1.0 / active_factors)

                        grad_scale = math.sqrt((1.0 - beta_eff) / beta_eff)

                    else:
                        grad_scale = math.sqrt((1.0 - beta2) / beta2)
                        beta_eff = beta2

                    for ind, S in enumerate(state['S_factors']):
                        if S is None:
                            continue

                        # Marginalize gradient to treat as (size, N)
                        G = p.grad.movedim(ind, 0).reshape(S.shape[0], -1) * grad_scale
                        if group["scale_G"]:
                            G = G * (grad_scale / math.sqrt(max(1, G.shape[1])))

                        # Compute total variance (trace norm of G G^T)
                        g_trace_norm = torch.sum(G * G)

                        # Find principal direction via fast power iteration on G G^T
                        x = torch.ones((G.shape[1], 1), device=G.device, dtype=G.dtype)
                        y = torch.mm(G, x)
                        y_norm = torch.norm(y)
                        if y_norm > eps:
                            y = y / y_norm
                        x = torch.mm(G.t(), y)
                        x_norm = torch.norm(x)
                        if x_norm > eps:
                            x = x / x_norm

                        # Extract the 1D principal vector g (shape: size)
                        g = torch.mm(G, x).squeeze(1)

                        # Scale g to match trace norm so total variance is preserved
                        g_norm = torch.norm(g)
                        if g_norm > eps:
                            g = g * (torch.sqrt(g_trace_norm) / g_norm)

                        # Compute u = S^T g using fast matrix-vector product
                        u = torch.mv(S.t(), g)
                        u_norm_sq = torch.sum(u * u)

                        if u_norm_sq > eps:
                            # Calculate alpha for the exact symmetric rank-1 square root update
                            if u_norm_sq < 1e-4:
                                alpha = 0.5 - 0.375 * u_norm_sq
                            else:
                                alpha = (1.0 - 1.0 / torch.sqrt(1.0 + u_norm_sq)) / u_norm_sq

                            # Compute su = S @ u
                            su = torch.mv(S, u)

                            # Exact rank-1 update of the inverse square root matrix
                            S.sub_(torch.outer(su, u), alpha=alpha)

                        # Apply the EMA decay restoration directly to S
                        S.mul_(1.0 / math.sqrt(beta_eff))

                # Compute preconditioned search direction d = S S^T g
                input = p.grad
                if beta1 != 0:
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(input)
                    input = state["exp_avg"]
                    input.lerp_(p.grad, 1-beta1)
                    if group["cautious"]:
                        input = input * (input * p.grad).gt_(0)

                if num_factors > 0:
                    # 1. Multiply by S^T across all valid dimensions
                    step_dir = apply_factors(input, state['S_factors'], transpose=True)
                    # 2. Multiply by S across all valid dimensions
                    step_dir = apply_factors(step_dir, state['S_factors'], transpose=False)
                else:
                    step_dir = input.clone()

                # Store evaluation points BEFORE stepping
                state['prev_theta'].copy_(p)
                state['prev_grad'].copy_(p.grad)

                # Parameter update
                params_with_grad.append(p)
                updates.append(step_dir)

                state['step'] += 1

            # Grafting application
            if len(updates) > 0 and group["gtue_mode"] != "disabled":
                opt_utils.graft_to_update_ema_(
                    self = self,
                    params_with_grad = params_with_grad,
                    updates_ = updates,
                    metric = group["gtue_metric"],
                    beta = group["gtue_beta"],
                    max_metric_growth=group["gtue_max_metric_growth"],
                    min_metric=group["gtue_min_metric"],
                    eps=group["eps"],
                    mode=group["gtue_mode"],
                )

            if len(updates) > 0:
                torch._foreach_mul_(updates, group["lr"])
                torch._foreach_sub_(params_with_grad, updates)

        return loss