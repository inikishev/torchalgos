'''qwen3.8 max'''
import torch
from torch.optim.optimizer import Optimizer


class SOLVENT(Optimizer):
    """
    SOLVENT: Second-Order Local rEsoLveNT optimizer.

    It applies a damped Kronecker/Sylvester resolvent to matrix-shaped
    parameters. This is an affine normal-form extension of whitening:
    instead of merely whitening the gradient, it approximately translates
    to the minimizer of the local Kronecker quadratic model and then
    normalizes the residual.

    For a matrix gradient G and curvature factors A, B, the ideal update
    direction solves:

        A Δ B + λ Δ = G

    which is equivalent to:

        vec(Δ) = (B ⊗ A + λ I)^{-1} vec(G).

    In generic mode, A and B are estimated from EMA gradient outer
    products:

        L = EMA(G G^T)
        R = EMA(G^T G)

    and their eigenvalues are raised to `factor_power`.

    factor_power = 0.25 gives whitening/polar-like behavior.
    factor_power = 0.50 gives a stronger Newton-like resolvent behavior.

    Args:
        params: iterable of parameters to optimize or dicts defining
            parameter groups.
        lr: learning rate.
        betas: coefficients for gradient and factor running averages.
        eps: numerical floor.
        weight_decay: decoupled weight decay.
        damping: relative damping for the resolvent.
        damping_floor: absolute damping floor.
        factor_power: power applied to eigenvalues of L and R.
            0.25 ~ whitening/polar-like.
            0.5 ~ stronger inverse-curvature behavior.
        factor_update_every: recompute eigendecompositions every this
            many steps.
        factor_warmup: use AdamW fallback for this many steps before
            enabling SOLVENT factors.
        max_full_dim: only use full matrix factors when both matrix
            dimensions are <= this value.
        max_rel_step: trust-region clip: update norm <= max_rel_step *
            max(param_norm, 1).
        bias_correction: bias-correct momentum and factor estimates.
        use_matrix: if False, always use AdamW fallback.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.95),
        eps: float = 1e-12,
        weight_decay: float = 0.0,
        damping: float = 1e-2,
        damping_floor: float = 1e-10,
        factor_power: float = 0.5,
        factor_update_every: int = 10,
        factor_warmup: int = 0,
        max_full_dim: int = 2048,
        max_rel_step: float = 0.1,
        bias_correction: bool = True,
        use_matrix: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if factor_power < 0.0:
            raise ValueError(f"factor_power must be nonnegative, got {factor_power}")
        if factor_update_every <= 0:
            factor_update_every = 1
        if factor_warmup < 0:
            factor_warmup = 0

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            damping=damping,
            damping_floor=damping_floor,
            factor_power=factor_power,
            factor_update_every=int(factor_update_every),
            factor_warmup=int(factor_warmup),
            max_full_dim=int(max_full_dim),
            max_rel_step=max_rel_step,
            bias_correction=bias_correction,
            use_matrix=use_matrix,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _safe_eigh(A: torch.Tensor, jitter: float = 1e-12):
        """
        Robust symmetric eigendecomposition for PSD matrices.

        Falls back to diagonal approximation if LAPACK/eigh fails.
        """
        A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        n = A.shape[-1]

        if n == 0:
            return (
                torch.empty(0, device=A.device, dtype=A.dtype),
                torch.empty((0, 0), device=A.device, dtype=A.dtype),
            )

        # Symmetrize and avoid mutating caller state.
        A = (A + A.transpose(-2, -1)).mul(0.5)
        A = A.clone()
        A.diagonal(dim1=-2, dim2=-1).add_(jitter)

        try:
            vals, vecs = torch.linalg.eigh(A, UPLO="L")
            vals = torch.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            vecs = torch.nan_to_num(vecs, nan=0.0, posinf=0.0, neginf=0.0)
            return vals, vecs
        except Exception:
            # Very conservative fallback.
            vals = torch.diagonal(A).clamp_min(0.0)
            vecs = torch.eye(n, device=A.device, dtype=A.dtype)
            return vals, vecs

    def _update_factors(self, state, group, step: int):
        """
        Recompute eigenbases for the left and right Kronecker factors.
        """
        beta2 = group["betas"][1]
        eps = group["eps"]
        power = group["factor_power"]

        L = state["L"]
        R = state["R"]

        if group["bias_correction"]:
            bc2 = max(1.0 - beta2 ** step, 1e-8)
            L_hat = L / bc2
            R_hat = R / bc2
        else:
            L_hat = L
            R_hat = R

        eig_l, U = self._safe_eigh(L_hat, jitter=eps)
        eig_r, V = self._safe_eigh(R_hat, jitter=eps)

        eig_l = eig_l.clamp_min(0.0)
        eig_r = eig_r.clamp_min(0.0)

        if power == 0.0:
            a = torch.ones_like(eig_l)
            b = torch.ones_like(eig_r)
        else:
            a = eig_l.pow(power)
            b = eig_r.pow(power)

        state["U"] = U
        state["V"] = V
        state["a"] = a
        state["b"] = b
        state["factor_step"] = step

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("SOLVENT does not support sparse gradients.")

                # Robustness: skip parameters with non-finite gradients.
                if not torch.isfinite(grad).all():
                    continue

                state = self.state[p]

                # State initialization.
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                    state["matrix_enabled"] = False
                    state["factor_step"] = 0

                    if group["use_matrix"] and p.dim() >= 2:
                        rows = p.shape[0]
                        cols = p.numel() // rows

                        if (
                            rows > 0
                            and cols > 0
                            and max(rows, cols) <= group["max_full_dim"]
                        ):
                            state["matrix_enabled"] = True
                            state["L"] = torch.zeros(
                                (rows, rows), device=p.device, dtype=torch.float32
                            )
                            state["R"] = torch.zeros(
                                (cols, cols), device=p.device, dtype=torch.float32
                            )
                            state["U"] = None
                            state["V"] = None
                            state["a"] = None
                            state["b"] = None

                state["step"] += 1
                step = state["step"]

                # Decoupled weight decay.
                if group["weight_decay"] != 0.0:
                    p.mul_(1.0 - lr * group["weight_decay"])

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # Momentum and Adam-style second moment for fallback.
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                used_solvent = False

                if state["matrix_enabled"]:
                    rows = p.shape[0]

                    # Update Kronecker factor estimates using the raw gradient.
                    G = grad.detach().reshape(rows, -1).to(torch.float32)

                    state["L"].mul_(beta2).add_(G @ G.t(), alpha=1.0 - beta2)
                    state["R"].mul_(beta2).add_(G.t() @ G, alpha=1.0 - beta2)

                    # Periodically refresh eigenbases.
                    due = (
                        state["factor_step"] == 0
                        or (step - state["factor_step"]) >= group["factor_update_every"]
                    )

                    if step >= group["factor_warmup"] and due:
                        self._update_factors(state, group, step)

                    if state["U"] is not None:
                        U = state["U"]
                        V = state["V"]
                        a = state["a"]
                        b = state["b"]

                        # Use bias-corrected momentum as the right-hand side.
                        M = exp_avg.reshape(rows, -1).to(torch.float32)
                        if group["bias_correction"]:
                            bc1 = max(1.0 - beta1 ** step, 1e-8)
                            M = M / bc1

                        # Project into Kronecker eigenbasis.
                        C = U.t() @ M @ V

                        # Adaptive damping scale.
                        a_pos = a[a > 0]
                        b_pos = b[b > 0]

                        if a_pos.numel() > 0:
                            a_scale = a_pos.mean()
                        else:
                            a_scale = torch.tensor(1.0, device=a.device, dtype=a.dtype)

                        if b_pos.numel() > 0:
                            b_scale = b_pos.mean()
                        else:
                            b_scale = torch.tensor(1.0, device=b.device, dtype=b.dtype)

                        scale = (a_scale * b_scale).clamp_min(eps)
                        lam = group["damping"] * scale + group["damping_floor"]

                        # Resolvent denominator: a_i b_j + lambda.
                        denom = torch.outer(a, b) + lam
                        denom = denom.clamp_min(eps)

                        X = C / denom

                        # Back to parameter space.
                        delta = U @ X @ V.t()
                        delta = delta.reshape(p.shape).to(p.dtype)

                        if torch.isfinite(delta).all():
                            # Simple trust-region clipping.
                            if group["max_rel_step"] is not None and group["max_rel_step"] > 0:
                                p_norm = float(
                                    torch.linalg.vector_norm(p.detach().float()).item()
                                )
                                d_norm = float(
                                    torch.linalg.vector_norm(delta.float()).item()
                                )
                                max_norm = group["max_rel_step"] * max(p_norm, 1.0)

                                if d_norm > max_norm:
                                    delta.mul_(max_norm / (d_norm + eps))

                            p.add_(delta, alpha=-lr)
                            used_solvent = True

                # Fallback: AdamW-style update for vectors, small tensors,
                # warmup, or if SOLVENT update failed.
                if not used_solvent:
                    m_hat = exp_avg
                    v_hat = exp_avg_sq

                    if group["bias_correction"]:
                        bc1 = max(1.0 - beta1 ** step, 1e-8)
                        bc2 = max(1.0 - beta2 ** step, 1e-8)
                        m_hat = exp_avg / bc1
                        v_hat = exp_avg_sq / bc2

                    denom = v_hat.sqrt().add_(eps)
                    p.addcdiv_(m_hat, denom, value=-lr)

        return loss