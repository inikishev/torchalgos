'''qwen 3.8 max'''
import torch
from torch.optim.optimizer import Optimizer
from typing import Optional


class PolarCancel(Optimizer):
    """
    PolarCancel: Preconditioned Polar Cancellation optimizer.

    For matrix-shaped parameters, it maintains momentum and a cheap
    row/column second-moment estimate, preconditioned by diagonal
    left/right curvature, then applies a Newton-Schulz polar transform.

    For non-matrix parameters, it falls back to AdamW, signed Adam,
    or plain momentum.

    Args:
        params: iterable of parameters or param groups.
        lr: learning rate.
        betas: coefficients for momentum and second-moment EMA.
        eps: numerical epsilon.
        weight_decay: decoupled weight decay.
        ns_steps: number of Newton-Schulz iterations.
        ns_coeffs: Newton-Schulz coefficients:
            "muon": accelerated coefficients used in Muon-style code.
            "quintic": exact-fixed-point quintic coefficients.
            or a length-3 tuple/list.
        second_moment:
            "rowcol": cheap row/column second moments for matrix params.
            "elementwise": full Adam-style second moment for matrix params.
            "none": no second moment; closer to Muon.
        post_scale:
            If True, maps the polarized direction back through the
            diagonal preconditioner.
        normalize_update:
            If True, normalizes update RMS to 1 before applying lr.
        vector_mode:
            Fallback for non-matrix params:
            "adamw", "sign", or "none".
        matrix_min_dim:
            Minimum tensor dim to consider matrix mode.
        max_polar_dim:
            If min(m,n) is larger than this, fall back to vector mode.
            Prevents enormous Newton-Schulz Gram matrices.
        max_aspect:
            If max(m,n)/min(m,n) is larger than this, fall back to vector
            mode. Useful to avoid applying polar updates to embeddings.
        update_clip:
            If > 0, elementwise clamp update after normalization.
        work_dtype:
            Dtype used inside Newton-Schulz. If None, float32 is used for
            half/bfloat16 states, otherwise the state dtype is kept.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.95, 0.98),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        ns_coeffs: str = "muon",
        second_moment: str = "rowcol",
        post_scale: bool = True,
        normalize_update: bool = True,
        vector_mode: str = "adamw",
        matrix_min_dim: int = 2,
        max_polar_dim: Optional[int] = 4096,
        max_aspect: Optional[float] = 16.0,
        update_clip: float = 0.0,
        work_dtype: Optional[torch.dtype] = None,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if second_moment not in {"rowcol", "elementwise", "none"}:
            raise ValueError(f"Invalid second_moment: {second_moment}")
        if vector_mode not in {"adamw", "sign", "none"}:
            raise ValueError(f"Invalid vector_mode: {vector_mode}")
        if ns_steps < 0:
            raise ValueError(f"Invalid ns_steps: {ns_steps}")
        if update_clip < 0.0:
            raise ValueError(f"Invalid update_clip: {update_clip}")

        if ns_coeffs == "muon":
            coeffs = (3.4445, -4.7750, 2.0315)
        elif ns_coeffs == "quintic":
            coeffs = (15.0 / 8.0, -5.0 / 4.0, 3.0 / 8.0)
        else:
            coeffs = tuple(ns_coeffs)
            if len(coeffs) != 3:
                raise ValueError("ns_coeffs must be 'muon', 'quintic', or length-3 tuple.")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            ns_coeffs=coeffs,
            second_moment=second_moment,
            post_scale=post_scale,
            normalize_update=normalize_update,
            vector_mode=vector_mode,
            matrix_min_dim=matrix_min_dim,
            max_polar_dim=max_polar_dim,
            max_aspect=max_aspect,
            update_clip=update_clip,
            work_dtype=work_dtype,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _state_dtype(dtype: torch.dtype) -> torch.dtype:
        # Keep master states in float32 for half/bfloat16 parameters.
        if dtype in (torch.float16, torch.bfloat16):
            return torch.float32
        return dtype

    @staticmethod
    def _newton_schulz_polar(
        X: torch.Tensor,
        steps: int,
        coeffs,
        work_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Approximate polar factor / zero-power transform via Newton-Schulz.

        Input:
            X: 2D tensor.
        Output:
            Same shape as X, approximately polarized.
        """
        orig_dtype = X.dtype

        if work_dtype is not None:
            X = X.to(work_dtype)
        elif X.dtype in (torch.float16, torch.bfloat16):
            X = X.float()

        m, n = X.shape
        if m == 0 or n == 0:
            return X.to(orig_dtype)

        transpose = m > n
        if transpose:
            X = X.t()

        # Now X has shape r x c with r <= c.
        norm = X.norm()
        if norm == 0:
            return torch.zeros_like(X).to(orig_dtype)

        # Keep singular values bounded for the polynomial iteration.
        X = X / (norm + 1e-7)

        a, b, c = coeffs

        for _ in range(steps):
            A = X @ X.t()
            # X <- a X + b A X + c A^2 X
            X = a * X + (b * A + c * (A @ A)) @ X

        if transpose:
            X = X.t()

        return X.to(orig_dtype)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("PolarCancel does not support sparse gradients.")

                # Decoupled weight decay.
                if wd != 0.0:
                    p.data.mul_(1.0 - lr * wd)

                state = self.state[p]

                # State initialization.
                if len(state) == 0:
                    state["step"] = 0
                    state_dtype = self._state_dtype(p.dtype)

                    is_matrix = False
                    m = n = None

                    if p.dim() >= group["matrix_min_dim"] and p.numel() > 1:
                        m = p.shape[0]
                        n = p.numel() // m

                        if m > 0 and n > 0:
                            small_dim = min(m, n)
                            aspect = max(m, n) / float(small_dim)

                            dim_ok = small_dim > 1
                            max_dim_ok = (
                                group["max_polar_dim"] is None
                                or small_dim <= group["max_polar_dim"]
                            )
                            aspect_ok = (
                                group["max_aspect"] is None
                                or aspect <= group["max_aspect"]
                            )

                            is_matrix = dim_ok and max_dim_ok and aspect_ok

                    state["is_matrix"] = is_matrix

                    if is_matrix:
                        state["m"] = m
                        state["n"] = n
                        state["M"] = torch.zeros(
                            (m, n), dtype=state_dtype, device=p.device
                        )

                        if group["second_moment"] == "elementwise":
                            state["V"] = torch.zeros(
                                (m, n), dtype=state_dtype, device=p.device
                            )
                        elif group["second_moment"] == "rowcol":
                            state["V_row"] = torch.zeros(
                                m, dtype=state_dtype, device=p.device
                            )
                            state["V_col"] = torch.zeros(
                                n, dtype=state_dtype, device=p.device
                            )
                    else:
                        state["M"] = torch.zeros_like(
                            p, dtype=state_dtype, device=p.device
                        )
                        if group["vector_mode"] != "none":
                            state["V"] = torch.zeros_like(
                                p, dtype=state_dtype, device=p.device
                            )

                state["step"] += 1
                step = state["step"]

                bc1 = 1.0 - beta1 ** step
                bc2 = 1.0 - beta2 ** step

                state_dtype = state["M"].dtype

                if state["is_matrix"]:
                    m = state["m"]
                    n = state["n"]

                    g = grad.detach().to(state_dtype).reshape(m, n)
                    M = state["M"]

                    # Momentum EMA.
                    M.mul_(beta1).add_(g, alpha=1.0 - beta1)
                    M_hat = M / bc1

                    sm = group["second_moment"]

                    if sm == "elementwise":
                        V = state["V"]
                        V.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                        V_hat = V / bc2

                        scale = torch.rsqrt(V_hat + eps)
                        Y = M_hat * scale

                        Z = self._newton_schulz_polar(
                            Y,
                            group["ns_steps"],
                            group["ns_coeffs"],
                            group["work_dtype"],
                        )

                        if group["post_scale"]:
                            U = Z * scale
                        else:
                            U = Z

                    elif sm == "rowcol":
                        V_row = state["V_row"]
                        V_col = state["V_col"]

                        g2 = g * g
                        row_avg = g2.mean(dim=1)
                        col_avg = g2.mean(dim=0)

                        V_row.mul_(beta2).add_(row_avg, alpha=1.0 - beta2)
                        V_col.mul_(beta2).add_(col_avg, alpha=1.0 - beta2)

                        V_row_hat = V_row / bc2
                        V_col_hat = V_col / bc2

                        row_scale = torch.rsqrt(V_row_hat + eps)
                        col_scale = torch.rsqrt(V_col_hat + eps)

                        # Preconditioned momentum.
                        Y = M_hat * row_scale.unsqueeze(1) * col_scale.unsqueeze(0)

                        # Polar cancellation.
                        Z = self._newton_schulz_polar(
                            Y,
                            group["ns_steps"],
                            group["ns_coeffs"],
                            group["work_dtype"],
                        )

                        if group["post_scale"]:
                            U = Z * row_scale.unsqueeze(1) * col_scale.unsqueeze(0)
                        else:
                            U = Z

                    else:
                        # No second moment: closer to Muon, but with bias correction
                        # and RMS normalization.
                        Y = M_hat
                        U = self._newton_schulz_polar(
                            Y,
                            group["ns_steps"],
                            group["ns_coeffs"],
                            group["work_dtype"],
                        )

                    if group["normalize_update"]:
                        rms = torch.sqrt(torch.mean(U * U))
                        if rms > eps:
                            U = U / rms

                    if group["update_clip"] > 0.0:
                        U = U.clamp_(-group["update_clip"], group["update_clip"])

                    p.data.add_(U.reshape(p.shape).to(p.dtype), alpha=-lr)

                else:
                    # Vector fallback.
                    g = grad.detach().to(state_dtype)
                    M = state["M"]
                    M.mul_(beta1).add_(g, alpha=1.0 - beta1)

                    if group["vector_mode"] == "none":
                        U = M / bc1
                        if group["normalize_update"]:
                            rms = torch.sqrt(torch.mean(U * U))
                            if rms > eps:
                                U = U / rms
                        if group["update_clip"] > 0.0:
                            U = U.clamp_(-group["update_clip"], group["update_clip"])
                        p.data.add_(U.reshape(p.shape).to(p.dtype), alpha=-lr)

                    else:
                        V = state["V"]
                        V.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                        M_hat = M / bc1
                        V_hat = V / bc2

                        if group["vector_mode"] == "adamw":
                            denom = torch.sqrt(V_hat) + eps
                            U = M_hat / denom
                            p.data.add_(U.reshape(p.shape).to(p.dtype), alpha=-lr)

                        elif group["vector_mode"] == "sign":
                            denom = torch.sqrt(V_hat) + eps
                            U = torch.tanh(M_hat / denom)

                            if group["normalize_update"]:
                                rms = torch.sqrt(torch.mean(U * U))
                                if rms > eps:
                                    U = U / rms

                            if group["update_clip"] > 0.0:
                                U = U.clamp_(-group["update_clip"], group["update_clip"])

                            p.data.add_(U.reshape(p.shape).to(p.dtype), alpha=-lr)

        return loss