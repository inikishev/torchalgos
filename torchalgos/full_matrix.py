"""Full-Matrix Adagrad and Adam."""

from collections.abc import Iterable
import torch


def regularize_eigh(
    L: torch.Tensor,
    Q: torch.Tensor,
    tol: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    finite = L.isfinite()
    if finite.any():
        L = L[finite]
        Q = Q[:, finite]
    else:
        return None, None

    L_max = L[-1] # L is sorted in ascending order

    if tol is not None:
        indices = L > tol * L_max
        L = L[indices]
        Q = Q[:, indices]

    return L, Q

def vec_to_tensors(vec: torch.Tensor, reference: Iterable[torch.Tensor]) -> list[torch.Tensor]:
    tensors = []
    cur = 0
    for r in reference:
        numel = r.numel()
        tensors.append(vec[cur:cur+numel].reshape_as(r))
        cur += numel
    return tensors


def compute_preconditioner(accumulator: torch.Tensor, reg: float, eigval_tol: float):
    I = torch.eye(accumulator.size(0), device=accumulator.device, dtype=accumulator.dtype)
    L, Q = torch.linalg.eigh(accumulator + I * reg) # pylint:disable=not-callable
    L, Q = regularize_eigh(L, Q, tol=eigval_tol)
    if L is not None and Q is not None:
        return Q @ L.rsqrt().diag_embed() @ Q.T

    # diagonal fallback
    return accumulator.diagonal().add(reg).rsqrt().diag_embed()


class FullMatrixAdagrad(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        lr_decay: float = 0,
        weight_decay: float = 0,
        reg: float = 1e-10,
        eigval_tol: float = 1e-6,
        precond_freq: int = 1,
    ):
        defaults = dict(
            lr = lr,
            lr_decay = lr_decay,
            weight_decay = weight_decay,
            reg = reg,
            eigval_tol = eigval_tol,
            precond_freq = precond_freq,
        )
        super().__init__(params, defaults)


    @torch.no_grad
    def step(self, closure=None): # pyright:ignore[reportIncompatibleMethodOverride]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            params_list = [p for p in group["params"] if p.requires_grad]
            state = self.state[params_list[0]]

            # Gather flat gradients
            grads_list = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params_list]
            g = torch.cat([g.ravel() for g in grads_list])

            # Initialize the state
            if "accumulator" not in state:
                state["accumulator"] = torch.eye(g.numel(), device=g.device, dtype=g.dtype)
                state["step"] = 0

            # Update accumulator
            state["accumulator"].add_(torch.outer(g, g))

            # Update preconditioner
            if state["step"] % group["precond_freq"] == 0:
                state["preconditioner"] = compute_preconditioner(
                    accumulator = state["accumulator"],
                    reg = group["reg"],
                    eigval_tol = group["eigval_tol"],
                )

            # Update parameters
            updates = vec_to_tensors(g @ state["preconditioner"], params_list)

            if group["weight_decay"] > 0.0:
                torch._foreach_add_(
                    updates,
                    torch._foreach_mul(group["params"], group["weight_decay"])
                )

            clr = group["lr"] / (1 + state["step"] * group["lr_decay"])
            torch._foreach_sub_(params_list, updates, alpha=clr)

            state["step"] += 1

        return loss

class FullMatrixAdam(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps = 1e-8,
        weight_decay: float = 0,
        reg: float = 1e-10,
        eigval_tol: float = 1e-6,
        precond_freq: int = 1,
    ):
        defaults = dict(
            lr = lr,
            betas = betas,
            eps = eps,
            weight_decay = weight_decay,
            reg = reg,
            eigval_tol = eigval_tol,
            precond_freq = precond_freq,
        )
        super().__init__(params, defaults)


    @torch.no_grad
    def step(self, closure=None): # pyright:ignore[reportIncompatibleMethodOverride]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            params_list = [p for p in group["params"] if p.requires_grad]
            state = self.state[params_list[0]]

            # Gather flat gradients
            grads_list = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params_list]
            g = torch.cat([g.ravel() for g in grads_list])

            # Initialize the state
            if "accumulator" not in state:
                state["exp_avg"] = torch.zeros_like(g)
                state["accumulator"] = torch.eye(g.numel(), device=g.device, dtype=g.dtype)
                state["step"] = 0

            # Update accumulator
            beta1, beta2 = group["betas"]
            state["exp_avg"].lerp_(g, 1-beta1)
            state["accumulator"].lerp_(torch.outer(g, g), 1-beta2)

            # Update preconditioner
            if state["step"] % group["precond_freq"] == 0:
                state["preconditioner"] = compute_preconditioner(
                    accumulator = state["accumulator"] / (1.0 - beta2 ** (state["step"] + 1)),
                    reg = group["reg"],
                    eigval_tol = group["eigval_tol"],
                )

            # Update parameters
            exp_avg = state["exp_avg"] / (1.0 - beta1 ** (state["step"] + 1))
            updates = vec_to_tensors(exp_avg @ state["preconditioner"], params_list)

            if group["weight_decay"] > 0.0:
                torch._foreach_add_(
                    updates,
                    torch._foreach_mul(group["params"], group["weight_decay"])
                )

            torch._foreach_sub_(params_list, updates, alpha=group["lr"])

            state["step"] += 1

        return loss
