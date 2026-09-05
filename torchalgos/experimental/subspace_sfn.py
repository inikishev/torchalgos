from collections import deque
from collections.abc import Iterable, Sequence

import torch


def regularize_eigh(
    L: torch.Tensor,
    Q: torch.Tensor,
    truncate: int | float | None = None,
    tol: float | None = None,
    damping: float = 0,
    rdamping: float = 0,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    """Applies regularization to eigendecomposition. Returns ``(L, Q)``.

    Args:
        L (torch.Tensor): eigenvalues, shape ``(rank,)``.
        Q (torch.Tensor): eigenvectors, shape ``(n, rank)``.
        truncate (int | None, optional):
            keeps top ``truncate`` eigenvalues. Defaults to None.
        tol (float | None, optional):
            all eigenvalues smaller than largest eigenvalue times ``tol`` are removed. Defaults to None.
        damping (float | None, optional): scalar added to eigenvalues. Defaults to 0.
        rdamping (float | None, optional): scalar multiplied by largest eigenvalue and added to eigenvalues. Defaults to 0.
    """
    # remove non-finite eigenvalues and small
    finite = L.isfinite()
    if tol is not None:
        finite = finite.logical_and(L > tol * L[-1]) # L is ordered in ascending order

    if finite.any():
        L = L[finite]
        Q = Q[:, finite]
    else:
        return None, None

    # truncate to rank
    if truncate is not None:
        if isinstance(truncate, float):
            truncate = max(round(finite.numel() * truncate), 1)

        L = L[-truncate:]
        Q = Q[:, -truncate:]

    # damping
    d = damping + rdamping * L[-1]
    if d != 0:
        L += d

    return L, Q

def get_L_Q(M: torch.Tensor, tol: float | None, damping: float = 0, truncate: int | float | None = None):
    """
    do:
    ```python
    M = torch.stack(history, dim=1)
    ```

    returns L ``(rank, )``, Q ``(ndim, rank)``.
    """

    MtM = M.T @ M
    if damping != 0:
        MtM.add_(torch.eye(MtM.size(0), device=MtM.device, dtype=MtM.dtype).mul_(damping))

    try:
        L, Q = torch.linalg.eigh(MtM) # pylint:disable=not-callable

        finite = L.isfinite()
        if tol is not None and tol != 0:
            finite = finite.logical_and(L > tol * L[-1])

        if finite.any():
            L = L[finite]
            Q = Q[:, finite]

        else:
            return None, None

        # L is ordered in ascending order
        if truncate is not None:
            L = L[-truncate:]
            Q = Q[:, -truncate:]

        U = (M @ Q) * L.rsqrt()
        return L, U

    except torch.linalg.LinAlgError:
        return None, None

def vec_to_tensors(vec: torch.Tensor, reference: Iterable[torch.Tensor]) -> list[torch.Tensor]:
    tensors = []
    cur = 0
    for r in reference:
        numel = r.numel()
        tensors.append(vec[cur:cur+numel].reshape_as(r))
        cur += numel
    return tensors


def flatten_jacobian(jacs: Sequence[torch.Tensor]) -> torch.Tensor:
    if not jacs:
        return torch.empty(0, 0)

    n_out = jacs[0].shape[0]
    return torch.cat([j.reshape(n_out, -1) for j in jacs], dim=1)

def _clip_magnitude(x: torch.Tensor, min: float):
    x = x.clone()
    x[x.abs() < min] = min
    return x

class SubspaceSFN(torch.optim.Optimizer):
    """Run saddle-free Newton with momentum in a subspace spanned by past ``history_size`` gradients.

    Args:
        params: params
        lr (float, optional): learning rate. Defaults to 1/3.
        history_size (int, optional): history size. Defaults to 10.
        g_beta (float | None, optional):
            optional beta for exponential moving average of gradients. Defaults to None.
        g_proj_beta (float | None, optional):
            optional beta for exponential moving average of projected gradients. Defaults to None.
        sub_tol (float, optional):
            regularizes projection matrix by removing eigenvalues smaller than largest
            eigenvalue times this value. Defaults to 1e-5.
        sub_damping (float, optional):
            scale of identity matrix added to covariance matrix. Defaults to 1e-8.
        sub_truncate (int | float | None, optional):
            truncates projection matrix to k largest eigenvalues. Defaults to None.
        H_tol (float, optional):
            regularizes hessian by removing eigenvalues smaller than largest
            eigenvalue times this value. Defaults to 1e-5.
        H_damping (float, optional):
            scale of identity matrix added to hessian. Defaults to 1e-8.. Defaults to 1e-8.
        H_truncate (int | float | None, optional):
            truncates hessian to k largest eigenvalues. Float means percentage
            (e.g. by default this keeps 90% largest eigenvalues). Defaults to 0.9.
        H_inv_beta (float | None, optional):
            optional beta for exponential moving average of regularized hessian inverse. Defaults to None.
        map_beta (float | None, optional):
            tracks exponential moving average of a diagonal matrix which
            maps gradient to preconditioned gradient, or its inverse. Defaults to None.
        map_clip (float, optional):
            clips denominator when computing diagonal mapping matrix. Defaults to 1e-2.
        map_damping (float, optional):
            value added to diagonal mapping matrix before applying it to gradient. Defaults to 1e-4.
        inverse_map (bool, optional):
            whether to track inverse of the mapping matrix. Defaults to True.
        mdev_clip (float | None, optional):
            clips mean absolute deviation of the update. Defaults to 1.
        value_clip (float | None, optional):
            clips update value. Defaults to 10.
        update_freq (int, optional):
            hessian update frequency. Defaults to 1.
    """
    def __init__(
        self,
        params,
        lr: float = 1,
        history_size: int = 10,

        g_beta: float | None = None,
        g_proj_beta: float | None = 0.95,
        cautious: bool = True,
        g_proj_clip: float | None = None,

        sub_tol: float = 1e-4,
        sub_damping: float = 1e-8,
        sub_truncate: int | float | None = None,

        H_tol: float = 1e-4,
        H_damping: float = 0.2,
        H_truncate: int | float | None = None,
        H_inv_beta: float | None = 0.9,
        H_remove_negative: bool = True,

        map_beta: float | None = None,
        map_clip: float = 1e-2,
        map_damping: float = 1e-4,
        inverse_map: bool = True,

        mdev_clip: float | None = 0.3,
        value_clip: float | None = 0.3,

        update_freq: int = 1,
    ):
        super().__init__(params, {})

        self.lr = lr
        self.history_size = history_size

        self.g_beta = g_beta
        self.g_proj_beta = g_proj_beta
        self.cautious = cautious
        self.g_proj_clip = g_proj_clip

        self.sub_tol = sub_tol
        self.sub_damping = sub_damping
        self.sub_truncate = sub_truncate

        self.H_tol = H_tol
        self.H_damping = H_damping
        self.H_truncate = H_truncate
        self.H_inv_beta = H_inv_beta
        self.H_remove_negative = H_remove_negative

        self.map_beta = map_beta
        self.map_clip = map_clip
        self.map_damping = map_damping
        self.inverse_map = inverse_map

        self.mdev_clip = mdev_clip
        self.value_clip = value_clip

        self.update_freq = update_freq

        self.exp_avg = None
        self.exp_avg_proj = None
        self.H = None
        self.H_inv = None
        self.g_map = None
        self.history = deque(maxlen=history_size)
        self.current_step = 0
        self.num_H_inv_updates = 0

        self.Q_g = None

    @torch.no_grad
    def step(self, closure): # pyright:ignore[reportIncompatibleMethodOverride] # pylint:disable=signature-differs

        with torch.enable_grad():
            loss = closure(False)
            p_list = [p for g in self.param_groups for p in g["params"] if p.requires_grad]
            g_list = torch.autograd.grad(loss, p_list, create_graph=True, allow_unused=True, materialize_grads=True)
            g = torch.cat([t.ravel() for t in g_list])

        self.history.append(g)

        # PCA
        L_g, Q_g = get_L_Q(torch.stack(tuple(self.history), -1), tol=self.sub_tol, damping=self.sub_damping, truncate=self.sub_truncate)

        if self.Q_g is None: # first step Q can't be None because all it does is normalizes the gradient
            assert Q_g is not None
            self.Q_g = Q_g

        if Q_g is not None:

            # reproject H_inv and exponential average
            C = Q_g.T @ self.Q_g

            if self.exp_avg_proj is not None:
                self.exp_avg_proj = C @ self.exp_avg_proj

            if self.H_inv is not None:
                self.H_inv = C @ self.H_inv @ C.T

            self.Q_g = Q_g

        Q_g = self.Q_g

        # project
        if self.g_beta is not None:
            if self.exp_avg is None:
                self.exp_avg = torch.zeros_like(g)
            self.exp_avg.lerp_(g, 1-self.g_beta)
            exp_avg = self.exp_avg
            if self.cautious:
                exp_avg = exp_avg * ((exp_avg.sign() * g.sign()) > 0)
            g_proj = Q_g.T @ exp_avg

        else:
            g_proj = Q_g.T @ g

        # run newton
        if self.current_step % self.update_freq == 0:

            I_H = torch.eye(g_proj.numel(), device=g_proj.device, dtype=g_proj.dtype)

            with torch.enable_grad():
                H_unproj = flatten_jacobian(torch.autograd.grad(g, p_list, (Q_g @ I_H).T, is_grads_batched=True))

            H = (Q_g.T @ H_unproj.T).T

            if self.H_damping != 0:
                H = H.add(I_H, alpha=self.H_damping)

            L_H, Q_H = torch.linalg.eigh(H) # pylint:disable=not-callable

            if not self.H_remove_negative: L_H = L_H.abs()
            L_H, Q_H = regularize_eigh(L_H, Q_H, truncate=self.H_truncate, tol=self.H_tol)

            if L_H is not None and Q_H is not None:
                self.num_H_inv_updates += 1

                H_inv = Q_H @ (L_H.reciprocal().diag_embed()) @ Q_H.T
                if self.H_inv is None:
                    self.H_inv = torch.zeros_like(H_inv)

                if self.H_inv_beta is not None:
                    self.H_inv.lerp_(H_inv, 1-self.H_inv_beta)
                    H_inv = self.H_inv / (1 - self.H_inv_beta ** self.num_H_inv_updates)

                else:
                    self.H_inv = H_inv

        H_inv = self.H_inv
        if H_inv is None:
            dir = self.Q_g  @ g_proj.clip(-0.1, 0.1)
            torch._foreach_sub_(p_list, vec_to_tensors(dir, p_list))
            return loss

        if self.g_proj_beta is not None:
            if self.exp_avg_proj is None:
                self.exp_avg_proj = torch.zeros_like(g_proj)
            self.exp_avg_proj.lerp_(g_proj, 1-self.g_proj_beta)

            exp_avg_proj = self.exp_avg_proj
            if self.cautious:
                exp_avg_proj = exp_avg_proj * ((exp_avg_proj.sign() * g_proj.sign()) > 0)

            g_proj = exp_avg_proj

        if self.g_proj_clip is not None:
            g_proj = g_proj.clip(-self.g_proj_clip, self.g_proj_clip)

        dir_proj = H_inv @ g_proj
        dir = self.Q_g @ dir_proj

        # lerp diagonal map
        if self.map_beta is not None:

            if self.inverse_map:
                g_map = dir / _clip_magnitude(g, self.map_clip)
            else:
                g_map = g / _clip_magnitude(dir, self.map_clip)

            if self.g_map is None:
                self.g_map = torch.zeros_like(g_map)

            self.g_map.lerp_(g_map, weight=1-self.map_beta)
            g_map = self.g_map / (1 - self.map_beta ** (self.current_step + 1))

            g_map = g_map.abs() + self.map_damping

            # apply map to grad or momentum
            if self.exp_avg is not None:
                g = self.exp_avg

            if self.inverse_map:
                dir = g * g_map

            else:
                dir = g / g_map

        if self.mdev_clip is not None:
            dir_mdev = dir.abs().mean()
            if dir_mdev > self.mdev_clip:
                dir *= self.mdev_clip / dir_mdev

        if self.value_clip is not None:
            dir = dir.clip(-self.value_clip, self.value_clip)

        torch._foreach_sub_(p_list, vec_to_tensors(dir, p_list), alpha=self.lr)
        self.current_step += 1
        return loss