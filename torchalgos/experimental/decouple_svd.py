from collections.abc import Callable, Sequence
import torch

def svd_init(p: torch.Tensor):
    assert p.ndim >= 2
    dtype = p.dtype
    U, S, Vh = torch.linalg.svd(p.to(torch.float64), full_matrices=False) # pylint:disable=not-callable
    # U - m * k
    # S - k * k
    # V^H - k * n
    # where k = min(m, n)

    # we need to mark requires_grad as True so that the optimizer doesn't skip them
    return U.to(dtype=dtype), S.to(dtype=dtype).requires_grad_(True), Vh.to(dtype=dtype).requires_grad_(True)

def ortho_mse(A: torch.Tensor):
    *b, m, n = A.shape
    if n > m:
        A = A.mH

    AtA = A.mH @ A
    I = torch.eye(AtA.size(-1), device=A.device, dtype=A.dtype)
    return torch.nn.functional.mse_loss(AtA, I)

def ortho_grad(A: torch.Tensor):
    """gradient of `mse(A.T @ A, I)`, A is rotated so that A.T @ A is smaller"""
    *b, m, n = A.shape
    transpose = n > m
    if transpose:
        A = A.mH
        m,n = n,m

    scalar = 4 / n**2
    G_unsc = A @ (A.mH @ A - torch.eye(n, device=A.device, dtype=A.dtype))

    if transpose:
        G_unsc = G_unsc.mH

    return scalar * G_unsc

class DecoupleSVD(torch.optim.Optimizer):
    """Decouple matrix parameters as U S V^H where U and V^H are updated to be more orthogonal,
    and non-matrix as magnitude * direction.

    Recommended - use POGO (Proximal One-step Geometric Orthoptimizer) on U and Vh, ortho_opt=None and w_ortho=0

    Args:
        params: parameters
        opt_U: optimizer for the U factor (near-orthogonal matrix).
        opt_S: optimizer for the S factor (scales vector).
        opt_Vh: optimizer for the Vh factor (near-orthogonal matrix).
        opt_dir: optimizer for directions of non-matrix parameters.
        opt_magn: optimizer for magnitudes of non-matrix parameters.
        opt_coupled: optimizer for non-decoupled parameters.
        opt_ortho: optimizer for updating U and Vh to be orthogonal, None to disable. Defaults to None.
        w_ortho: multiplier for orthogonality gradient added to U and Vh gradients. Defaults to 0.
        eps: epsilon for normalization, if None, uses machine epsilon for dtype of the param. Defaults to None.
        log_stats: whether to log orthogonality and mangitudes.
    """
    def __init__(
        self,
        params,
        opt_U: Callable[[list[torch.Tensor]], torch.optim.Optimizer],
        opt_S: Callable[[list[torch.Tensor]], torch.optim.Optimizer],
        opt_Vh: Callable[[list[torch.Tensor]], torch.optim.Optimizer],
        opt_dir: Callable[[list[torch.Tensor]], torch.optim.Optimizer],
        opt_magn: Callable[[list[torch.Tensor]], torch.optim.Optimizer],
        opt_coupled: Callable[[list[torch.Tensor]], torch.optim.Optimizer] | None = None,
        opt_ortho: Callable[[list[torch.Tensor]], torch.optim.Optimizer] | None = None,
        w_ortho: float = 0,
        eps: float | None = None,
        decouple: bool = True,
        force_md_decouple: bool = False,
        log_stats: bool = False,
    ):
        defaults = dict(decouple=decouple, force_md_decouple=force_md_decouple, eps=eps, w_ortho=w_ortho, log_stats=log_stats)

        self.opt_U_fn = opt_U
        self.opt_S_fn = opt_S
        self.opt_Vh_fn = opt_Vh
        self.opt_dir_fn = opt_dir
        self.opt_magn_fn = opt_magn
        self.opt_coupled_fn = opt_coupled
        self.opt_ortho_fn = opt_ortho

        self.opt_U = None
        self.opt_S = None
        self.opt_Vh = None
        self.opt_dir = None
        self.opt_magn = None
        self.opt_coupled = None
        self.opt_ortho = None

        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None): # pyright:ignore[reportIncompatibleMethodOverride]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Initialize
        svd_params = []
        U_list = []
        S_list = []
        Vh_list = []

        md_params = [] # for non-matrix parameters where we decouple magnitude and direction
        dir_list = []
        magn_list = []

        coupled_params = [] # for non-decoupled parameters

        # Gather params
        for group in self.param_groups:
            for p in group["params"]:

                if p.grad is None:
                    continue

                if not group["decouple"]:
                    coupled_params.append(p)
                    continue

                state = self.state[p]
                eps = group["eps"]
                if eps is None: eps = torch.finfo(p.dtype).eps
                state["eps"] = eps

                # Non-matrix parameter
                if group["force_md_decouple"] or sum(1 for dim in p.size() if dim > 1) < 2:
                    md_params.append(p)

                    if "dir" not in state:
                        state["magn"] = torch.linalg.vector_norm(p).clip(min=eps).requires_grad_(True) # pylint:disable=not-callable
                        state["dir"] = (p / state["magn"]).requires_grad_(True)

                    dir_list.append(state["dir"])
                    magn_list.append(state["magn"])

                    # gradients
                    state["dir"].grad = p.grad * state["magn"]
                    state["magn"].grad = (p.grad * state["dir"]).sum()

                    if group["log_stats"]:
                        if "magn_history" not in state: state["magn_history"] = []
                        state["magn_history"].append(state["magn"].item())

                else:
                    svd_params.append(p)

                    # Matrix parameter
                    if "U" not in state:
                        state["U"], state["S"], state["Vh"] = svd_init(p)

                    U, S, Vh = state["U"], state["S"], state["Vh"]
                    U_list.append(U) # *, m, k
                    S_list.append(S) # *, k
                    Vh_list.append(Vh) # * k, n

                    # gradients
                    S_col = S.unsqueeze(-1) # *, k, 1
                    U.grad = p.grad @ (S_col * Vh).mH
                    S.grad = ((U.mH @ p.grad) * Vh).sum(-1)
                    Vh.grad = (S_col * U.mH) @ p.grad

                    # inject orthogonality penalty
                    if group["w_ortho"] != 0:
                        U.grad += ortho_grad(U) * group["w_ortho"]
                        Vh.grad += ortho_grad(Vh) * group["w_ortho"]

                    if group["log_stats"]:
                        if "U_ortho_history" not in state:
                            state["U_ortho_history"] = []
                            state["Vh_ortho_history"] = []

                        state["U_ortho_history"].append(ortho_mse(U).item())
                        state["Vh_ortho_history"].append(ortho_mse(Vh).item())

        # initialize optimizers
        if self.opt_U is None and len(svd_params) > 0:
            self.opt_U = self.opt_U_fn(U_list)
            self.opt_S = self.opt_S_fn(S_list)
            self.opt_Vh = self.opt_Vh_fn(Vh_list)

            if self.opt_ortho_fn is not None:
                self.opt_ortho = self.opt_ortho_fn(U_list + Vh_list)

        if self.opt_dir is None and len(md_params) > 0:
            self.opt_dir = self.opt_dir_fn(dir_list)
            self.opt_magn = self.opt_magn_fn(magn_list)

        if self.opt_coupled is None and len(coupled_params) > 0:
            assert self.opt_coupled_fn is not None
            self.opt_coupled = self.opt_coupled_fn(coupled_params)

        # step with the optimizers
        if self.opt_U is not None:
            assert self.opt_S is not None and self.opt_Vh is not None
            self.opt_U.step()
            self.opt_S.step()
            self.opt_Vh.step()

        if self.opt_dir is not None:
            assert self.opt_magn is not None
            self.opt_dir.step()
            self.opt_magn.step()

        if self.opt_coupled is not None:
            self.opt_coupled.step()

        # Set new parameters
        for p, U, S, Vh in zip(svd_params, U_list, S_list, Vh_list, strict=True):
            p.set_(U @ torch.diag_embed(S, dim1=-2, dim2=-1) @ Vh)

        for p, dir, magn in zip(md_params, dir_list, magn_list, strict=True):
            p.set_(dir * magn)

            # keep dir on unit sphere
            state = self.state[p]
            dir = dir / torch.linalg.vector_norm(dir).clip(min=state["eps"]) # pylint:disable=not-callable

        # step with orthogonality opt if defined
        if self.opt_ortho is not None:
            for m in U_list + Vh_list:
                m.grad = ortho_grad(m)
            self.opt_ortho.step()

        # clear factor gradients to save memory
        for t in U_list + S_list + Vh_list + dir_list + magn_list:
            t.grad = None

        return loss

    def plot_ortho(self):
        import matplotlib.pyplot as plt

        params = [p for g in self.param_groups for p in g['params']]

        U_histories = []
        Vh_histories = []

        for p in params:
            state = self.state[p]
            if "U_ortho_history" in state:
                U_histories.append(state["U_ortho_history"])
                Vh_histories.append(state["Vh_ortho_history"])

        for i,(u,v) in enumerate(zip(U_histories, Vh_histories, strict=True)):
            plt.plot(u, label=f"U-{i}")
            plt.plot(v, label=f"Vh-{i}")
