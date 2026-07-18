
import torch


@torch.compile
def zeropower_warmstart(G: torch.Tensor, prev_X: torch.Tensor | None,
                         warmstart_coeff=0.3, ns_iters=1,
                         a=3.4445, b=-4.7750, c=2.0315):
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    if prev_X is not None and X.shape == prev_X.shape:
        prev = prev_X.bfloat16()
        X = (1 - warmstart_coeff) * X + warmstart_coeff * prev
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(ns_iters):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def lion_(grads: list[torch.Tensor], exp_avg_: list[torch.Tensor], beta1: list[float], beta2: list[float],):
    update = torch._foreach_lerp(exp_avg_, grads, [1-b for b in beta1])
    torch._foreach_sign_(update)
    torch._foreach_lerp_(exp_avg_, grads, [1-b for b in beta2])
    return list(update)

def make_muon_param_groups_for_emb(model: torch.nn.Module):
    muon_params = []
    lion_params = []

    for module in model.modules():
        if isinstance(module, (torch.nn.Embedding, torch.nn.EmbeddingBag)):
            lion_params.extend(module.parameters(recurse=False))
        else:
            muon_params.extend(module.parameters(recurse=False))

    params = [
        {'params': muon_params},
        {'params': lion_params, "force_lion": True},
    ]

    return [d for d in params if len(d["params"]) > 0]

class WSMuon(torch.optim.Optimizer):
    """Muon which blends new gradient with previous orthogonalized gradient and runs 1 ns iter.
    This includes Lion fallback for <=1d params.  I thought Lion makes sense as a fallback,
    alternatively use `torchalgos.SPlus` which works on 1d params (unlike canonical SPlus).

    This sucks though its better than I imagined. I think the POGO optimizer will work better for tracking the buffer.

    Args:
        params: Iterable of tensors to optimize, or pass the model itself
            which automatically uses lion for embeddings.
        lr: Learning rate. Defaults to 3e-3.
        warmstart_coeff: how much of previous orthogonalized gradient to blend into normalized new gradient.
            0.3 is tuned and works better than other values.
        ns_iters: number of newton schulz iters per step. only 1 works well.
        lion_scale: multiplier for lion learning rate.
        lion_betas: betas in lion. Defaults to (0.9, 0.99).
        force_lion: force lion (to use in param groups on embeddings and other weird params). Defaults to False.
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        warmstart_coeff: float = 0.3,
        ns_iters: int = 1,
        lion_scale: float = 0.01,
        lion_betas: tuple[float,float] = (0.9, 0.99),
        force_lion: bool = False,
        weight_decay: float = 0,
    ):
        defaults = dict(
            lr = lr,
            warmstart_coeff = warmstart_coeff,
            ns_iters = ns_iters,
            lion_scale = lion_scale,
            lion_betas = lion_betas,
            force_lion = force_lion,
            weight_decay = weight_decay,
        )
        if isinstance(params, torch.nn.Module):
            params = make_muon_param_groups_for_emb(params)

        super().__init__(params, defaults)


    @torch.no_grad
    def step(self, closure=None): # pyright:ignore[reportIncompatibleMethodOverride]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        muon_params = []
        muon_updates = []
        muon_lrs = []

        lion_params = []
        lion_grads = []
        lion_exp_avgs = []
        lion_beta1s = []
        lion_beta2s = []
        lion_lrs = []

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue

                state = self.state[p]

                if p.ndim <= 1 or group["force_lion"]:
                    if "exp_avg" not in state: state["exp_avg"] = p.grad.sign()
                    lion_params.append(p)
                    lion_grads.append(p.grad)
                    lion_exp_avgs.append(state["exp_avg"])
                    lion_beta1s.append(group["lion_betas"][0])
                    lion_beta2s.append(group["lion_betas"][1])
                    lion_lrs.append(group["lr"] * group["lion_scale"])

                else:
                    state["X"] = zeropower_warmstart(
                        p.grad,
                        prev_X = state.get("X", None),
                        warmstart_coeff = group["warmstart_coeff"],
                        ns_iters = group["ns_iters"]
                    )
                    muon_params.append(p)
                    muon_updates.append(state["X"])
                    muon_lrs.append(group["lr"])

            if len(lion_grads) == 0: lion_updates = []
            else: lion_updates = lion_(lion_grads, lion_exp_avgs, lion_beta1s, lion_beta2s)

            params = muon_params + lion_params
            updates = muon_updates + lion_updates
            lrs = muon_lrs + lion_lrs

            if group["weight_decay"] != 0.0:
                torch._foreach_add_(
                    updates,
                    torch._foreach_mul(params, group["weight_decay"])
                )

            torch._foreach_sub_(params, torch._foreach_mul(updates, lrs))

        return loss
