"""Idea is to orthogonalize fast EMA to slow EMA. So I was trying to get a useful optimizer out of this for a while but nothing worked well. Then I passed the idea and all of my code to Qwen 3.7 max and this is a much better result."""
import torch
from torch.optim import Optimizer


class OrthoAdam(Optimizer):
    """
    OrthoAdam: A stable, scale-invariant, 2nd-order-like optimizer that projects the
    Adam update orthogonal to the high-frequency "fast" modes (valley walls) per-tensor.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        beta_slow=0.999,
        eps=1e-8,
        weight_decay=0.0,
        w=0.9,
        cautious: bool = True,
    ):
        defaults = dict(lr=lr, betas=betas, beta_slow=beta_slow, eps=eps,
                        weight_decay=weight_decay, w=w, cautious=cautious)
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None): # pyright:ignore[reportIncompatibleMethodOverride]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            beta_slow = group['beta_slow']
            w = group['w']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg_fast'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_slow'] = grad.clone() * 1e-5
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                step = state['step'] + 1
                state['step'] = step

                exp_avg_fast = state['exp_avg_fast']
                exp_avg_slow = state['exp_avg_slow']
                exp_avg_sq = state['exp_avg_sq']

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # 1. Standard Adam + Slow EMA updates
                exp_avg_fast.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_slow.mul_(beta_slow).add_(grad, alpha=1 - beta_slow)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 2. Bias Corrections
                m_fast_hat = exp_avg_fast / (1 - beta1 ** step)
                m_slow_hat = exp_avg_slow / (1 - beta_slow ** step)
                v_hat = exp_avg_sq / (1 - beta2 ** step)

                # 3. Preconditioned Directions (Scale Invariance)
                denom = v_hat.sqrt().add(eps)
                u_fast = m_fast_hat / denom
                u_slow = m_slow_hat / denom

                # 4. Isolate the Fast Mode (Transverse Oscillation Vector)
                diff = u_fast - u_slow

                # 5. Stable Per-Tensor Orthogonalization
                u_norm = u_fast.norm()
                diff_norm = diff.norm()

                dot_product = (u_fast * diff).sum()
                denom_norms = u_norm * diff_norm

                # If the difference is negligible, do not orthogonalize (rho = 0)
                rho = torch.where(denom_norms > 1e-8, dot_product / (denom_norms + 1e-12), torch.zeros_like(dot_product))
                rho = rho.clamp(-1.0, 1.0) # Mathematically strict bound

                diff_hat = diff / (diff_norm + 1e-12)

                # 6. Compute Final Orthogonal Update
                # Magnitude mathematically bounded by [sqrt(1-w^2), 1] * |u_fast|
                u_ortho = u_fast - (w * rho * u_norm) * diff_hat

                if group["cautious"]:
                    u_ortho *= (u_ortho * grad).gt_(0)

                p.add_(u_ortho, alpha=-group['lr'])

        return loss