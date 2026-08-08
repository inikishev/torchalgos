"""gemini said its muon (its worse than moun)"""

import torch
import torch.optim as optim

class Aether(optim.Optimizer):
    """
    AETHER: Adaptive Extraction of Transform-Harmonic Equivalent Roots.

    A spectral phase-only optimizer that achieves generalized spatial whitening
    in O(N log N) time, completely bypassing the O(N^3) MatMul bottlenecks of
    Newton-Schulz (Muon) and SVD (SOAP).
    """
    def __init__(self, params, lr=1e-3, beta=0.95, eps=1e-8, weight_decay=0.0, optimize_1d:bool=False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")

        defaults = dict(lr=lr, beta=beta, eps=eps, weight_decay=weight_decay,optimize_1d=optimize_1d)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Aether does not support sparse gradients.')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum'] = torch.zeros_like(p)

                state['step'] += 1
                momentum = state['momentum']

                # Decoupled Weight Decay (AdamW style)
                if weight_decay != 0:
                    p.data.mul_(1.0 - lr * weight_decay)

                # Update Momentum (Exponential Moving Average)
                momentum.mul_(beta).add_(grad, alpha=1.0 - beta)

                # ==========================================================
                # CORE AETHER ALGORITHM
                # ==========================================================
                if p.dim() >= 2 or (group["optimize_1d"] and p.dim() >= 1):
                    # 1. Transform to the easier function (Frequency Domain)
                    # The FFT applies useful butterfly cancellations, running in O(N log N)
                    dims = tuple(range(p.dim()))

                    # Compute Real N-Dimensional FFT (handles arbitrary layer dimensions)
                    F_M = torch.fft.rfftn(momentum.to(torch.float32), dim=dims)

                    # 2. Spectral Phase-Only Extraction
                    # Normalizing the magnitude equalizes energy across all structural bases,
                    # enforcing a theoretical Dirac-delta spatial autocorrelation (Perfect Whitening).
                    mag = torch.abs(F_M)
                    F_update = F_M / (mag + eps)

                    # 3. Inverse Transform back to Spatial Domain
                    update = torch.fft.irfftn(F_update, s=p.shape, dim=dims).to(p.dtype)

                    # 4. Energy Restoration
                    # The phase extraction reduces the variance of the matrix elements to 1/N.
                    # We scale by sqrt(N) to enforce an RMS of 1.0, aligning the update size
                    # identically with standard Adam/Muon step magnitudes.
                    numel = p.numel()
                    update.mul_(numel ** 0.5)
                else:
                    # Fallback for 1D parameters (biases, layernorms):
                    # Standard RMS normalized momentum
                    rms = momentum.norm() / (p.numel() ** 0.5)
                    update = momentum / (rms + eps)

                # Apply gradient step
                p.add_(update, alpha=-lr)

        return loss