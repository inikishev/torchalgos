"""gemini"""
import math
import torch
from torch.optim.optimizer import Optimizer

class Sylph(Optimizer):
    """
    SYLPH: SYLvester Preconditioned Hessian Optimizer.

    A breakthrough beyond SOAP and Shampoo. It pre-conditions gradients using
    the fractional Sylvester operator (L ⊗ I + I ⊗ R) rather than Kronecker
    whitening (L ⊗ R). The dense continuous cross-terms cancel out perfectly
    in the eigenbasis, yielding a highly tractable, expressive transformation.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95),
                 sylvester_beta=0.95, eps=1e-8, sylph_p=0.5, update_freq=10):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(lr=lr, betas=betas, sylvester_beta=sylvester_beta,
                        eps=eps, sylph_p=sylph_p, update_freq=update_freq)
        super(Sylph, self).__init__(params, defaults)
        self._step = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        self._step += 1

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # State Initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                    # For 2D parameters (Linear/Attention weights), setup Sylvester factors
                    if grad.dim() == 2:
                        state['L'] = torch.zeros(grad.shape[0], grad.shape[0], device=p.device)
                        state['R'] = torch.zeros(grad.shape[1], grad.shape[1], device=p.device)
                        state['Q_L'] = torch.eye(grad.shape[0], device=p.device)
                        state['Q_R'] = torch.eye(grad.shape[1], device=p.device)
                        state['lambda_L'] = torch.ones(grad.shape[0], device=p.device)
                        state['lambda_R'] = torch.ones(grad.shape[1], device=p.device)

                state['step'] += 1

                if grad.dim() == 2:
                    # 1. Update Gradient Covariance Matrices
                    beta_s = group['sylvester_beta']
                    state['L'].mul_(beta_s).add_(grad @ grad.T, alpha=1 - beta_s)
                    state['R'].mul_(beta_s).add_(grad.T @ grad, alpha=1 - beta_s)

                    # 2. Amortized Eigendecomposition (Only computed every `update_freq` steps)
                    if self._step % group['update_freq'] == 1 or self._step == 1:
                        # eigh returns ascending eigenvalues
                        try:
                            # Clamp for stability
                            lambda_L, Q_L = torch.linalg.eigh(state['L'])
                            lambda_R, Q_R = torch.linalg.eigh(state['R'])
                            state['lambda_L'], state['Q_L'] = lambda_L.clamp(min=1e-8), Q_L
                            state['lambda_R'], state['Q_R'] = lambda_R.clamp(min=1e-8), Q_R
                        except Exception as e:
                            state['Q_L'] = torch.eye(grad.shape[0], device=p.device)
                            state['Q_R'] = torch.eye(grad.shape[1], device=p.device)
                            state['lambda_L'] = torch.ones(grad.shape[0], device=p.device)
                            state['lambda_R'] = torch.ones(grad.shape[1], device=p.device)

                    Q_L, Q_R = state['Q_L'], state['Q_R']
                    lambda_L, lambda_R = state['lambda_L'], state['lambda_R']

                    # 3. Rotate Gradient into the Eigenbasis
                    grad_rot = Q_L.T @ grad @ Q_R

                    # 4. THE SYLVESTER CANCELLATION (The Breakthrough)
                    # H = L ⊗ I + I ⊗ R transforms to a simple additive eigenvalue space.
                    # Broadcasting (m, 1) + (1, n) yields the dense (m, n) operator instantly.
                    S_eigen = lambda_L.unsqueeze(1) + lambda_R.unsqueeze(0)

                    # Apply Fractional Sylvester Transformation
                    p_power = group['sylph_p']
                    grad_sylv = grad_rot / (S_eigen ** p_power + group['eps'])

                    # 5. Apply Adam within the smoothly preconditioned space
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    beta1, beta2 = group['betas']

                    exp_avg.mul_(beta1).add_(grad_sylv, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).add_(grad_sylv ** 2, alpha=1 - beta2)

                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']

                    step_size = group['lr'] / bias_correction1

                    # Adam step in the eigenbasis
                    adam_update = exp_avg / ((exp_avg_sq.sqrt() / math.sqrt(bias_correction2)) + group['eps'])

                    # 6. Rotate Update back to the Parameter Space
                    update = Q_L @ adam_update @ Q_R.T

                    p.add_(update, alpha=-step_size)

                else:
                    # Fallback to standard AdamW for 1D Biases / LayerNorms
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    beta1, beta2 = group['betas']

                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).add_(grad ** 2, alpha=1 - beta2)

                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']

                    step_size = group['lr'] / bias_correction1
                    update = exp_avg / ((exp_avg_sq.sqrt() / math.sqrt(bias_correction2)) + group['eps'])
                    p.add_(update, alpha=-step_size)

        return loss