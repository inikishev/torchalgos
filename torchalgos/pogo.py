"""POGO From https://github.com/adrianjav/pogo Note That This Code Is GPL License."""
import math

import torch

# ============================================================================ #
#                                 POGO Helpers                                 #
# ============================================================================ #

def _solve_quartic_equation(coefs: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-3
    assert coefs.shape[1] == 5

    coefs = coefs.cfloat()
    coefs = coefs[:, 1:] / coefs[:, :1]
    B, C, D, E = coefs.T

    a = -3.0 / 8.0 * B**2 + C
    b = B**3 / 8.0 - B * C / 2.0 + D
    c = -3.0 / 256.0 * B**4 + C * B**2 / 16.0 - B * D / 4.0 + E

    is_b_zero = b.real.abs() < epsilon

    opt1 = torch.sqrt(a**2 - 4 * c)
    sol_b_zero = [torch.sqrt((-a + (-1) ** i * opt1) / 2) for i in [0, 1]]
    sol_b_zero.extend([-x for x in sol_b_zero])
    sol_b_zero = torch.stack(sol_b_zero, dim=-1)
    sol_b_zero -= B.unsqueeze(-1) / 4.0

    P = -a**2 / 12.0 - c
    Q = -a**3 / 108.0 + a * c / 3.0 - b**2 / 8.0
    R = -Q / 2.0 + torch.sqrt(Q**2 / 4.0 + P**3 / 27.0)
    U = torch.pow(R, 1 / 3.0)

    y = -5.0 / 6.0 * a + torch.where(
        U == 0,
        -torch.pow(Q, 1 / 3.0),
        U - P / (3 * U),
    )
    W = torch.sqrt(a + 2 * y)

    opt1 = 2 * b / W
    sol_b_nonzero = [torch.sqrt(-(3 * a + 2 * y + (-1) ** i * opt1)) for i in [0, 1]]
    sol_b_nonzero.extend([-x for x in sol_b_nonzero])
    sol_b_nonzero = torch.stack(sol_b_nonzero, dim=-1)
    sol_b_nonzero[:, [0, 2]] += W.unsqueeze(-1)
    sol_b_nonzero[:, [1, 3]] -= W.unsqueeze(-1)
    sol_b_nonzero = sol_b_nonzero / 2.0 - B.unsqueeze(-1) / 4.0

    solution = torch.where(
        is_b_zero.unsqueeze(-1),
        sol_b_zero,
        sol_b_nonzero,
    )
    return solution


def _solve_cubic_equation(coefs: torch.Tensor) -> torch.Tensor:
    assert coefs.shape[1] == 4
    coefs = coefs.cfloat()
    a, b, c, d = coefs.T

    d0 = b**2 - 3 * a * c
    d1 = 2 * b**3 - 9 * a * b * c + 27 * a**2 * d

    C = torch.pow((d1 + torch.sqrt(d1**2 - 4 * d0**3)) / 2.0, 1 / 3.0)
    psi = (-1 + math.sqrt(3) * 1j) / 2.0

    solution = [-(b + psi**k * C + d0 / (psi**k * C)) / (3 * a) for k in [0, 1, 2]]
    return torch.stack(solution, dim=-1)


def _solve_quadratic_equation(coefs: torch.Tensor) -> torch.Tensor:
    assert coefs.shape[1] == 3
    coefs = coefs.cfloat()
    a, b, c = coefs.T
    disc = torch.sqrt(b**2 - 4 * a * c)
    solution = torch.stack([disc, -disc], dim=-1)
    return (solution - b.unsqueeze(-1)) / (2 * a.unsqueeze(-1))


def _solve_monic_equation(coefs: torch.Tensor) -> torch.Tensor:
    assert coefs.shape[1] == 2
    coefs = coefs.cfloat()
    a, b = coefs.T
    return (-b / a).unsqueeze(-1)


def _solve_poly_equation(coefs: torch.Tensor) -> torch.Tensor:
    eps = 1e-4
    assert coefs.shape[1] == 5

    quartic = coefs[:, 0].abs() > eps
    cubic = ~quartic & (coefs[:, 1].abs() > eps)
    quadratic = ~quartic & ~cubic & (coefs[:, 2].abs() > eps)
    monic = ~quartic & ~cubic & ~quadratic

    sol = torch.empty((coefs.shape[0], 4), device=coefs.device, dtype=torch.cfloat)
    if quartic.any():
        sol[quartic] = _solve_quartic_equation(coefs[quartic])
    if cubic.any():
        sol_cubic = _solve_cubic_equation(coefs[cubic][..., 1:])
        sol[cubic] = torch.cat((sol_cubic, sol_cubic[:, :1]), dim=-1)
    if quadratic.any():
        sol_quadratic = _solve_quadratic_equation(coefs[quadratic][..., 2:])
        sol[quadratic] = torch.cat((sol_quadratic, sol_quadratic[:, :1]), dim=-1)
    if monic.any():
        sol_monic = _solve_monic_equation(coefs[monic][..., 3:])
        sol[monic] = torch.cat([sol_monic for _ in range(4)], dim=-1)

    return sol


def compute_lambda(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    p = A.shape[-2]
    I = torch.eye(p, device=A.device, dtype=A.dtype).expand(A.shape[0], p, p)

    AAT = torch.bmm(A, A.conj().transpose(-1, -2)) - I
    BBT = torch.bmm(B, B.conj().transpose(-1, -2))
    ABT = torch.bmm(A, B.conj().transpose(-1, -2)) + torch.bmm(B, A.conj().transpose(-1, -2))

    foo = lambda x, y: (x * y).sum(dim=(-1, -2))
    distance = foo(AAT, AAT)
    coefs = torch.stack(
        [
            foo(BBT, BBT),
            2 * foo(ABT, BBT),
            2 * foo(BBT, AAT) + foo(ABT, ABT),
            2 * foo(AAT, ABT),
            distance,
        ],
        dim=-1,
    )

    mask = coefs[:, 0] == 0
    coefs[mask, :-1] = 1.0

    coefs = coefs / coefs.abs().max(dim=-1, keepdim=True)[0]
    roots = _solve_poly_equation(coefs)

    if torch.is_complex(A):
        indices = roots.abs().argmin(dim=1)
        lambda_regul = roots.flatten()[torch.arange(0, roots.shape[0] * 4, 4, device=indices.device) + indices]
    else:
        indices = roots.imag.abs().argmin(dim=1)
        lambda_regul = roots.real.flatten()[torch.arange(0, roots.shape[0] * 4, 4, device=indices.device) + indices]
    return lambda_regul


def pogo_step(
    Q: torch.Tensor,
    grad: torch.Tensor,
    lr: float = 0.1,
    exact_landing: bool = False,
) -> torch.Tensor:
    """Executes a POGO step on O(n)."""
    Q_b = Q.unsqueeze(0) if Q.ndim == 2 else Q
    G_b = grad.unsqueeze(0) if grad.ndim == 2 else grad

    # Relative gradient on Stiefel/Orthogonal manifold
    XXtG = torch.bmm(torch.bmm(Q_b, Q_b.conj().transpose(-1, -2)), G_b)
    XGtX = torch.bmm(torch.bmm(Q_b, G_b.conj().transpose(-1, -2)), Q_b)
    rel_grad = 0.5 * (XXtG - XGtX)

    # Midpoint
    A = Q_b - lr * rel_grad
    B = A - torch.bmm(torch.bmm(A, A.conj().transpose(-1, -2)), A)

    if exact_landing:
        lambda_regul = compute_lambda(A, B).view(-1, 1, 1)
    else:
        lambda_regul = 0.5

    Q_next = A + lambda_regul * B
    return Q_next.squeeze(0) if Q.ndim == 2 else Q_next

