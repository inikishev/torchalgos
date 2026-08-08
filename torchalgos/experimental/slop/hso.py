"""qwen3.8 max"""
import math

import torch
from torch.optim.optimizer import Optimizer


_EULER_GAMMA = 0.5772156649015328606

# Half-normal target: |Z| where Z ~ N(0, 1)
_HALF_NORMAL_MEDIAN = 0.6744897501960817
_HALF_NORMAL_MEAN = math.sqrt(2.0 / math.pi)
_HALF_NORMAL_LOG_MEAN = -0.5 * (_EULER_GAMMA + math.log(2.0))
_HALF_NORMAL_LOG_STD = math.pi / math.sqrt(8.0)

# Unit-variance signed Laplace target:
# Laplace(0, b) with variance 2 b^2 = 1 => b = 1 / sqrt(2).
# Magnitude is exponential with scale b.
_LAPLACE_B = 1.0 / math.sqrt(2.0)
_LAPLACE_MEDIAN = _LAPLACE_B * math.log(2.0)
_LAPLACE_MEAN = _LAPLACE_B
_LAPLACE_LOG_MEAN = math.log(_LAPLACE_B) - _EULER_GAMMA
_LAPLACE_LOG_STD = math.pi / math.sqrt(6.0)


_TARGET_INFO = {
    "half_normal": {
        "median": _HALF_NORMAL_MEDIAN,
        "mean": _HALF_NORMAL_MEAN,
        "log_mean": _HALF_NORMAL_LOG_MEAN,
        "log_std": _HALF_NORMAL_LOG_STD,
        "quantile": lambda p: math.sqrt(2.0) * torch.erfinv(p),
    },
    "half_laplace": {
        "median": _LAPLACE_MEDIAN,
        "mean": _LAPLACE_MEAN,
        "log_mean": _LAPLACE_LOG_MEAN,
        "log_std": _LAPLACE_LOG_STD,
        "quantile": lambda p: -_LAPLACE_B * torch.log1p(-p),
    },
}


def _clip(x: torch.Tensor, clip: float | None) -> torch.Tensor:
    if clip is None or clip <= 0:
        return x
    return x.clamp(-clip, clip)


def _target_quantile(p: torch.Tensor, target: str) -> torch.Tensor:
    """
    Target quantile function for gradient magnitudes.

    For half_normal:
        Q(p) = sqrt(2) * erfinv(p)

    p should be in (0, 1).
    """
    p = p.clamp(1e-7, 1.0 - 1e-7)
    return _TARGET_INFO[target]["quantile"](p)


def _sign_scaled(
    grad: torch.Tensor,
    target: str,
    eps: float,
    clip: float | None,
) -> torch.Tensor:
    """
    First-step fallback: keep sign, use target median magnitude.
    """
    info = _TARGET_INFO[target]
    out = torch.where(
        grad.abs() > eps,
        grad.sign() * info["median"],
        torch.zeros_like(grad),
    )
    return _clip(out, clip)


def _robust_normalize(
    grad: torch.Tensor,
    history: torch.Tensor,
    filled: int,
    eps: float,
    target: str,
    clip: float | None,
) -> torch.Tensor:
    """
    Normalize gradient by a robust recent scale.

    Uses median(|g|) if available, otherwise mean(|g|).
    The scale is chosen so that a half-normal-like magnitude would have
    approximately unit target scale.
    """
    if filled <= 0:
        return _sign_scaled(grad, target, eps, clip)

    info = _TARGET_INFO[target]
    abs_hist = history.narrow(0, 0, filled).abs()

    med = abs_hist.median(dim=0).values
    mean = abs_hist.mean(dim=0)

    scale_med = med / info["median"]
    scale_mean = mean / info["mean"]

    scale = torch.where(med > eps, scale_med, scale_mean)
    safe = scale > eps

    out = torch.where(
        safe,
        grad / scale.clamp_min(eps),
        grad.sign() * info["median"],
    )

    return _clip(out, clip)


def _quantile_shape(
    grad: torch.Tensor,
    history: torch.Tensor,
    filled: int,
    eps: float,
    target: str,
    ignore_zeros: bool,
    clip: float | None,
) -> torch.Tensor:
    """
    Nonlinear elementwise quantile shaping.

    For each scalar coordinate:
        1. Look at recent |g|.
        2. Compute the empirical quantile of current |g|.
        3. Map that quantile to the target magnitude quantile.
        4. Preserve the sign of g.

    This matches the whole unsigned magnitude distribution, not just moments.
    """
    if filled <= 0:
        return _sign_scaled(grad, target, eps, clip)

    info = _TARGET_INFO[target]
    abs_hist = history.narrow(0, 0, filled).abs()
    cur_abs = grad.abs()

    if ignore_zeros:
        mask = abs_hist > eps
        denom = mask.sum(dim=0).to(grad.dtype)

        less = ((abs_hist < cur_abs) & mask).sum(dim=0).to(grad.dtype)
        equal = ((abs_hist == cur_abs) & mask).sum(dim=0).to(grad.dtype)
    else:
        denom = torch.full_like(cur_abs, float(filled), dtype=grad.dtype)
        less = (abs_hist < cur_abs).sum(dim=0).to(grad.dtype)
        equal = (abs_hist == cur_abs).sum(dim=0).to(grad.dtype)

    # Average ranks for ties.
    rank = less + 0.5 * equal

    valid = denom > 0
    p = rank / denom.clamp_min(1.0)
    p = torch.where(valid, p, torch.full_like(p, 0.5))

    # Empirical quantile clipping.
    min_p = max(0.5 / max(filled, 1), 1e-7)
    max_p = min(1.0 - 0.5 / max(filled, 1), 1.0 - 1e-7)

    if min_p >= max_p:
        # filled == 1
        p = torch.full_like(p, 0.5)
    else:
        p = p.clamp(min_p, max_p)

    q = _target_quantile(p, target)

    out = grad.sign() * q

    # If there were no usable nonzero history entries, fall back to median.
    fallback = grad.sign() * info["median"]
    out = torch.where(valid, out, fallback)

    # Exact zeros stay zero.
    out = torch.where(cur_abs > eps, out, torch.zeros_like(out))

    return _clip(out, clip)


def _logmoment_shape(
    grad: torch.Tensor,
    history: torch.Tensor,
    filled: int,
    eps: float,
    target: str,
    clip: float | None,
    max_power: float = 5.0,
    max_exp: float = 20.0,
) -> torch.Tensor:
    """
    Cheaper parametric shaping.

    Fits:
        h(r) = exp(a log(r) + b)

    so that log |h(g)| approximately matches the target log-magnitude
    mean and standard deviation.
    """
    if filled <= 0:
        return _sign_scaled(grad, target, eps, clip)

    info = _TARGET_INFO[target]
    abs_hist = history.narrow(0, 0, filled).abs()

    mask = abs_hist > eps
    count = mask.sum(dim=0).to(grad.dtype)
    valid = count > 0
    count_safe = count.clamp_min(1.0)

    log_hist = torch.log(abs_hist.clamp_min(eps))
    log_hist = torch.where(mask, log_hist, torch.zeros_like(log_hist))

    mean = log_hist.sum(dim=0) / count_safe

    diff = log_hist - mean
    diff = torch.where(mask, diff, torch.zeros_like(diff))

    var = (diff * diff).sum(dim=0) / count_safe
    std = var.clamp_min(1e-8).sqrt()

    power = (info["log_std"] / std).clamp(min=0.1, max=max_power)
    bias = info["log_mean"] - power * mean

    log_cur = torch.log(grad.abs().clamp_min(eps))
    log_out = power * log_cur + bias
    log_out = log_out.clamp(-max_exp, max_exp)

    out = grad.sign() * torch.exp(log_out)

    fallback = grad.sign() * info["median"]
    out = torch.where(valid, out, fallback)

    out = torch.where(grad.abs() > eps, out, torch.zeros_like(out))

    return _clip(out, clip)


class HSO(Optimizer):
    """
    History-shaped gradient optimizer.

    This optimizer maintains a recent gradient history for each scalar
    parameter and applies an elementwise nonlinear map to the gradient.

    Methods:
        - "quantile":
            Nonparametric quantile matching of |g| to a target magnitude
            distribution. Matches all unsigned quantiles/moments implicitly.

        - "logmoment":
            Parametric power transform fitted using log(|g|) mean and variance.
            Cheaper and smoother, but less flexible.

        - "none":
            Only robust scale normalization, no higher-order shaping.

    Targets:
        - "half_normal":
            Magnitude of N(0, 1). Natural isotropic-quadratic proxy.

        - "half_laplace":
            Magnitude of unit-variance Laplace. Slightly heavier-tailed.

    Notes:
        - This is a prototype.
        - Elementwise history is memory-heavy.
        - The target distribution is a heuristic proxy for "well-conditioned
          isotropic gradient behavior".
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta: float = 0.9,
        weight_decay: float = 0.0,
        history_size: int = 100,
        warmup: int = 20,
        method: str = "quantile",
        shape_weight: float = 0.8,
        target: str = "half_normal",
        ignore_zeros: bool = True,
        eps: float = 1e-12,
        clip: float = 10.0,
        gate: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= beta < 1.0):
            raise ValueError(f"Invalid beta: {beta}. Must be in [0, 1).")
        if history_size < 0:
            raise ValueError(f"Invalid history_size: {history_size}")
        if warmup < 0:
            raise ValueError(f"Invalid warmup: {warmup}")
        if eps <= 0:
            eps = 1e-12
        if target not in _TARGET_INFO:
            raise ValueError(
                f"Unknown target {target}. "
                f"Available: {list(_TARGET_INFO.keys())}"
            )
        if method not in {"quantile", "logmoment", "none"}:
            raise ValueError(
                f"Unknown method {method}. "
                "Available: 'quantile', 'logmoment', 'none'."
            )

        shape_weight = min(max(float(shape_weight), 0.0), 1.0)

        defaults = dict(
            lr=lr,
            beta=beta,
            weight_decay=weight_decay,
            history_size=int(history_size),
            warmup=int(warmup),
            method=method,
            shape_weight=shape_weight,
            target=target,
            ignore_zeros=ignore_zeros,
            eps=eps,
            clip=clip,
            gate=gate,
        )

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            wd = group["weight_decay"]
            K = int(group["history_size"])
            warmup = int(group["warmup"])
            method = group["method"]
            target = group["target"]
            eps = group["eps"]
            clip = group["clip"]
            shape_weight = group["shape_weight"]
            ignore_zeros = group["ignore_zeros"]
            use_gate = group["gate"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise RuntimeError(
                        "HistoryShapedOptimizer does not support sparse gradients."
                    )

                grad = p.grad.detach().to(torch.float32)

                # Skip non-finite gradients rather than corrupting history.
                if not torch.isfinite(grad).all():
                    continue

                state = self.state[p]

                # Initialize or repair state.
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, dtype=torch.float32, device=p.device
                    )

                    if K > 0:
                        state["history"] = torch.zeros(
                            (K,) + p.shape,
                            dtype=torch.float32,
                            device=p.device,
                        )
                    else:
                        state["history"] = None

                    state["ptr"] = 0
                    state["filled"] = 0

                else:
                    if state["exp_avg"].shape != p.shape:
                        state["exp_avg"] = torch.zeros_like(
                            p, dtype=torch.float32, device=p.device
                        )
                        state["step"] = 0

                    if K > 0:
                        needs_reinit = (
                            state.get("history") is None
                            or state["history"].shape[0] != K
                            or state["history"].shape[1:] != p.shape
                        )
                        if needs_reinit:
                            state["history"] = torch.zeros(
                                (K,) + p.shape,
                                dtype=torch.float32,
                                device=p.device,
                            )
                            state["ptr"] = 0
                            state["filled"] = 0
                    else:
                        state["history"] = None
                        state["ptr"] = 0
                        state["filled"] = 0

                # Decoupled weight decay.
                if wd != 0:
                    p.mul_(1.0 - lr * wd)

                filled = state["filled"] if K > 0 else 0

                if K > 0 and filled > 0:
                    hist = state["history"]

                    # Robustly normalized baseline.
                    raw = _robust_normalize(
                        grad=grad,
                        history=hist,
                        filled=filled,
                        eps=eps,
                        target=target,
                        clip=clip,
                    )

                    # Higher-order shaped candidate.
                    if method == "quantile":
                        shaped_candidate = _quantile_shape(
                            grad=grad,
                            history=hist,
                            filled=filled,
                            eps=eps,
                            target=target,
                            ignore_zeros=ignore_zeros,
                            clip=clip,
                        )
                    elif method == "logmoment":
                        shaped_candidate = _logmoment_shape(
                            grad=grad,
                            history=hist,
                            filled=filled,
                            eps=eps,
                            target=target,
                            clip=clip,
                        )
                    elif method == "none":
                        shaped_candidate = raw
                    else:
                        raise ValueError(f"Unknown method: {method}")

                    progress = 1.0 if warmup <= 0 else min(
                        1.0, filled / float(warmup)
                    )
                    alpha = shape_weight * progress

                    # Gate: do not strongly shape already tiny robust gradients.
                    if use_gate:
                        info = _TARGET_INFO[target]
                        gate_value = (
                            raw.abs() / max(info["median"], eps)
                        ).clamp(0.0, 1.0)
                        alpha = alpha * gate_value

                    shaped = (1.0 - alpha) * raw + alpha * shaped_candidate

                else:
                    # No history yet: sign-scaled fallback.
                    shaped = _sign_scaled(grad, target, eps, clip)

                shaped = _clip(shaped, clip)

                # Momentum.
                exp_avg = state["exp_avg"]
                exp_avg.mul_(beta).add_(shaped, alpha=1.0 - beta)

                state["step"] += 1

                if beta > 0.0:
                    bias_correction = 1.0 - beta ** state["step"]
                    update = exp_avg / bias_correction
                else:
                    update = exp_avg

                p.add_(update.to(p.dtype), alpha=-lr)

                # Store current gradient after using causal history.
                if K > 0:
                    state["history"][state["ptr"]].copy_(grad)
                    state["ptr"] = (state["ptr"] + 1) % K
                    state["filled"] = min(state["filled"] + 1, K)

        return loss


@torch.no_grad()
def history_qualities(
    history: torch.Tensor | None,
    eps: float = 1e-12,
    target: str = "half_normal",
) -> dict[str, float]:
    """
    Compute diagnostic unsigned / bias-resistant qualities from a gradient
    history buffer.

    This is useful for monitoring whether recent gradients look like the
    target distribution.

    The returned quantities are mostly sign-invariant:
        - nonzero_fraction
        - magnitude quantiles
        - magnitude quantile ratios
        - log absolute moments
        - sign-flip rate

    Signed mean and signed odd moments are intentionally omitted because
    they are highly biased along optimization trajectories.
    """
    if history is None or history.numel() == 0 or history.shape[0] == 0:
        return {}

    history = history.detach().float()
    abs_hist = history.abs()
    mask = abs_hist > eps

    out: dict[str, float] = {
        "nonzero_fraction": mask.float().mean().item(),
    }

    vals = abs_hist[mask]
    if vals.numel() == 0:
        return out

    # Magnitude quantiles.
    q_levels = torch.tensor(
        [0.25, 0.50, 0.75, 0.90, 0.95],
        device=vals.device,
        dtype=vals.dtype,
    )
    qs = torch.quantile(vals, q_levels)
    q25, q50, q75, q90, q95 = [float(x) for x in qs]

    info = _TARGET_INFO[target]
    target_q50 = float(info["median"])
    target_q90 = float(
        _target_quantile(torch.tensor(0.90, device=vals.device), target).item()
    )
    target_q95 = float(
        _target_quantile(torch.tensor(0.95, device=vals.device), target).item()
    )

    out.update(
        {
            "q25": q25,
            "q50": q50,
            "q75": q75,
            "q90": q90,
            "q95": q95,
            "q90/q50": q90 / max(q50, eps),
            "q95/q50": q95 / max(q50, eps),
            "target_q50": target_q50,
            "target_q90/q50": target_q90 / max(target_q50, eps),
            "target_q95/q50": target_q95 / max(target_q50, eps),
        }
    )

    # Log absolute moments.
    #
    # Instead of computing E[|g|^p], which can overflow, compute:
    #
    #   log E[|g|^p] = logsumexp(p log |g|) - log(count)
    #
    count = mask.sum(dim=0).float()
    valid_elem = count > 0

    if bool(valid_elem.any()):
        log_abs = torch.log(abs_hist.clamp_min(eps))
        log_abs = torch.where(
            mask,
            log_abs,
            torch.full_like(log_abs, math.log(eps)),
        )

        count_safe = count.clamp_min(1.0)

        for p in (1, 2, 3, 4):
            x = p * log_abs
            x = x.masked_fill(~mask, -1e30)
            lse = torch.logsumexp(x, dim=0)
            log_moment = lse - torch.log(count_safe)
            out[f"log_abs_moment_p{p}"] = log_moment[valid_elem].mean().item()

    # Sign-flip rate.
    #
    # This is a dynamic sign quality. It is less biased than E[sign(g)]
    # because it measures oscillation/noise rather than one-sided position.
    if history.shape[0] > 1:
        signs = torch.sign(history)
        both_nonzero = (signs[1:] != 0) & (signs[:-1] != 0)

        if bool(both_nonzero.any()):
            flips = (signs[1:] * signs[:-1] < 0) & both_nonzero
            out["sign_flip_rate"] = (
                flips.float().sum().item() / both_nonzero.float().sum().item()
            )
        else:
            out["sign_flip_rate"] = float("nan")

    return out