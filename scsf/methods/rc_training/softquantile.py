"""Soft (smooth, differentiable-in-threshold) coverage quantiles.

``solve_soft_thresholds`` finds, for each coverage ``c``, the threshold
``t_c`` satisfying ``mean_i a_ic(t_c) == c`` under the standard formulation

    a_ic(t) = 1 - sigmoid((t - t_i) / T_s)   (accept when t_i > t).

The solution is found by bisection on the monotone ``mean(a) - c`` residual
(per coverage, with bracket expansion). ``SoftCoverageThreshold`` wraps
``solve_soft_thresholds`` in a differentiable ``torch.autograd.Function``:
gradients flow into the score ``s`` and are zero for the coverage vector, i.e.

    dh/ds_j = a_jc (1 - a_jc) / (sum_i a_ic (1 - a_ic))            (per coverage)

so the hard quantile functional ``h_c(s)`` is relaxed into a weighted
subgradient. All internal solves run on a detached buffer; the custom backward
re-solves at the forward-stored coverages and returns the closed-form grads.
"""

from __future__ import annotations

import torch

_DEFAULT_TOL = 1e-6
_MAX_ITERS = 200
_BRACKET_GROWTH = 2.0
_INIT_BRACKET = 1.0
_BRACKET_MAX = 1e6


def _accept_probs(scores, t, Ts):
    """``a_ic = 1 - sigmoid((t - s_i)/T_s)`` under temperature ``Ts``."""
    return (1.0 - (t.unsqueeze(1) - scores.unsqueeze(0)) / Ts).sigmoid()


def soft_threshold_residual(scores, coverages, Ts, thresholds):
    """``mean_i a_ic(t_c) - c`` per coverage (positive means accept too much)."""
    return _accept_probs(scores, thresholds, Ts).mean(1) - coverages


def solve_soft_thresholds(scores, coverages, Ts, tol=_DEFAULT_TOL,
                          max_iters=_MAX_ITERS):
    """Bisection over each coverage's ``t_c``; returns ``(n_c,)`` thresholds.

    Deterministic bisection against the monotone residual; identical inputs
    give identical roots (integer-identical undirected computation). All
    computations run on detached copies, so the returned thresholds carry no
    autograd graph.
    """
    s = scores.detach().flatten()
    c = coverages.detach().flatten()
    if not (0.0 < c).all() or not (c < 1.0).all():
        raise ValueError("coverages must be strictly within (0, 1)")
    if s.numel() == 0:
        raise ValueError("empty score set")
    lo = torch.full_like(c, -_INIT_BRACKET)
    hi = torch.full_like(c, _INIT_BRACKET)
    res = soft_threshold_residual(s, c, Ts, lo)
    needs = (res > 0).bool()
    while needs.any() and hi.abs().max() < _BRACKET_MAX:
        lo = torch.where(needs, lo - (hi - lo) * (_BRACKET_GROWTH - 1.0), lo)
        hi = torch.where(needs, hi * _BRACKET_GROWTH, hi)
        res = soft_threshold_residual(s, c, Ts, lo)
        needs = (res > 0).bool()
    for _ in range(int(max_iters)):
        mid = 0.5 * (lo + hi)
        res = soft_threshold_residual(s, c, Ts, mid)
        lo = torch.where(res > 0, mid, lo)
        hi = torch.where(res > 0, hi, mid)
    return 0.5 * (lo + hi)


class SoftCoverageThreshold(torch.autograd.Function):
    """Differentiable-in-s threshold solve; zero gradient for the coverage."""

    @staticmethod
    def forward(ctx, scores, coverages, Ts):
        ctx.save_for_backward(scores.detach(), coverages.detach())
        ctx.Ts = float(Ts.detach())
        return solve_soft_thresholds(scores, coverages, Ts)

    @staticmethod
    def backward(ctx, grad_h):
        s, c = ctx.saved_tensors
        t = solve_soft_thresholds(s, c, torch.tensor(ctx.Ts, device=s.device))
        a = _accept_probs(s, t, ctx.Ts)                     # (n_c, B)
        flat = a * (1.0 - a)                                # a_ic (1 - a_ic)
        denom = flat.sum(1, keepdim=True).clamp_min(1e-12)
        # vjp of the coverage-scaled subgradient: grad_s[j] = sum_c grad_h_c
        # * a_jc(1 - a_jc) / (T_s * sum_i a_ic(1 - a_ic)).
        term = flat / (ctx.Ts * denom)                      # (n_c, B)
        grad_s = torch.einsum("c,cb->b", grad_h.detach(), term)
        return grad_s, torch.zeros_like(grad_h), None


def soft_coverage_threshold(scores, coverages, Ts):
    """``SoftCoverageThreshold`` applied against the forward-stored coverages."""
    return SoftCoverageThreshold.apply(scores, coverages, Ts)


__all__ = [
    "SoftCoverageThreshold",
    "soft_coverage_threshold",
    "soft_threshold_residual",
    "solve_soft_thresholds",
]