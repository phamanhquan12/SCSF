"""Differentiable selective-classification surrogate for SAGE-DS utility.

The exact metrics in :mod:`scsf.metrics.selective` are *evaluation* objects:
they sort by confidence and are non-differentiable. SAGE-DS needs a cost whose
gradient (``g_sel``) tells the controller whether the auxiliary-supervision
direction on the backbone improves the shape of the confidence-to-risk ordering
on held-out data. This module provides that differentiable surrogate, built on
the same confidence>keep / ``u = -confidence`` / error-positive orientation as
the exact sampler, and documents its relation to the Zhou et al. population
AURC estimator.

Relationship to Zhou et al. (population) AURC
---------------------------------------------
Zhou, Zhang, et al. treat AURC as ``E[risk | coverage]`` over the *population*
confidence distribution and estimate it from the empirical risk-coverage curve
(``E[ risk( Coverage_t ) ]`` over ``t``). Their estimator is exactly the
finite-sample prefix-AURC implemented in :func:`scsf.metrics.selective.aurc`
in the limit of a large sample; the surrogate below is its smooth relaxation so
a gradient can flow into the model. With hard 0/1 errors and stage-to-zero
temperature it collapses (over the coverage grid) to the exact hard AURC used
by the evaluator.

Surrogate form
--------------
For a batch of confidence scores ``c_i`` (higher = keep) we build, for a
coverage level ``q`` (hard quantile ``t_q`` of ``c``, detached), a **soft
gated mask** with smoothing temperature ``tau``::

    w_{i,q} = sigmoid( (c_i - t_q) / tau )        # "keep" weight of sample i
    R_q      = sum_i w_i * e_i / sum_i w_i        # differentiable selective risk

The surrogate is the mean of ``R_q`` over the coverage grid (mirroring the
prefix-mean definition of AURC as the expectation over coverages of the
coverage-specific risk, as in Zhou et al.). The thresholds and any internal
references to the sorted confidence are detached so they are constants w.r.t.
the differentiated parameters; only confidence (hence only model parameters)
carries gradient on the primary path. The error term ``e_i`` may be the hard,
detached 0/1 error (gradient flows only through the ranking term) or a smooth
"error likelihood" ``1 - TCP`` (default; gradient also flows through the risk
magnitude), selectable via ``error_mode``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .selective import COVERAGE_GRID_PERCENT


def _quantile(x: torch.Tensor, q: float) -> torch.Tensor:
    """Detached scalar ``q``-quantile of a 1-D tensor (higher = confident)."""
    x = x.detach()
    n = x.numel()
    if n == 0:
        return torch.zeros((), device=x.device)
    k = int(min(n - 1, max(0, math.floor(q * n))))
    return x.kthvalue(int(k) + 1, dim=0).values


def soft_selective_risk(confidence, error, tau: float,
                        coverages=None) -> torch.Tensor:
    """Differentiable selective risk averaged over a coverage grid.

    ``confidence`` (B,) higher = keep (carries gradient w.r.t. the model).
    ``error`` (B,) error likelihood of each sample (0/1 hard or smooth proxy).
    ``tau`` smoothing temperature > 0. Returns a scalar, lower = better.
    """
    if confidence.numel() == 0:
        return torch.zeros((), device=confidence.device)
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    coverages = list(COVERAGE_GRID_PERCENT if coverages is None else coverages)
    out = torch.zeros((), device=confidence.device)
    for q in coverages:
        q = min(max(q / 100.0, 0.0), 1.0)
        if q == 0.0:
            continue
        t = _quantile(confidence, 1.0 - q)          # keep the top q fraction
        w = torch.sigmoid((confidence - t) / tau)
        denom = w.sum().clamp_min(1e-12)
        out = out + (w * error).sum() / denom
    return out / max(len(coverages), 1)


def soft_aurc_surrogate(logits, targets, confidence=None, tau: float = 0.1,
                        coverages=None, error_mode: str = "proxy") -> torch.Tensor:
    """Differentiable surrogate of AURC from terminal logits.

    Args:
        logits: (B, C) main-class logits (gradient carrier).
        targets: (B,) int labels.
        confidence: optional (B,) confidence; defaults to MSP(logits).
        tau: smoothing temperature.
        coverages: optional list of coverage *percents*; defaults to the locked
            contract grid.
        error_mode: 'proxy' uses the smooth ``1 - softmax_true`` error
            likelihood (default); 'hard' detaches a 0/1 error indicator.

    Returns a scalar ``> 0``; lower = more selective-competent model.
    """
    if confidence is None:
        confidence = F.softmax(logits, dim=1).max(dim=1).values
    with torch.no_grad():
        p_true = torch.softmax(logits, dim=1).gather(
            1, targets.view(-1, 1)).squeeze(1)
    if error_mode == "proxy":
        err = 1.0 - p_true
    elif error_mode == "hard":
        err = (logits.argmax(dim=1) != targets).detach().to(logits.dtype)
    else:
        raise ValueError(f"unknown error_mode {error_mode!r}")
    return soft_selective_risk(confidence, err, tau, coverages)


def selective_surrogate_gradient(soft_aurc_val, params, retain_graph=True):
    """Gradients of the surrogate scalar wrt a list of params."""
    grads = torch.autograd.grad(
        soft_aurc_val, params, create_graph=False, retain_graph=retain_graph,
        allow_unused=True, materialize_grads=True,
    )
    return grads


__all__ = [
    "soft_selective_risk",
    "soft_aurc_surrogate",
    "selective_surrogate_gradient",
    "_quantile",
]