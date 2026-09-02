"""Iterative local boundary approximation (DeepFool-style) for audit.

This is **not** an adversarial-distance or robustness guarantee. It is a
carefully documented, local, linearized walk toward the *decision boundary in
input space*, used only to cross-check that the analytic feature-space radii
from :mod:`scsf.depthfrag.geometry` carry the same fragility signal. Because
the analytic radius lives in feature space at a tap while the iterative walk
accumulates a norm in input-pixel space, the comparison is reported as
correlation plus a scale-normalized relative error, never as an equality.

Algorithm (documented equivalent of DeepFool for the nearest competitor)
-------------------------------------------------------------------------
Starting from ``x`` with true label ``y`` and current prediction ``k``:

1. If ``k != y`` we have already crossed the boundary (recorded, distance 0).
2. Let ``c`` be the *current nearest competitor*: the non-``y`` class with
   the largest logit. Write ``f = z_y - z_c`` (positive while correct).
3. Linearize around ``x_t``: ``f(x_t + d) ~= f(x_t) + <g, d>`` with
   ``g = d f / d x_t``. The minimal-norm displacement to the local plane
   ``f = 0`` is ``d = -f g / ||g||^2``, whose norm ``|f| / ||g||`` is the
   local linear boundary distance.
4. Step once along ``d`` (no line search — the classic DeepFool pricing),
   accumulate ``||d||``, and repeat up to ``max_steps`` times.

Only the current nearest competitor is considered each iteration (instead of
the full min-over-classes DeepFool step), which we document explicitly; the
walk therefore measures a local, nearest-competitor boundary distance. The
backward passes here use plain ``autograd.grad`` on the input: first order,
no graph retained, no second-order quantities enter the audit.

Reports (see :func:`iterative_boundary_audit`) collect per-sample accumulated
distance, steps used, and whether the walk flipped the prediction, plus the
mean wall cost across the audited subset.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


def iterative_boundary(backbone, x_t: torch.Tensor, y: int, max_steps: int = 50,
                       tol: float = 1e-6, eps_denom: float = 1e-12) -> Dict[str, float]:
    """Per-example DeepFool-style walk; returns dict (dist, steps, flipped)."""
    param_dtype = next(backbone.parameters()).dtype
    x_t = x_t.to(dtype=param_dtype).clone().requires_grad_(True).unsqueeze(0)
    dist = 0.0
    steps = 0
    flipped = False
    y = int(y)
    while steps < max_steps:
        if not x_t.requires_grad:
            x_t = x_t.detach().requires_grad_(True)
        with torch.enable_grad():
            bo = backbone(x_t)
            z = bo.logits[:, :].squeeze(0)
            pred = int(z.argmax().detach().item())

        if pred != y:
            flipped = (pred != y)
            break
        # nearest competitor class (max logit among c != y)
        zy = float(z[y].detach())
        mask = torch.arange(z.shape[0], device=z.device) != y
        zc = float(z.detach()[mask].max())
        f = zy - zc
        if f <= 0.0:
            flipped = False  # at/beyond boundary under linearization
            break
        with torch.enable_grad():
            g = torch.autograd.grad(z[y] - z[mask].max(), x_t, retain_graph=False)[0]
        denom = float(g.pow(2).sum()) + eps_denom
        g_norm = float(g.norm().item())
        if g_norm < 1e-30:
            break
        step_len = f / g_norm if g_norm > 0 else 0.0
        d = (-f / denom) * g
        x_t = (x_t + d).detach().requires_grad_(True)
        dist += float(d.norm().item())
        steps += 1
        if step_len < tol:
            break
    return {"dist": dist, "steps": steps, "flipped": bool(flipped)}


def iterative_boundary_audit(backbone, x: torch.Tensor, y: torch.Tensor,
                             max_steps: int = 50, batch_size: int = 32,
                             tol: float = 1e-6) -> Dict[str, object]:
    """Run the walk over a fixed subset; return per-sample + aggregate results.

    Returns
    -------
    per_sample : np.ndarray of (dist, steps, flipped) records
    summary    : dict with mean steps, mean flipped (label-flip success),
                 mean dist, and wall-clock ms.
    """
    was_training = backbone.training
    backbone.eval()
    records: List[dict] = []
    t0 = time.perf_counter()
    for i in range(x.shape[0]):
        xi = x[i].to(next(backbone.parameters()).device)
        r = iterative_boundary(backbone, xi, int(y[i]), max_steps=max_steps, tol=tol)
        records.append(r)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    if was_training:
        backbone.train()
    n = len(records)
    per_sample = np.asarray(
        [(r["dist"], r["steps"], float(r["flipped"])) for r in records],
        dtype=[("dist", "f8"), ("steps", "i8"), ("flipped", "f8")],
    )
    summary = {
        "n": int(n),
        "mean_dist": float(np.mean(per_sample["dist"])),
        "mean_steps": float(np.mean(per_sample["steps"])),
        "std_steps": float(np.std(per_sample["steps"])),
        "label_flip_success": float(np.mean(per_sample["flipped"])),
        "wall_ms": float(wall_ms),
        "ms_per_sample": float(wall_ms / max(n, 1)),
    }
    return {"per_sample": per_sample, "summary": summary}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank (Spearman) correlation; NaN-safe on constant inputs."""
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.size < 2 or b.size < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def relative_scale_error(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    """Scale-normalized relative error ``mean(|aN - bN| / max(|bN|, eps))``.

    Both arrays are divided by their median absolute value before computing
    the per-sample relative error, so the (physically different) analytic and
    iterative quantities are compared on a common relative scale.
    """
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.size == 0 or b.size == 0:
        return float("nan")
    ma = np.median(np.abs(a)) + eps
    mb = np.median(np.abs(b)) + eps
    aN = a / ma
    bN = b / mb
    return float(np.mean(np.abs(aN - bN) / (np.abs(bN) + eps)))


def compare_analytic_iterative(analytic: np.ndarray, iterative: np.ndarray) -> Dict[str, float]:
    """Bundle the required audit numbers for one (analytic, iterative) pair."""
    return {
        "spearman": spearman(analytic, iterative),
        "relative_error": relative_scale_error(analytic, iterative),
        "n": int(len(np.asarray(analytic).reshape(-1))),
    }


__all__ = [
    "iterative_boundary",
    "iterative_boundary_audit",
    "compare_analytic_iterative",
    "relative_scale_error",
    "spearman",
]