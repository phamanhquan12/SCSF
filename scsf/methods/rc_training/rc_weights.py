"""Exact harmonic RC weights ``W_n`` and the coverage-window variant.

All helpers return **detached training weights**: ranks, hard errors and
weights are stop-gradient; the useful gradient flows through CE, ranking or
soft-acceptance losses elsewhere. The fixed-rank RC identities describe one
intervention in isolation, not a finite-step SGD guarantee.

Normalization and clipping order is explicit (and test-locked):

1. ``harmonic_weights`` computes ``w_raw = W_n(r)`` on the stable-ordered
   detached confidence.
2. ``normalize_clip`` normalizes to mean-1 **over the incorrect set**, then
   clips large values at ``clip_max``.

Sum vs mean vs raw sums are distinguished in telemetry names by the caller.
Empty masks / empty pair sets produce finite zero values (the losses handle
that); this module never produces NaN for valid positive weights.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch


def rank_of(confidences, ids=None):
    """Stable 1-based rank by descending confidence; ties by ascending id.

    ``ids`` is the deterministic secondary key (independent of correctness),
    matching the evaluator's tie rule. Returns a ``(n,)`` long tensor built
    from detached copies, so it carries no autograd graph.
    """
    conf = confidences.detach().flatten()
    n = conf.shape[0]
    if ids is None:
        ids = torch.arange(n, device=conf.device)
    ids = ids.detach().flatten()
    order = torch.lexsort((ids, -conf))
    rank = torch.empty(n, dtype=torch.long, device=conf.device)
    rank[order] = torch.arange(1, n + 1, device=conf.device)
    return rank


def harmonic_weights_at_ranks(n, ranks):
    """``W_n(r) = (1/n) * sum_{k=r..n} 1/k`` gathered at 1-based ``ranks``."""
    n = int(n)
    inv = 1.0 / torch.arange(1, n + 1, dtype=torch.float32, device=ranks.device)
    suffix = torch.cumsum(inv.flip(0), dim=0).flip(0)   # suffix[r-1] = sum_{k=r..n} 1/k
    return suffix[ranks.clamp(1, n) - 1] / n


def harmonic_weights(confidences, ids=None, clip_max: Optional[float] = None):
    """Exact harmonic weights ``W_n(r_i)`` for a batch."""
    n = confidences.shape[0]
    ranks = rank_of(confidences, ids)
    w = harmonic_weights_at_ranks(n, ranks)
    if clip_max is not None:
        w = torch.clamp(w, max=float(clip_max))
    return w


def normalize_mean1(weights, mask):
    """Mean-1 normalization over the masked subset (e.g. incorrect examples).

    Returns ``(weights_normed, applied: bool)``; ``applied`` is ``False`` when
    the mask is empty (the returned tensor is a clone, never NaN).
    """
    m = mask.detach().bool()
    if not m.any():
        return weights.detach().clone(), False
    mean = weights.detach()[m].mean()
    if not bool(mean > 0):
        return weights.detach().clone(), False
    return weights.detach() / mean, True


def normalize_clip(weights, mask, clip_max: Optional[float] = None):
    """Documented order: mean-1 over the masked set, then clip large values."""
    out, _ = normalize_mean1(weights, mask)
    if clip_max is not None:
        out = torch.clamp(out, max=float(clip_max))
    return out


def coverage_window_weights(n: int, ks: Sequence[int], device=None):
    """Partial-RC weights: ``Wp(r) = sum_{k in ks, k >= r} 1/k``.

    With a single ``k = max(1, floor(c*n))`` this makes
    ``sum_i e_i * Wp(r_i)`` exactly the risk at coverage ``c`` (prefix error
    count over ``k``), so the coverage-window variant targets a partial RC
    instead of full AURC.
    """
    n = int(n)
    dev = device or torch.device("cpu")
    inv = 1.0 / torch.arange(1, n + 1, dtype=torch.float32, device=dev)
    sel = torch.zeros(n, dtype=torch.float32, device=dev)
    for k in ks:
        k = int(min(max(k, 1), n))
        sel[k - 1] += inv[k - 1]
    return torch.cumsum(sel.flip(0), dim=0).flip(0)


__all__ = [
    "coverage_window_weights",
    "harmonic_weights",
    "harmonic_weights_at_ranks",
    "normalize_clip",
    "normalize_mean1",
    "rank_of",
]