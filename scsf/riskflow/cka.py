"""Linear CKA and pairwise-correlation diagnostics shared by RiskFlow.

These routines measure how redundant a set of per-depth score/innovation
columns are. The novelty mechanism of RiskFlow is supported only if the
residual innovations are *less* redundant across depth than independent
intermediate heads, while the cumulative risk is *more* predictive. The two
redundancy measures here are used in the required comparisons:

* ``pairwise_linear_cka(mat) -> (L, L)`` — linear CKA (centered) between every
  pair of columns of an ``(N, L)`` score matrix.
* ``cross_depth_correlation(mat) -> (L, L)`` — Pearson correlation between
  pairs of columns, robust to a zero-variance column (returns ``nan`` pinned
  to ``0.0`` so a constant column is reported *not* perfectly redundant).

Both are pure-numpy so they can also be run on exported artifacts without
torch.
"""

from __future__ import annotations

import numpy as np


def _center(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=0, keepdims=True)


def pairwise_linear_cka(mat: np.ndarray) -> np.ndarray:
    """Linear (centered) CKA matrix between columns of ``mat``.

    ``mat`` is ``(N, L)``: one row per example, one column per depth. Uses the
    centered (linear) kernel so the result is the standard centered linear CKA
    between column features: CKA(i,j) = <K_i,K_j>_F / (||K_i||_F ||K_j||_F)
    with K_i = X_i X_i^T, which reduces to (X_i^T X_j)^2 / (||X_i||^2 ||X_j||^2).
    """
    x = _center(np.asarray(mat, dtype=np.float64))   # (N, L)
    L = x.shape[1]
    cols = [x[:, i] for i in range(L)]
    out = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            d_ij = float(cols[i] @ cols[j])
            den = float((cols[i] @ cols[i]) * (cols[j] @ cols[j]))
            out[i, j] = (d_ij * d_ij / den) if den > 1e-12 else (1.0 if i == j else 0.0)
    return out


def cross_depth_correlation(mat: np.ndarray) -> np.ndarray:
    """Pearson correlation between columns of ``mat``, zero variance->0.0.

    A zero-variance column has undefined correlation; reporting ``nan`` here
    is fine but we map it to ``0.0`` (off-diagonal) so downstream summaries
    never poison the aggregate. The diagonal stays 1.0.
    """
    mat = np.asarray(mat, dtype=np.float64)
    L = mat.shape[1]
    out = np.full((L, L), np.nan)
    for i in range(L):
        for j in range(L):
            a, b = mat[:, i], mat[:, j]
            va, vb = float(np.var(a)), float(np.var(b))
            if va < 1e-12 or vb < 1e-12:
                out[i, j] = 1.0 if i == j else 0.0
                continue
            out[i, j] = float(np.corrcoef(a, b)[0, 1])
    return out


def off_diagonal_mean(mat: np.ndarray) -> float:
    """Mean of the strictly off-diagonal entries (redundancy summary)."""
    mat = np.asarray(mat, dtype=np.float64)
    n = mat.shape[0]
    if n < 2:
        return float("nan")
    iu = np.triu_indices(n, k=1)
    return float(np.abs(mat[iu]).mean())


__all__ = [
    "cross_depth_correlation",
    "off_diagonal_mean",
    "pairwise_linear_cka",
]
