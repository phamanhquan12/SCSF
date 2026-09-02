"""Rank-based AUC / average-precision helpers (numpy only, no sklearn).

Orientation convention for failure prediction: ``score`` is the *risk /
uncertainty* score, higher = more likely to be the positive class. The caller
is responsible for converting confidence to uncertainty (``u = -confidence``).
Average-rank ties are used so tied scores contribute exactly their fractional
rank (this is what makes the ROC curve fully determined for tied scores).

Degenerate cases return NaN (undefined) — never a fabricated number:
  * AUROC with zero positives or zero negatives,
  * AUPR with zero positives.
"""

from __future__ import annotations

import numpy as np


def _average_ranks(sorted_vals: np.ndarray) -> np.ndarray:
    """Return average ranks (1-based) for values sorted ascending.

    Runs of equal values share the mean of their adjacent integer ranks.
    """
    n = len(sorted_vals)
    ranks = np.arange(1, n + 1, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            avg = (ranks[i] + ranks[j]) / 2.0
            ranks[i : j + 1] = avg
        i = j + 1
    return ranks


def roc_auc(risk_scores: np.ndarray, positive: np.ndarray) -> float:
    """ROC-AUC with risk_scores as the ranking score; higher→positive.

    Equivalent to the Mann-Whitney U statistic over ranks; identical to
    ``sklearn.metrics.roc_auc_score`` up to float rounding.
    """
    risk_scores = np.asarray(risk_scores, dtype=float).reshape(-1)
    positive = np.asarray(positive).reshape(-1).astype(bool)
    if risk_scores.shape[0] != positive.shape[0]:
        raise ValueError("score/label length mismatch")
    n_pos = int(positive.sum())
    n_neg = len(positive) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(risk_scores, kind="stable")           # ascending score
    sorted_risk = risk_scores[order]
    avg_ranks_sorted = _average_ranks(sorted_risk)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    ranks_per_sample = avg_ranks_sorted[inv]
    positive_ranks_sum = float(ranks_per_sample[positive].sum())
    auc = (positive_ranks_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def average_precision(risk_scores: np.ndarray, positive: np.ndarray, ids=None) -> float:
    """Average precision of the risk score over the positive class.

    Relevant items (positives) are ranked by descending risk; the value is the
    mean precision at each relevant prefix (PASCAL-VOC style, ties ordered by
    ascending id so the definition is fully deterministic).
    """
    risk_scores = np.asarray(risk_scores, dtype=float).reshape(-1)
    positive = np.asarray(positive).reshape(-1).astype(bool)
    if risk_scores.shape[0] != positive.shape[0]:
        raise ValueError("score/label length mismatch")
    n_pos = int(positive.sum())
    if n_pos == 0:
        return float("nan")
    ids = np.arange(len(positive)) if ids is None else np.asarray(ids).reshape(-1)
    order = np.lexsort((ids, -risk_scores))                  # desc score, asc id
    pos_order = positive[order]
    n_ranked = 0
    n_pos_so_far = 0
    precisions = []
    for is_pos in pos_order:
        n_ranked += 1
        if is_pos:
            n_pos_so_far += 1
            precisions.append(n_pos_so_far / n_ranked)
    if not precisions:
        return float("nan")
    return float(np.mean(precisions))