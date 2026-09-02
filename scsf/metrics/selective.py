"""Exact selective-classification metrics (the empirical contract, §4).

Conventions
-----------
* ``confidence`` : higher = more confident (scores sort **descending**).
* ``uncertainty`` = ``-confidence`` (or a native risk score).
* ``error = 1[prediction != label]`` is the **positive** class throughout.
* ``ids`` is the deterministic secondary sort key (ascending) so tied scores
  are fully deterministic, per contract §4.

Definitions implemented (unit-tested with hand-computed cases):
* ``selective_risk_at_coverages``: accept exactly ``k = max(1, floor(q*N/100))``
  most-confident samples for every q in the contract grid
  ``100,99,95,90,85,...,5,1``.
* ``aurc``: empirical AURC over **all** accepted prefixes k=1..N
  (mean risk, not a trapezoid over the hard-coverage grid).
* ``auroc_error``: failure-detection AUROC, ``u = -confidence``, error positive.
* ``aupr_error``: average precision over the same ordering.
* ``excess_aurc``: ``aurc - optimal AURC at the same empirical error rate``.
* ``per_class_aurc`` / ``worst_class_aurc``: class-restricted AURC
  (scores of that class's examples only), worst = largest (bad) AURC.
* ``risk_coverage_curve``: full empirical curve for all prefix sizes.

Degenerate cases (all-correct / all-wrong / single class) return NaN for
undefined quantities rather than a fabricated number.
"""

from __future__ import annotations

import numpy as np

from .roc import average_precision, roc_auc

#: Locked hard-coverage grid from the empirical contract (§4).
COVERAGE_GRID_PERCENT = [100, 99, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50,
                         45, 40, 35, 30, 25, 20, 15, 10, 5, 1]


def stable_confidence_order(confidence, ids=None, n=None):
    """Indices ordering samples most-confident-first with id tie-break.

    Returns an int array of length n giving the descending-confidence order;
    ties are broken by ascending sample id (fully deterministic).
    """
    confidence = np.asarray(confidence, dtype=float).reshape(-1)
    n = len(confidence) if n is None else int(n)
    if ids is None:
        ids = np.arange(n)
    else:
        ids = np.asarray(ids).reshape(-1)
    return np.lexsort((ids, -confidence))


def errors(labels, predictions):
    labels = np.asarray(labels).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)
    if labels.shape[0] != predictions.shape[0]:
        raise ValueError("length mismatch")
    return (predictions != labels).astype(int)


def risk_coverage_curve(labels, predictions, confidence, ids=None):
    """Full empirical risk-coverage curve after descending-score sorting.

    Returns (coverage_frac, risk) arrays over accepted-prefix sizes 1..N.
    """
    n = len(np.asarray(labels).reshape(-1))
    if n == 0:
        raise ValueError("empty sample set")
    err = errors(labels, predictions)
    order = stable_confidence_order(confidence, ids, n)
    err_sorted = err[order]
    cum_err = np.cumsum(err_sorted, dtype=float)
    k = np.arange(1, n + 1, dtype=float)
    risk = cum_err / k
    return k / n, risk


def aurc(labels, predictions, confidence, ids=None) -> float:
    """Empirical AURC over all accepted prefixes k = 1..N (lower is better)."""
    _, risk = risk_coverage_curve(labels, predictions, confidence, ids)
    return float(np.mean(risk))


def auroc_error(labels, predictions, confidence, ids=None) -> float:
    """Failure-detection AUROC with error as positive class (higher better)."""
    err = errors(labels, predictions)
    uncertainty = -np.asarray(confidence, dtype=float).reshape(-1)
    return roc_auc(uncertainty, err.astype(bool))


def aupr_error(labels, predictions, confidence, ids=None) -> float:
    """AUPR for error detection with error as positive class."""
    err = errors(labels, predictions)
    uncertainty = -np.asarray(confidence, dtype=float).reshape(-1)
    return average_precision(uncertainty, err.astype(bool), ids)


def selective_risk_at_coverages(labels, predictions, confidence, ids=None,
                                coverages=None):
    """Error fraction at the hard-coverage grid (contract §4).

    Returns a list of dicts:
      {coverage, k, n, accepted_frac, risk}
    with k = max(1, floor(q*N/100)) and risk = error fraction among those k.
    """
    coverages = list(COVERAGE_GRID_PERCENT if coverages is None else coverages)
    n = len(np.asarray(labels).reshape(-1))
    err = errors(labels, predictions)
    order = stable_confidence_order(confidence, ids, n)
    err_sorted = err[order]
    cum_err = np.cumsum(err_sorted, dtype=float)
    out = []
    for q in coverages:
        k = max(1, int(np.floor(q * n / 100.0)))
        k = min(k, n)
        risk = float(cum_err[k - 1]) / k
        out.append(
            {
                "coverage": int(q),
                "k": int(k),
                "n": int(n),
                "accepted_frac": float(k) / n,
                "risk": float(risk),
            }
        )
    return out


def optimal_aurc(num_errors: int, n: int) -> float:
    """Optimal AURC at the same empirical error rate (perfect selector)."""
    n = int(n)
    e = int(num_errors)
    if e > n or e < 0:
        raise ValueError("num_errors out of range")
    k = np.arange(1, n + 1, dtype=float)
    opt_risk = np.maximum(0.0, k - (n - e)) / k
    return float(np.mean(opt_risk))


def excess_aurc(labels, predictions, confidence, ids=None) -> float:
    """AURC minus the optimal AURC at the same empirical error rate."""
    n = len(np.asarray(labels).reshape(-1))
    e = int(errors(labels, predictions).sum())
    return aurc(labels, predictions, confidence, ids) - optimal_aurc(e, n)


def per_class_aurc(labels, predictions, confidence, ids=None, num_classes=None):
    """Class-restricted AURC (each class's own samples scored by itself)."""
    labels = np.asarray(labels).reshape(-1)
    num_classes = int(labels.max()) + 1 if num_classes is None else int(num_classes)
    out = {}
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            out[c] = float("nan")
            continue
        out[c] = aurc(labels[mask], np.asarray(predictions).reshape(-1)[mask],
                      np.asarray(confidence).reshape(-1)[mask],
                      None if ids is None else np.asarray(ids).reshape(-1)[mask])
    return out


def worst_class_aurc(labels, predictions, confidence, ids=None, num_classes=None) -> float:
    """Worst-class AURC = max over classes of class-restricted AURC."""
    vals = per_class_aurc(labels, predictions, confidence, ids, num_classes).values()
    finite = [v for v in vals if not np.isnan(v)]
    if not finite:
        return float("nan")
    return float(max(finite))


def all_metrics(labels, predictions, confidence, ids=None, num_classes=None):
    """Convenience bundle of every metric the evaluator reports."""
    labels = np.asarray(labels).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)
    n = len(labels)
    risk = float(errors(labels, predictions).mean())
    pc = per_class_aurc(labels, predictions, confidence, ids, num_classes)
    return {
        "n": int(n),
        "acc": float((labels == predictions).mean()),
        "err": risk,
        "aurc": aurc(labels, predictions, confidence, ids),
        "auroc_error": auroc_error(labels, predictions, confidence, ids),
        "aupr_error": aupr_error(labels, predictions, confidence, ids),
        "excess_aurc": excess_aurc(labels, predictions, confidence, ids),
        "mean_class_aurc": float(np.nanmean(list(pc.values()))),
        "worst_class_aurc": worst_class_aurc(labels, predictions, confidence, ids, num_classes),
        "per_class_aurc": {int(k): float(v) for k, v in pc.items()},
    }