"""Validation-fitted oracles (diagnostic upper bounds, not deployment scores).

The DepthFrag extractor compares its geometry scores against two oracle
aggregations of the raw depth profile. Both are fitted **on the validation
split only** and applied to the untouched test split; the oracle is a
diagnostic sanity check for how much signal the raw profile carries, never the
proposed deployment method.

* ``oracle_lin``   — standardized linear least-squares fit of the error
  indicator on the per-site profile.
* ``oracle_logit`` — standardized ridge logistic regression (L-BFGS-B via
  scipy.optimize) on the same features.

Both return a *confidence-style* score: the oracle predicts uncertainty
(``<error probability> = 1[wrong]`` ---- 1[incorrect]), so confidence is the
negative prediction. Only the ordering counts for the metrics.
"""

from __future__ import annotations

import functools
import os
from typing import Dict, Optional, Tuple

import numpy as np


def _standardize(profile: np.ndarray, mean: Optional[np.ndarray] = None,
                 std: Optional[np.ndarray] = None, eps: float = 1e-9):
    profile = np.asarray(profile, dtype=float)
    if profile.ndim == 1:
        profile = profile.reshape(-1, 1)
    if mean is None:
        mean = profile.mean(axis=0)
    if std is None:
        std = profile.std(axis=0) + eps
    X = (profile - mean) / std
    return np.concatenate([X, np.ones((profile.shape[0], 1))], axis=1), mean, std


def _errors(labels, predictions):
    labels = np.asarray(labels).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)
    return (predictions != labels).astype(float)


class FitOracle:
    """Standardized linear/logistic oracle fitted on a single split."""

    def __init__(self, variant: str = "logit", ridge: float = 1e-3):
        if variant not in ("lin", "logit"):
            raise ValueError(f"variant must be lin or logit, got {variant!r}")
        self.variant = variant
        self.ridge = float(ridge)
        self.mean = None
        self.std = None
        self.coef = None

    def fit(self, profile, labels, predictions) -> "FitOracle":
        err = _errors(labels, predictions)
        X, self.mean, self.std = _standardize(profile)
        if self.variant == "lin":
            A = X.T @ X + self.ridge * np.eye(X.shape[1])
            self.coef = np.linalg.solve(A, X.T @ err)
        else:
            self.coef = self._fit_logistic(X, err)
        return self

    def _fit_logistic(self, X, err, iters: int = 400):
        from scipy.optimize import minimize

        w = np.zeros(X.shape[1])

        def loss(w):
            z = X @ w
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            ll = -(err * np.log(p + 1e-12) + (1 - err) * np.log(1 - p + 1e-12)).mean()
            reg = 0.5 * self.ridge * (w[:-1] ** 2).sum()
            return ll + reg

        def grad(w):
            z = X @ w
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            g = X.T @ (p - err) / len(err) + self.ridge * np.concatenate(
                [w[:-1], [0.0]])
            return g

        res = minimize(loss, w, jac=grad, method="L-BFGS-B",
                       options={"maxiter": iters})
        return res.x

    def confidence(self, profile) -> np.ndarray:
        """Confidence-style score = negative predicted uncertainty."""
        if self.coef is None:
            raise RuntimeError("fit() must run before confidence()")
        X, _, _ = _standardize(profile, self.mean, self.std)
        if self.variant == "lin":
            pred = X @ self.coef
        else:
            pred = 1.0 / (1.0 + np.exp(-np.clip(X @ self.coef, -30, 30)))
        return -pred


def fit_and_apply(profile_fit, labels_fit, predictions_fit,
                  profile_apply, variant: str = "logit") -> Tuple[np.ndarray, Dict]:
    """Fit on (fit) features, return confidence on the (apply) split."""
    oracle = FitOracle(variant)
    oracle.fit(profile_fit, labels_fit, predictions_fit)
    conf = oracle.confidence(profile_apply)
    return conf, {"variant": variant, "mean": list(oracle.mean),
                  "std": list(oracle.std), "coef": list(oracle.coef)}


@functools.lru_cache(maxsize=1)
def _scipy_available() -> bool:
    try:
        import scipy  # noqa: F401
        return True
    except Exception:
        return False


__all__ = ["FitOracle", "fit_and_apply", "_errors"]