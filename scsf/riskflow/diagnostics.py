"""RiskFlow diagnostics: per-depth exports, redundancy, and trajectory plots.

The novelty claim of RiskFlow rests on the measurements here: residual
innovations must be *less* redundant across depth than independent-head
outputs, while cumulative risk is *more* predictive. This module exposes that
evidence.

* ``export_trace`` — per-example/per-depth numpy arrays of gate values,
  innovation logits/vectors, cumulative risk logits, and hard/soft
  pseudo-residual targets.
* ``redundancy_report`` — cross-depth correlation and linear CKA for (a)
  independent-head outputs and (b) RiskFlow innovations, with a one-row
  summary suitable for a ledger.
* ``assign_category`` — fixed config rule for the four requested trajectory
  categories (easy-correct, ambiguous-correct, high-confidence-wrong,
  corrupted). Assignment uses the config thresholds, never hand selection.
* ``save_trajectory_plots`` — matplotlib time-series plots for one example of
  each category.

Every function is pure-numpy/torch-agnostic where possible so it can also run
on saved artifacts.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from .cka import cross_depth_correlation, pairwise_linear_cka

CATEGORIES = ("easy_correct", "ambiguous_correct", "high_conf_wrong", "corrupted")


def assign_category(hard_error: np.ndarray, final_risk: np.ndarray,
                    cat_lo: float = 0.3, cat_hi: float = 0.7) -> np.ndarray:
    """Fixed rule: map each example to a trajectory category.

    ``final_risk`` is the final risk logit ``s_L`` (higher -> higher risk):
    * easy-correct         : correct and ``risk < cat_lo``
    * ambiguous-correct    : correct and ``cat_lo <= risk < cat_hi``
    * high-confidence-wrong: wrong and ``risk < cat_lo``
    * corrupted            : ``risk >= cat_hi`` (highest-risk pile)
    """
    err = np.asarray(hard_error, dtype=bool)
    risk = np.asarray(final_risk, dtype=np.float64)
    out = np.empty(risk.shape[0], dtype=object)
    out[~err & (risk < cat_lo)] = "easy_correct"
    out[~err & (risk >= cat_lo) & (risk < cat_hi)] = "ambiguous_correct"
    out[err & (risk < cat_lo)] = "high_conf_wrong"
    out[risk >= cat_hi] = "corrupted"
    return out


def export_trace(trace, hard_error=None, soft_target=None) -> Dict[str, np.ndarray]:
    """Convert a ``RiskFlowTrace`` into per-depth numpy arrays.

    Returns a dict with keys (all numpy, float32 unless noted)::

        site_names, s_hard (L+1,B) [, s_soft (L+1,B)], innov_hard (L,B),
        [innov_soft (L,B),] gates (L,B) [, deltas (L,B,D),]
        eps_hard (L,B) [, eps_soft (L,B),] final_s_hard (B,), hard_error (B,)
        [, final_s_soft (B,), soft_target (B,)]
    """
    out: Dict[str, np.ndarray] = {
        "site_names": np.asarray(trace.site_names, dtype=object),
        "s_hard": trace.s_hard.detach().cpu().numpy().astype(np.float32),
        "final_s_hard": trace.final_s_hard.detach().cpu().numpy().astype(np.float32),
        "innov_hard": trace.innov_hard.detach().cpu().numpy().astype(np.float32)
        if trace.innov_hard is not None else None,
        "gates": trace.gates.detach().cpu().numpy().astype(np.float32)
        if trace.gates is not None else None,
        "deltas": trace.deltas.detach().cpu().numpy().astype(np.float32)
        if trace.deltas is not None else None,
        "eps_hard": trace.eps_hard.detach().cpu().numpy().astype(np.float32)
        if trace.eps_hard is not None else None,
        "prediction": trace.prediction.detach().cpu().numpy().astype(np.int64),
    }
    if trace.s_soft is not None:
        out["s_soft"] = trace.s_soft.detach().cpu().numpy().astype(np.float32)
        out["final_s_soft"] = trace.final_s_soft.detach().cpu().numpy().astype(np.float32)
        out["innov_soft"] = trace.innov_soft.detach().cpu().numpy().astype(np.float32)
        out["eps_soft"] = trace.eps_soft.detach().cpu().numpy().astype(np.float32)
    h = hard_error if hard_error is not None else (trace.hard_error.detach().cpu().numpy()
                                                   if trace.hard_error is not None else None)
    if h is not None:
        out["hard_error"] = np.asarray(h).astype(np.float32)
    st = soft_target if soft_target is not None else (
        trace.soft_target.detach().cpu().numpy() if trace.soft_target is not None else None)
    if st is not None:
        out["soft_target"] = np.asarray(st).astype(np.float32)
    for k in list(out):
        if out[k] is None:
            del out[k]
    return out


def redundancy_report(independent_heads: np.ndarray, innovations: np.ndarray,
                      cumulative_risk: np.ndarray) -> Dict[str, float]:
    """Cross-depth correlation + linear CKA for the two competing structures.

    ``independent_heads`` is ``(N, L)`` of independent-head outputs (the
    ``heads`` mode).
    ``innovations`` is ``(N, L)`` of RiskFlow residual logits (innov_hard).
    ``cumulative_risk`` is ``(N, L)`` of RiskFlow cumulative risk logits
    (the ``s_hard`` rows after each depth).

    Returns a summary dict used in the ledger / final report.
    """
    cka_ind = pairwise_linear_cka(independent_heads)
    cka_inn = pairwise_linear_cka(innovations)
    corr_ind = cross_depth_correlation(independent_heads)
    corr_inn = cross_depth_correlation(innovations)
    return {
        "cka_offdiag_mean_independent_heads": float(
            _odm(cka_ind)),
        "cka_offdiag_mean_innovations": float(_odm(cka_inn)),
        "corr_offdiag_mean_independent_heads": float(_odm(corr_ind)),
        "corr_offdiag_mean_innovations": float(_odm(corr_inn)),
        "innov_redundancy_ratio_cka": float(_safe(_odm(cka_inn)) / _safe(_odm(cka_ind))),
        "innov_redundancy_ratio_corr": float(_safe(_odm(corr_inn)) / _safe(_odm(corr_ind))),
        "cumulative_risk_corr_final": float(
            np.corrcoef(cumulative_risk[:, 0], cumulative_risk[:, -1])[0, 1])
        if cumulative_risk.shape[1] >= 2 and cumulative_risk.shape[0] >= 2 and
        np.var(cumulative_risk[:, 0]) > 1e-12 and np.var(cumulative_risk[:, -1]) > 1e-12
        else 0.0,
    }


def _odm(mat: np.ndarray) -> float:
    from .cka import off_diagonal_mean
    return float(off_diagonal_mean(mat))


def _safe(x: float) -> float:
    return float(x) if np.isfinite(x) else float("nan")


def save_trajectory_plots(data: Dict[str, np.ndarray], out_dir: str,
                          category_key: Optional[np.ndarray] = None,
                          hard_error: Optional[np.ndarray] = None,
                          final_risk: Optional[np.ndarray] = None) -> List[str]:
    """Plot per-example trajectories: one representative per category.

    ``data`` is the output of :func:`export_trace`. If ``category_key`` is
    None, categories are derived with :func:`assign_category` from
    ``hard_error`` and ``final_risk`` using default thresholds.

    Returns the list of saved PNG paths (one per category). If matplotlib is
    unavailable, returns an empty list (plots are optional diagnostics).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:                       # pragma: no cover - optional dep
        print("[riskflow] matplotlib unavailable; skipping trajectory plots")
        return []

    os.makedirs(out_dir, exist_ok=True)
    if category_key is None:
        category_key = assign_category(hard_error, final_risk)
    sites = list(data.get("site_names", []))
    x = np.arange(len(sites)) if sites else np.arange(
        data.get("s_hard", np.zeros((1, 1))).shape[0] - 1)

    saved = []
    for cat in CATEGORIES:
        idx = np.where(category_key == cat)[0]
        if len(idx) == 0:
            continue
        i = int(idx[0])
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
        s_hard = data["s_hard"][:, i] if "s_hard" in data else None
        if s_hard is not None:
            axes[0, 0].plot(np.arange(s_hard.shape[0]), s_hard, "o-", label="s_hard")
            axes[0, 0].set_title(f"{cat} [ex {i}] cumulative risk logit")
        if "innov_hard" in data:
            axes[0, 1].plot(x, data["innov_hard"][:, i], "s-", label="innov_hard")
            axes[0, 1].set_title("innovation logits (s_l - s_{l-1})")
        if "gates" in data and data["gates"] is not None:
            axes[1, 0].plot(x, data["gates"][:, i], "^--", label="gate")
            axes[1, 0].set_title("per-depth sample-dependent gates")
        if "deltas" in data and data["deltas"] is not None:
            axes[1, 1].plot(x,
                            np.linalg.norm(data["deltas"][:, i], axis=-1), "d-",
                            label="||delta_r_l||")
            axes[1, 1].set_title("innovation vector norm")
        for a in axes.flat:
            a.legend(loc="best", fontsize=8)
            a.grid(True, alpha=0.3)
        fig.suptitle(f"RiskFlow trajectory: {cat} (example {i})", fontsize=12)
        fig.tight_layout()
        path = os.path.join(out_dir, f"trajectory_{cat}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        saved.append(path)
    return saved


__all__ = [
    "CATEGORIES",
    "assign_category",
    "export_trace",
    "redundancy_report",
    "save_trajectory_plots",
]
