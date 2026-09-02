"""Frozen-checkpoint DepthFrag geometry extraction + score-variant evaluation.

``DepthFragExtractor`` loads an ordinary checkpoint's backbone and, with the
backbone in eval role, computes sample-level signed radius profiles
(``sample id + site name -> signed relative radius`` plus absolute radius and
regression target) over a split. The prompt-locked score ladder is then
evaluated: MSP, terminal logit margin, terminal normalized radius, one
intermediate radius, raw full-depth profile, minimum / soft-minimum / mean
radius, and the validation-fitted standardized linear/logistic oracles — all
compared on the untouched test split.

The extraction itself is first-order only (``autograd.grad`` with
``create_graph=False``) and releases the graph after each site batch; the
returned profiles are detached numpy arrays. No second-order graph is
retained.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..data import build_dataloader
from ..engine.checkpoint import CheckpointManager
from ..metrics import all_metrics

__all__ = [
    "DepthFragExtractor",
    "score_variants_from_profiles",
    "evaluate_variants",
    "SCORE_VARIANTS",
]

SCORE_VARIANTS = ("msp", "logit_margin", "term_radius", "min_radius",
                  "softmin_radius", "mean_radius", "mid_radius",
                  "oracle_lin", "oracle_logit")


class DepthFragExtractor:
    """Sample-level signed-radius profiles from an ordinary checkpoint."""

    def __init__(self, cfg: dict, run_dir: str, checkpoint: str = "selected",
                 device: str | None = None, p: float = 2, q: float = 2,
                 eps: float = 1e-12, target_kind: str = "signed_log1p",
                 mode: str = "fast", exact_microbatch: int = 1,
                 mid_roles: Sequence[str] = ("top_l2",)):
        from ..methods import build_method

        dev = torch.device(device or cfg["train"].get("device", "cpu"))
        self.cfg = cfg
        self.run_dir = run_dir
        self.device = dev
        self.p = float(p)
        self.q = float(q)
        self.eps = float(eps)
        self.target_kind = target_kind
        self.mode = mode

        manager = CheckpointManager(run_dir)
        if not manager.exists(checkpoint):
            raise FileNotFoundError(f"checkpoint {checkpoint!r} missing in {run_dir}")
        payload = manager.load(checkpoint, map_location=dev)
        cfg["train"]["device"] = str(dev)
        self.method = build_method(cfg.get("method_name", "ce"), cfg)
        self.method.load_state_dict(payload["model_state"])
        self.method.to(dev)
        self.method.eval()

        self.backbone = self.method.backbone
        self.num_classes = int(cfg["data"]["num_classes"])
        self.site_names = list(self.backbone.taps.keys())
        self.terminal_site = self.backbone.roles.get("top_l1", self.site_names[-1])
        self.mid_sites = [self.backbone.roles.get(r, r) for r in mid_roles]

        from ..depthfrag.geometry import SiteRadiiComputer

        self.computer = SiteRadiiComputer(
            self.backbone, self.site_names, p=p, q=q, eps=eps,
            target_kind=target_kind, mode=mode, exact_microbatch=exact_microbatch,
        )

    # -- profile extraction -------------------------------------------------
    def profile_split(self, split: str, subset: Optional[int] = None,
                      return_logits: bool = True, num_workers: int = 0,
                      batch_size: Optional[int] = None) -> dict:
        """Extract per-sample profiles for a split; returns the artifact dict."""
        loader = build_dataloader(self.cfg, split, shuffle=False,
                                  return_indices=split != "test",
                                  num_workers=num_workers, batch_size=batch_size)
        ids_l, y_l, pred_l, marg_l, logits_l = [], [], [], [], []
        per = {s: {"rho": [], "rel": [], "target": []} for s in self.site_names}
        t0 = time.perf_counter()
        n = 0
        for batch in loader:
            x = batch[0].to(self.device)
            y = batch[1].to(self.device)
            rb = self.computer.compute(x, y, role="eval", return_logits=return_logits)
            ids_l.append(np.asarray(batch[2]) if split != "test"
                         else np.arange(n, n + x.shape[0]))
            y_l.append(y.detach().cpu().numpy())
            pred_l.append(rb.prediction.detach().cpu().numpy())
            marg_l.append(rb.margin.detach().cpu().numpy())
            if return_logits:
                logits_l.append(rb.logits.detach().cpu().numpy())
            for s in self.site_names:
                per[s]["rho"].append(rb.rho[s].detach().cpu().numpy())
                per[s]["rel"].append(rb.rel[s].detach().cpu().numpy())
                per[s]["target"].append(rb.target[s].detach().cpu().numpy())
            n += x.shape[0]
            if subset is not None and n >= subset:
                break
        wall_s = time.perf_counter() - t0

        ids = np.concatenate(ids_l)[:n]
        y = np.concatenate(y_l)[:n]
        pred = np.concatenate(pred_l)[:n]
        margin = np.concatenate(marg_l)[:n]
        logits = np.concatenate(logits_l)[:n] if return_logits else None
        rho = np.stack([np.concatenate(per[s]["rho"])[:n] for s in self.site_names], axis=1)
        rel = np.stack([np.concatenate(per[s]["rel"])[:n] for s in self.site_names], axis=1)
        target = np.stack([np.concatenate(per[s]["target"])[:n] for s in self.site_names], axis=1)
        return {
            "split": split, "n": int(n),
            "ids": ids, "labels": y, "predictions": pred, "margins": margin,
            "logits": logits,
            "site_names": list(self.site_names),
            "rho": rho, "rel": rel, "target": target,
            "wall_s": float(wall_s),
        }

    # -- iterative cross-check ---------------------------------------------
    def iterative_audit(self, profile: dict, subset: Optional[int] = None,
                        max_steps: int = 50, seed: int = 0) -> dict:
        """Analytic vs iterative local-boundary comparison on fixed val subset.

        Uses one subset of *validation* samples (the analytic radii from the
        relative-radius profile and the DeepFool-style walk in input space).
        """
        from ..depthfrag.iterative import (
            compare_analytic_iterative,
            iterative_boundary_audit,
        )

        ids = profile["ids"]
        n = len(ids)
        rng = np.random.RandomState(seed)
        choose = np.arange(n)
        if subset is not None:
            choose = rng.choice(n, size=min(int(subset), n), replace=False)
            choose.sort()
        ids_sub = ids[choose]
        analytic_profile = profile["rel"][choose]
        in_size = self.cfg["backbones"][self.cfg["backbone"]].get("input_size", 32)
        loader = build_dataloader(self.cfg, "val", shuffle=False, return_indices=True,
                                  num_workers=0)
        xs = np.empty((len(choose), 3, in_size, in_size))
        ys = np.empty(len(choose), dtype=int)
        ptr = 0
        sub_set = set(int(i) for i in ids_sub)
        for batch in loader:
            b_ids = np.asarray(batch[2])
            keep = [k for k in range(len(b_ids)) if int(b_ids[k]) in sub_set]
            if not keep:
                continue
            for k in keep:
                xs[ptr] = batch[0][k].numpy()
                ys[ptr] = int(batch[1][k])
                ptr += 1
        xs = torch.from_numpy(xs[:ptr])
        ys_t = torch.from_numpy(ys[:ptr])

        audit = iterative_boundary_audit(self.backbone, xs, ys_t,
                                         max_steps=max_steps)
        iter_dist = audit["per_sample"]["dist"]
        term_idx = self.site_names.index(self.terminal_site)
        analytic_term = analytic_profile[:, term_idx]
        analytic_min = analytic_profile.min(axis=1)
        out = {
            "subset_n": int(ptr),
            "summary": audit["summary"],
            "analytic_vs_iter": {
                "terminal_radius": compare_analytic_iterative(analytic_term, iter_dist),
                "min_radius": compare_analytic_iterative(analytic_min, iter_dist),
            },
        }
        return out

    # -- persistence --------------------------------------------------------
    def save_artifacts(self, out_dir: str, prof, extra: dict):
        os.makedirs(out_dir, exist_ok=True)
        np.savez(
            os.path.join(out_dir, f"profiles_{prof['split']}.npz"),
            ids=prof["ids"], labels=prof["labels"], predictions=prof["predictions"],
            margins=prof["margins"], site_names=np.asarray(prof["site_names"]),
            rho=prof["rho"], rel=prof["rel"], target=prof["target"],
            logits=prof["logits"] if prof["logits"] is not None else np.empty(0),
            wall_s=np.asarray(prof["wall_s"]),
        )
        with open(os.path.join(out_dir, "extract.json"), "w") as f:
            json.dump(extra, f, indent=2, sort_keys=True, default=str)


# ---------------------------------------------------------------------------
# score variants
# ---------------------------------------------------------------------------
def score_variants_from_profiles(prof: dict, terminal_site: Optional[str] = None,
                                 mid_sites: Sequence[str] = ()) -> dict:
    """Compute every confidence-style score from a profile.

    Confidence direction follows the harness convention (higher = keep).
    Radii and margins are used directly (larger radius is more robust). The
    oracles are fitted separately (see :func:`evaluate_variants`).
    """
    rel = prof["rel"]
    L = rel.shape[1]
    sites = list(prof["site_names"])
    scores: Dict[str, np.ndarray] = {}
    if prof["logits"] is not None:
        logits = prof["logits"]
        p = np.clip(np.exp(logits - logits.max(axis=1, keepdims=True)), 1e-45, 1)
        p /= p.sum(axis=1, keepdims=True)
        scores["msp"] = p.max(axis=1)
        top2 = np.sort(np.partition(logits, -2, axis=1)[:, -2:], axis=1)[:, ::-1]
        scores["logit_margin"] = top2[:, 0] - top2[:, 1]
    term_idx = sites.index(terminal_site) if terminal_site in sites else L - 1
    scores["term_radius"] = np.asarray(rel)[:, term_idx].copy()
    scores["min_radius"] = rel.min(axis=1)
    scores["softmin_radius"] = _soft_min_np(rel)
    scores["mean_radius"] = rel.mean(axis=1)
    for i in range(L):
        scores[f"site_{sites[i]}"] = np.asarray(rel)[:, i].copy()
    for r in mid_sites:
        if r in sites:
            scores["mid_radius"] = np.asarray(rel)[:, sites.index(r)].copy()
    return scores


def _soft_min_np(rel: np.ndarray, tau: float = 2.0) -> np.ndarray:
    rel = np.asarray(rel)
    w = np.exp(-tau * rel)
    w /= w.sum(axis=1, keepdims=True)
    return (w * rel).sum(axis=1)


def evaluate_variants(prof_val: dict, prof_test: dict,
                      terminal_site: Optional[str] = None,
                      mid_sites: Sequence[str] = (),
                      variant_names: Sequence[str] = SCORE_VARIANTS) -> dict:
    """Validation-fitted variant metrics on val + untouched test.

    The linear/logistic oracles are fitted on the validation profile **only**
    and then applied to both splits; ``prof_test`` is never used for fitting.
    """
    sv_val = score_variants_from_profiles(prof_val, terminal_site, mid_sites)
    sv_test = score_variants_from_profiles(prof_test, terminal_site, mid_sites)

    labels_v, labels_t = prof_val["labels"], prof_test["labels"]
    ids_v, ids_t = prof_val["ids"], prof_test["ids"]
    rows: Dict[str, dict] = {}
    for name in variant_names:
        if name.startswith("oracle"):
            continue
        if name not in sv_val:
            continue
        for split, prof, sv, ids in (("val", prof_val, sv_val, ids_v),
                                     ("test", prof_test, sv_test, ids_t)):
            m = all_metrics(labels_v if split == "val" else labels_t,
                            prof["predictions"], sv[name], ids, num_classes=None)
            rows.setdefault(name, {})[split] = {
                "aurc": m["aurc"], "auroc_error": m["auroc_error"],
                "n": int(m["n"]),
            }

    for variant in ("lin", "logit"):
        from .oracle import fit_and_apply

        conf_v, _ = fit_and_apply(prof_val["rel"], labels_v, prof_val["predictions"],
                                  prof_val["rel"], variant)
        conf_t, _ = fit_and_apply(prof_val["rel"], labels_v, prof_val["predictions"],
                                  prof_test["rel"], variant)
        name = f"oracle_{variant}"
        for split, conf, prof, ids in (("val", conf_v, prof_val, ids_v),
                                       ("test", conf_t, prof_test, ids_t)):
            m = all_metrics(labels_v if split == "val" else labels_t,
                            prof["predictions"], conf, ids, num_classes=None)
            rows.setdefault(name, {})[split] = {
                "aurc": m["aurc"], "auroc_error": m["auroc_error"],
                "n": int(m["n"]),
            }

    return {"variants": rows, "scores_val": sv_val, "scores_test": sv_test}

__all__ = sorted(__all__)