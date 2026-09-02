"""DepthFrag geometry: depth-wise signed decision-fragility radii.

For each example with true class ``y`` and tapped representation ``h_l`` at
site ``l`` (a registry tap), define the terminal true-class margin

    m = z_y - max_{c != y} z_c

and the per-site fragility geometry

    g_l             = d m / d h_l          (vector-Jacobian product)
    rho_l           = m / (||g_l||_q + eps)
    relative_rho_l  = rho_l / (||h_l||_p + eps)
    target_l        = sign(relative_rho_l) * log1p(|relative_rho_l|)

Defaults are ``p = q = 2`` (Euclidean); ``p = inf, q = 1`` are supported.
Radii stay **signed**: incorrect examples (negative margin) keep a negative
radius so they are distinguishable instead of being clipped to zero.
``target_l`` is always detached before probe/distillation losses.

BatchNorm treatment
-------------------
``autograd.grad(m.sum(), h_l)`` is a *batch* VJP; it equals the stack of
per-example Jacobian-vector products exactly when the forward graph is
block-diagonal across examples. Training-mode BatchNorm couples examples
through the batch statistics and invalidates that reading. Two modes are
exposed:

* ``fast``  — one batched forward with BatchNorm pinned to *eval* stats
  (parameters remain differentiable) and a single ``autograd.grad``; the
  gradients are per-example Jacobians for any network whose only
  cross-example operator is BatchNorm (documented assumption).
* ``exact`` — per-example Jacobians via ``torch.func.functional_call`` on a
  configurable small microbatch: each example is functionally evaluated on
  its own, so no example is ever coupled to another (valid for any network,
  regardless of BatchNorm statistics mode).

The BatchNorm coupling itself is demonstrated (not hidden) by a dedicated
test comparing ``fast`` with BN in training mode against ``exact``.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..backbones import Backbone, MultiHook, adaptive_flatten

__all__ = [
    "true_class_margin",
    "per_example_norm",
    "radii_from_site",
    "target_transform",
    "site_gradients",
    "aggregate_profile",
    "soft_min_profile",
    "cvar_lower_tail",
    "pool_tap",
    "SiteRadiiComputer",
    "RadiiBatch",
    "AGG_FUNCS",
    "TARGET_KINDS",
]

TARGET_KINDS = ("signed_log1p", "absolute", "clipped")
AGG_FUNCS = ("soft_min", "cvar", "mean", "min", "terminal")


# ---------------------------------------------------------------------------
# margin + geometry primitives
# ---------------------------------------------------------------------------
def true_class_margin(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Terminal true-class margin ``z_y - max_{c != y} z_c`` (per example).

    ``max_{c != y}`` is ``z_{c*}`` with ``c*`` the class carrying the largest
    logit among classes other than the target — i.e. the top-2 logit when the
    target is the argmax, otherwise the top logit.
    """
    z_y = logits.gather(1, targets.view(-1, 1)).squeeze(1)
    if logits.size(1) >= 2:
        top2 = torch.topk(logits, k=2, dim=1).values
        is_argmax = logits.argmax(dim=1) == targets
        other = torch.where(is_argmax, top2[:, 1], top2[:, 0])
    else:
        other = z_y
    return z_y - other


def per_example_norm(t: torch.Tensor, p) -> torch.Tensor:
    """Per-example norm of a (B, ...) tensor, flattening non-batch axes.

    ``p`` may be ``2``, ``1`` (``q=1`` support) or ``float('inf')``.
    """
    f = t.reshape(t.shape[0], -1)
    if p == float("inf"):
        return f.abs().amax(dim=1)
    return f.norm(p=float(p), dim=1)


def radii_from_site(m: torch.Tensor, g_l, h_l, p: float = 2, q: float = 2,
                    eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-site signed radii given margin ``m``, site gradient ``g_l``, feature ``h_l``.

    Returns ``(rho, relative_rho)`` exactly as
    ``m / (||g_l||_q + eps)`` and ``rho / (||h_l||_p + eps)``. Values stay
    signed; the ``eps`` floors keep the formula finite when ``||g_l|| == 0``.
    A ``None`` gradient (site not on the margin's path) yields ``0`` radii.
    """
    if g_l is None:
        return torch.zeros_like(m), torch.zeros_like(m)
    g = per_example_norm(g_l, q)
    rho = m / (g + eps)
    rel = rho / (per_example_norm(h_l, p) + eps)
    return rho, rel


def target_transform(rel: torch.Tensor, kind: str = "signed_log1p",
                     cap: float = 1.0) -> torch.Tensor:
    """Compress a signed relative radius into a scalar regression target.

    ``signed_log1p`` : ``sign(r) * log1p(|r|)`` (default, keeps sign).
    ``absolute``     : ``log1p(|r|)`` (sensitivity control that erases the
                       correct/incorrect distinction).
    ``clipped``      : signed, with ``|r|`` clipped to ``cap`` before
                       ``log1p`` (sensitivity control on magnitude).
    """
    if kind == "absolute":
        return torch.log1p(rel.abs())
    if kind == "clipped":
        return torch.sign(rel) * torch.log1p(torch.clamp(rel.abs(), max=float(cap)))
    if kind == "signed_log1p":
        return torch.sign(rel) * torch.log1p(rel.abs())
    raise ValueError(f"unknown target_kind {kind!r}; choose from {TARGET_KINDS}")


def site_gradients(m: torch.Tensor, features: Dict[str, torch.Tensor],
                   sites: Sequence[str]) -> Dict[str, Optional[torch.Tensor]]:
    """Batch VJP ``d m.sum() / d h_l`` for every site (first order only).

    ``create_graph=False`` nodes both the intended first-order semantics and
    the "no accidental second-order graph retention" property. The graph is
    retained across all but the last site and then freed, so nothing 2nd-order
    is kept alive afterwards.
    """
    out: Dict[str, Optional[torch.Tensor]] = {}
    for i, s in enumerate(sites):
        h = features[s]
        if h is None or not h.requires_grad:
            out[s] = None
            continue
        g = torch.autograd.grad(
            m.sum(), h, retain_graph=(i < len(sites) - 1), create_graph=False,
            allow_unused=True, materialize_grads=True,
        )[0]
        out[s] = g
    return out


# ---------------------------------------------------------------------------
# pooling + aggregation
# ---------------------------------------------------------------------------
def pool_tap(feature: torch.Tensor, token: str = "cls") -> torch.Tensor:
    """Reduce a tap feature to a fixed ``(B, D)`` vector.

    4-D CNN maps -> global-average-pool then flatten; 3-D token sequences
    (ViT) -> CLS token (or token mean, config-selectable); 2-D stays as-is.
    """
    if feature.dim() == 2:
        return feature
    if feature.dim() == 4:
        return adaptive_flatten(feature, out=1)
    if feature.dim() == 3:
        if token == "cls":
            return feature[:, 0]
        if token == "mean":
            return feature.mean(dim=1)
        raise ValueError(f"unknown ViT token pooling {token!r}")
    raise ValueError(f"unsupported tap feature dim {feature.dim()}")


def soft_min_profile(profile: torch.Tensor, tau: float = 2.0) -> torch.Tensor:
    """Differentiable soft minimum over the depth profile.

    Weight concentrates on the *smallest* (most fragile) values via
    ``w = softmax(-tau * profile)``; tau -> +inf recovers the hard min.
    """
    w = F.softmax(-float(tau) * profile, dim=-1)
    return (w * profile).sum(dim=-1)


def cvar_lower_tail(profile: torch.Tensor, frac: float = 0.25) -> torch.Tensor:
    """Lower-tail CVaR: mean of the ``frac``-quantile *smallest* radii."""
    k = max(1, int(math.ceil(float(frac) * profile.size(-1))))
    small = torch.topk(-profile, k=k, dim=-1).values
    return small.mean(dim=-1)


def mean_profile(profile: torch.Tensor) -> torch.Tensor:
    return profile.mean(dim=-1)


def min_profile(profile: torch.Tensor) -> torch.Tensor:
    return profile.min(dim=-1).values


def terminal_profile(profile: torch.Tensor) -> torch.Tensor:
    return profile[..., -1]


def aggregate_profile(profile: torch.Tensor, agg: str = "soft_min",
                      tau: float = 2.0, frac: float = 0.25) -> torch.Tensor:
    """Aggregate a ``(B, L)`` radius-profile column into one scalar per example."""
    if agg == "soft_min":
        return soft_min_profile(profile, tau)
    if agg == "cvar":
        return cvar_lower_tail(profile, frac)
    if agg == "mean":
        return mean_profile(profile)
    if agg == "min":
        return min_profile(profile)
    if agg == "terminal":
        return terminal_profile(profile)
    raise ValueError(f"unknown aggregate {agg!r}; choose from {AGG_FUNCS}")


# ---------------------------------------------------------------------------
# batch computation
# ---------------------------------------------------------------------------
class RadiiBatch:
    """Output of one batch's radius extraction (all tensors detached)."""

    __slots__ = ("margin", "prediction", "rho", "rel", "target", "site_names",
                 "logits", "x_requires_grad_used")

    def __init__(self, margin, prediction, rho: Dict[str, torch.Tensor],
                 rel: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor],
                 site_names: List[str], logits: Optional[torch.Tensor] = None):
        self.margin = margin
        self.prediction = prediction
        self.rho = rho
        self.rel = rel
        self.target = target
        self.site_names = list(site_names)
        self.logits = logits


class SiteRadiiComputer:
    """Extract per-site signed radii from a backbone forward.

    ``mode="fast"`` one batched forward (BatchNorm in eval role unless
    ``role="train"``) + one ``autograd.grad`` per site; ``mode="exact"`` runs
    per-example forwards through ``torch.func.functional_call`` so no example
    is ever coupled to another. Both return detached tensors.

    The BatchNorm statistics role is a parameter, not hidden state: the
    BatchNorm-coupling *demonstration* test uses ``role="train"`` (batch
    statistics) precisely to show that the naive batch VJP then differs from
    the per-example reading.
    """

    def __init__(self, backbone: Backbone, site_names: Sequence[str],
                 p: float = 2, q: float = 2, eps: float = 1e-12,
                 target_kind: str = "signed_log1p", clip_cap: float = 1.0,
                 mode: str = "fast", exact_microbatch: int = 1):
        if mode not in ("fast", "exact"):
            raise ValueError(f"mode must be fast or exact, got {mode!r}")
        self.backbone = backbone
        self.site_names = list(site_names)
        self.p = float(p)
        self.q = float(q)
        self.eps = float(eps)
        self.target_kind = str(target_kind)
        self.clip_cap = float(clip_cap)
        self.mode = mode
        self.exact_microbatch = max(int(exact_microbatch), 1)
        self.device = next(backbone.parameters()).device

    # -- public -----------------------------------------------------------
    def compute(self, x: torch.Tensor, y: torch.Tensor, role: str = "eval",
                return_logits: bool = False) -> RadiiBatch:
        """Signed radius profile for a batch.

        ``role='eval'`` pins BatchNorm to running/eval statistics
        (parameters remain differentiable); ``role='train'`` keeps batch
        statistics (only for the coupling demonstration).
        """
        if self.mode == "exact":
            return self._compute_exact(x, y, role, return_logits)
        return self._compute_fast(x, y, role, return_logits)

    # -- fast: one batched forward + batch VJP ------------------------------
    def _compute_fast(self, x, y, role, return_logits):
        was_training = self.backbone.training
        _set_role(self.backbone, role)
        xg = x.detach().clone().requires_grad_(True)
        store: Dict[str, torch.Tensor] = {}
        hooks = MultiHook(self.backbone.taps, store)
        try:
            with torch.enable_grad():
                bo = self.backbone(xg)
                m = true_class_margin(bo.logits, y)
                gs = site_gradients(m, {s: store.get(s) for s in self.site_names},
                                    self.site_names)
                rho, rel, target = {}, {}, {}
                for s in self.site_names:
                    r, rr = radii_from_site(m, gs[s], store[s], self.p, self.q,
                                            self.eps)
                    rho[s] = r.detach()
                    rel[s] = rr.detach()
                    target[s] = target_transform(
                        rr.detach(), self.target_kind, self.clip_cap)
                out = RadiiBatch(
                    margin=m.detach(),
                    prediction=bo.logits.detach().argmax(dim=1),
                    rho=rho, rel=rel, target=target,
                    site_names=self.site_names,
                    logits=bo.logits.detach() if return_logits else None,
                )
        finally:
            hooks.remove()
            _set_role(self.backbone, was_training)
        return out

    # -- exact: per-example Jacobians via torch.func -------------------------
    def _compute_exact(self, x, y, role, return_logits):
        from torch.func import functional_call

        params = {n: p for n, p in self.backbone.named_parameters()}
        buffers = {n: b for n, b in self.backbone.named_buffers()}
        was_training = self.backbone.training
        _set_role(self.backbone, role)

        margins, preds, logits_all = [], [], []
        rho = {s: [] for s in self.site_names}
        rel = {s: [] for s in self.site_names}
        target = {s: [] for s in self.site_names}
        n = x.shape[0]
        try:
            for start in range(0, n, self.exact_microbatch):
                stop = min(start + self.exact_microbatch, n)
                for i in range(start, stop):
                    x_i = x[i:i + 1].detach().clone().requires_grad_(True)
                    y_i = y[i:i + 1]
                    store: Dict[str, torch.Tensor] = {}
                    hooks = MultiHook(self.backbone.taps, store)
                    with torch.enable_grad():
                        bo = functional_call(self.backbone, (params, buffers), x_i)
                        m_i = true_class_margin(bo.logits, y_i)
                        gs = site_gradients(m_i, {s: store.get(s) for s in self.site_names},
                                            self.site_names)
                        for s in self.site_names:
                            r, rr = radii_from_site(m_i, gs[s], store[s],
                                                    self.p, self.q, self.eps)
                            rho[s].append(r.detach().cpu())
                            rel[s].append(rr.detach().cpu())
                            target[s].append(target_transform(
                                rr.detach(), self.target_kind, self.clip_cap).cpu())
                        margins.append(m_i.detach().cpu())
                        preds.append(bo.logits.detach().argmax(dim=1).cpu())
                        if return_logits:
                            logits_all.append(bo.logits.detach().cpu())
                    hooks.remove()
                    del store
        finally:
            _set_role(self.backbone, was_training)

        def _cat(seq):
            return torch.cat(seq, dim=0) if seq else torch.empty(0)

        return RadiiBatch(
            margin=_cat(margins),
            prediction=_cat(preds),
            rho={s: _cat(rho[s]) for s in self.site_names},
            rel={s: _cat(rel[s]) for s in self.site_names},
            target={s: _cat(target[s]) for s in self.site_names},
            site_names=self.site_names,
            logits=_cat(logits_all) if return_logits else None,
        )


def _set_role(model, role):
    """Pin the module to train/eval. Accepts role strings ('train'/'eval') and
    the booleans produced by ``nn.Module.training`` (restore path)."""
    if role is True or role == "train":
        model.train()
    elif role is False or role == "eval":
        model.eval()
    else:
        raise ValueError(f"role must be 'train'/'eval' or a bool, got {role!r}")