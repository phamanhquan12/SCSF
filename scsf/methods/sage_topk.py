"""SAGE-TopK: budgeted top-k selective deep supervision.

Preregistered protocol: ``docs/SAGE_TOPK_PROTOCOL.md``.

Method identity (protocol section 3): automatic enumeration of registered
backbone block-boundary candidates; a short profiling stage (epochs 0-4);
fixed Top-K selection *after* profiling; normalized classification-compatible
auxiliary directions; a small convex allocation problem. It has no learned
controller, no stochastic gates, no hard-concrete mechanism, no auxiliary
confidence MLP, no robust/class-weighted selective objective, and no amortized
solver.

Profiling (epochs 0-4)
----------------------
* the backbone trains with ordinary final-head CE;
* every companion head trains on **detached** backbone features (head
  parameters only, so profiling never routes auxiliary gradients into the
  backbone);
* every ``utility_interval``-th training batch the classification-compatible
  projected auxiliary gradients of every candidate are measured against the
  SAGE-V2 global selective surrogate gradient (``soft_aurc_surrogate`` on a
  deterministic disjoint meta batch) and the **mean cosine utility** and
  descriptive variance are accumulated per candidate;
* at the end of epoch 4 the two candidates with highest mean utility are
  selected (exact ties broken by registration order) and frozen.

Training (epoch 5+)
-------------------
* only the selected companion heads compute auxiliary objectives/gradients;
  unselected heads receive no gradient and their parameters are untouched;
* per selected site the same-batch CE-safety projection (the exact SAGE-V2
  identity) and unit normalization produce ``v_l``;
* the convex allocation

      min  0.5 lambda^T G lambda - b^T lambda
      s.t. lambda >= 0, sum(lambda) <= B        (G = V^T V, b = V^T s)

  is solved exactly by deterministic enumeration for K = 2 and certified
  before application (finite values, feasibility, objective no-worse-than
  ``lambda = 0``, selective alignment, numeric mixture CE compatibility);
  any failed check yields the **zero-update fallback** ``lambda = 0``;
* the backbone update ``g_CE + rho * ||g_CE|| * V lambda`` (``rho = 1``) is
  routed via the dot-product loss trick (same mechanism as SAGE v1/v2).

The selective target is refreshed every ``utility_interval`` batches and on
the first post-profiling batch; the detached normalized target ``s`` is cached
between refreshes and its age (batches since refresh) is logged. The descent
certificate is relative to the gradient the solver used — between refreshes
that is the **cached** target, not the current selective gradient.

Inference is unchanged: plain terminal MSP from the final backbone classifier;
companion heads and all allocation machinery are stripped from the deployment
graph.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import build_dataset
from ..metrics.surrogate import soft_aurc_surrogate
from .base import Method, MethodPrediction
from .sage_ds import _cat, _flatten, _pool_tap, project_aux
from .sage_ds_v2 import cosine_utility
from .scores import compute_scores

__all__ = [
    "SageTopKMethod",
    "LinearAuxHead",
    "normalize_direction",
    "classification_compatible_direction",
    "select_topk_sites",
    "solve_topk_allocation",
    "allocation_certificate",
]

EPS = 1e-8
TOL = 1e-6


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------
class LinearAuxHead(nn.Module):
    """Linear companion classifier ``Linear(d, C)`` (protocol section 4).

    No LayerNorm, hidden layer, or nonlinear MLP: the head is a single affine
    map on the native-pooled features. Training-only instrument.
    """

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def normalize_direction(g_flat: torch.Tensor, eps: float = EPS):
    """Unit-normalize a flat gradient; ``(v, norm, is_zero)``.

    Zero-norm directions are handled explicitly: the returned vector is all
    zero and ``is_zero`` is True (its allocation coordinate then contributes
    nothing).
    """
    n = float(torch.norm(g_flat).item())
    if n <= eps:
        return torch.zeros_like(g_flat), 0.0, True
    return g_flat / n, n, False


def classification_compatible_direction(g_aux_flat: torch.Tensor,
                                        g0_flat: torch.Tensor,
                                        eps: float = EPS):
    """CE-safety projection + unit normalization of an aux direction.

    Removes any component of ``g_aux`` opposing the current CE gradient
    ``g0`` (the exact SAGE-V2 projection identity), then normalizes::

        v = proj(g_aux) / (||proj(g_aux)|| + eps)

    Returns ``(v, align_before, align_after, is_zero)`` where the alignment
    values are ``<g_aux, g0>`` before and ``<v, g0>`` after projection.
    """
    safe, align_before = project_aux(g_aux_flat, g0_flat, eps=eps)
    align_after = float(torch.dot(safe, g0_flat).item())
    v, _, is_zero = normalize_direction(safe, eps)
    return v, align_before, align_after, is_zero


def select_topk_sites(means: torch.Tensor, k: int) -> List[int]:
    """Deterministic top-k selection by mean cosine utility.

    Exact ties are broken by stable candidate (registration) order: the sort
    key is ``(-mean, index)``.
    """
    k = max(0, min(int(k), int(means.numel())))
    order = sorted(range(int(means.numel())), key=lambda i: (-float(means[i]), i))
    return order[:k]


# ---------------------------------------------------------------------------
# deterministic K <= 2 convex allocation
# ---------------------------------------------------------------------------
def _qp_objective(lam: torch.Tensor, G: torch.Tensor, b: torch.Tensor) -> float:
    return float((0.5 * lam @ G @ lam - b @ lam).item())


def solve_topk_allocation(G: torch.Tensor, b: torch.Tensor, B: float = 1.0,
                          tol: float = TOL, eps: float = 1e-12) -> Dict:
    """Exact deterministic enumerative QP for K <= 2 (protocol section 8).

        min  0.5 lambda^T G lambda - b^T lambda
        s.t. lambda >= 0,  sum(lambda) <= B

    The feasible region for K = 2 is the right triangle ``{lam >= 0,
    sum <= B}``. A convex quadratic attains its minimum either at the
    unconstrained point (if feasible) or on a face. Every face is a compact
    1-D quadratic with a closed-form clamped optimum:

    * the interior point ``G^{-1} b``;
    * the coordinate faces ``lambda_i = 0`` (free coordinate clamped in
      ``[0, B]``), including their endpoints;
    * the budget face ``sum(lambda) = B`` (segment quadratic in one free
      parameter).

    Singular/collinear Gram matrices are handled **without** silently adding a
    ridge penalty that changes the objective: zero rows/columns simply degrade
    to the linear subproblem on the remaining supporting direction(s) and the
    enumeration is still over the same faces.
    """
    K = int(b.numel())
    if G.shape != (K, K):
        raise ValueError(f"Gram must be {K}x{K}, got {tuple(G.shape)}")
    if K > 2:
        raise NotImplementedError(
            "sage_topk protocol locks K=2; solver enumeration supports K <= 2")

    G64 = G.double()
    b64 = b.double()
    dev = G.device

    candidates: List[Tuple[torch.Tensor, float]] = []
    zero = torch.zeros(K, dtype=G64.dtype, device=dev)
    candidates.append((zero.clone(), _qp_objective(zero, G64, b64)))

    if K == 1:
        g = float(G64[0, 0])
        bb = float(b64[0])
        lam = B if bb > 0 else 0.0
        if g > eps:
            lam = min(max(bb / g, 0.0), B)
        l1 = torch.tensor([lam], dtype=G64.dtype, device=dev)
        candidates.append((l1, _qp_objective(l1, G64, b64)))
    else:
        # interior (unconstrained) point, if feasible
        try:
            l_int = torch.linalg.solve(G64, b64)
            if (float(l_int.min().item()) >= -tol
                    and float(l_int.sum().item()) <= B + tol):
                candidates.append((l_int.detach().clone(),
                                   _qp_objective(l_int, G64, b64)))
        except Exception:
            pass

        # coordinate faces: lam_i = 0, free coordinate in [0, B]
        for j in range(K):
            gj = float(G64[j, j])
            bj = float(b64[j])
            lam = B if bj > 0 else 0.0
            if gj > eps:
                lam = min(max(bj / gj, 0.0), B)
            lf = torch.zeros(K, dtype=G64.dtype, device=dev)
            lf[j] = lam
            candidates.append((lf.clone(), _qp_objective(lf, G64, b64)))

        # budget face: lam_1 + lam_2 = B, quadratic in t = lam_1
        g00, g01, g11 = (float(G64[0, 0]), float(G64[0, 1]), float(G64[1, 1]))
        A = 0.5 * (g00 - 2.0 * g01 + g11)
        c = B * (g01 - g11) - float(b64[0]) + float(b64[1])
        t = 0.0
        if A > eps:
            t = min(max(-c / (2.0 * A), 0.0), B)
        elif c < 0:
            t = B
        ls = torch.tensor([t, B - t], dtype=G64.dtype, device=dev)
        candidates.append((ls, _qp_objective(ls, G64, b64)))

    best = min(candidates, key=lambda cc: cc[1])
    lam, objective = best
    return {
        "lambda": lam.detach().clone().float(),
        "objective": objective,
        "n_candidates": len(candidates),
        "status": "enum",
    }


def allocation_certificate(lam: torch.Tensor, G: torch.Tensor, b: torch.Tensor,
                           B: float = 1.0, tol: float = TOL) -> Dict[str, float]:
    """Approve-or-reject predicates for an allocation (protocol section 8).

    Checks: finite values; ``min(lambda) >= -tol``; ``sum(lambda) <= B + tol``;
    selective alignment ``b^T lambda >= -tol``; objective no worse than
    ``lambda = 0`` (objective(0) = 0) within tolerance. Returns the predicates
    plus ``ok``.
    """
    obj = _qp_objective(lam, G, b)
    # objective(0) uses the same (original precision) G/b, so this is the
    # same quantity the solver minimized.
    finite = bool(torch.isfinite(lam).all()) and math.isfinite(obj)
    nonneg = float(lam.min().item()) if lam.numel() else 0.0
    total = float(lam.sum().item())
    b_lam = float((lam @ b).item())
    ok = bool(finite and nonneg >= -tol and total <= B + tol
              and b_lam >= -tol and obj <= tol)
    return {
        "objective": obj, "min_lambda": nonneg, "sum_lambda": total,
        "b_lambda": b_lam, "finite": finite, "ok": ok,
    }


# ---------------------------------------------------------------------------
# the method
# ---------------------------------------------------------------------------
class SageTopKMethod(Method):
    """Budgeted top-k selective deep supervision (``sage_topk``)."""

    method_name = "sage_topk"

    def default_score(self) -> str:
        return "msp"

    def default_scores(self):
        return ("msp", "entropy", "energy", "logit_margin")

    def __init__(self, train_cfg: dict):
        super().__init__(train_cfg)
        m = train_cfg["method"]
        self.k = int(m.get("k", 2))
        self.profiling_epochs = int(m.get("profiling_epochs", 5))
        self.utility_interval = int(m.get("utility_interval", 50))
        self.B = float(m.get("B", 1.0))
        self.rho = float(m.get("rho", 1.0))
        self.surrogate_tau = float(m.get("surrogate_tau", 0.3))
        self.projection_eps = float(m.get("projection_eps", EPS))
        self.tol = float(m.get("cert_tol", TOL))
        self.token = str(m.get("token", "cls"))

        if self.k > 2:
            raise ValueError(
                f"sage_topk protocol locks K <= 2 (deterministic enumeration); "
                f"got k={self.k}")

        # candidates come exclusively from the backbone adapter registry
        self.site_names = list(self.backbone.taps.keys())
        probe = self._probe_site_dims()
        self.aux_heads = nn.ModuleDict(
            {s: LinearAuxHead(probe[s], self.num_classes) for s in self.site_names}
        )
        self._aux_params = [p for h in self.aux_heads.values()
                            for p in h.parameters()]
        self._aux_param_ids = {id(p): i for i, p in enumerate(self._aux_params)}

        # utility/backbone gradient parameter set: every trainable backbone param
        self._utility_params = [(n, p) for n, p in self.backbone.named_parameters()
                                if p.requires_grad]
        self._n_backbone = sum(int(p.numel()) for _, p in self._utility_params)

        # -- exact-resume state lives in registered buffers (checkpoints) -----
        self.register_buffer("_profiling", torch.tensor(True))
        self.register_buffer("_selected",
                             torch.full((self.k,), -1, dtype=torch.long))
        n_sites = len(self.site_names)
        self.register_buffer("_utility_sum", torch.zeros(n_sites))
        self.register_buffer("_utility_sumsq", torch.zeros(n_sites))
        self.register_buffer("_utility_n", torch.zeros((), dtype=torch.long))
        self.register_buffer("_s_cached",
                             torch.zeros(self._n_backbone, dtype=torch.float32))
        self.register_buffer("_s_set", torch.tensor(False))
        self.register_buffer("_s_refresh_step", torch.tensor(-1, dtype=torch.long))
        self.register_buffer("_s_target_age", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_zero_updates", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_allocation_steps", torch.tensor(0, dtype=torch.long))

        # -- timing / telemetry ---------------------------------------------
        self._timers = {name: 0.0 for name in
                        ("backbone_ce", "aux_grads", "meta_refresh", "gram",
                         "qp_cert")}
        self._timer_cnt = {name: 0 for name in self._timers}
        self._timers_prev = dict(self._timers)
        self._timer_cnt_prev = dict(self._timer_cnt)

        self._step_aux_acc: Dict[str, float] = {s: 0.0 for s in self.site_names}
        self._step_aux_loss: Dict[str, float] = {s: 0.0 for s in self.site_names}
        self._step_aux_mass = 0.0
        self._step_lambda_sum = 0.0
        self._step_target_age = 0
        self._step_lambda_acc: Dict[str, float] = {s: 0.0 for s in self.site_names}

        self._meta_ds = None
        self._epoch = 0
        self._epoch_steps = 0
        self._log: List[dict] = []
        self._step_rows: List[dict] = []
        self._util_rows: List[dict] = []
        self._profile_stats: Optional[dict] = None
        self._last_gJ_norm = 0.0
        self._selection_fallback = False

    # ------------------------------------------------------------------ init
    def _probe_site_dims(self) -> Dict[str, int]:
        with torch.no_grad(), self.probe_mode():
            bo = self.backbone(
                torch.zeros(1, self.backbone.channels, self.backbone.input_size,
                            self.backbone.input_size)
            )
        return {s: int(_pool_tap(bo.features[s], self.token).shape[-1])
                for s in self.site_names}

    def _meta_dataset(self):
        if self._meta_ds is None:
            self._meta_ds = build_dataset(self.cfg, "val")
        return self._meta_ds

    def _meta_batch(self, step: int, device):
        """Deterministic, stateless meta batch: a pure function of the global
        step so an exact resume draws the same meta batch at the same step."""
        ds = self._meta_dataset()
        bs = int(self.cfg["train"].get("batch_size", 64))
        n = len(ds)
        n_batches = max(1, math.ceil(n / bs))
        i = step % n_batches
        xs, ys = [], []
        for j in range(i * bs, min(n, (i + 1) * bs)):
            x, y = ds[j][0], ds[j][1]
            xs.append(torch.as_tensor(x, device=device))
            ys.append(torch.as_tensor(y, device=device))
        return torch.stack(xs), torch.stack(ys), i

    # ------------------------------------------------------------- inference
    def predict_batch(self, x):
        bo = self.backbone(x)
        logits = bo.logits[:, : self.num_classes]
        scores = compute_scores(logits, self.default_scores())
        conf = scores["msp"]  # plain MSP is the only primary score (locked)
        scores["sage_conf"] = conf
        return MethodPrediction(logits, logits.argmax(dim=1), conf, scores)

    def inference_modules(self):
        # companion heads + allocation machinery excluded from deployment
        return [self.backbone]

    def selected_sites(self) -> List[str]:
        """Currently selected candidate names (registration order by index)."""
        return [self.site_names[i] for i in self._selected.tolist()
                if int(i) >= 0]

    def optimizer_specs(self):
        t = self.cfg["train"]
        params = list(self.backbone.parameters()) + self._aux_params
        params = [p for p in params if p.requires_grad]
        return [{
            "params": params,
            "kind": t.get("optimizer", "sgd"),
            "lr": float(t["lr"]),
            "momentum": float(t.get("momentum", 0.9)),
            "weight_decay": float(t.get("weight_decay", 5e-4)),
        }]

    # -------------------------------------------------------------- training
    def train_loss(self, batch, state):
        device = next(self.backbone.parameters()).device
        self._epoch_steps += 1
        x = batch[0].to(device)
        y = batch[1].to(device)
        bo = self.backbone(x)
        ce_t = F.cross_entropy(bo.logits[:, : self.num_classes], y)
        if bool(self._profiling):
            return self._profiling_loss(bo, y, ce_t, state)
        return self._allocated_loss(bo, y, ce_t, state)

    # -- profiling epochs 0-4 ------------------------------------------------
    def _profiling_loss(self, bo, y, ce_t, state):
        """Backbone CE + per-head CE on *detached* features (measurement-only
        auxiliary gradients; never routed into the backbone)."""
        device = ce_t.device
        out = {}
        aux_total = torch.zeros((), device=device)
        for s in self.site_names:
            h = self.aux_heads[s]
            feat = _pool_tap(bo.features[s], self.token)
            l_aux = F.cross_entropy(h(feat.detach()), y)
            with torch.no_grad():
                acc = (h(feat).detach().argmax(1) == y).float().mean()
                out[f"aux_acc_{s}"] = acc
                self._step_aux_acc[s] += float(acc)
                self._step_aux_loss[s] += float(l_aux.detach())
            out[f"aux_loss_{s}"] = l_aux.detach()
            aux_total = aux_total + l_aux
        out["ce"] = ce_t
        out["sage_aux"] = aux_total

        step = int(getattr(state, "batch_index", 0))
        if step > 0 and step % self.utility_interval == 0:
            self._measure_profiling(bo, y, ce_t, step)
        return out

    def _measure_profiling(self, bo, y, ce_t, step):
        """Candidate utility measurement (protocol section 5): the projected
        per-site auxiliary gradients on the current training batch vs. the
        SAGE-V2 selective-gradient direction on a disjoint meta batch. Pure
        measurement: nothing is applied to the backbone."""
        device = ce_t.device
        params = [p for _, p in self._utility_params]
        t0 = time.perf_counter()
        g0 = torch.autograd.grad(ce_t, params, retain_graph=True,
                                 allow_unused=True, materialize_grads=True)
        self._tick("backbone_ce", t0)
        g0_flat = _cat([_flatten(g) for g in g0]).detach()
        g0_n = float(torch.norm(g0_flat).item())

        g_sel = self._refresh_selective(device, step)
        gJ_n = self._last_gJ_norm

        row = {"step": int(step), "epoch": int(self._epoch),
               "phase": "profiling", "g0_norm": g0_n, "gJ_norm": gJ_n,
               "meta_batch": int(self._last_meta_index)}
        self._utility_n.add_(1)
        for s in self.site_names:
            h = self.aux_heads[s]
            feat = _pool_tap(bo.features[s], self.token)
            l_aux = F.cross_entropy(h(feat), y)   # attached: measurement only
            g_lb = torch.autograd.grad(l_aux, params, retain_graph=True,
                                       allow_unused=True, materialize_grads=True)
            gl_flat = _cat([_flatten(g) for g in g_lb]).detach()
            tilde, _ = project_aux(gl_flat, g0_flat, eps=self.projection_eps)
            tilde = tilde.detach()
            til_n = float(torch.norm(tilde).item())
            raw = float(torch.dot(g_sel, tilde).item())
            u_cos = cosine_utility(raw, gJ_n, til_n, eps=self.projection_eps)
            i = self.site_names.index(s)
            self._utility_sum[i].add_(u_cos)
            self._utility_sumsq[i].add_(u_cos * u_cos)
            row[f"cos_{s}"] = u_cos
            row[f"raw_{s}"] = raw
            row[f"til_norm_{s}"] = til_n
        self._util_rows.append(row)

    # -- training epoch 5+ ----------------------------------------------------
    def _allocated_loss(self, bo, y, ce_t, state):
        device = ce_t.device
        params = [p for _, p in self._utility_params]
        step = int(getattr(state, "batch_index", 0))
        self._allocation_steps.add_(1)
        out = {}

        t0 = time.perf_counter()
        g0 = torch.autograd.grad(ce_t, params, retain_graph=True,
                                 allow_unused=True, materialize_grads=True)
        self._tick("backbone_ce", t0)
        g0_flat = _cat([_flatten(g) for g in g0]).detach()
        g0_n = float(torch.norm(g0_flat).item())

        if self._should_refresh(step):
            self._refresh_selective(device, step)
        else:
            self._s_target_age.add_(1)
        s_flat = self._s_cached.detach()

        selected = [int(i) for i in self._selected.tolist() if int(i) >= 0]
        vs: List[torch.Tensor] = []
        auxhead_g: Dict[int, torch.Tensor] = {}
        for s_idx in selected:
            s = self.site_names[s_idx]
            h = self.aux_heads[s]
            feat = _pool_tap(bo.features[s], self.token)
            l_aux = F.cross_entropy(h(feat), y)
            t0 = time.perf_counter()
            g_lb = torch.autograd.grad(l_aux, params, retain_graph=True,
                                       allow_unused=True, materialize_grads=True)
            self._tick("aux_grads", t0)
            gl_flat = _cat([_flatten(g) for g in g_lb]).detach()
            v, align_before, align_after, is_zero = \
                classification_compatible_direction(gl_flat, g0_flat,
                                                    eps=self.projection_eps)
            vs.append(v)

            gh = torch.autograd.grad(l_aux, list(h.parameters()),
                                     retain_graph=True, allow_unused=True,
                                     materialize_grads=True)
            for gi, p in enumerate(list(h.parameters())):
                if gh[gi] is not None:
                    auxhead_g[self._aux_param_ids[id(p)]] = gh[gi]

            with torch.no_grad():
                acc = (h(feat).detach().argmax(1) == y).float().mean()
                out[f"aux_acc_{s}"] = acc
                self._step_aux_acc[s] += float(acc)
                self._step_aux_loss[s] += float(l_aux.detach())
            out[f"aux_loss_{s}"] = l_aux.detach()
            out[f"aux_align_{s}"] = align_after

        lambd = torch.zeros(0, device=device)
        mix = torch.zeros(self._n_backbone, device=device)
        cert = None
        zero = False
        if vs:
            t0 = time.perf_counter()
            V = torch.stack(vs)
            G = V @ V.t()
            bvec = V @ s_flat
            self._tick("gram", t0)
            t0 = time.perf_counter()
            sol = solve_topk_allocation(G, bvec, B=self.B, tol=self.tol)
            self._tick("qp_cert", t0)
            lambd = sol["lambda"].to(device)
            mix = V.t() @ lambd
            mix_n = float(torch.norm(mix).item())
            ce_dot = float(torch.dot(mix, g0_flat).item())
            ce_compat = (ce_dot / (mix_n * g0_n + self.projection_eps)
                         if mix_n > 0.0 else 1.0)
            cert = allocation_certificate(lambd, G, bvec, B=self.B, tol=self.tol)
            ok = bool(cert["ok"] and ce_compat >= -self.tol and mix_n > 0.0)
            if ok:
                self._step_aux_mass += mix_n
                self._step_lambda_sum += float(lambd.sum().item())
                for s_idx, s in enumerate(
                        [self.site_names[i] for i in selected if i >= 0]):
                    self._step_lambda_acc[s] += float(lambd[s_idx].item())
            else:
                zero = True
                self._zero_updates.add_(1)
                lambd = torch.zeros_like(lambd)
                cert = None
        else:
            zero = True
            self._zero_updates.add_(1)
        if zero:
            mix = torch.zeros(self._n_backbone, device=device)

        aux_flat = self.rho * g0_n * mix
        routed = torch.zeros((), device=device)
        acc = 0
        for p, g0p in zip(params, g0):
            n = p.numel()
            g_desired = (g0p if g0p is not None else torch.zeros_like(p)) + \
                aux_flat[acc:acc + n].reshape_as(p)
            routed = routed + torch.sum(p * g_desired.detach())
            acc += n
        for i, p in enumerate(self._aux_params):
            if i in auxhead_g:
                routed = routed + torch.sum(p * auxhead_g[i].detach())

        out["routed"] = routed
        out["ce"] = ce_t.detach()
        self._step_target_age += int(self._s_target_age)

        row = {
            "epoch": int(self._epoch), "step": int(step),
            "lambda": [float(l) for l in lambd.tolist()] if lambd.numel() else [],
            "sum_lambda": float(lambd.sum().item()) if lambd.numel() else 0.0,
            "zero": bool(zero), "target_age": int(self._s_target_age),
            "refresh_step": int(self._s_refresh_step),
            "selected": [self.site_names[i] for i in selected],
        }
        if cert is not None:
            row["cert_ok"] = bool(cert["ok"])
            row["cert_b_lambda"] = cert["b_lambda"]
            row["cert_obj"] = cert["objective"]
            row["mix_norm"] = float(torch.norm(mix).item())
        self._step_rows.append(row)
        return out

    def _should_refresh(self, step: int) -> bool:
        if step > 0 and step % self.utility_interval == 0:
            return True
        if self._epoch == self.profiling_epochs and step == 0:
            return True
        return False

    def _refresh_selective(self, device, step: int) -> torch.Tensor:
        """SAGE-V2 global selective surrogate gradient on a disjoint meta batch
        (eval mode, deterministic); caches the detached normalized target."""
        xm, ym, meta_index = self._meta_batch(step, device)
        self._last_meta_index = int(meta_index)
        params = [p for _, p in self._utility_params]
        was_training = self.training
        self.eval()
        try:
            t0 = time.perf_counter()
            bo = self.backbone(xm)
            logits = bo.logits[:, : self.num_classes]
            surrogate = soft_aurc_surrogate(logits, ym, tau=self.surrogate_tau)
            gJ = torch.autograd.grad(surrogate, params, retain_graph=False,
                                     allow_unused=True, materialize_grads=True)
            self._tick("meta_refresh", t0)
        finally:
            self.train(was_training)
        gJ_flat = _cat([_flatten(g) for g in gJ]).detach()
        gJ_n = float(torch.norm(gJ_flat).item())
        s, _, is_zero = normalize_direction(gJ_flat, self.projection_eps)
        self._s_cached.copy_(s)
        self._s_set.copy_(not is_zero)
        self._s_refresh_step.copy_(step)
        self._s_target_age.copy_(0)
        self._last_gJ_norm = gJ_n
        row = {
            "step": int(step), "epoch": int(self._epoch),
            "phase": "refresh", "gJ_norm": gJ_n,
            "meta_batch": int(meta_index), "target_zero": bool(is_zero),
        }
        if not bool(self._profiling):
            selected = [int(i) for i in self._selected.tolist() if int(i) >= 0]
            row["selected"] = [self.site_names[i] for i in selected]
        self._util_rows.append(row)
        return s

    # -- epoch hooks ---------------------------------------------------------
    def on_epoch_start(self, epoch: int):
        self._epoch = int(epoch)
        self._profiling.copy_(self._epoch < self.profiling_epochs)
        self._timers_prev = dict(self._timers)
        self._timer_cnt_prev = dict(self._timer_cnt)

    def on_epoch_end(self, epoch: int, val_metrics: dict):
        if epoch == self.profiling_epochs - 1:
            self._finalize_selection()
        self._write_epoch_log(epoch)
        self._flush_rounds()
        if epoch == int(self.cfg["train"]["epochs"]) - 1:
            self._write_end_manifest()
        self._step_aux_acc = {s: 0.0 for s in self.site_names}
        self._step_aux_loss = {s: 0.0 for s in self.site_names}
        self._step_aux_mass = 0.0
        self._step_lambda_sum = 0.0
        self._step_target_age = 0
        self._step_lambda_acc = {s: 0.0 for s in self.site_names}
        self._epoch_steps = 0

    def _finalize_selection(self):
        n = int(self._utility_n)
        means = torch.zeros(len(self.site_names))
        variances = torch.zeros(len(self.site_names))
        if n > 0:
            means = self._utility_sum.detach().cpu().clone() / float(n)
            variances = (self._utility_sumsq.detach().cpu().clone() / float(n)
                         - means * means).clamp_min(0.0)
        else:
            self._selection_fallback = True
        idx = select_topk_sites(means, self.k)
        with torch.no_grad():
            self._selected.copy_(torch.tensor(idx, dtype=torch.long))
        self._profile_stats = {
            "epoch": int(self._epoch),
            "n_measurements": int(n),
            "means": {s: float(means[i]) for i, s in enumerate(self.site_names)},
            "variances": {s: float(variances[i])
                          for i, s in enumerate(self.site_names)},
            "selected": [self.site_names[i] for i in idx],
            "indices": idx,
            "fallback_empty_measurements": bool(self._selection_fallback),
        }
        try:
            run_dir = os.path.join(self.cfg["results_root"], self.cfg["run_name"])
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "sage_topk_selection.json"), "w") as f:
                json.dump(self._profile_stats, f, indent=2, sort_keys=True)
        except Exception:
            pass

    def _timing_deltas(self) -> Dict[str, dict]:
        out = {}
        for name in self._timers:
            ms = self._timers[name] - self._timers_prev[name]
            cnt = self._timer_cnt[name] - self._timer_cnt_prev[name]
            out[name] = {
                "ms_cum": float(ms),
                "calls": int(cnt),
                "ms_per_call": float(ms / cnt) if cnt else 0.0,
            }
        return out

    def _write_epoch_log(self, epoch: int):
        n_alloc = max(int(self._allocation_steps), 1)
        row = {
            "epoch": int(epoch),
            "phase": "profiling" if self._epoch < self.profiling_epochs else "alloc",
            "timings": self._timing_deltas(),
        }
        if not bool(self._profiling):
            row["aux_mass_mean"] = float(self._step_aux_mass / n_alloc)
            row["lambda_sum_mean"] = float(self._step_lambda_sum / n_alloc)
            row["zero_updates"] = int(self._zero_updates)
            row["allocation_steps"] = int(self._allocation_steps)
            row["zero_update_frac"] = float(int(self._zero_updates) / n_alloc)
            row["lambda"] = {s: float(self._step_lambda_acc[s] / n_alloc)
                             for s in self.site_names}
            row["target_age_mean_cum"] = float(self._step_target_age / n_alloc)
        if self._profile_stats is not None:
            row["profile"] = {
                "selected": self._profile_stats["selected"],
                "means": self._profile_stats["means"],
                "variances": self._profile_stats["variances"],
                "n_measurements": self._profile_stats["n_measurements"],
            }
        n = max(self._epoch_steps, 1)
        for s in self.site_names:
            row[f"aux_acc_{s}"] = float(self._step_aux_acc[s] / n)
            row[f"aux_loss_{s}"] = float(self._step_aux_loss[s] / n)
        self._log.append(row)
        self._write_log()

    def _write_log(self):
        try:
            run_dir = os.path.join(self.cfg["results_root"], self.cfg["run_name"])
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "sage_topk.jsonl"), "a") as f:
                for r in self._log:
                    f.write(json.dumps(r, default=str) + "\n")
            self._log.clear()
        except Exception:
            self._log.clear()

    def _flush_rounds(self):
        try:
            run_dir = os.path.join(self.cfg["results_root"], self.cfg["run_name"])
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "sage_topk_steps.jsonl"), "a") as f:
                for r in self._step_rows:
                    f.write(json.dumps(r, default=str) + "\n")
            with open(os.path.join(run_dir, "sage_topk_utility.jsonl"), "a") as f:
                for r in self._util_rows:
                    f.write(json.dumps(r, default=str) + "\n")
            self._step_rows.clear()
            self._util_rows.clear()
        except Exception:
            self._step_rows.clear()
            self._util_rows.clear()

    def _write_end_manifest(self):
        try:
            run_dir = os.path.join(self.cfg["results_root"], self.cfg["run_name"])
            os.makedirs(run_dir, exist_ok=True)
            total_ms = float(sum(self._timers.values()))
            doc = {
                "selected": (self._profile_stats or {}).get("selected", []),
                "profile": self._profile_stats,
                "zero_updates": int(self._zero_updates),
                "allocation_steps": int(self._allocation_steps),
                "zero_update_frac": float(int(self._zero_updates) / max(
                    int(self._allocation_steps), 1)),
                "timings_ms_total": {k: float(v) for k, v in self._timers.items()},
                "timings_calls": {k: int(v) for k, v in self._timer_cnt.items()},
                "timings_ms_total_sum": total_ms,
            }
            with open(os.path.join(run_dir, "sage_topk_manifest.json"), "w") as f:
                json.dump(doc, f, indent=2, sort_keys=True)
        except Exception:
            pass

    def _tick(self, name: str, t0: float) -> None:
        self._timers[name] += (time.perf_counter() - t0) * 1000.0
        self._timer_cnt[name] += 1


# ---------------------------------------------------------------------------
# reachability/aux-only backward introspection (architecture-neutral)
# ---------------------------------------------------------------------------
def params_reached_by_aux(method: SageTopKMethod, site: str, num_examples: int = 2):
    """Backbone parameter names receiving gradient from site ``site``'s head.

    Same semantics as :func:`scsf.methods.sage_ds.params_reached_by_aux` but
    for the linear companion heads.
    """
    if site not in method.backbone.taps:
        raise KeyError(f"{site!r} is not a registered tap of "
                       f"{type(method.backbone).__name__}")
    dev = next(method.backbone.parameters()).device
    was_training = method.training
    method.eval()
    for p in method.backbone.parameters():
        p.grad = None
    x = torch.randn(num_examples, method.backbone.channels,
                    method.backbone.input_size, method.backbone.input_size,
                    device=dev)
    y = torch.randint(0, method.num_classes, (num_examples,), device=dev)
    with torch.enable_grad():
        bo = method.backbone(x)
        feat = _pool_tap(bo.features[site], method.token)
        loss = F.cross_entropy(method.aux_heads[site](feat), y)
        loss.backward()
    reached = [n for n, p in method.backbone.named_parameters()
               if p.grad is not None and bool(torch.any(p.grad != 0))]
    for p in method.backbone.parameters():
        p.grad = None
    method.train(was_training)
    return reached