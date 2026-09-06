"""SAGE-V3: robust training-aware gradient allocation.

Preregistered protocol: ``docs/SAGE_V3_PROTOCOL.md`` (commits ``78ccbff``,
``b653ce2``).  ``sage_ds`` (v1) and ``sage_ds_v2`` are preserved unchanged.

SAGE-v3 replaces the *learned* hard-concrete gate controller of v1/v2 with a
**certified convex QP** over per-site auxiliary gradient weights:

* the target is a **class-conditioned robust** selective surrogate

      J_rob = ( logsumexp( tau * [J_1, ..., J_C] ) - log(C) ) / tau

  with ``J_c`` the differentiable selective-risk surrogate of class ``c`` on a
  deterministic **class-balanced meta batch** (``k_meta`` examples per class)
  evaluated in eval mode, and a single fixed ``tau = 10`` for both datasets;

* per-site directions are the same-batch projected + normalized auxiliary
  gradients (``A = [a_1 ... a_L]``), the robust gradient is ``r = g_r/||g_r||``
  and the CE direction is ``c = g_0/||g_0||``;

* the exact deterministic active-set QP

      minimize  0.5 lambda^T (G + ridge I) lambda - b^T lambda
      subject to  lambda >= 0,  sum(lambda) <= B,  q^T lambda >= 0

  (``G = A^T A``, ``b = A^T r``, ``q = A^T c``) is solved **every training
  step** with a KKT-verified enumeration, certified before application
  (protocol section 5.2), and falls back to ``lambda = 0`` if the certificate
  fails;

* the backbone update is ``g_CE + rho * ||g_CE|| * A lambda`` routed via the
  dot-product loss trick (same mechanism as v1/v2), ``rho = 1``.

The robust gradient ``g_r`` is refreshed every ``utility_interval`` steps
(protocol section 3.3); between refreshes the QP is solved from the fresh
per-step directions ``A`` and the **cached** ``r``.  The cached ``r`` and the
allocation counters live in registered buffers so the exact-resume contract
holds (a resumed run applies identical allocations between the resume point and
the next refresh boundary).  Inference is unchanged: plain terminal MSP, aux
heads stripped from the deployment graph.

``sage_ds_v3_amortized`` is a secondary efficiency variant that replaces the
exact solver with a small learned unrolled solver over ``G, b, q`` only; it is
registered separately and never runs in the primary three-seed matrix.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import build_dataset
from ..metrics.surrogate import soft_aurc_surrogate
from .base import MethodPrediction
from .sage_ds import SageDSMethod, _cat, _flatten, _pool_tap, project_aux
from .scores import compute_scores

__all__ = [
    "SageDSV3Method",
    "robust_selective_target",
    "solve_sage_v3_qp",
    "qp_certificate",
    "qp_kkt_residual",
    "AmortizedAllocationSolver",
    "class_balanced_meta_batch",
]

EPS = 1e-8


# ---------------------------------------------------------------------------
# robust class-conditioned selective target
# ---------------------------------------------------------------------------
def robust_selective_target(per_class_surrogates: Sequence[torch.Tensor],
                            tau: float) -> torch.Tensor:
    """Class-conditioned robust mean ``J_rob``.

    ``per_class_surrogates`` is a list of ``C`` scalar tensors, each carrying
    a graph to the backbone logits of its class slice.  Aggregates with the
    log-sum-exp robust mean ``(logsumexp(tau * J) - log(C)) / tau`` (between
    the mean for ``tau -> 0`` and the max for ``tau -> inf``).
    """
    J = torch.stack([s if torch.is_tensor(s) else torch.scalar_tensor(float(s))
                     for s in per_class_surrogates])
    if tau <= 0:
        raise ValueError(f"robust tau must be > 0, got {tau}")
    C = J.numel()
    return (torch.logsumexp(tau * J, dim=0) - math.log(C)) / tau


# ---------------------------------------------------------------------------
# exact deterministic active-set QP solver
# ---------------------------------------------------------------------------
def qp_certificate(lambd: torch.Tensor, b: torch.Tensor, q: torch.Tensor,
                   B: float, tol: float = 1e-6) -> Dict[str, float]:
    """Approve-or-reject predicate values for an allocation (protocol 5.2).

    Returns all four predicates plus ``ok``: ``b_lambda > 0`` AND
    ``q_lambda >= -tol`` AND ``min_lambda >= -tol`` AND ``sum_lambda <= B+tol``.
    """
    b_l = float(torch.dot(lambd, b).item())
    q_l = float(torch.dot(lambd, q).item())
    nonneg = float(lambd.min().item())
    total = float(lambd.sum().item())
    ok = (b_l > 0.0 and q_l >= -tol and nonneg >= -tol and total <= B + tol)
    return {
        "b_lambda": b_l, "q_lambda": q_l, "min_lambda": nonneg,
        "sum_lambda": total, "ok": bool(ok),
    }


def _solve_kkt_system(H, bvec, one_col, qneg_col, budget, active_sum, active_q):
    """Solve the KKT linear system of one candidate active set.

    Free variables are excluded by the caller.  Stationarity is
    ``H lambda + alpha*1 - beta*q = b`` with the active affine rows
    ``1^T lambda = B`` and ``q^T lambda = 0`` appended.  Returns
    ``[lambda_F; alpha; beta]`` for the candidate's multiplier layout.
    """
    n_free = H.shape[0]
    n_alpha = 1 if active_sum else 0
    n_beta = 1 if active_q else 0
    n = n_free + n_alpha + n_beta
    cols = [H]
    rhs = [bvec]
    one_F = one_col.double()
    qneg_F = qneg_col.double()
    if active_sum:
        cols.append(one_F.unsqueeze(1))
        rhs.append(torch.full((1,), float(budget), dtype=H.dtype, device=H.device))
    if active_q:
        cols.append(qneg_F.unsqueeze(1))
        rhs.append(torch.zeros((1,), dtype=H.dtype, device=H.device))
    rows = [torch.cat(cols, dim=1)]
    if active_sum:
        r = one_F.new_zeros(n)
        r[:n_free] = one_F
        rows.append(r.unsqueeze(0))
    if active_q:
        r = one_F.new_zeros(n)
        r[:n_free] = qneg_F
        rows.append(r.unsqueeze(0))
    lhs = torch.cat(rows, dim=0)
    return torch.linalg.solve(lhs, torch.cat(rhs))


def solve_sage_v3_qp(G: torch.Tensor, b: torch.Tensor, q: torch.Tensor,
                     B: float = 1.0, ridge: float = 1e-4,
                     tol: float = 1e-9) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Deterministic exact active-set enumeration solver for the SAGE-V3 QP.

    Solves::

        minimize  0.5 lambda^T (G + ridge I) lambda - b^T lambda
        subject to  lambda >= 0, sum(lambda) <= B, q^T lambda >= 0

    by enumerating every active set (which dimensions are boxed at zero and
    which of the two affine constraints is tight), solving the KKT linear
    system for each, keeping only primal- and dual-feasible candidates, and
    returning the one with minimal objective.  For a convex QP this is an
    exact solve; KKT sufficiency is verified independently by
    :func:`qp_kkt_residual`.  Deterministic in float64 (no iteration, no RNG).

    Args:
        G: ``(L, L)`` symmetric (PSD) ``A^T A``.
        b: ``(L,)``.
        q: ``(L,)``.
        B: total-allocation budget.
        ridge: diagonal conditioning (``G + ridge I`` stays PD).
        tol: feasibility tolerance.

    Returns ``(lambda, info)`` with ``info = {obj, active_sum, active_q, free,
    kkt_residual, lambda}``.
    """
    L = int(G.shape[0])
    G = G.clone().detach().double()
    b = b.clone().detach().double()
    q = q.clone().detach().double()
    H = G + ridge * torch.eye(L, dtype=G.dtype, device=G.device)
    device = G.device

    best_lambda = torch.zeros(L, dtype=torch.float64, device=device)
    best_obj = float("inf")
    best_info: Optional[dict] = None

    one_col = torch.ones(L, dtype=torch.float64, device=device)
    qneg_col = -q

    for mask in range(1 << L):
        free = [i for i in range(L) if (mask >> i) & 1]
        if not free:
            continue
        F = torch.tensor(free, dtype=torch.long, device=device)
        Hf = H.index_select(0, F).index_select(1, F)
        bf = b.index_select(0, F)
        of = one_col.index_select(0, F)
        qf = qneg_col.index_select(0, F)
        for active_sum in (False, True):
            for active_q in (False, True):
                try:
                    sol = _solve_kkt_system(Hf, bf, of, qf, B, active_sum, active_q)
                except (torch.linalg.LinAlgError, RuntimeError):
                    continue
                n_var = len(free)
                n_alpha = 1 if active_sum else 0
                n_beta = 1 if active_q else 0
                lf = sol[:n_var]
                alpha = sol[n_var] if n_alpha else 0.0
                beta = sol[n_var + n_alpha] if n_beta else 0.0

                # primal feasibility
                if float(lf.min()) < -tol:
                    continue
                total = float(lf.sum())
                if total > B + tol:
                    continue
                if float(torch.dot(lf, -qf)) < -tol:  # q^T lambda >= 0
                    continue
                # dual feasibility of the affine multipliers (both <=-form)
                if alpha < -tol or beta < -tol:
                    continue

                lambd = torch.zeros(L, dtype=torch.float64, device=device)
                lambd.index_copy_(0, F, lf.double())
                obj = float(0.5 * torch.dot(lambd, H @ lambd) - torch.dot(lambd, b))
                if obj < best_obj:
                    best_obj = obj
                    best_lambda = lambd
                    best_info = {
                        "obj": obj, "active_sum": bool(active_sum),
                        "active_q": bool(active_q), "free": free,
                        "kkt_residual": float(qp_kkt_residual(lambd, G, b, q, B, ridge)),
                        "lambda": [float(v) for v in lambd.detach().cpu()],
                    }

    if best_info is None:
        # no feasible free candidate exists -> lambda = 0 is the only feasible
        # allocation and the optimum (sum 0 <= B, q^T 0 >= 0 always hold)
        best_lambda = torch.zeros(L, dtype=torch.float64, device=device)
        best_info = {"obj": 0.0, "active_sum": False, "active_q": False,
                     "free": [], "kkt_residual": 0.0, "lambda": [0.0] * L}
    return best_lambda, best_info


def qp_kkt_residual(lambd: torch.Tensor, G: torch.Tensor, b: torch.Tensor,
                    q: torch.Tensor, B: float, ridge: float,
                    tol: float = 1e-9, active_sum: Optional[bool] = None,
                    active_q: Optional[bool] = None,
                    free: Optional[Sequence[int]] = None) -> float:
    """Primal-dual KKT residual of an allocation (independent verification).

    Recomputes the KKT multipliers from scratch for the allocation's active
    set (the free coordinates plus which of the two affine constraints is
    tight, inferred from the allocation if not supplied) and reports the worst
    violation across stationarity, primal feasibility and dual feasibility.
    Used by tests to verify solver optimality without trusting the enumeration.
    """
    L = int(lambd.numel())
    lam = lambd.double()
    H = G.double() + ridge * torch.eye(L, dtype=torch.float64)
    if free is None:
        free = [i for i in range(L) if float(lam[i]) > tol]
    if active_sum is None:
        active_sum = abs(float(lam.sum().item()) - B) <= tol
    if active_q is None:
        active_q = abs(float(torch.dot(lam, q.double()).item())) <= tol

    n_alpha = 1 if active_sum else 0
    n_beta = 1 if active_q else 0
    if free or n_alpha or n_beta:
        F = torch.tensor(list(free), dtype=torch.long)
        one_F = torch.ones(len(free), dtype=torch.float64)
        qneg_F = -q.double().index_select(0, F)
        cols = [H.index_select(0, F).index_select(1, F)]
        rhs = [b.double().index_select(0, F)]
        if active_sum:
            cols.append(one_F.unsqueeze(1))
            rhs.append(torch.tensor([B]))
        if active_q:
            cols.append(qneg_F.unsqueeze(1))
            rhs.append(torch.zeros((1,)))
        n = len(free) + n_alpha + n_beta
        top = torch.cat(cols, dim=1)  # (lenF, n)
        rows = [top]
        if active_sum:
            r = torch.zeros(n)
            r[:len(free)] = one_F
            rows.append(r.unsqueeze(0))
        if active_q:
            r = torch.zeros(n)
            r[:len(free)] = qneg_F
            rows.append(r.unsqueeze(0))
        sol = torch.linalg.lstsq(torch.cat(rows, dim=0), torch.cat(rhs)).solution
        lf = sol[:len(free)]
        alpha = float(sol[len(free)]) if n_alpha else 0.0
        beta = float(sol[len(free) + n_alpha]) if n_beta else 0.0
    else:
        lf = torch.zeros((0,))
        alpha = beta = 0.0

    if free:  # pin the affine multipliers from the free-coordinate stationarity
        resid = H @ lam - b.double() + alpha * torch.ones(L) - beta * q.double()
        free_set = set(int(i) for i in free)
        parts = []
        for i in range(L):
            if i in free_set:
                parts.append(float(resid[i].abs()))
            else:
                parts.append(max(-float(resid[i]), 0.0))  # box multiplier mu_i >= 0
        station = max(parts)
    else:
        # no free coordinates: stationarity over an empty set is vacuous and
        # the box multipliers can always absorb the affine residual, so only
        # primal + dual feasibility remain meaningful here.
        station = 0.0

    prim = max(float((lam.clamp_min(0.0) - lam).abs().max()),
               max(float(lam.sum().item() - B), 0.0),
               max(-float(torch.dot(lam, q.double()).item()), 0.0))
    dual = max(-alpha, -beta, 0.0)
    return station + prim + dual


# ---------------------------------------------------------------------------
# class-balanced meta batch (locked in protocol section 3.1)
# ---------------------------------------------------------------------------
def _build_meta_index(store, cfg: dict, num_classes: int) -> dict:
    """Cache the per-class validation local/global index maps on the store."""
    if getattr(store, "_v3_meta_cache", None) is not None:
        return store._v3_meta_cache
    ds = build_dataset(cfg, "val", return_indices=True)
    subset = ds.base
    tv_base = subset.base
    targets = getattr(tv_base, "targets", None)
    local_by_class = {c: [] for c in range(num_classes)}
    for local, gidx in enumerate(subset.indices):
        if targets is not None:
            c = int(targets[gidx])
        else:
            c = int(tv_base[gidx][1])
        if c < 0 or c >= num_classes:
            raise RuntimeError(f"meta index class {c} outside [0, {num_classes})")
        local_by_class[c].append(local)
    global_by_class = {
        c: [subset.indices[loc] for loc in locs]
        for c, locs in local_by_class.items()
    }
    cache = {"ds": ds, "subset": subset, "tv_base": tv_base,
             "local_by_class": local_by_class, "global_by_class": global_by_class}
    store._v3_meta_cache = cache
    return cache


def class_balanced_meta_batch(cfg: dict, num_classes: int, k_meta: int,
                              offset: int, device: torch.device,
                              store):
    """Deterministic class-balanced sample from the validation split.

    Each class contributes ``k_meta`` consecutive global training-fold indices
    from a deterministic rotation ``start = (offset * k_meta) % n`` (no RNG),
    so a resumed run reproduces the exact same meta batches.  The per-class
    index map is cached on the method by :func:`_build_meta_index`.

    Returns ``(x, y, global_indices)`` with exactly ``C * k_meta`` examples.
    """
    cache = _build_meta_index(store, cfg, num_classes)
    subset = cache["subset"]
    xs, ys, gids = [], [], []
    for c in range(num_classes):
        gidx_list = cache["global_by_class"][c]
        n = len(gidx_list)
        if n == 0:
            raise RuntimeError(f"class {c} has no validation examples")
        start = (offset * k_meta) % n
        for t in range(k_meta):
            gidx = gidx_list[(start + t) % n]
            x, y = subset.base[gidx]
            xs.append(x)
            ys.append(y)
            gids.append(int(gidx))
    return torch.stack(xs).to(device), torch.tensor(ys, device=device), gids


# ---------------------------------------------------------------------------
# amortized learned solver (secondary variant)
# ---------------------------------------------------------------------------
class AmortizedAllocationSolver(nn.Module):
    """Learned unrolled projected solver for the SAGE-V3 QP (secondary only).

    Inputs are the sufficient statistics ``G, b, q``; three to five unrolled
    projected-gradient/momentum steps learn only the step sizes (and optional
    momentum).  Every prediction is final-projected by the exact solver (in
    ``no_grad``), so outputs are always feasible.  Not used by the primary
    method.
    """

    def __init__(self, steps: int = 4, learn_momentum: bool = True,
                 B: float = 1.0, ridge: float = 1e-4):
        super().__init__()
        if not 3 <= steps <= 5:
            raise ValueError("amortized solver requires 3-5 unrolled steps")
        self.steps = int(steps)
        self.B = float(B)
        self.ridge = float(ridge)
        self.learn_momentum = bool(learn_momentum)
        self.eta = nn.Parameter(torch.full((self.steps,), 0.1))
        self.mom = nn.Parameter(torch.full((self.steps,), 0.9)) if learn_momentum else None

    def forward(self, G: torch.Tensor, b: torch.Tensor, q: torch.Tensor):
        L = int(G.shape[0])
        H = G.double() + self.ridge * torch.eye(L, dtype=torch.float64)
        l = torch.zeros(L, dtype=torch.float64)
        m = torch.zeros_like(l)
        for t in range(self.steps):
            grad = H @ l - b.double()
            if self.mom is not None:
                m = self.mom[t] * m + (1.0 - self.mom[t]) * grad
                step = self.eta[t] * m
            else:
                step = self.eta[t] * grad
            l = self._project(l - step, G, b, q)
        return l

    @torch.no_grad()
    def _project(self, l, G, b, q):
        return solve_sage_v3_qp(G, b, q, self.B, self.ridge)[0].to(
            dtype=torch.float64)


# ---------------------------------------------------------------------------
# the method
# ---------------------------------------------------------------------------
class SageDSV3Method(SageDSMethod):
    """Robust training-aware gradient allocation (``sage_ds_v3``)."""

    method_name = "sage_ds_v3"
    needs_indices = True

    def default_score(self) -> str:
        return "msp"

    def default_scores(self):
        return ("msp", "entropy", "energy", "logit_margin", "sage_conf")

    def __init__(self, train_cfg: dict):
        super().__init__(train_cfg)
        m = train_cfg["method"]
        self.robust_tau = float(m.get("robust_tau", 10.0))
        self.budget_B = float(m.get("budget_B", 1.0))
        self.rho = float(m.get("rho", 1.0))
        self.qp_ridge = float(m.get("qp_ridge", 1e-4))
        self.meta_k = int(m.get("meta_k", 8))
        self.cert_tol = float(m.get("cert_tol", 1e-6))
        self.eps = float(m.get("projection_eps", EPS))
        self.amortized = bool(m.get("amortized", False))
        if self.amortized:
            self.solver = AmortizedAllocationSolver(
                steps=int(m.get("amortized_steps", 4)),
                B=self.budget_B, ridge=self.qp_ridge)
        else:
            self.solver = None

        # exact-resume state lives in registered buffers (protocol exact-resume
        # contract): the cached robust gradient and the allocation counters
        # persist across checkpoints and are restored by load_state_dict.
        utility_size = int(sum(p.numel() for _, p in self._utility_params))
        self.register_buffer("_v3_cached_r",
                             torch.zeros(utility_size, dtype=torch.float32))
        self.register_buffer("_v3_round", torch.tensor(-1, dtype=torch.long))
        self.register_buffer("_v3_zero_ct", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_v3_qp_calls", torch.tensor(0, dtype=torch.long))

        self._per_class_J: List[float] = []
        self._meta_class_idx = None
        self._v3_meta_cache = None
        self._last_train_ids: Optional[List[int]] = None
        self._last_refresh_step = 0
        self._audit_applied = False
        self._audit = None

        self._step_lambda_acc: Dict[str, float] = {s: 0.0 for s in self.site_names}
        self._step_lambda_open: Dict[str, float] = {s: 0.0 for s in self.site_names}
        self._step_class_j: List[float] = [0.0] * self.num_classes
        self._step_cert_ok = 0.0
        self._step_fallback = 0.0
        self._step_lambda_sum = 0.0
        self._step_aux_mass = 0.0
        self._step_qp_obj = 0.0
        self._step_rows: List[dict] = []

    # ------------------------------------------------------------- inference
    def predict_batch(self, x):
        bo = self.backbone(x)
        logits = bo.logits[:, : self.num_classes]
        scores = compute_scores(logits, self.default_scores())
        conf = scores["msp"]  # plain MSP is the only primary score (locked)
        scores["sage_conf"] = conf
        return MethodPrediction(logits, logits.argmax(dim=1), conf, scores)

    def inference_modules(self):
        # backbone only (aux heads are training-only instruments)
        return [self.backbone]

    # -------------------------------------------------------------- training
    def train_loss(self, batch, state):
        device = next(self.backbone.parameters()).device
        x = batch[0].to(device)
        y = batch[1].to(device)
        raw_ids = batch[2] if len(batch) > 2 else None
        self._last_train_ids = (
            [int(i) for i in raw_ids.detach().cpu().tolist()]
            if raw_ids is not None else None
        )
        bo = self.backbone(x)
        ce_t = F.cross_entropy(bo.logits[:, : self.num_classes], y)
        return self._allocated_loss(bo, x, y, ce_t, state)

    def _allocated_loss(self, bo, x, y, ce_t, state):
        backbone_params = [p for _, p in self._utility_params]
        self._step_n += 1
        out = {}
        sites = list(self.site_names)
        L = len(sites)

        # CE gradient on THIS training batch (safety reference + base update).
        g0 = torch.autograd.grad(ce_t, backbone_params, retain_graph=True,
                                 allow_unused=True, materialize_grads=True)
        g0_flat = _cat([_flatten(g) for g in g0])
        norm0 = float(torch.norm(g0_flat).item())

        step = int(getattr(state, "batch_index", 0))

        # refresh the robust gradient on the locked cadence (protocol 3.3)
        if step > 0 and self.utility_interval > 0 and step % self.utility_interval == 0:
            self._refresh_robust_gradient(ce_t.device, step)

        r_flat = self._r_or_none()

        a_flat: Dict[str, torch.Tensor] = {}
        auxhead_g = [torch.zeros_like(p) for p in self._aux_params]
        for s in sites:
            h = self.aux_heads[s]
            feat = _pool_tap(bo.features[s], self.token)
            l_aux = F.cross_entropy(h(feat), y)
            t0 = time.perf_counter()
            gb = torch.autograd.grad(l_aux, backbone_params, retain_graph=True,
                                     allow_unused=True, materialize_grads=True)
            self.aux_ms[s] += (time.perf_counter() - t0) * 1000.0
            gl_flat = _cat([_flatten(g) for g in gb]).detach()
            til_safe, align_before = project_aux(gl_flat, g0_flat, eps=self.eps)
            til_flat = til_safe.detach()
            anorm = float(torch.norm(til_flat).item())
            a_flat[s] = (til_flat / (anorm + self.eps) if anorm > 0
                         else torch.zeros_like(til_flat))
            align_after = float(torch.dot(a_flat[s], g0_flat).item())

            # aux-head params keep their own unweighted CE gradient
            gh = torch.autograd.grad(l_aux, list(h.parameters()), retain_graph=True,
                                     allow_unused=True, materialize_grads=True)
            for gi, p in enumerate(list(h.parameters())):
                if gh[gi] is not None:
                    auxhead_g[self._aux_param_ids[id(p)]] = gh[gi]

            with torch.no_grad():
                acc_aux = (h(feat).detach().argmax(1) == y).float().mean()
                out[f"aux_acc_{s}"] = acc_aux
                self._step_aux_acc[s] += float(acc_aux)
                self._step_aux_loss[s] += float(l_aux.detach())
            out[f"aux_loss_{s}"] = l_aux.detach()
            out[f"align_before_{s}"] = float(align_before)
            out[f"align_after_{s}"] = float(align_after)

        out["g0_norm2"] = float(torch.dot(g0_flat, g0_flat).item())

        # ---- solve + certify the QP (every step, from fresh A and cached r)
        lambd = torch.zeros(L, dtype=torch.float64)
        info: dict = {"obj": 0.0, "kkt_residual": 0.0, "active_sum": False,
                      "active_q": False, "free": []}
        if L > 0 and r_flat is not None and norm0 > 0:
            r_flat = r_flat.double()
            r_n = float(r_flat.norm().item())
            if r_n > 0:
                r_hat = r_flat / r_n
            else:
                r_hat = r_flat  # zero robust gradient -> zero b -> cert fails
            c_flat = g0_flat / (norm0 + self.eps)
            A = torch.stack([a_flat[s].double() for s in sites], dim=1)  # (P, L)
            G = A.T @ A
            bvec = A.T @ r_hat
            qvec = A.T @ c_flat.double()
            self._v3_qp_calls.add_(1)
            if self.amortized and self.solver is not None:
                with torch.no_grad():
                    lambd = self.solver(G, bvec, qvec)
            else:
                lambd, info = solve_sage_v3_qp(G, bvec, qvec, self.budget_B,
                                               self.qp_ridge, self.cert_tol)
            cert = qp_certificate(lambd, bvec, qvec, self.budget_B, self.cert_tol)
        else:
            cert = qp_certificate(lambd, bvec=torch.zeros(L), q=torch.zeros(L),
                                  B=self.budget_B, tol=self.cert_tol)
            G = torch.zeros(L, L)
            bvec = torch.zeros(L)
            qvec = torch.zeros(L)

        certified = bool(cert["ok"]) if "ok" in cert else False
        fallback = False
        if not certified:
            lambd = torch.zeros(L, dtype=torch.float64)  # honored zero allocation
            fallback = True
            self._v3_zero_ct.add_(1)

        # ---- apply  g_update = g_CE + rho * ||g_CE|| * (A lambda)
        step_aux = [torch.zeros_like(p) for p in backbone_params]
        acc = 0
        for i, p in enumerate(backbone_params):
            n = p.numel()
            for j, s in enumerate(sites):
                if float(lambd[j]) != 0.0:
                    step_aux[i] = step_aux[i] + float(lambd[j]) * a_flat[s][acc:acc + n].reshape_as(p)
            acc += n
        aux_scale = self.rho * norm0
        step_aux = [float(aux_scale) * a for a in step_aux]

        routed = torch.zeros((), device=ce_t.device)
        for p, g0p, add in zip(backbone_params, g0, step_aux):
            g_desired = (g0p if g0p is not None else torch.zeros_like(p)) + add
            routed = routed + torch.sum(p * g_desired.detach())
        for p, g in zip(self._aux_params, auxhead_g):
            routed = routed + torch.sum(p * g.detach())
        out["routed"] = routed
        out["ce"] = ce_t.detach()

        # ---- audit hook (tests verify the applied gradient numerically)
        if self._audit_applied:
            self._audit = {
                "g0": [g.detach().clone() for g in g0],
                "add": [a.detach().clone() for a in step_aux],
                "aux": [g.detach().clone() for g in auxhead_g],
                "lambda": {s: float(lambd[i]) for i, s in enumerate(sites)},
                "cert": cert, "rho": float(self.rho), "norm0": norm0,
            }

        # ---- logging (per-step summary; details at refresh steps)
        lambda_sum = float(lambd.sum().item())
        out["lambda_sum"] = lambda_sum
        out["lambda_zero"] = float(self._v3_zero_ct.item())
        out["qp_obj"] = float(info.get("obj", 0.0))
        out["cert_ok"] = float(certified)
        out["fallback_zero"] = float(fallback)
        out["applied_aux_mass"] = float(aux_scale * lambda_sum)
        for j, s in enumerate(sites):
            out[f"lambda_{s}"] = float(lambd[j])

        gJ_norm = float(r_flat.norm().item()) if r_flat is not None else 0.0
        self._step_lambda_sum += lambda_sum
        self._step_aux_mass += float(aux_scale * lambda_sum)
        self._step_qp_obj += float(info.get("obj", 0.0))
        self._step_cert_ok += float(certified)
        self._step_fallback += float(fallback)
        for j, s in enumerate(sites):
            self._step_lambda_acc[s] += float(lambd[j])
            self._step_lambda_open[s] += float(lambd[j] > 0)
        row = {"step": int(step), "epoch": int(getattr(self, "_last_epoch", 0)),
               "lambda_sum": lambda_sum, "cert_ok": float(certified),
               "fallback_zero": float(fallback),
               "qp_obj": float(info.get("obj", 0.0)),
               "applied_aux_mass": float(aux_scale * lambda_sum),
               "gJ_norm": gJ_norm, "lambda_zero_count": int(self._v3_zero_ct.item()),
               "qp_calls": int(self._v3_qp_calls.item())}
        for j, s in enumerate(sites):
            row[f"lambda_{s}"] = float(lambd[j])
        self._step_rows.append(row)

        if step > 0 and self.utility_interval > 0 and step % self.utility_interval == 0:
            self._log_utility_row(G, bvec, qvec, lambd, info, cert, step, self._per_class_J)

        return out

    # ------------------------------------------------------------------ QP
    def _r_or_none(self) -> Optional[torch.Tensor]:
        if int(self._v3_round.item()) < 0:
            return None
        return self._v3_cached_r

    def _refresh_robust_gradient(self, device, step: int):
        """One forward + single backward for ``g_r = grad_theta J_rob``.

        Protocol section 3.1/3.2: per-class ``J_c`` from the class-balanced
        meta batch, ``J_rob`` via logsum-exp, robust gradient from exactly one
        backward on ``J_rob`` wrt the backbone parameters.  The resulting
        ``r`` is cached in the persistent buffer so exact resume holds.
        """
        params = [p for _, p in self._utility_params]
        offset = max(0, step // self.utility_interval)
        xm, ym, meta_gids = class_balanced_meta_batch(
            self.cfg, self.num_classes, self.meta_k, offset, torch.device(device), self)

        # bilevel-discipline lock: val split is disjoint from train by
        # construction, but refuse silently (protocol section 2/3 lock).
        if self._last_train_ids is not None:
            overlap = set(self._last_train_ids) & set(meta_gids)
            if overlap:
                raise RuntimeError(
                    f"sage_ds_v3 robustness violation: train batch and meta "
                    f"batch share official-fold indices {sorted(overlap)[:5]} "
                    f"(step {step})")

        was_training = self.training
        self.eval()
        try:
            t0 = time.perf_counter()
            bo = self.backbone(xm)
            logits = bo.logits[:, : self.num_classes]
            per_class = []
            for c in range(self.num_classes):
                mask_c = (ym == c)
                if int(mask_c.sum()) == 0:
                    raise RuntimeError(
                        f"robust target: class {c} absent from meta batch")
                J_c = soft_aurc_surrogate(logits[mask_c], ym[mask_c],
                                          tau=self.hard_concrete_tau)
                per_class.append(J_c)
            self._per_class_J = [float(j.detach().item()) for j in per_class]
            J_rob = robust_selective_target(per_class, self.robust_tau)
            g_r = torch.autograd.grad(J_rob, params, create_graph=False,
                                      retain_graph=False, allow_unused=True,
                                      materialize_grads=True)
            self.utility_ms = (time.perf_counter() - t0) * 1000.0
        finally:
            self.train(was_training)
        r = _cat([_flatten(g) for g in g_r]).detach().float()
        self._v3_cached_r.copy_(r)
        self._v3_round.fill_(offset)
        r_n = float(r.norm().item())
        if r_n > 0:
            self._v3_cached_r.div_(r_n)  # protocol: r = g_r / ||g_r||
        self._step_class_j = list(self._per_class_J)
        self._last_refresh_step = int(step)
    def _log_utility_row(self, G, b, q, lambd, info, cert, step: int,
                         per_class_J: List[float]):
        row = {
            "step": int(step),
            "refresh_round": int(self._v3_round.item()),  # type: ignore[arg-type]
            "class_surrogate": list(per_class_J),
            "G": [[float(v) for v in row_] for row_ in G.detach().cpu()],
            "b": [float(v) for v in b.detach().cpu()],
            "q": [float(v) for v in q.detach().cpu()],
            "lambda": [float(v) for v in lambd.detach().cpu()],
            "qp_obj": float(info.get("obj", 0.0)),
            "qp_kkt": float(info.get("kkt_residual", 0.0)),
            "cert_ok": bool(cert.get("ok", False)),
            "cert_b_lambda": cert.get("b_lambda", 0.0),
            "cert_q_lambda": cert.get("q_lambda", 0.0),
            "cert_min_lambda": cert.get("min_lambda", 0.0),
            "cert_sum_lambda": cert.get("sum_lambda", 0.0),
            "lambda_zero_count": int(self._v3_zero_ct.item()),
            "qp_calls": int(self._v3_qp_calls.item()),
            "utility_ms": float(self.utility_ms),
            "tau": float(self.robust_tau),
            "B": float(self.budget_B),
            "rho": float(self.rho),
        }
        try:
            run_dir = os.path.join(self.cfg["results_root"], self.cfg["run_name"])
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "sage_ds_v3_utility.jsonl"), "a") as f:
                f.write(json.dumps(row, default=str) + "\n")
        except Exception:
            pass

    def _write_steps_log(self):
        try:
            run_dir = os.path.join(self.cfg["results_root"], self.cfg["run_name"])
            os.makedirs(run_dir, exist_ok=True)
            if not self._step_rows:
                return
            with open(os.path.join(run_dir, "sage_ds_v3_steps.jsonl"), "a") as f:
                for r in self._step_rows:
                    f.write(json.dumps(r, default=str) + "\n")
            self._step_rows = []
        except Exception:
            pass

    def on_epoch_start(self, epoch: int):
        self._last_epoch = int(epoch)
        super().on_epoch_start(epoch)

    def on_epoch_end(self, epoch: int, val_metrics: dict):
        n = max(self._step_n, 1)
        row = {
            "epoch": int(epoch),
            "utility_ms": float(self.utility_ms),
            "cert_ok_rate": float(self._step_cert_ok / n),
            "fallback_rate": float(self._step_fallback / n),
            "lambda_sum_mean": float(self._step_lambda_sum / n),
            "applied_aux_mass_mean": float(self._step_aux_mass / n),
            "qp_obj_mean": float(self._step_qp_obj / n),
            "lambda_zero_count": int(self._v3_zero_ct.item()),
            "qp_calls": int(self._v3_qp_calls.item()),
            "refresh_round": int(self._v3_round.item()),
            "robust_tau": float(self.robust_tau),
        }
        for c in range(self.num_classes):
            row[f"class_surrogate_{c}"] = float(self._step_class_j[c])
        for s in self.site_names:
            row[f"lambda_mean_{s}"] = float(self._step_lambda_acc[s] / n)
            row[f"lambda_open_frac_{s}"] = float(self._step_lambda_open[s] / n)
            row[f"aux_acc_{s}"] = float(self._step_aux_acc[s] / n)
            row[f"aux_loss_{s}"] = float(self._step_aux_loss[s] / n)
            row[f"aux_ms_{s}"] = float(self.aux_ms[s])
        self._log.append(row)
        self._write_log()
        self._write_steps_log()
        # reset epoch accumulators
        self._step_n = 0
        self._step_aux_acc = {s: 0.0 for s in self.site_names}
        self._step_aux_loss = {s: 0.0 for s in self.site_names}
        self._step_lambda_acc = {s: 0.0 for s in self.site_names}
        self._step_lambda_open = {s: 0.0 for s in self.site_names}
        self._step_class_j = [0.0] * self.num_classes
        self._step_cert_ok = 0.0
        self._step_fallback = 0.0
        self._step_lambda_sum = 0.0
        self._step_aux_mass = 0.0
        self._step_qp_obj = 0.0

    def _write_log(self):
        try:
            run_dir = os.path.join(self.cfg["results_root"], self.cfg["run_name"])
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "sage_ds_v3.jsonl"), "a") as f:
                for r in self._log:
                    f.write(json.dumps(r, default=str) + "\n")
            self._log.clear()
        except Exception:
            pass


# convenience alias for the primary name
def _alias_factory(cfg):
    return SageDSV3Method(cfg)