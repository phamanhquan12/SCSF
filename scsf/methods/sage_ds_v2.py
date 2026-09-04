"""SAGE-V2: bilevel-utility gated selective supervision.

Preregistered protocol: ``docs/SAGE_V2_PROTOCOL.md``.

SAGE-V2 corrects :mod:`scsf.methods.sage_ds` (v1).  v1 estimated the per-site
selective utility ``U_l = <g_sel, g_l>`` with both gradients on *one* validation
batch and projected the *combined* auxiliary gradient against an EMA of past CE
gradients.  SAGE-V2 uses the true bilevel structure:

* the per-site supervision direction is the **training-side** gradient
  ``g_l_train = grad L_l_aux(B_train)``, projected per site against *this
  batch's* CE gradient ``g0_train = grad L_CE(B_train)`` (CE-safety projection,
  exact per-step and per-site);
* the selective objective is graded by ``g_J_meta = grad AURC~(B_meta)`` from a
  **disjoint held-out meta batch** evaluated in eval mode;
* the raw utility ``U_raw_l = <g_J_meta, tilde_g_l>`` is retained and logged,
  and the **cosine** utility ``U_cos_l = U_raw_l / (||g_J_meta|| ||tilde_g_l||
  + eps)`` drives the hard-concrete gate controller.

Distinctness of the two sides is enforced at runtime: the method requests the
global training-fold index of every batch (``needs_indices = True``) and the
utility estimate raises if the train batch and the meta batch share a sample.

Applied gradient on the backbone at every step::

    g_applied = g0_train + sum_l (z_l * s) * tilde_g_l

with ``z_l`` a sampled hard-concrete gate and ``s`` one fixed global
supervision scale.  Auxiliary-head parameters receive their own unweighted CE
gradient.  The inference score is unchanged: plain terminal MSP, aux heads and
the controller stripped from the deployment graph.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from ..data import build_dataloader
from ..metrics.surrogate import soft_aurc_surrogate
from .base import MethodPrediction
from .sage_ds import SageDSMethod, _cat, _flatten, _pool_tap, project_aux
from .scores import compute_scores

__all__ = [
    "SageDSV2Method",
    "bilevel_utilities",
    "cosine_utility",
    "support_fraction",
]

EPS = 1e-8

_VALID_TOPO = ("sparse_utility_safe",)


# ---------------------------------------------------------------------------
# utility math (unit-testable without a full method)
# ---------------------------------------------------------------------------
def support_fraction(g_flat: Optional[torch.Tensor]) -> float:
    """Fraction of nonzero entries of a (flattened) gradient vector."""
    if g_flat is None or g_flat.numel() == 0:
        return 0.0
    return float(torch.count_nonzero(g_flat).double() / g_flat.numel())


def cosine_utility(u_raw: float, g_meta_norm: float, g_proj_norm: float,
                   eps: float = EPS) -> float:
    """Scale-free cosine utility ``U_cos = U_raw / (||gJ|| ||tilde|| + eps)``.

    Zero-safe: with a zero gradient pair both the numerator and denominator
    vanish, and the ``eps`` in the denominator keeps the result a finite 0.0.
    """
    denom = g_meta_norm * g_proj_norm + eps
    if denom <= 0.0:
        return 0.0
    return float(u_raw) / float(denom)


def bilevel_utilities(g_meta: torch.Tensor, g_aux: torch.Tensor,
                      g_proj: torch.Tensor, eps: float = EPS) -> Dict[str, float]:
    """All per-site bilevel utility quantities for one site.

    Args:
        g_meta: flat ``g_J_meta`` (selective surrogate gradient, meta batch).
        g_aux: flat unprojected ``g_l_train`` (aux gradient, train batch).
        g_proj: flat projected ``tilde_g_l`` (CE-safe, train batch).
        eps: projection/cosine epsilon.

    Returns ``{unprojected, raw, cos, gJ_norm, gl_norm, tilde_norm}`` where
    ``raw`` = ``<g_meta, g_proj>`` (``U_raw``), ``cos`` = cosine utility,
    ``unprojected`` = ``<g_meta, g_aux>`` (v1-style pairing, now with disjoint
    batches), all finite.
    """
    gJ_n = float(torch.norm(g_meta).item())
    gl_n = float(torch.norm(g_aux).item())
    tilde_n = float(torch.norm(g_proj).item())
    u_unproj = float(torch.dot(g_meta, g_aux).item())
    u_raw = float(torch.dot(g_meta, g_proj).item())
    return {
        "unprojected": u_unproj,
        "raw": u_raw,
        "cos": cosine_utility(u_raw, gJ_n, tilde_n, eps),
        "gJ_norm": gJ_n,
        "gl_norm": gl_n,
        "tilde_norm": tilde_n,
    }


# ---------------------------------------------------------------------------
# the method
# ---------------------------------------------------------------------------
class SageDSV2Method(SageDSMethod):
    """Bilevel-utility gated selective supervision (``sage_ds_v2``)."""

    method_name = "sage_ds_v2"
    #: the data loader returns the global training-fold index of every batch so
    #: the bilevel sides can be proven disjoint (protocol section 2/3).
    needs_indices = True

    def __init__(self, train_cfg: dict):
        super().__init__(train_cfg)
        topo = str(train_cfg["method"].get("topology", "sparse_utility_safe"))
        if topo not in _VALID_TOPO:
            raise ValueError(
                f"sage_ds_v2 only supports topology in {_VALID_TOPO}, got {topo!r}")
        self.eps = float(train_cfg["method"].get("projection_eps", EPS))
        self._last_train_ids: Optional[List[int]] = None
        self._audit_applied = False
        self._audit: Optional[dict] = None

    # ------------------------------------------------------------- inference
    def predict_batch(self, x):
        bo = self.backbone(x)
        logits = bo.logits[:, : self.num_classes]
        scores = compute_scores(logits, self.default_scores())
        conf = scores["msp"]  # plain MSP is the only primary score (locked)
        scores["sage_conf"] = conf
        return MethodPrediction(logits, logits.argmax(dim=1), conf, scores)

    def inference_modules(self):
        # backbone only: the risk head is not part of the v2 protocol
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
        return self._bilevel_loss(bo, x, y, ce_t, state)

    def _bilevel_loss(self, bo, x, y, ce_t, state):
        backbone_params = [p for _, p in self._utility_params]
        self._step_n += 1
        out = {}

        # CE gradient on THIS training batch (CE-safety reference + part of the
        # applied gradient).  No EMA: the safety inequality is exact per step.
        g0 = torch.autograd.grad(ce_t, backbone_params, retain_graph=True,
                                 allow_unused=True, materialize_grads=True)
        g0_flat = _cat([_flatten(g) for g in g0])

        sampled = self.controller.sample_all()
        step_aux = [torch.zeros_like(p) for p in backbone_params]
        auxhead_g = [torch.zeros_like(p) for p in self._aux_params]
        g_l_raw: Dict[str, torch.Tensor] = {}
        tilde: Dict[str, torch.Tensor] = {}

        for s in self.site_names:
            h = self.aux_heads[s]
            feat = _pool_tap(bo.features[s], self.token)
            l_aux = F.cross_entropy(h(feat), y)
            # gate weight (detached: the gate is manual, its gradient must never
            # flow through the routed training loss — locked by a smoke test)
            w = sampled[s].detach() * self.supervision_scale
            t0 = time.perf_counter()
            gb = torch.autograd.grad(l_aux, backbone_params, retain_graph=True,
                                     allow_unused=True, materialize_grads=True)
            self.aux_ms[s] += (time.perf_counter() - t0) * 1000.0
            gl_flat = _cat([_flatten(g) for g in gb]).detach()
            g_l_raw[s] = gl_flat
            til_safe, align_before = project_aux(gl_flat, g0_flat, eps=self.eps)
            til_flat = til_safe.detach()
            tilde[s] = til_flat
            align_after = float(torch.dot(til_flat, g0_flat).item())

            acc = 0
            for i, p in enumerate(backbone_params):
                n = p.numel()
                step_aux[i] = step_aux[i] + w * til_flat[acc:acc + n].reshape_as(p)
                acc += n

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
                self._step_l0[s] += float(self.controller.gates[s].l0_norm())
            out[f"aux_loss_{s}"] = l_aux.detach()
            out[f"gatep_{s}"] = self.controller.gate_prob(s).detach()
            out[f"gate_{s}"] = (w.detach() / self.supervision_scale).clamp(0.0, 1.0)
            out[f"align_before_{s}"] = float(align_before)
            out[f"align_after_{s}"] = float(align_after)

        mean_before = float(sum(out[f"align_before_{s}"] for s in self.site_names)
                            / max(len(self.site_names), 1))
        mean_after = float(sum(out[f"align_after_{s}"] for s in self.site_names)
                           / max(len(self.site_names), 1))
        self._last_align = (mean_before, mean_after)
        out["align_before"] = mean_before
        out["align_after"] = mean_after

        # applied gradient = CE + sum_l (z_l * s) * tilde_g_l, routed directly
        routed = torch.zeros((), device=ce_t.device)
        for p, g0p, add in zip(backbone_params, g0, step_aux):
            g_desired = (g0p if g0p is not None else torch.zeros_like(p)) + add
            routed = routed + torch.sum(p * g_desired.detach())
        for p, g in zip(self._aux_params, auxhead_g):
            routed = routed + torch.sum(p * g.detach())
        out["routed"] = routed
        out["ce"] = ce_t.detach()

        if self._audit_applied:
            self._audit = {
                "g0": [g.detach().clone() for g in g0],
                "add": [a.detach().clone() for a in step_aux],
                "aux": [g.detach().clone() for g in auxhead_g],
                "scale": float(self.supervision_scale),
                "z": {s: float(sampled[s]) for s in self.site_names},
                "tilde": {s: tilde[s].detach().clone() for s in self.site_names},
            }

        step = int(getattr(state, "batch_index", 0))
        if self._should_estimate_utility(step):
            self._estimate_utilities(ce_t.device, g0_flat, g_l_raw, tilde,
                                     sampled, step)
        return out

    # ------------------------------------------------------------- utility
    def _meta_batch(self, device):
        """Fresh deterministic batch from the held-out validation split."""
        if self._val_loader is None:
            self._val_loader = build_dataloader(self.cfg, "val", shuffle=False,
                                                return_indices=True, num_workers=0)
            self._val_iter = iter(self._val_loader)
        try:
            batch = next(self._val_iter)
        except StopIteration:
            self._val_iter = iter(self._val_loader)
            batch = next(self._val_iter)
        ids = batch[2] if len(batch) > 2 else None
        meta_ids = ([int(i) for i in ids.detach().cpu().tolist()]
                    if ids is not None else None)
        return batch[0].to(device), batch[1].to(device), meta_ids

    def _estimate_utilities(self, device, g0_flat, g_l_raw, tilde, sampled,
                            step: int):
        xm, ym, meta_ids = self._meta_batch(device)
        params = [p for _, p in self._utility_params]
        was_training = self.training
        self.eval()
        try:
            t0 = time.perf_counter()
            bo = self.backbone(xm)
            logits = bo.logits[:, : self.num_classes]
            surrogate = soft_aurc_surrogate(logits, ym, tau=self.hard_concrete_tau)
            g_J = torch.autograd.grad(surrogate, params, retain_graph=True,
                                      allow_unused=True, materialize_grads=True)
            self.utility_ms = (time.perf_counter() - t0) * 1000.0
        finally:
            self.train(was_training)

        # bilevel discipline lock: the meta batch must never overlap the train
        # batch whose gradients were used on the training side.
        if self._last_train_ids is not None and meta_ids is not None:
            overlap = set(self._last_train_ids) & set(meta_ids)
            if overlap:
                raise RuntimeError(
                    f"sage_ds_v2 bilevel violation: train and meta batches share "
                    f"official-fold indices {sorted(overlap)[:5]}... (step {step})")

        gJ_flat = _cat([_flatten(g) for g in g_J]).detach()
        gJ_n = float(torch.norm(gJ_flat).item())
        g0_n = float(torch.norm(g0_flat).item())

        cos_vals = []
        row = {"step": int(step), "gJ_norm": gJ_n, "g0_norm": g0_n}
        for s in self.site_names:
            u = bilevel_utilities(gJ_flat, g_l_raw[s], tilde[s], eps=self.eps)
            u_cos = u["cos"]
            cos_vals.append(u_cos)
            row[f"raw_unprojected_utility_{s}"] = u["unprojected"]
            row[f"raw_utility_{s}"] = u["raw"]
            row[f"cos_utility_{s}"] = u_cos
            row[f"gl_norm_{s}"] = u["gl_norm"]
            row[f"tilde_gl_norm_{s}"] = u["tilde_norm"]
            row[f"support_frac_{s}"] = support_fraction(tilde[s])
            row[f"align_before_{s}"] = float(torch.dot(g_l_raw[s], g0_flat).item())
            row[f"align_after_{s}"] = float(torch.dot(tilde[s], g0_flat).item())
            row[f"gatep_{s}"] = float(self.controller.gate_prob(s).detach().cpu())
            row[f"sampled_gate_{s}"] = float(sampled[s].detach().cpu())
            row[f"eff_aux_w_{s}"] = float(sampled[s].detach().cpu()) * self.supervision_scale
        row["meta_bs"] = int(xm.shape[0]) if xm is not None else 0
        row["train_ids"] = self._last_train_ids
        row["meta_ids"] = meta_ids

        # gate control consumes the cosine utility; U_raw is retained in logs
        self.controller.update_utility_ema(cos_vals)
        self.controller.step_from_utility(self.controller_lr, self.sparsity_cost,
                                          self.strength_cap)
        for s in self.site_names:
            row[f"uema_{s}"] = self.controller.utility_ema_dict()[s]
        self._write_utility_log(row)

    # ------------------------------------------------------------------ logs
    def _write_utility_log(self, row):
        try:
            run_dir = os.path.join(self.cfg["results_root"], self.cfg["run_name"])
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "sage_ds_v2_utility.jsonl"), "a") as f:
                f.write(json.dumps(row, default=str) + "\n")
        except Exception:
            pass

    def _write_log(self):
        try:
            run_dir = os.path.join(self.cfg["results_root"], self.cfg["run_name"])
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "sage_ds_v2.jsonl"), "a") as f:
                for r in self._log:
                    f.write(json.dumps(r, default=str) + "\n")
            self._log.clear()
        except Exception:
            pass