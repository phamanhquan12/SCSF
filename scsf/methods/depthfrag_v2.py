"""DepthFrag-v2: stable whitened depth fragility (EMA teacher targets).

Method structure (see docs/DEPTHFRAG_V2_PROTOCOL.md, preregistered)
-------------------------------------------------------------------
* A true EMA teacher ``theta_T <- nu*theta_T + (1-nu)*theta_S`` (nu = 0.999,
  locked) is maintained from the first optimizer step. The teacher supplies
  every geometry target; the student never distills its own decision surface.
* Per site ``l`` a diagonal feature-covariance EMA ``Sigma_l`` is computed on
  **teacher** features (nu_v = 0.01, locked) and the fragility radius is the
  **whitened** margin-grad ratio
      rho_l = relu(m_T) / (sqrt(g_l^T Sigma_l g_l) + eps)
  with ``m_T`` the true-class teacher margin and ``g_l`` the first-order
  gradient of ``m_T`` wrt the pooled teacher representation at site ``l``.
* ``rho_l`` is clipped to training-only EMA quantiles ``[rho_p1, rho_p99]``
  and transformed ``target_l = sign(rho)*log1p(|rho|)``; the clip bounds the
  value even for tiny/normalized gradients (never infinite; degenerate sites
  are flagged and log a constant zero target).
* Epochs 0-24 inclusive: the backbone receives CE gradients only; probe/head
  inputs are detached from the backbone path (targets always come from the
  EMA teacher). Epoch 25 onward the probe/head gradients flow end-to-end.
  Nothing (optimizer / scheduler / EMA / model) resets at the boundary.
* Final confidence is the label-free ``head(final_embedding)`` (plus terminal
  logits); probes and their autograd machinery never enter the prediction
  path. True-label geometry on val/test is an **oracle diagnostic** channel
  only and never feeds the primary metrics/score.

Nice properties preserved from v1: correct per-site logging (each site logs
its own prediction and its own target), the target forward keeps BN evaluation
role (per-example independence), and a small training-only input is used for
clip statistics so inference stays label-free.
"""

from __future__ import annotations

import copy
import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones import MultiHook
from ..depthfrag.geometry import (
    aggregate_profile,
    pool_tap,
    target_transform,
    true_class_margin,
)
from .base import Method, MethodPrediction
from .depthfrag import FragHead, FragProbe
from .scores import compute_scores

__all__ = [
    "DepthFragV2Method",
    "whitened_rho",
    "teacher_whitened_targets",
    "iterative_boundary_distance",
    "compute_boundary_direction",
    "run_oracle_diagnostic",
]


def whitened_rho(margin: torch.Tensor, grad: torch.Tensor,
                 sigma: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Whitened fragility radius ``relu(m) / (sqrt(g^T Sigma g) + eps)``.

    ``margin``: (B,) teacher true-class margin.  ``grad``: (B, D) first-order
    gradient of the margin wrt the pooled teacher feature.  ``sigma``: (D,)
    per-dimension teacher feature variance (EMA).  Channel rescaling
    invariance: ``g -> a*g`` with ``Sigma -> a^2 Sigma`` leaves ``rho``
    unchanged (4.2 of the protocol).
    """
    denom = torch.sqrt((grad * grad * sigma).sum(dim=-1)) + eps
    return torch.relu(margin) / denom


def _pool_feature(feat: torch.Tensor, token: str) -> torch.Tensor:
    feat = pool_tap(feat, token)
    return feat


def pooled_gradient(g_raw: torch.Tensor, raw: torch.Tensor,
                    token: str = "cls") -> torch.Tensor:
    """Convert a gradient wrt the raw tap into ``d (.)/d pooled_feature``.

    Backbone blocks mutate tap outputs **in place** (version counter advances),
    so gradients read out of a *pooled* tensor built afterwards are silently
    dropped by autograd.  Computing the gradient wrt the raw tap is reliable
    (the graph node stays live); this helper then converts it to pooled
    coordinates analytically:

    * 4-D feature (CNN):  ``h_c = mean_px(raw_c)`` so
      ``d/dh_c = sum_px d/raw_{c,px}``.
    * 3-D token (ViT):    ``h = raw[:, 0]`` (``cls``) / ``mean`` -> identity /
      ``sum`` over tokens.
    * 2-D:                identity.
    """
    if raw.dim() == 4:
        return g_raw.reshape(g_raw.shape[0], g_raw.shape[1], -1).sum(dim=-1)
    if raw.dim() == 3:
        if token == "cls":
            return g_raw[:, 0]
        if token == "mean":
            return g_raw.sum(dim=1)
        raise ValueError(f"unknown ViT token pooling {token!r}")
    return g_raw


def pooled_vjps(logits: torch.Tensor, raw: torch.Tensor,
                token: str = "cls") -> torch.Tensor:
    """Pooled-coordinate Jacobian rows ``d z_c / d h``, stacked ``(B, C, D)``.

    One backward per logit against the raw tap, converted with
    :func:`pooled_gradient`.  Used by the oracle boundary walk.
    """
    C = logits.shape[1]
    rows = []
    for c in range(C):
        g = torch.autograd.grad(
            logits[:, c].sum(), raw, retain_graph=True,
            allow_unused=True, materialize_grads=True)[0]
        rows.append(pooled_gradient(g, raw, token))
    return torch.stack(rows, dim=1)


def teacher_whitened_targets(margin: torch.Tensor,
                             grads: Dict[str, torch.Tensor],
                             sigmas: Dict[str, torch.Tensor],
                             rho_p1: Dict[str, torch.Tensor],
                             rho_p99: Dict[str, torch.Tensor],
                             site_names: Sequence[str],
                             eps: float = 1e-12) -> Tuple[Dict[str, torch.Tensor], ...]:
    """Per-site clipped signed-log1p ``target_l`` for the given geometry.

    ``rho_p1/rho_p99`` are training-only EMA quantiles (persistent buffers).
    Returns ``(target, rho_clip, denom)`` dicts keyed by site and a flag dict
    ``degenerate`` for sites whose covariance has never seen variation
    (``Sigma == 0`` -> constant zero target, finite, logged).
    """
    target: Dict[str, torch.Tensor] = {}
    rho_clip: Dict[str, torch.Tensor] = {}
    denom: Dict[str, torch.Tensor] = {}
    degenerate: Dict[str, bool] = {}
    for s in site_names:
        rho = whitened_rho(margin, grads[s], sigmas[s], eps)
        p1 = float(rho_p1[s].item())
        p99 = float(rho_p99[s].item())
        if p99 < p1:
            p1, p99 = p99, p1
        rc = torch.clamp(rho, min=p1, max=p99)
        denom[s] = torch.sqrt((grads[s] * grads[s] * sigmas[s]).sum(dim=-1)) + eps
        rho_clip[s] = rc
        target[s] = torch.sign(rc) * torch.log1p(rc.abs())
        degenerate[s] = bool(float(sigmas[s].sum().item()) == 0.0)
    return target, rho_clip, denom, degenerate


def compute_boundary_direction(teacher_logits: torch.Tensor,
                               raw_feature: torch.Tensor,
                               token: str,
                               u_l: torch.Tensor) -> torch.Tensor:
    """Linearized logit direction ``J_l u_l`` of moving the pooled feature
    along ``u_l``.

    ``J_l`` is the Jacobian of the teacher logits wrt the pooled feature at
    site ``l`` (all ``C`` logits), computed through the raw tap (in-place
    safe) and converted to pooled coordinates.  Returns ``(B, C)``: the
    per-logit change per unit displacement in the normalized gradient
    direction.
    """
    vjps = pooled_vjps(teacher_logits, raw_feature, token)  # (B, C, D)
    return (vjps * u_l.unsqueeze(1)).sum(dim=-1)


def iterative_boundary_distance(z0: torch.Tensor, dz: torch.Tensor,
                                pred0: torch.Tensor, step: float = 0.25,
                                max_steps: int = 256) -> torch.Tensor:
    """Iterative displacement bound along the linearized logit direction.

    Walks ``z(s) = z0 + s*dz`` in small steps until the predicted class
    changes, then bisects to the flip boundary.  Returns the displacement
    ``s*`` (a lower bound on the true nonlinear boundary distance).  Examples
    that never flip within ``max_steps`` are reported as ``max_steps*step``
    (indicative, not a certified bound).
    """
    B = z0.shape[0]
    changed = pred0.new_zeros(B, dtype=torch.bool)
    s_low = z0.new_zeros(B)
    s_hi = z0.new_zeros(B)
    z = z0.clone()
    s = 0.0
    for _ in range(max_steps):
        s += step
        z = z + step * dz
        flipped = (z.argmax(dim=1) != pred0)
        newly = flipped & ~changed
        if newly.any():
            s_hi[newly] = s
            s_low[newly] = s - step
            changed |= newly
        if changed.all():
            break
    s_hi[~changed] = max_steps * step
    for _ in range(24):  # bisection to the boundary for flipped examples
        if not changed.any():
            break
        mid = (s_low + s_hi) / 2.0
        z_mid = z0 + dz * mid.unsqueeze(1)
        flipped = z_mid.argmax(dim=1) != pred0
        move = flipped & changed
        if move.any():
            s_hi[move] = mid[move]
        else:
            s_low[~flipped & changed] = mid[~flipped & changed]
    return s_hi


class DepthFragV2Method(Method):
    """Distill stable whitened depth fragility into a terminal score."""

    method_name = "depthfrag_v2"

    def default_score(self) -> str:
        return "depthfrag"

    def default_scores(self):
        return ("msp", "entropy", "energy", "logit_margin", "depthfrag")

    def __init__(self, train_cfg: dict):
        super().__init__(train_cfg)
        m = train_cfg["method"]
        self.eps = float(m.get("eps", 1e-12))
        self.token = str(m.get("token", "cls"))
        self.agg = str(m.get("agg", "soft_min"))
        self.agg_tau = float(m.get("agg_tau", 2.0))
        self.probe_hidden = int(m.get("probe_hidden", 32))
        self.huber_delta = float(m.get("huber_delta", 1.0))
        self.use_probes = bool(m.get("use_probes", True))
        self.use_head = bool(m.get("use_head", True))
        self.nu = float(m.get("nu", 0.999))
        self.nu_v = float(m.get("nu_v", 0.01))
        self.warmup_epochs = int(m.get("warmup_epochs", 25))
        self.cvar_frac = float(m.get("cvar_frac", 0.25))

        role_pr = m.get("probe_sites", "all")
        if role_pr == "all":
            self.site_names = list(self.backbone.taps.keys())
        else:
            role_list = list(role_pr)
            self.site_names = [self.backbone.roles[r] for r in role_list]
        self.terminal_site = self.backbone.roles.get(
            m.get("terminal_site", "top_l1"),
            m.get("terminal_site", "top_l1"))

        dims = self._probe_site_dims()
        self.probes = nn.ModuleDict(
            {s: FragProbe(dims[s], self.probe_hidden) for s in self.site_names}
        )
        self._probe_params = [
            p for pr in self.probes.values() for p in pr.parameters()
        ]
        self.head = FragHead(self.backbone.final_dim, self.probe_hidden)
        self._head_params = list(self.head.parameters())

        # EMA teacher: deep copy of the student backbone (same init), kept as a
        # submodule so it checkpoints/restores bit-exact; never trainable.
        self.teacher = copy.deepcopy(self.backbone)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # persistent per-site buffers (exact resume / bit-exact restore).
        for s in self.site_names:
            self.register_buffer(f"_v2_sigma_{s.replace('.', '_')}",
                                 torch.zeros(dims[s]))
            self.register_buffer(f"_v2_rho_p1_{s.replace('.', '_')}",
                                 torch.zeros(1))
            self.register_buffer(f"_v2_rho_p99_{s.replace('.', '_')}",
                                 torch.ones(1) * 1.0)
            self.register_buffer(f"_v2_steps_{s.replace('.', '_')}",
                                 torch.zeros(1))
        self._step_n = 0
        self._step_head = 0.0
        self._step_head_pred = 0.0
        self._step_agg_t = 0.0
        self._step_agg_p = 0.0
        self._step_term_margin = 0.0
        self._step_denom: Dict[str, float] = {s: 0.0 for s in self.site_names}
        self._step_rho: Dict[str, float] = {s: 0.0 for s in self.site_names}
        self._step_q_pred: Dict[str, float] = {s: 0.0 for s in self.site_names}
        self._step_q_target: Dict[str, float] = {s: 0.0 for s in self.site_names}
        self._step_probe: Dict[str, float] = {s: 0.0 for s in self.site_names}
        self._target_ms = 0.0
        self._log: List[dict] = []
        self._sigma_cache: Optional[Dict[str, torch.Tensor]] = None

    # ------------------------------------------------------------------ init
    def _probe_site_dims(self) -> Dict[str, int]:
        with torch.no_grad(), self.probe_mode():
            bo = self.backbone(
                torch.zeros(1, self.backbone.channels,
                            self.backbone.input_size, self.backbone.input_size)
            )
        return {s: int(_pool_feature(bo.features[s], self.token).shape[-1])
                for s in self.site_names}

    def _sigma(self, s: str) -> torch.Tensor:
        return self.get_buffer(f"_v2_sigma_{s.replace('.', '_')}")

    def _rho_p1(self, s: str) -> torch.Tensor:
        return self.get_buffer(f"_v2_rho_p1_{s.replace('.', '_')}")

    def _rho_p99(self, s: str) -> torch.Tensor:
        return self.get_buffer(f"_v2_rho_p99_{s.replace('.', '_')}")

    def _steps(self, s: str) -> torch.Tensor:
        return self.get_buffer(f"_v2_steps_{s.replace('.', '_')}")

    # ------------------------------------------------------------- inference
    def predict_batch(self, x):
        bo = self.backbone(x)
        logits = bo.logits[:, : self.num_classes]
        scores = compute_scores(logits, self.default_scores())
        if self.use_head:
            scores["depthfrag"] = self.head(bo.final_embedding)
        else:
            scores["depthfrag"] = compute_scores(
                logits, ("logit_margin",))["logit_margin"]
        conf = self._pick(scores)
        return MethodPrediction(logits, logits.argmax(dim=1), conf, scores)

    def _pick(self, scores):
        if self.score in scores:
            return scores[self.score]
        if "depthfrag" in scores:
            return scores["depthfrag"]
        return scores["msp"]

    def stripped_predict_batch(self, x):
        return self.predict_batch(x)

    def to_deployment(self) -> nn.Module:
        return self

    def inference_modules(self):
        mods = [self.backbone]
        if self.use_head:
            mods.append(self.head)
        return mods

    def optimizer_specs(self):
        t = self.cfg["train"]
        params = list(self.backbone.parameters()) + self._probe_params
        if self.use_head:
            params = params + self._head_params
        params = [p for p in params if p.requires_grad]
        return [{
            "params": params,
            "kind": t.get("optimizer", "sgd"),
            "lr": float(t["lr"]),
            "momentum": float(t.get("momentum", 0.9)),
            "weight_decay": float(t.get("weight_decay", 5e-4)),
        }]

    # ---------------------------------------------------------- EMA teacher
    def _ema_update_teacher(self):
        """``theta_T <- nu*theta_T + (1-nu)*theta_S`` incl. BN buffers.

        Maintained from the first optimizer step: called at the start of every
        ``train_loss`` after the first step, so at step ``t`` the teacher is
        the EMA of the student state up to step ``t-1``.  Step-0 teacher
        coincides with the freshly-synchronized student initial state.
        """
        with torch.no_grad():
            for (tn, tp), (sn, sp) in zip(
                    self.teacher.named_parameters(),
                    self.backbone.named_parameters()):
                tp.mul_(self.nu).add_(sp, alpha=1.0 - self.nu)
            for (tn, tb), (sn, sb) in zip(
                    self.teacher.named_buffers(),
                    self.backbone.named_buffers()):
                if not tb.is_floating_point():
                    continue
                tb.mul_(self.nu).add_(sb, alpha=1.0 - self.nu)

    # -------------------------------------------------------------- training
    def train_loss(self, batch, state):
        device = next(self.backbone.parameters()).device
        x = batch[0].to(device)
        y = batch[1].to(device)

        step = int(getattr(state, "batch_index", 0))
        epoch = int(getattr(state, "epoch", 0))
        in_warmup = epoch < self.warmup_epochs

        if step > 0 or epoch > 0:
            self._ema_update_teacher()

        t0 = time.perf_counter()
        targets, rho_clip, denom, degenerate, margins = \
            self._targets_eval_forward(x, y, update_stats=not in_warmup)
        self._target_ms += (time.perf_counter() - t0) * 1000.0

        bo = self.backbone(x)
        logits = bo.logits[:, : self.num_classes]
        ce = F.cross_entropy(logits, y)
        out = {"ce": ce}

        true_profile = torch.stack([targets[s] for s in self.site_names], dim=1)
        agg_target = aggregate_profile(true_profile, self.agg,
                                       self.agg_tau, self.cvar_frac)

        if self.use_probes:
            pl = torch.zeros((), device=device)
            pred_columns = []
            qs: Dict[str, torch.Tensor] = {}
            for s in self.site_names:
                feat = _pool_feature(bo.features[s], self.token)
                if in_warmup:
                    feat = feat.detach()
                q = self.probes[s](feat)
                qs[s] = q
                pred_columns.append(q)
                pl = pl + F.huber_loss(q, targets[s].detach(),
                                       delta=self.huber_delta)
            pred_profile = torch.stack(pred_columns, dim=1)
            with torch.no_grad():
                agg_pred = aggregate_profile(pred_profile, self.agg,
                                             self.agg_tau, self.cvar_frac)
                self._step_agg_t = getattr(self, "_step_agg_t", 0.0) + \
                    float(agg_target.detach().mean())
                self._step_agg_p = getattr(self, "_step_agg_p", 0.0) + \
                    float(agg_pred.detach().mean())
                for s in self.site_names:
                    self._step_probe[s] += float(
                        F.huber_loss(qs[s].detach(), targets[s].detach(),
                                     delta=self.huber_delta))
                    self._step_q_pred[s] += float(qs[s].detach().mean())
                    self._step_q_target[s] += float(targets[s].detach().mean())
            out["depthfrag_probe"] = pl / len(self.site_names)

        if self.use_head:
            head_input = bo.final_embedding
            if in_warmup:
                head_input = head_input.detach()
            head_score = self.head(head_input)
            out["depthfrag_head"] = F.huber_loss(
                head_score, agg_target.detach(), delta=self.huber_delta)
            self._step_head += 1.0

        # decomposition (mandated section 8), accumulated under no_grad
        with torch.no_grad():
            self._step_n += 1
            self._step_term_margin += float(margins.detach().mean())
            for s in self.site_names:
                self._step_denom[s] += float(denom[s].detach().mean())
                self._step_rho[s] += float(rho_clip[s].detach().mean())
            if self.use_head:
                self._step_head_pred += float(head_score.detach().mean())
        return out

    def _targets_eval_forward(self, x, y, update_stats: bool):
        """Detached teacher targets with BatchNorm in eval (teacher) role.

        The teacher runs eval-mode (running stats) for per-example independence
        of the geometry; ``Sigma_l`` and the quantile EMA are updated from
        teacher pooled features (training-only when ``update_stats``).
        """
        store: Dict[str, torch.Tensor] = {}
        hooks = MultiHook(self.teacher.taps, store)
        was_teacher_eval = not self.teacher.training
        xg = x.detach().clone().requires_grad_(True)
        self.teacher.eval()
        try:
            with torch.enable_grad():
                bo = self.teacher(xg)
                m = true_class_margin(bo.logits[:, : self.num_classes], y)
                pooled = {}
                grads = {}
                for i, s in enumerate(self.site_names):
                    raw = store[s]
                    pooled[s] = _pool_feature(raw, self.token)
                    g_raw = torch.autograd.grad(
                        m.sum(), raw, retain_graph=(
                            i < len(self.site_names) - 1),
                        create_graph=False, allow_unused=True,
                        materialize_grads=True)[0]
                    grads[s] = pooled_gradient(g_raw, raw, self.token)
                    del g_raw
                self._update_stats(pooled, m, grads, update_stats)
                sigmas = {s: self._sigma(s).to(m.dtype) for s in self.site_names}
                target, rho_clip, denom, degenerate = teacher_whitened_targets(
                    m, grads, sigmas,
                    {s: self._rho_p1(s).to(m.dtype) for s in self.site_names},
                    {s: self._rho_p99(s).to(m.dtype) for s in self.site_names},
                    self.site_names, self.eps)
        finally:
            hooks.remove()
            self.teacher.train(not was_teacher_eval)
        out_target = {s: target[s].detach() for s in self.site_names}
        out_rho = {s: rho_clip[s].detach() for s in self.site_names}
        out_denom = {s: denom[s].detach() for s in self.site_names}
        return out_target, out_rho, out_denom, degenerate, m.detach()

    def _update_stats(self, pooled, m, grads, update_stats: bool):
        with torch.no_grad():
            for s in self.site_names:
                sigma = self._sigma(s)
                cnt = self._steps(s)
                if update_stats:
                    h2 = (pooled[s].detach() ** 2).mean(dim=0)
                    sigma.mul_(1.0 - self.nu_v).add_(h2, alpha=self.nu_v)
                    cnt.add_(1.0)
                if float(cnt.item()) <= 2:
                    continue
                rho = whitened_rho(m.detach(), grads[s].detach(), sigma.detach(),
                                   self.eps)
                if rho.numel() == 0:
                    continue
                rho = rho.detach()
                p1 = torch.quantile(rho, q=0.01)
                p99 = torch.quantile(rho, q=0.99)
                b1 = self._rho_p1(s)
                b9 = self._rho_p99(s)
                b1.mul_(1.0 - self.nu_v).add_(p1, alpha=self.nu_v)
                b9.mul_(1.0 - self.nu_v).add_(p99, alpha=self.nu_v)

    # ------------------------------------------------------------------ logs
    def on_epoch_end(self, epoch: int, val_metrics: dict):
        n = max(self._step_n, 1)
        in_warmup = epoch < self.warmup_epochs
        row = {
            "epoch": int(epoch),
            "target_ms": float(self._target_ms),
            "head_steps": int(self._step_head),
            "warmup": in_warmup,
            "aux_grads_to_backbone": not in_warmup,
            "terminal_margin": float(getattr(self, "_step_term_margin", 0.0) / n),
            "agg_target": float(getattr(self, "_step_agg_t", 0.0) / n),
            "agg_pred": float(getattr(self, "_step_agg_p", 0.0) / n),
        }
        for s in self.site_names:
            row[f"probe_huber_{s}"] = float(self._step_probe[s] / n)
            row[f"grad_denom_{s}"] = float(self._step_denom[s] / n)
            row[f"margin_grad_ratio_{s}"] = float(self._step_rho[s] / n)
            row[f"q_pred_{s}"] = float(self._step_q_pred[s] / n)
            row[f"q_target_{s}"] = float(self._step_q_target[s] / n)
            row[f"sigma_norm_{s}"] = float(self._sigma(s).norm().item())
            row[f"rho_p1_{s}"] = float(self._rho_p1(s).item())
            row[f"rho_p99_{s}"] = float(self._rho_p99(s).item())
        self._log.append(row)
        self._write_log()
        self._reset_step_stats()

    def _reset_step_stats(self):
        self._step_n = 0
        self._step_head = 0.0
        self._step_head_pred = 0.0
        self._step_agg_t = 0.0
        self._step_agg_p = 0.0
        self._step_term_margin = 0.0
        self._step_denom = {s: 0.0 for s in self.site_names}
        self._step_rho = {s: 0.0 for s in self.site_names}
        self._step_q_pred = {s: 0.0 for s in self.site_names}
        self._step_q_target = {s: 0.0 for s in self.site_names}
        self._step_probe = {s: 0.0 for s in self.site_names}
        self._target_ms = 0.0

    def _write_log(self):
        try:
            run_dir = os.path.join(self.cfg["results_root"], self.cfg["run_name"])
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "depthfrag_v2.jsonl"), "a") as f:
                for r in self._log:
                    f.write(json.dumps(r, default=str) + "\n")
            self._log.clear()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# fixed oracle diagnostic (protocol section 9) — runs once after training, on
# a validation subset only; results are logged to a separate
# ``depthfrag_v2_oracle_diag.json`` channel, never to the primary registry.
# ---------------------------------------------------------------------------
def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() < 3:
        return float("nan")
    a = a.detach().cpu()
    b = b.detach().cpu()
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    if float(ra.std()) == 0.0 or float(rb.std()) == 0.0:
        return float("nan")
    return float(torch.corrcoef(torch.stack([ra, rb]))[0, 1].item())


def run_oracle_diagnostic(run_dir: str, checkpoint: str = "selected",
                          device: Optional[str] = None,
                          per_class_examples: int = 50,
                          step: float = 0.25, max_steps: int = 256) -> dict:
    """Post-training fixed diagnostic: analytic vs iterative boundary distance.

    Subsets the validation fold to (up to) ``per_class_examples`` examples per
    class, then per site compares the analytic whitened ``rho_l`` against an
    iterative boundary-distance walk along the whitened gradient direction.
    Writes ``<run_dir>/depthfrag_v2_oracle_diag.json``.
    """
    from ..data.cifar import build_dataset
    from ..engine.checkpoint import CheckpointManager
    from .factory import build_method

    with open(os.path.join(run_dir, "cfg.json")) as f:
        cfg = json.load(f)
    dev = torch.device(device or cfg["train"].get("device", "cpu"))
    cfg["train"]["device"] = str(dev)
    manager = CheckpointManager(run_dir)
    payload = manager.load(checkpoint, map_location=dev)
    method = build_method(cfg["method_name"], cfg)
    method.load_state_dict(payload["model_state"])
    method.to(dev)
    method.eval()
    method.teacher.eval()

    ds = build_dataset(cfg, "val", return_indices=True)
    subset = ds.base
    tv_base = subset.base
    targets = getattr(tv_base, "targets", None)
    if targets is None:
        raise RuntimeError("val dataset has no targets; oracle needs labels")
    counts: Dict[int, int] = {}
    picked = []
    for local, gidx in enumerate(subset.indices):
        c = int(targets[gidx])
        if counts.get(c, 0) < per_class_examples:
            picked.append((local, gidx))
            counts[c] = counts.get(c, 0) + 1

    analytic = {s: [] for s in method.site_names}
    iterative = {s: [] for s in method.site_names}
    degenerate: Dict[str, bool] = {s: False for s in method.site_names}
    batch = 64
    with torch.no_grad():
        for start in range(0, len(picked), batch):
            locals_ = [p[0] for p in picked[start:start + batch]]
            x = torch.stack([ds[local][0] for local in locals_]).to(dev)
            x.requires_grad_(True)
            yt = torch.tensor([int(targets[subset.indices[local]])
                               for local in locals_], device=dev)
            store: Dict[str, torch.Tensor] = {}
            hooks = MultiHook(method.teacher.taps, store)
            try:
                with torch.enable_grad():
                    bo = method.teacher(x)
                    m = true_class_margin(bo.logits[:, : method.num_classes], yt)
                    grads = {}
                    pooled = {}
                    for i, s in enumerate(method.site_names):
                        raw = store[s]
                        pooled[s] = _pool_feature(raw, method.token)
                        g_raw = torch.autograd.grad(
                            m.sum(), raw, retain_graph=True,
                            create_graph=False, allow_unused=True,
                            materialize_grads=True)[0]
                        grads[s] = pooled_gradient(g_raw, raw, method.token)
                    for s in method.site_names:
                        sig = method._sigma(s).to(m.dtype)
                        denom = torch.sqrt(
                            (grads[s] * grads[s] * sig).sum(dim=-1)) + method.eps
                        rho = torch.relu(m) / denom
                        u_w = grads[s] / denom.unsqueeze(1)
                        dz = compute_boundary_direction(
                            bo.logits[:, : method.num_classes], store[s],
                            method.token, u_w)
                        dist = iterative_boundary_distance(
                            bo.logits[:, : method.num_classes], dz,
                            bo.logits[:, : method.num_classes].argmax(dim=1),
                            step=step, max_steps=max_steps)
                        analytic[s].append(rho.detach().cpu())
                        iterative[s].append(dist.detach().cpu())
                        if float(sig.sum().item()) == 0.0:
                            degenerate[s] = True
            finally:
                hooks.remove()

    result = {"run_dir": run_dir, "checkpoint": checkpoint,
              "per_class_examples": per_class_examples, "step": step,
              "max_steps": max_steps, "sites": {}}
    for s in method.site_names:
        a = torch.cat(analytic[s])
        b = torch.cat(iterative[s])
        if degenerate[s] or float(a.std()) == 0.0:
            corr = float("nan")
        else:
            corr = _spearman(a, b)
        result["sites"][s] = {
            "spearman": corr, "n": int(a.numel()),
            "degenerate": degenerate[s],
            "analytic_rho_mean": float(a.mean().item()),
            "iterative_dist_mean": float(b.mean().item()),
        }
    import numpy as np
    result["full_vectors"] = {
        s: {"analytic_rho": np.asarray(torch.cat(analytic[s])).tolist(),
            "iterative_dist": np.asarray(torch.cat(iterative[s])).tolist()}
        for s in method.site_names
    }
    out_path = os.path.join(run_dir, "depthfrag_v2_oracle_diag.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True, default=str)
    return result


def _oracle_cli(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    run_dir = argv[0] if argv else "."
    kw = {}
    for a in argv[1:]:
        k, _, v = a.partition("=")
        if k == "device":
            kw["device"] = v
        elif k == "per_class_examples":
            kw["per_class_examples"] = int(v)
        elif k == "checkpoint":
            kw["checkpoint"] = v
    return run_oracle_diagnostic(run_dir, **kw)


if __name__ == "__main__":
    _oracle_cli()