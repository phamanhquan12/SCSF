"""SAGE-DS: learn a selective supervision topology via out-of-sample utility.

SAGE-DS = Supervision Auxiliaries Gated by Estimated Differentiable-Selective
utility.

High-level idea
---------------
Standard deep supervision attaches an auxiliary classifier at a site and adds
its cross-entropy to the backbone loss, assuming every (or a fixed set of)
site helps. SAGE-DS instead:

1. attaches a lightweight auxiliary classifier to **every candidate site** the
   backbone registry exposes (``backbone.taps``);
2. every ``utility_interval`` steps, on a fresh held-out **validation** batch,
   estimates a first-order *selective utility* for each site::

       theta' = theta - eta * g_l
       J_sel(theta') ~= J_sel(theta) - eta * <g_sel, g_l>
       U_l = <g_sel, g_l>

   where ``g_l`` is the auxiliary-supervision gradient on the backbone and
   ``g_sel`` is the gradient of a differentiable selective-ranking/AURC
   surrogate (:mod:`scsf.metrics.surrogate`). ``U_l > 0`` predicts that pulling
   the backbone along ``g_l`` improves held-out selective competence.
3. a sparse **hard-concrete** gate per site is modulated by an EMA of that
   utility (probability grows with positive held-out utility, and pays a
   configurable sparsity cost), producing a **sparse, time-varying supervision
   topology**;
4. an optional classification-safety projection removes the component of the
   combined auxiliary gradient that conflicts with the running CE gradient
   (``g0_ema``), so auxiliary supervision is only applied when it does not
   fight the main classifier.

The auxiliary heads are **training-only instruments**, stripped at inference;
the default inference confidence is plain MSP from the terminal logits (an
optional tiny terminal risk head is an ablation, not the default).

This is an independent method in the common harness; it never special-cases a
particular backbone (ResNet/VGG/...); sites come exclusively from
``backbone.taps`` and the semantic ``roles`` map.
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

from ..backbones import adaptive_flatten
from ..data import build_dataloader
from ..metrics.surrogate import soft_aurc_surrogate
from .base import Method, MethodPrediction
from .scores import compute_scores

__all__ = [
    "SageDSMethod",
    "AuxHead",
    "HardConcreteGate",
    "Controller",
    "project_aux",
    "params_reached_by_aux",
    "selective_utility",
]

TOPO_NATIVE = {"fixed_late", "all_equal", "learned_dense"}
TOPO_CONTROLLER = {"sparse_utility", "sparse_utility_safe"}
ALL_TOPO = TOPO_NATIVE | TOPO_CONTROLLER


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------
def _pool_tap(feature: torch.Tensor, token: str = "cls") -> torch.Tensor:
    """Reduce a tap feature to a fixed (B, D) vector for an aux head.

    * 4-D CNN maps -> global-average-pool then flatten;
    * 3-D token sequences (ViT) -> CLS token (or token mean) selected by config;
    * 2-D vectors -> returned unchanged.
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


class AuxHead(nn.Module):
    """Lightweight per-site auxiliary classifier (training-only instrument)."""

    def __init__(self, in_features: int, num_classes: int, hidden: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.proj = nn.Sequential(nn.Linear(in_features, hidden), nn.ReLU(inplace=True))
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.proj(self.norm(x)))


class HardConcreteGate(nn.Module):
    """A single hard-concrete (differentiable L0) gate in [0, 1].

    ``log_alpha`` is the unconstrained controller parameter; the expected gate
    probability is ``p = sigmoid(log_alpha)``. ``sample()`` draws a stretched
    binary-concrete value in [0, 1] so the supervision weight applied on a
    training step is a realized sparse gate; ``l0_norm()`` is the differentiable
    ``P(gate > 0)`` surrogate cost used for the sparsity accounting/log.
    """

    def __init__(self, tau: float = 0.3, stretch: Tuple[float, float] = (-0.1, 1.1)):
        super().__init__()
        if tau <= 0:
            raise ValueError(f"hard-concrete tau must be > 0, got {tau}")
        self.tau = float(tau)
        self.lo, self.hi = float(stretch[0]), float(stretch[1])
        self.log_alpha = nn.Parameter(torch.zeros(()))

    def expected(self) -> torch.Tensor:
        """Differentiable expected gate probability in (0, 1)."""
        return torch.sigmoid(self.log_alpha)

    def log_prob_gate_open(self) -> torch.Tensor:
        return F.logsigmoid(self.log_alpha - self.tau * math.log(-self.lo / self.hi))

    def l0_norm(self) -> torch.Tensor:
        """Differentiable L0 surrogate ``P(gate open)`` in (0, 1)."""
        return torch.exp(self.log_prob_gate_open())

    def sample(self, rs: Optional[torch.Generator] = None) -> torch.Tensor:
        """Reparameterized hard-concrete sample in [0, 1]."""
        eps = 1e-6
        u = torch.rand(
            self.log_alpha.shape,
            device=self.log_alpha.device,
            dtype=self.log_alpha.dtype,
            generator=rs,
        ).clamp_min(eps).clamp_max(1 - eps)
        s = torch.sigmoid(
            (self.log_alpha + torch.log(u) - torch.log(1.0 - u)) / self.tau
        )
        z = s * (self.hi - self.lo) + self.lo
        return z.clamp(0.0, 1.0)


class Controller(nn.Module):
    """Per-site EMA selective utility + hard-concrete gate probabilities.

    ``log_alpha_l`` is the gate's controller parameter; ``utility_ema_l`` is
    the EMA of the held-out selective utility; ``ui_step`` counts utility
    observations. Utility and gate state are module state so checkpoints
    preserve them exactly.
    """

    def __init__(self, site_names: Sequence[str], tau: float = 0.3, beta: float = 0.99):
        super().__init__()
        self.site_names = list(site_names)
        self.tau = float(tau)
        self.beta = float(beta)
        self.gates = nn.ModuleDict({s: HardConcreteGate(tau=tau) for s in self.site_names})
        self.register_buffer("utility_ema", torch.zeros(len(self.site_names)))
        self.register_buffer("ui_step", torch.zeros((), dtype=torch.long))

    def gate_prob(self, site: str) -> torch.Tensor:
        return self.gates[site].expected()

    def gate_probs(self) -> Dict[str, float]:
        return {s: float(self.gates[s].expected().detach().cpu()) for s in self.site_names}

    def sample_all(self, rs: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
        return {s: self.gates[s].sample(rs) for s in self.site_names}

    @torch.no_grad()
    def update_utility_ema(self, utilities: Sequence[Optional[float]]):
        """Stable EMA of relative per-site selective utility.

        Utility dot products can span many orders of magnitude as the
        backbone changes scale.  The controller only consumes their relative
        values, so normalize each observation vector before updating the raw
        EMA.  Bias correction is applied when reading the buffer; storing the
        corrected value back into the EMA would repeatedly divide old state
        and eventually overflow.
        """
        vals = []
        for utility in utilities:
            cur = float(utility) if utility is not None else 0.0
            vals.append(cur if math.isfinite(cur) else 0.0)
        scale = max((abs(cur) for cur in vals), default=0.0)
        if scale > 0.0:
            vals = [cur / scale for cur in vals]
        self.ui_step += 1
        for i, cur in enumerate(vals):
            self.utility_ema[i].mul_(self.beta).add_((1.0 - self.beta) * cur)

    def _corrected_utility(self) -> torch.Tensor:
        step = int(self.ui_step)
        if step <= 0:
            return self.utility_ema
        return self.utility_ema / (1.0 - self.beta ** step)

    @torch.no_grad()
    def step_from_utility(self, controller_lr: float, sparsity_cost: float, cap: float):
        """Controller move: ``log_alpha += lr * (u_norm - sparsity_cost)``.

        ``u_norm`` normalizes the per-site EMA utilities to [-1, 1] so one noisy
        dot product cannot dominate; each update is additionally clipped to
        ``[-cap, cap]`` (per-site strength cap).
        """
        utility = self._corrected_utility()
        umax = float(utility.abs().max())
        for i, s in enumerate(self.site_names):
            raw = float(utility[i])
            u_norm = raw / umax if umax > 1e-12 else 0.0
            delta = controller_lr * (u_norm - sparsity_cost)
            self.gates[s].log_alpha.data.add_(
                float(torch.clamp(torch.tensor(delta), -cap, cap)))

    def utility_ema_dict(self) -> Dict[str, float]:
        utility = self._corrected_utility()
        return {s: float(utility[i].detach().cpu())
                for i, s in enumerate(self.site_names)}


# ---------------------------------------------------------------------------
# projection helper
# ---------------------------------------------------------------------------
def project_aux(g_aux: torch.Tensor, g0_ema: torch.Tensor, eps: float = 1e-8):
    """Classification-safety projection of an aux gradient against ``g0_ema``.

    If ``<g_aux, g0_ema> < 0`` (the aux direction conflicts with the running
    CE descent) remove that conflicting component::

        g_safe = g_aux - (<g_aux, g0_ema> / (||g0_ema||^2 + eps)) * g0_ema

    Otherwise ``g_safe = g_aux``. Returns ``(g_safe, alignment)`` with
    ``alignment = <g_aux, g0_ema>`` *before* projection.
    """
    dot = torch.sum(g_aux * g0_ema)
    if dot < 0.0:
        nrm = torch.sum(g0_ema * g0_ema) + eps
        g_aux = g_aux - (dot / nrm) * g0_ema
    return g_aux, float(dot)


# ---------------------------------------------------------------------------
# the method
# ---------------------------------------------------------------------------
class SageDSMethod(Method):
    """Learn a sparse time-varying supervision topology by selective utility."""

    method_name = "sage_ds"

    def default_score(self) -> str:
        return "msp"

    def default_scores(self):
        return ("msp", "entropy", "energy", "logit_margin", "sage_conf")

    def __init__(self, train_cfg: dict):
        super().__init__(train_cfg)
        m = train_cfg["method"]
        self.topology = str(m.get("topology", "sparse_utility_safe"))
        if self.topology not in ALL_TOPO:
            raise ValueError(f"unknown sage_ds topology {self.topology!r}")
        self.safety = bool(m.get("safety", self.topology == "sparse_utility_safe"))
        self.utility = bool(m.get("utility", self.topology in TOPO_CONTROLLER))
        self.supervision_scale = float(m.get("supervision_scale", 1.0))
        self.utility_interval = int(m.get("utility_interval", 50))
        self.controller_lr = float(m.get("controller_lr", 0.05))
        self.sparsity_cost = float(m.get("sparsity_cost", 0.5))
        self.utility_ema_beta = float(m.get("utility_ema_beta", 0.99))
        self.hard_concrete_tau = float(m.get("hard_concrete_tau", 0.3))
        self.strength_cap = float(m.get("strength_cap", 0.5))
        self.token = str(m.get("token", "cls"))
        self.use_risk_head = bool(m.get("use_risk_head", False))
        self.fixed_sites = list(m.get("fixed_sites", ["top_l2", "top_l1"]))
        self.candidate_roles = list(m.get("candidate_roles", [])) or None

        # candidate sites come from the backbone registry (never hard-coded)
        self.site_names = self._resolve_sites()
        probe = self._probe_site_dims()
        self.aux_heads = nn.ModuleDict(
            {s: AuxHead(probe[s], self.num_classes) for s in self.site_names}
        )
        self._aux_params = [p for h in self.aux_heads.values() for p in h.parameters()]
        self._aux_param_ids = {id(p): i for i, p in enumerate(self._aux_params)}
        if self.use_risk_head:
            self.risk_head = AuxHead(self.backbone.final_dim, 1, hidden=16)
        else:
            self.risk_head = None

        self.controller = Controller(self.site_names, tau=self.hard_concrete_tau,
                                     beta=self.utility_ema_beta)
        if self.topology == "fixed_late":
            late = [self.backbone.roles[r] for r in self.fixed_sites]
            self._site_weights = {s: (1.0 if s in late else 0.0) for s in self.site_names}
            self._learn_w = None
        elif self.topology == "all_equal":
            self._site_weights = {s: 1.0 for s in self.site_names}
            self._learn_w = None
        elif self.topology == "learned_dense":
            self._site_weights = None
            self._learn_w = nn.ParameterDict(
                {s: nn.Parameter(torch.zeros(())) for s in self.site_names}
            )
        else:
            self._site_weights = None
            self._learn_w = None

        # utility gradient parameter set (default: every shared backbone param)
        self._set_utility_params(m.get("grad_subset", None))

        self.g0_ema: Optional[torch.Tensor] = None
        self.utility_ms = 0.0
        self.aux_ms: Dict[str, float] = {s: 0.0 for s in self.site_names}
        self.result_gate_history: Dict[str, List[float]] = {s: [] for s in self.site_names}
        self.result_epoch_history: List[int] = []
        self._val_loader = None
        self._val_iter = None
        self._log: List[dict] = []
        self._step_aux_acc: Dict[str, float] = {s: 0.0 for s in self.site_names}
        self._step_aux_loss: Dict[str, float] = {s: 0.0 for s in self.site_names}
        self._step_n = 0
        self._step_l0: Dict[str, float] = {s: 0.0 for s in self.site_names}
        self._last_align = (0.0, 0.0)

    # ------------------------------------------------------------------ init
    def _resolve_sites(self) -> List[str]:
        if self.candidate_roles:
            return [self.backbone.roles[r] for r in self.candidate_roles]
        return list(self.backbone.taps.keys())

    def _probe_site_dims(self) -> Dict[str, int]:
        with torch.no_grad(), self.probe_mode():
            bo = self.backbone(
                torch.zeros(1, self.backbone.channels, self.backbone.input_size,
                            self.backbone.input_size)
            )
        return {s: int(_pool_tap(bo.features[s], self.token).shape[-1]) for s in self.site_names}

    def _set_utility_params(self, subset: Optional[List[str]]):
        """Utility-gradient params: all (default) or a logged name-prefix subset.

        When a subset is configured the *unused* parameters are recorded so the
        cost-control choice is reproducible and auditable.
        """
        prefix = list(subset) if subset else None
        self._utility_params: List[Tuple[str, nn.Parameter]] = []
        self._unused_params: List[str] = []
        for n, p in self.backbone.named_parameters():
            if not p.requires_grad:
                continue
            if prefix is not None and not any(n.startswith(pre) for pre in prefix):
                self._unused_params.append(n)
                continue
            self._utility_params.append((n, p))

    # ------------------------------------------------------------- inference
    def predict_batch(self, x):
        bo = self.backbone(x)
        logits = bo.logits[:, : self.num_classes]
        scores = compute_scores(logits, self.default_scores())
        conf = scores["msp"]  # default inference confidence: terminal MSP
        if self.use_risk_head:
            with torch.no_grad():
                r = self.risk_head(_pool_tap(bo.final_embedding, self.token)).squeeze(-1)
                scores["sage_conf"] = -r  # risk head: higher risk -> lower confidence
        else:
            scores["sage_conf"] = conf
        return MethodPrediction(logits, logits.argmax(dim=1), conf, scores)

    def stripped_predict_batch(self, x):
        """Deployment-graph inference (aux heads/controller are absent)."""
        return self.predict_batch(x)

    def to_deployment(self) -> nn.Module:
        # aux heads + controller are already absent from the prediction path
        return self

    def inference_modules(self):
        mods = [self.backbone]
        if self.use_risk_head:
            mods.append(self.risk_head)
        return mods

    def optimizer_specs(self):
        """Backbone + aux heads (SGD); controller gates are updated manually.

        The hard-concrete ``log_alpha`` parameters are driven by the
        out-of-sample utility controller, not by the batch-loss optimizer, so
        they are intentionally left out of the param groups.
        """
        t = self.cfg["train"]
        params = list(self.backbone.parameters()) + self._aux_params
        if self._learn_w is not None:
            params = params + list(self._learn_w.parameters())
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
        x = batch[0].to(device)
        y = batch[1].to(device)
        bo = self.backbone(x)
        ce_t = F.cross_entropy(bo.logits[:, : self.num_classes], y)
        if self.topology in TOPO_NATIVE:
            return self._native_loss(bo, x, y, ce_t)
        return self._controller_loss(bo, x, y, ce_t, state)

    def _native_loss(self, bo, x, y, ce_t):
        """fixed_late / all_equal / learned_dense: native, unprojected aux CE."""
        out = {}
        aux = torch.zeros((), device=ce_t.device)
        for s in self.site_names:
            h = self.aux_heads[s]
            feat = _pool_tap(bo.features[s], self.token)
            l_aux = F.cross_entropy(h(feat), y)
            if self.topology == "learned_dense":
                w = torch.sigmoid(self._learn_w[s]) * self.supervision_scale
            else:
                w = self._site_weights[s]
            with torch.no_grad():
                out[f"aux_acc_{s}"] = (h(feat).detach().argmax(1) == y).float().mean()
            out[f"aux_loss_{s}"] = l_aux.detach()
            if w > 0:
                aux = aux + w * l_aux
        out["ce"] = ce_t
        out["sage_aux"] = aux
        return out

    def _controller_loss(self, bo, x, y, ce_t, state):
        backbone_params = [p for _, p in self._utility_params]
        self._step_n += 1
        g0 = torch.autograd.grad(ce_t, backbone_params, retain_graph=True,
                                 allow_unused=True, materialize_grads=True)
        g0_flat = [_flatten(g) for g in g0]
        if self.safety:
            if self.g0_ema is None:
                self.g0_ema = _cat(g0_flat).clone().detach()
            else:
                self.g0_ema.mul_(self.utility_ema_beta).add_(
                    _cat(g0_flat).detach() * (1.0 - self.utility_ema_beta))

        sampled = self.controller.sample_all()
        g_aux = [torch.zeros_like(p) for p in backbone_params]
        auxhead_g = [torch.zeros_like(p) for p in self._aux_params]
        out = {}
        for s in self.site_names:
            h = self.aux_heads[s]
            feat = _pool_tap(bo.features[s], self.token)
            l_aux = F.cross_entropy(h(feat), y)
            w = sampled[s] * self.supervision_scale
            t0 = time.perf_counter()
            gb = torch.autograd.grad(w * l_aux, backbone_params, retain_graph=True,
                                     allow_unused=True, materialize_grads=True)
            self.aux_ms[s] += (time.perf_counter() - t0) * 1000.0
            for i, g in enumerate(gb):
                if g is not None:
                    g_aux[i] = g_aux[i] + g
            # aux-head params keep their *unprojected* own-CE gradient
            gh = torch.autograd.grad(l_aux, list(h.parameters()), retain_graph=True,
                                     allow_unused=True, materialize_grads=True)
            for gi, p in enumerate(list(h.parameters())):
                if gh[gi] is not None:
                    auxhead_g[self._aux_param_ids[id(p)]] = gh[gi]
            with torch.no_grad():
                acc = (h(feat).detach().argmax(1) == y).float().mean()
                out[f"aux_acc_{s}"] = acc
                self._step_aux_acc[s] += float(acc)
                self._step_aux_loss[s] += float(l_aux.detach())
                self._step_l0[s] += float(self.controller.gates[s].l0_norm())
            out[f"aux_loss_{s}"] = l_aux.detach()
            out[f"gatep_{s}"] = self.controller.gate_prob(s).detach()
            out[f"gate_{s}"] = (w.detach() / self.supervision_scale).clamp(0.0, 1.0)

        # classify-alignment + optional classification-safety projection
        align_before = _cat([_flatten(g) for g in g_aux])
        if self.safety and self.g0_ema is not None:
            safe, align = project_aux(align_before, self.g0_ema)
            align_after = float(torch.sum(safe * self.g0_ema))
        else:
            safe, align = align_before, 0.0
            align_after = 0.0
        self._last_align = (float(align), float(align_after))
        out["align_before"] = float(align)
        out["align_after"] = float(align_after)

        # re-scatter the projected combined aux gradient per-param
        safe_flat = _flatten(safe)
        g_safe = []
        acc = 0
        for p in backbone_params:
            n = p.numel()
            g_safe.append(safe_flat[acc:acc + n].reshape_as(p).clone())
            acc += n

        # route the desired gradient directly onto the parameters
        routed = torch.zeros((), device=ce_t.device)
        for p, g0p, gap in zip(backbone_params, g0, g_safe):
            g_desired = (g0p if g0p is not None else torch.zeros_like(p)) + gap
            routed = routed + torch.sum(p * g_desired.detach())
        for p, g in zip(self._aux_params, auxhead_g):
            routed = routed + torch.sum(p * g.detach())
        out["routed"] = routed
        out["ce"] = ce_t.detach()

        step = int(getattr(state, "batch_index", 0))
        if self._should_estimate_utility(step):
            self._estimate_utilities(ce_t.device)
        return out

    # ------------------------------------------------------------- utility
    def _should_estimate_utility(self, step: int) -> bool:
        return bool(self.utility and self.utility_interval > 0
                    and step > 0 and step % self.utility_interval == 0)

    def _val_batch(self, device):
        if self._val_loader is None:
            self._val_loader = build_dataloader(self.cfg, "val", shuffle=False,
                                                return_indices=True, num_workers=0)
            self._val_iter = iter(self._val_loader)
        try:
            batch = next(self._val_iter)
        except StopIteration:
            self._val_iter = iter(self._val_loader)
            batch = next(self._val_iter)
        return batch[0].to(device), batch[1].to(device)

    def _estimate_utilities(self, device):
        """Estimate per-site ``U_l = <g_sel, g_l>`` on a fresh val batch."""
        xv, yv = self._val_batch(device)
        params = [p for _, p in self._utility_params]
        was_training = self.training
        self.eval()
        try:
            bo = self.backbone(xv)
            logits = bo.logits[:, : self.num_classes]
            t0 = time.perf_counter()
            surrogate = soft_aurc_surrogate(logits, yv, tau=self.hard_concrete_tau)
            g_sel = torch.autograd.grad(surrogate, params, retain_graph=True,
                                        allow_unused=True, materialize_grads=True)
            self.utility_ms = (time.perf_counter() - t0) * 1000.0
            utilities = []
            for s in self.site_names:
                h = self.aux_heads[s]
                feat = _pool_tap(bo.features[s], self.token)
                l_aux = F.cross_entropy(h(feat), yv)
                g_l = torch.autograd.grad(l_aux, params, retain_graph=True,
                                          allow_unused=True, materialize_grads=True)
                utilities.append(float(selective_utility(g_sel, g_l).detach().cpu()))
        finally:
            self.train(was_training)
        self.controller.update_utility_ema(utilities)
        self.controller.step_from_utility(self.controller_lr, self.sparsity_cost,
                                          self.strength_cap)

    # ------------------------------------------------------------------ logs
    def on_epoch_end(self, epoch: int, val_metrics: dict):
        n = max(self._step_n, 1)
        sparsity_penalty = sum(
            self._step_l0[s] / n * self.sparsity_cost for s in self.site_names
        )
        row = {
            "epoch": int(epoch),
            "utility_ms": float(self.utility_ms),
            "unused_params": list(self._unused_params),
            "align_before": self._last_align[0],
            "align_after": self._last_align[1],
            "sparsity_penalty": float(sparsity_penalty),
        }
        for s in self.site_names:
            row[f"gatep_{s}"] = float(self.controller.gate_prob(s).detach().cpu())
            row[f"uema_{s}"] = self.controller.utility_ema_dict()[s]
            row[f"aux_ms_{s}"] = float(self.aux_ms[s])
            row[f"aux_acc_{s}"] = float(self._step_aux_acc[s] / n)
            row[f"aux_loss_{s}"] = float(self._step_aux_loss[s] / n)
            row[f"l0_{s}"] = float(self._step_l0[s] / n)
            self.result_gate_history[s].append(row[f"gatep_{s}"])
        self.result_epoch_history.append(int(epoch))
        self._log.append(row)
        self._step_n = 0
        self._step_aux_acc = {s: 0.0 for s in self.site_names}
        self._step_aux_loss = {s: 0.0 for s in self.site_names}
        self._step_l0 = {s: 0.0 for s in self.site_names}
        self._write_log()

    def _write_log(self):
        try:
            run_dir = os.path.join(self.cfg["results_root"], self.cfg["run_name"])
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "sage_ds.jsonl"), "a") as f:
                for r in self._log:
                    f.write(json.dumps(r, default=str) + "\n")
            self._log.clear()
        except Exception:
            pass

    def heatmap(self):
        """Per-epoch x per-site gate-probability matrix (values + labels)."""
        import numpy as np

        epochs = list(self.result_epoch_history)
        sites = list(self.site_names)
        mat = np.zeros((max(len(epochs), 1), len(sites)))
        for j, s in enumerate(sites):
            for i, v in enumerate(self.result_gate_history[s]):
                mat[i, j] = float(v)
        return {"epochs": epochs, "sites": sites, "gate_prob": mat}

    def save_heatmap(self, path: str):
        import numpy as np

        hm = self.heatmap()
        np.savez(path, epochs=np.asarray(hm["epochs"]),
                 sites=np.asarray(hm["sites"]), gate_prob=hm["gate_prob"])


# ---------------------------------------------------------------------------
# parameter-reachability introspection (aux-only backward prefix test)
# ---------------------------------------------------------------------------
def params_reached_by_aux(method: SageDSMethod, site: str, num_examples: int = 2):
    """Backbone parameter names receiving gradient from site ``site``'s aux head.

    Architecture neutral: performs one forward + auxiliary-head loss backward
    on a tiny synthetic batch and records which backbone parameters (by name)
    end up with a non-zero gradient. Only taps from the registry are accepted.
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
                    method.backbone.input_size, method.backbone.input_size, device=dev)
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


def _flatten(t):
    return t.reshape(-1)


def _cat(seq):
    if not seq:
        return torch.zeros((), device="cpu")
    return torch.cat([_flatten(t) for t in seq], dim=0)


def selective_utility(g_sel, g_l):
    """First-order selective utility ``U_l = <g_sel, g_l>``.

    With ``theta' = theta - eta * g_l``, ``J_sel(theta') - J_sel(theta)`` is
    approximated by ``-eta * U_l``, so ``U_l > 0`` predicts improvement. See the
    finite-difference unit test for the sign lock.
    """
    return sum(
        torch.sum(a * b) for a, b in zip(g_sel, g_l)
        if a is not None and b is not None
    )
