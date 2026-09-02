"""RiskFlow: persistent sequential failure-evidence state via conditional innovations.

RiskFlow maintains a small per-example **risk state** that is updated across
the ordered backbone taps instead of being a weighted sum of independent
intermediate confidence heads. Each depth contributes only the *new* failure
evidence not already explained by the previous cumulative state::

    r_0 = base_state                  (learned vector, broadcast over examples)
    delta_r_l, gate_l = psi(adapter_l(h_l), r_{l-1})
    r_l = r_{l-1} + sigmoid(gate_l) * delta_r_l
    s_hard_l = readout_hard(r_l)      (error-risk channel, default score)
    s_soft_l = readout_soft(r_l)      (difficulty channel, auxiliary)
    q_l     = sigmoid(s_hard_l)

The update cell ``psi`` and the readout heads are shared across depth; only the
**input adapters** are architecture-specific (pooled stage features for CNNs,
CLS / token-mean for ViT), so nothing special-cases a backbone.

Innovation supervision
----------------------
With ``e = 1[pred != y]`` (detached) the hard-channel pseudo-residual at depth
``l`` is ``eps_l = stopgrad(e - sigmoid(s_hard_{l-1}))``. The scalar logit
contribution induced by ``delta_r_l`` is ``s_l - s_{l-1}`` (exactly
``readout(gate_l * delta_r_l)``), trained with Huber loss to match ``eps_l``.
The target is detached so later stages never backprop through it, while the
current innovation loss may shape its adapter/backbone features. A second
continuous channel uses the **normalized true-label NLL** as its detached target
and is trained with an analogous pseudo-residual; it is an auxiliary training
signal and never the inference score. A terminal proper-scoring (BCE) loss on
the final ``s_L`` keeps the cumulative state meaningful. Redundant updates
across depth are penalized with a small, configurable cross-example
decorrelation term that is robust to zero-variance columns.

Modes (the ablation ladder from the empirical contract)
-------------------------------------------------------
1. ``concat`` — SCSF-style concatenation head (all pooled taps + embedding).
2. ``heads``  — independent per-depth error heads + learned weighted sum.
3. ``cum``    — shared cumulative state, **no** residual targets.
4. ``resid``  — cumulative state + residual innovation targets, **fixed** gates.
5. ``riskflow`` (default) — residual innovation + **sample-dependent** gates.
6. ``riskflow_frozen`` — frozen-backbone RiskFlow (control).
7. ``riskflow_hard`` — hard-error channel only (soft channel disabled).

State size is ``state_dim`` (default 64). Deployment overhead is nonzero and
is reported directly (parameters, MACs, latency, memory).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones import adaptive_flatten
from .base import Method, MethodPrediction
from .scores import compute_scores

VARIANTS = ("concat", "heads", "cum", "resid", "riskflow")
GATE_MODES = ("fixed", "sample")


def pool_tap(feature: torch.Tensor, token: str = "cls") -> torch.Tensor:
    """Reduce a tap feature to a fixed (B, D) vector.

    * 4-D CNN maps -> global-average-pool then flatten;
    * 3-D token sequences (ViT) -> CLS token (or token mean) from config;
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


class InputAdapter(nn.Module):
    """Architecture-specific pooled-feature projection to the state space."""

    def __init__(self, in_features: int, state_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.proj = nn.Linear(in_features, state_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(h))


class BaseState(nn.Module):
    """Learned base-risk state vector ``r_0`` (a persistent module so it is
    counted in deployment parameters via ``inference_modules()``)."""

    def __init__(self, state_dim: int):
        super().__init__()
        self.vector = nn.Parameter(torch.zeros(state_dim))

    def forward(self):
        return self.vector


class RiskCell(nn.Module):
    """Shared update cell ``psi(h, r_{l-1}) -> (delta_r_l, gate_logit_l)``.

    ``use_gate_net`` is False for fixed-gate modes; the gate then comes from a
    per-depth learnable scalar in the method so the gate is sample-independent.
    """

    def __init__(self, state_dim: int, hidden: int = 64, use_gate_net: bool = True):
        super().__init__()
        self.state_dim = int(state_dim)
        z = 2 * self.state_dim
        self.upd = nn.Sequential(
            nn.Linear(z, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, self.state_dim)
        )
        self.use_gate_net = bool(use_gate_net)
        if use_gate_net:
            self.gate_net = nn.Linear(z, 1)

    def delta(self, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return self.upd(torch.cat([h, r], dim=-1))

    def gate_logit(self, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        if not self.use_gate_net:
            raise RuntimeError("gate_net not present in fixed-gate mode")
        return self.gate_net(torch.cat([h, r], dim=-1)).squeeze(-1)


class RiskFlowTrace:
    """Per-example, per-depth export of the RiskFlow forward pass."""

    __slots__ = (
        "site_names", "logits", "prediction", "s_hard", "s_soft",
        "innov_hard", "innov_soft", "gates", "deltas", "eps_hard", "eps_soft",
        "final_s_hard", "final_s_soft", "hard_error", "soft_target",
    )

    def __init__(self, site_names, logits, prediction, s_hard, s_soft, innov_hard,
                 innov_soft, gates, deltas, eps_hard, eps_soft, final_s_hard,
                 final_s_soft, hard_error, soft_target):
        self.site_names = list(site_names)
        self.logits = logits
        self.prediction = prediction
        self.s_hard = s_hard
        self.s_soft = s_soft
        self.innov_hard = innov_hard
        self.innov_soft = innov_soft
        self.gates = gates
        self.deltas = deltas
        self.eps_hard = eps_hard
        self.eps_soft = eps_soft
        self.final_s_hard = final_s_hard
        self.final_s_soft = final_s_soft
        self.hard_error = hard_error
        self.soft_target = soft_target


def decorrelation_penalty(deltas: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Average off-diagonal redundancy of innovation vectors across a batch.

    ``deltas`` is ``(L, B, D)``. For each example the depth innovations are
    L2-normalized (with an ``eps`` floor so a zero innovation is never an
    invalid NaN), the per-example Gram is computed and the mean absolute
    off-diagonal entry is returned. This is a cross-example penalty in the
    sense that it is averaged over the batch, and it avoids the fragile scalar
    correlation that breaks when a column has zero variance.
    """
    if deltas.ndim != 3:
        raise ValueError(f"deltas must be (L, B, D), got {tuple(deltas.shape)}")
    L, B, D = deltas.shape
    if L < 2:
        return torch.zeros((), device=deltas.device)
    n = deltas.norm(dim=-1, keepdim=True).clamp_min(eps)
    d = deltas / n                                        # (L, B, D)
    gram = torch.einsum("lbd,mbd->lmb", d, d)             # (L, L, B)
    iu = torch.triu_indices(L, L, offset=1)
    off = gram[iu[0], iu[1]]                              # (K, B)
    return off.abs().mean()


class RiskFlowMethod(Method):
    """Accumulate conditional failure innovations into a persistent state."""

    method_name = "riskflow"

    def default_score(self) -> str:
        return "riskflow"

    def default_scores(self):
        return ("msp", "entropy", "energy", "logit_margin", "riskflow")

    def __init__(self, train_cfg: dict):
        super().__init__(train_cfg)
        m = train_cfg["method"]
        self.variant = str(m.get("mode", "riskflow"))
        if self.variant not in VARIANTS:
            raise ValueError(f"unknown riskflow.mode {self.variant!r}; "
                             f"choose {list(VARIANTS)}")
        self.use_soft = bool(m.get("use_soft", self.variant == "riskflow"))
        self.freeze_backbone = bool(m.get("freeze_backbone", False))
        self.state_dim = int(m.get("state_dim", 64))
        self.cell_hidden = int(m.get("cell_hidden", 64))
        self.token = str(m.get("token", "cls"))
        self.innov_scale = float(m.get("innov_scale", 1.0))
        self.term_scale = float(m.get("term_scale", 1.0))
        self.decorr_scale = float(m.get("decorr_scale", 0.01))
        self.huber_delta = float(m.get("huber_delta", 1.0))
        self.cat_lo = float(m.get("cat_lo", 0.3))
        self.cat_hi = float(m.get("cat_hi", 0.7))

        # gate mode: 'sample' only for the default riskflow; fixed otherwise
        self.gate_mode = "sample" if self.variant == "riskflow" else "fixed"

        # candidate sites from the registry, in deterministic registry order
        self.site_names = list(self.backbone.taps.keys())
        probe = self._probe_site_dims()

        self.adapters = nn.ModuleDict(
            {s: InputAdapter(probe[s], self.state_dim) for s in self.site_names}
        )
        self._adapter_params = [p for a in self.adapters.values() for p in a.parameters()]

        if self.variant == "concat":
            concat_in = sum(probe[s] for s in self.site_names) + self.backbone.final_dim
            self.concat_head = nn.Sequential(
                nn.Linear(concat_in, 256), nn.ReLU(inplace=True), nn.Linear(256, 1),
            )
            self.cell = None
            self.readout_hard = None
            self.readout_soft = None
            self.base_state = None
            self.fixed_gates = None
            self.head_weights = None
            self.heads = None
        elif self.variant == "heads":
            self.heads = nn.ModuleDict(
                {s: nn.Linear(self.state_dim, 1) for s in self.site_names}
            )
            self.head_weights = nn.Parameter(torch.zeros(len(self.site_names)))
            self.cell = None
            self.readout_hard = None
            self.readout_soft = None
            self.base_state = None
            self.fixed_gates = None
            self.concat_head = None
        else:  # cum / resid / riskflow : recurrent state
            self.cell = RiskCell(self.state_dim, self.cell_hidden,
                                 use_gate_net=(self.gate_mode == "sample"))
            self.readout_hard = nn.Linear(self.state_dim, 1)
            self.readout_soft = nn.Linear(self.state_dim, 1) if self.use_soft else None
            self.base_state = BaseState(self.state_dim)
            if self.gate_mode == "fixed":
                self.fixed_gates = nn.ParameterList(
                    [nn.Parameter(torch.zeros(())) for _ in self.site_names]
                )
            else:
                self.fixed_gates = None
            self.concat_head = None
            self.heads = None
            self.head_weights = None

        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------ init
    def _probe_site_dims(self) -> Dict[str, int]:
        with torch.no_grad(), self.probe_mode():
            bo = self.backbone(
                torch.zeros(1, self.backbone.channels, self.backbone.input_size,
                            self.backbone.input_size)
            )
        return {s: int(pool_tap(bo.features[s], self.token).shape[-1])
                for s in self.site_names}

    # ------------------------------------------------------------------ flow
    def _flow(self, bo, y=None) -> RiskFlowTrace:
        logits = bo.logits[:, : self.num_classes]
        prediction = logits.argmax(dim=1)
        B = logits.shape[0]

        e = None
        d = None
        if y is not None:
            e = (prediction != y).float().detach()
            # This continuous difficulty target can exceed one for a
            # confidently wrong prediction.  It is therefore supervised by
            # robust regression below, never as a Bernoulli probability.
            d = (F.nll_loss(F.log_softmax(logits, dim=1), y, reduction="none")
                 / math.log(self.num_classes)).detach()

        if self.variant == "concat":
            pooled = [pool_tap(bo.features[s], self.token) for s in self.site_names]
            s = self.concat_head(torch.cat(pooled + [bo.final_embedding], dim=1)).squeeze(-1)
            return RiskFlowTrace(
                self.site_names, logits, prediction,
                s.unsqueeze(0), None, None, None, None, None, None, None,
                s, None, e, d)

        if self.variant == "heads":
            cols = []
            for s in self.site_names:
                h = pool_tap(bo.features[s], self.token)
                cols.append(self.heads[s](self.adapters[s](h)).squeeze(-1))
            S = torch.stack(cols, dim=0)              # (L, B)
            w = F.softmax(self.head_weights, dim=0)   # (L,)
            final = (w[:, None] * S).sum(dim=0)
            return RiskFlowTrace(
                self.site_names, logits, prediction,
                S, None, None, None, None, None, None, None,
                final, None, e, d)

        # ---- recurrent: cum / resid / riskflow ----
        gates = []
        deltas = []
        s_hard_cols = []
        s_soft_cols = []
        r = self.base_state.vector.expand(B, self.state_dim)
        s_hard_cols.append(self.readout_hard(r).squeeze(-1))
        if self.use_soft:
            s_soft_cols.append(self.readout_soft(r).squeeze(-1))
        for l, s in enumerate(self.site_names):
            h = self.adapters[s](pool_tap(bo.features[s], self.token))
            delta = self.cell.delta(h, r)
            if self.gate_mode == "sample":
                gate = torch.sigmoid(self.cell.gate_logit(h, r))
            else:
                gate = torch.sigmoid(self.fixed_gates[l]).expand(B)
            r = r + gate[:, None] * delta
            gates.append(gate)
            deltas.append(delta)
            s_hard_cols.append(self.readout_hard(r).squeeze(-1))
            if self.use_soft:
                s_soft_cols.append(self.readout_soft(r).squeeze(-1))

        s_hard = torch.stack(s_hard_cols, dim=0)      # (L+1, B); row 0 = base
        s_soft = torch.stack(s_soft_cols, dim=0) if self.use_soft else None
        gates = torch.stack(gates, dim=0)             # (L, B)
        deltas = torch.stack(deltas, dim=0)           # (L, B, D)
        innov_hard = (s_hard[1:] - s_hard[:-1])       # (L, B)
        innov_soft = (s_soft[1:] - s_soft[:-1]) if self.use_soft else None

        final_s_hard = s_hard[-1]
        final_s_soft = s_soft[-1] if self.use_soft else None

        eps_hard = None
        eps_soft = None
        if e is not None:
            eps_hard = (e[None, :] - torch.sigmoid(s_hard[:-1])).detach()   # (L, B)
            if self.use_soft:
                eps_soft = (d[None, :] - s_soft[:-1]).detach()

        return RiskFlowTrace(
            self.site_names, logits, prediction,
            s_hard, s_soft, innov_hard, innov_soft, gates, deltas,
            eps_hard, eps_soft, final_s_hard, final_s_soft, e, d)

    # ------------------------------------------------------------- inference
    def predict_batch(self, x):
        bo = self.backbone(x)
        logits = bo.logits[:, : self.num_classes]
        with torch.no_grad():
            flow = self._flow(bo, y=None)
            riskflow = -flow.final_s_hard
        scores = compute_scores(logits, self.default_scores())
        scores["riskflow"] = riskflow
        if flow.final_s_soft is not None:
            scores["riskflow_soft"] = -flow.final_s_soft.detach()
        return MethodPrediction(logits, flow.prediction, riskflow, scores)

    def predict_with_trace(self, x, y=None):
        """Inference-time trace for diagnostics (targets present if y given)."""
        bo = self.backbone(x)
        logits = bo.logits[:, : self.num_classes]
        flow = self._flow(bo, y=y)
        riskflow = -flow.final_s_hard
        scores = compute_scores(logits, self.default_scores())
        scores["riskflow"] = riskflow
        if flow.final_s_soft is not None:
            scores["riskflow_soft"] = -flow.final_s_soft.detach()
        mp = MethodPrediction(logits, flow.prediction, riskflow, scores)
        return mp, flow

    def stripped_predict_batch(self, x):
        return self.predict_batch(x)

    def to_deployment(self) -> nn.Module:
        return self

    def inference_modules(self):
        mods = [self.backbone]
        if self.variant == "concat":
            mods.append(self.concat_head)
        elif self.variant == "heads":
            mods.append(self.heads)
        else:
            mods += [self.adapters, self.cell, self.readout_hard, self.base_state]
        return mods

    def optimizer_specs(self):
        t = self.cfg["train"]
        params = [p for p in self.parameters() if p.requires_grad]
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
        logits = bo.logits[:, : self.num_classes]
        ce = F.cross_entropy(logits, y)
        out = {"ce": ce}
        flow = self._flow(bo, y=y)
        e = flow.hard_error
        d = flow.soft_target

        if self.variant == "concat":
            out["rf_term_hard"] = self.term_scale * F.binary_cross_entropy_with_logits(
                flow.final_s_hard, e)
            return out

        if self.variant == "heads":
            per = sum(
                F.binary_cross_entropy_with_logits(flow.s_hard[l], e)
                for l in range(len(self.site_names))
            ) / len(self.site_names)
            out["rf_head"] = self.term_scale * (
                per + F.binary_cross_entropy_with_logits(flow.final_s_hard, e))
            return out

        # ---- recurrent ----
        if self.variant in ("resid", "riskflow"):
            res_hard = F.huber_loss(flow.innov_hard, flow.eps_hard, delta=self.huber_delta)
            out["rf_innov_hard"] = self.innov_scale * res_hard
            if self.use_soft:
                res_soft = F.huber_loss(flow.innov_soft, flow.eps_soft, delta=self.huber_delta)
                out["rf_innov_soft"] = self.innov_scale * res_soft

        out["rf_term_hard"] = self.term_scale * F.binary_cross_entropy_with_logits(
            flow.final_s_hard, e)
        if self.use_soft:
            out["rf_term_soft"] = self.term_scale * F.huber_loss(
                flow.final_s_soft, d, delta=self.huber_delta)

        if self.decorr_scale > 0 and flow.deltas is not None:
            out["rf_decorr"] = self.decorr_scale * decorrelation_penalty(flow.deltas)
        return out


__all__ = [
    "RiskCell",
    "BaseState",
    "InputAdapter",
    "RiskFlowTrace",
    "RiskFlowMethod",
    "decorrelation_penalty",
    "pool_tap",
]
