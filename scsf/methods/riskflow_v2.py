"""RiskFlow-V2: stage-wise risk refinement with signed bounded innovations.

RiskFlow-v1 models depth as a sequence of risk updates through a
**multiplicative gate** (`r = r + sigmoid(gate) * delta`) and supervises each
innovation against a pseudo-residual of the *live* model's hard error.  The
gate forces innovations to be widening, and the targets drift every step.

RiskFlow-V2 (protocol ``docs/RISKFLOW_V2_PROTOCOL.md``) removes the gate and
the decorrelation penalty from the primary method and makes each stage an
identifiable **signed, bounded** logit increment with a fixed EMA teacher::

    s_l = stop_gradient(s_{l-1}) + Delta_max * tanh(a_l)

- ``s_l`` is the stage-l cumulative failure-risk logit (scalar per example).
- ``a_l`` is an unbounded adapter cell output (the pre-bound innovation),
  mapped to ``[-Delta_max, +Delta_max]`` by ``tanh`` — later layers may
  increase *or* decrease risk.
- Targets come from an EMA copy of the student backbone (environment-agnostic
  fixed teacher): hard ``e_T = 1[argmax f_T(x) != y]`` for every stage's BCE,
  and a dense normalised ``d_T = -log p_T(y|x) / log(C)`` for a bounded soft
  Huber channel that only stabilises representations.
- Stop-gradient routing keeps previous-stage risk modules as sequential
  readout points (their BCE term cannot rewrite them), while the current
  stage's loss may still shape the backbone prefix feeding it.
- Inference is label-free; confidence = ``1 - sigmoid(s_L)`` (higher is
  more trusted).

The multiplicative gate is preserved only as the ``riskflow_v2_gate``
ablation (``remove_gate: false``).
"""

from __future__ import annotations

import copy
import json
import math
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Method, MethodPrediction
from .riskflow import InputAdapter, pool_tap
from .scores import compute_scores


def _soft_target(teacher_logits: torch.Tensor, y: torch.Tensor,
                 num_classes: int) -> torch.Tensor:
    """Normalised dense target ``d_T = -log p_T(y|x) / log(C) in [0, 1]``."""
    lp = F.log_softmax(teacher_logits[:, : num_classes], dim=1)
    d = -lp.gather(1, y.view(-1, 1)).squeeze(1) / math.log(num_classes)
    return d.clamp(min=0.0, max=1.0 + 1e-6)


class _BaseScale(nn.Module):
    """Learned per-mode base (logit offset) scalars.

    A 0-d ``nn.Parameter`` per mode, wrapped in a module so the base logits
    are part of checkpoint state and param accounting (``inference_modules``).
    """

    def __init__(self, use_soft: bool):
        super().__init__()
        self.hard = nn.Parameter(torch.zeros(()))
        self.soft = nn.Parameter(torch.zeros(())) if use_soft else None


class InnovationCell(nn.Module):
    """Per-stage pre-bound innovation cell (hard channel).

    Maps the shared adaptor state ``a`` (``state_dim``) to a scalar
    pre-bound hard innovation; with ``gate=True`` (ablation only) the
    innovation is additionally multiplied by a sample-dependent
    ``sigmoid(gate_logit)`` restoring v1's multiplicative gate.
    """

    def __init__(self, state_dim: int, hidden: int = 64, gate: bool = False):
        super().__init__()
        self.upd = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.gate = bool(gate)
        if self.gate:
            self.gate_net = nn.Linear(state_dim, 1)

    def forward(self, a: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        raw = self.upd(a).squeeze(-1)
        if self.gate:
            gate = torch.sigmoid(self.gate_net(a)).squeeze(-1)
        else:
            gate = None
        return raw, gate


class SoftCell(nn.Module):
    """Per-stage pre-bound innovation cell for the auxiliary soft channel."""

    def __init__(self, state_dim: int, hidden: int = 64):
        super().__init__()
        self.upd = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.upd(a).squeeze(-1)


class RiskFlowV2Trace:
    """Per-example, per-stage export of the RiskFlow-V2 forward pass."""

    __slots__ = (
        "site_names", "logits", "prediction",
        "s_hard", "s_soft", "innov_hard", "innov_soft",
        "hard_error", "soft_target", "final_s_hard",
    )

    def __init__(self, site_names, logits, prediction, s_hard, s_soft,
                 innov_hard, innov_soft, hard_error, soft_target):
        self.site_names = list(site_names)
        self.logits = logits
        self.prediction = prediction
        self.s_hard = s_hard                      # (L+1, B)
        self.s_soft = s_soft                      # (L+1, B) or None
        self.innov_hard = innov_hard              # (L, B)
        self.innov_soft = innov_soft              # (L, B) or None
        self.hard_error = hard_error              # (B,) or None
        self.soft_target = soft_target            # (B,) or None
        self.final_s_hard = s_hard[-1]

    def stage_hard_logit(self, l: int) -> torch.Tensor:
        return self.s_hard[l]

    def stage_soft_logit(self, l: int) -> torch.Tensor:
        return self.s_soft[l]


class RiskFlowV2Method(Method):
    """Accumulate signed bounded risk logits across depth (BCE-supervised)."""

    method_name = "riskflow_v2"

    def default_score(self) -> str:
        return "riskflow_v2"

    def default_scores(self):
        return ("msp", "entropy", "energy", "logit_margin", "riskflow_v2")

    def __init__(self, train_cfg: dict):
        super().__init__(train_cfg)
        m = train_cfg["method"]
        self.remove_gate = bool(m.get("remove_gate", True))
        self.variant = "riskflow_v2" if self.remove_gate else "riskflow_v2_gate"
        self.use_soft = bool(m.get("use_soft", True))
        self.state_dim = int(m.get("state_dim", 64))
        self.cell_hidden = int(m.get("cell_hidden", 64))
        self.token = str(m.get("token", "cls"))
        self.delta_max = float(m.get("delta_max", 2.0))
        self.huber_delta = float(m.get("huber_delta", 1.0))
        self.nu = float(m.get("nu", 0.999))
        self.log_every = int(m.get("log_every", 25))

        self.site_names = list(self.backbone.taps.keys())
        probe = self._probe_site_dims()

        self.adapters = nn.ModuleDict(
            {s: InputAdapter(probe[s], self.state_dim) for s in self.site_names}
        )
        self.cells = nn.ModuleDict(
            {s: InnovationCell(self.state_dim, self.cell_hidden,
                               gate=not self.remove_gate)
             for s in self.site_names}
        )
        self.cells_soft = nn.ModuleDict(
            {s: SoftCell(self.state_dim, self.cell_hidden) for s in self.site_names}
        ) if self.use_soft else None
        self.base = _BaseScale(self.use_soft)
        self.base_hard = self.base.hard
        self.base_soft = self.base.soft

        # EMA teacher (submodule so checkpoints restore it bit-exact).
        self.teacher = copy.deepcopy(self.backbone)
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.deployment_overhead = self._overhead(probe)
        self._reset_step_stats()

    # ---------------------------------------------------------------- init
    def _probe_site_dims(self) -> Dict[str, int]:
        with torch.no_grad(), self.probe_mode():
            bo = self.backbone(
                torch.zeros(1, self.backbone.channels, self.backbone.input_size,
                            self.backbone.input_size)
            )
        return {s: int(pool_tap(bo.features[s], self.token).shape[-1])
                for s in self.site_names}

    def _overhead(self, probe: Dict[str, int]) -> dict:
        macs = 0
        per_site = {}
        for s in self.site_names:
            adapter = probe[s] * self.state_dim + self.state_dim
            cell = self.state_dim * self.cell_hidden + 2 * self.cell_hidden
            per_site[s] = {"adapter_macs": adapter, "cell_macs": cell,
                           "adapter_params": (probe[s] + 1) * self.state_dim,
                           "cell_params": (self.state_dim + 1) * self.cell_hidden
                           + (self.cell_hidden + 1) * 2}
            macs += adapter + cell
        return {"macs_per_example": macs, "per_site": per_site}

    # ----------------------------------------------------------------- flow
    def _flow(self, bo, y=None, teacher_logits=None) -> RiskFlowV2Trace:
        logits = bo.logits[:, : self.num_classes]
        prediction = logits.argmax(dim=1)
        B = logits.shape[0]

        e = None
        d = None
        if y is not None and teacher_logits is not None:
            e = (teacher_logits[:, : self.num_classes].argmax(dim=1)
                 != y).float().detach()
            d = _soft_target(teacher_logits, y, self.num_classes).detach()

        s_hard_cols = [self.base_hard.expand(B)]
        s_soft_cols = [self.base_soft.expand(B)] if self.use_soft else None
        innov_hard = []
        innov_soft = [] if self.use_soft else None
        for s in self.site_names:
            a = self.adapters[s](pool_tap(bo.features[s], self.token))
            raw, gate = self.cells[s](a)
            innov = self.delta_max * torch.tanh(raw)
            if gate is not None:
                innov = innov * gate
            innov_hard.append(innov)
            s_hard_cols.append(s_hard_cols[-1].detach() + innov)
            if self.use_soft:
                raw_s = self.cells_soft[s](a)
                innov_s = self.delta_max * torch.tanh(raw_s)
                innov_soft.append(innov_s)
                s_soft_cols.append(s_soft_cols[-1].detach() + innov_s)

        s_hard = torch.stack(s_hard_cols, dim=0)
        s_soft = torch.stack(s_soft_cols, dim=0) if self.use_soft else None
        innov_hard = torch.stack(innov_hard, dim=0) if innov_hard else None
        innov_soft = torch.stack(innov_soft, dim=0) if innov_soft else None
        return RiskFlowV2Trace(self.site_names, logits, prediction,
                               s_hard, s_soft, innov_hard, innov_soft, e, d)

    # ----------------------------------------------------------- inference
    def _risk_confidence(self, flow: RiskFlowV2Trace) -> torch.Tensor:
        return 1.0 - torch.sigmoid(flow.final_s_hard.detach())

    def predict_batch(self, x):
        bo = self.backbone(x)
        logits = bo.logits[:, : self.num_classes]
        with torch.no_grad():
            flow = self._flow(bo, y=None)
        conf = self._risk_confidence(flow)
        scores = compute_scores(logits, self.default_scores())
        scores["riskflow_v2"] = conf
        return MethodPrediction(logits, logits.argmax(dim=1), conf, scores)

    def predict_with_trace(self, x, y=None):
        """Inference-time trace (targets present if ``y`` and teacher given)."""
        bo = self.backbone(x)
        logits = bo.logits[:, : self.num_classes]
        tl = None
        if y is not None:
            tl = self._teacher_logits(x)
        flow = self._flow(bo, y=y, teacher_logits=tl)
        conf = self._risk_confidence(flow)
        scores = compute_scores(logits, self.default_scores())
        scores["riskflow_v2"] = conf
        mp = MethodPrediction(logits, logits.argmax(dim=1), conf, scores)
        return mp, flow

    def stripped_predict_batch(self, x):
        return self.predict_batch(x)

    def to_deployment(self) -> nn.Module:
        return self

    def inference_modules(self):
        mods = [self.backbone, self.adapters, self.cells, self.base]
        if self.use_soft:
            mods.append(self.cells_soft)
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

    # ------------------------------------------------------------ EMA teacher
    def _ema_update_teacher(self):
        """``theta_T <- nu*theta_T + (1-nu)*theta_S`` incl. BN buffers.

        Maintained from the first optimizer step (same contract as
        DepthFrag-V2): called at the start of every ``train_loss`` after the
        first step, so at step ``t`` the teacher is the EMA of the student up
        to step ``t-1``.
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

    def _teacher_logits(self, x):
        with torch.no_grad():
            return self.teacher(x).logits.detach()

    # -------------------------------------------------------------- training
    def train_loss(self, batch, state):
        device = next(self.backbone.parameters()).device
        x = batch[0].to(device)
        y = batch[1].to(device)

        step = int(getattr(state, "batch_index", 0) or 0)
        epoch = int(getattr(state, "epoch", 0) or 0)
        if step > 0 or epoch > 0:
            self._ema_update_teacher()

        t0 = time.perf_counter()
        bo = self.backbone(x)
        logits = bo.logits[:, : self.num_classes]
        ce = F.cross_entropy(logits, y)
        out = {"ce": ce}

        tl = self._teacher_logits(x)
        flow = self._flow(bo, y=y, teacher_logits=tl)
        e = flow.hard_error
        n_stage = len(self.site_names) + 1
        stage_bce = sum(
            F.binary_cross_entropy_with_logits(flow.s_hard[l], e)
            for l in range(n_stage)
        ) / n_stage
        out["rfv2_stage_bce"] = stage_bce

        if self.use_soft:
            d = flow.soft_target
            soft_huber = sum(
                F.huber_loss(flow.s_soft[l], d, delta=self.huber_delta)
                for l in range(n_stage)
            ) / n_stage
            out["rfv2_soft_huber"] = soft_huber

        self._acc_log(flow, step, out)
        self._target_ms += (time.perf_counter() - t0) * 1000.0

        # periodic per-stage gradient-mass logging (extra backprops; gated)
        if self.log_every and step > 0 and step % self.log_every == 0:
            try:
                self._acc_grad_mass(flow, e)
            except Exception:
                pass
        return out

    # ------------------------------------------------------------------ logs
    def _acc_log(self, flow: RiskFlowV2Trace, step: int, out: dict):
        self._step_n += 1
        self._step_ce += float(out["ce"].detach())
        n_stage = len(self.site_names) + 1
        for l in range(n_stage):
            self._step_bce[l] += float(
                F.binary_cross_entropy_with_logits(
                    flow.s_hard[l].detach(), flow.hard_error.detach()))
        if self.use_soft:
            for l in range(n_stage):
                self._step_huber[l] += float(
                    F.huber_loss(flow.s_soft[l].detach(),
                                 flow.soft_target.detach(),
                                 delta=self.huber_delta))
        if flow.hard_error is not None:
            self._step_teacher_err += float(flow.hard_error.mean().detach())
            self._step_soft_mean += float(flow.soft_target.mean().detach())
        if flow.innov_hard is not None:
            for i, s in enumerate(self.site_names):
                self._step_pos[i] += float((flow.innov_hard[i] > 0).sum())
                self._step_neg[i] += float((flow.innov_hard[i] < 0).sum())
                self._step_innov[i] += float(flow.innov_hard[i].mean())

    def _acc_grad_mass(self, flow: RiskFlowV2Trace, e):
        for i, s in enumerate(self.site_names):
            L_i = F.binary_cross_entropy_with_logits(flow.s_hard[i + 1], e)
            g_cell = torch.autograd.grad(
                L_i, list(self.cells[s].parameters()), retain_graph=True,
                allow_unused=True)
            g_ada = torch.autograd.grad(
                L_i, list(self.adapters[s].parameters()), retain_graph=True,
                allow_unused=True)
            self._step_grad_cell[i] += sum(
                float(g.norm().item()) for g in g_cell if g is not None)
            self._step_grad_ada[i] += sum(
                float(g.norm().item()) for g in g_ada if g is not None)

    def _reset_step_stats(self):
        self._step_n = 0
        self._step_ce = 0.0
        self._step_teacher_err = 0.0
        self._step_soft_mean = 0.0
        self._target_ms = 0.0
        n_stage = len(self.site_names) + 1
        self._step_bce = {l: 0.0 for l in range(n_stage)}
        self._step_huber = {l: 0.0 for l in range(n_stage)}
        n = len(self.site_names)
        self._step_pos = [0.0] * n
        self._step_neg = [0.0] * n
        self._step_innov = [0.0] * n
        self._step_grad_cell = [0.0] * n
        self._step_grad_ada = [0.0] * n

    def on_epoch_end(self, epoch: int, val_metrics: dict):
        n = max(self._step_n, 1)
        row = {
            "epoch": int(epoch),
            "steps": self._step_n,
            "ce": float(self._step_ce / n),
            "teacher_err": float(self._step_teacher_err / n),
            "soft_target_mean": float(self._step_soft_mean / n),
            "target_ms": float(self._target_ms / n),
            "deployment_macs_per_example": int(self.deployment_overhead["macs_per_example"]),
        }
        for l, s in enumerate(self.site_names):
            row[f"bce_{s}"] = float(self._step_bce[l] / n)
            row[f"huber_{s}"] = float(self._step_huber[l] / n)
            row[f"innov_mean_{s}"] = float(self._step_innov[l] / n)
            row[f"pos_frac_{s}"] = float(self._step_pos[l] / max(self._step_pos[l] + self._step_neg[l], 1))
            row[f"grad_cell_{s}"] = float(self._step_grad_cell[l] / n)
            row[f"grad_adapter_{s}"] = float(self._step_grad_ada[l] / n)
        row["bce_terminal"] = float(self._step_bce[len(self.site_names)] / n)
        self._write_log([row])
        self._reset_step_stats()

    def _write_log(self, rows: List[dict]):
        try:
            run_dir = os.path.join(self.cfg["results_root"], self.cfg["run_name"])
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "riskflow_v2.jsonl"), "a") as f:
                for r in rows:
                    f.write(json.dumps(r, default=str) + "\n")
        except Exception:
            pass


__all__ = [
    "InnovationCell",
    "SoftCell",
    "RiskFlowV2Trace",
    "RiskFlowV2Method",
    "_soft_target",
    "pool_tap",
]