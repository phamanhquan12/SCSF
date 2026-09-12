"""DTR-SCSF — Depth-Transition Repair (§6 spec; review pp. 8–11).

Four-state transition head and conditional shallow→final distillation. A probe
classifier (GAP -> Linear(C)) reads the ``top_l2`` tap. ``tS`` = probe correct
on **both** views, ``tL`` = final classifier correct on the primary view; the
``TransitionCalibrator`` head emits the fixed state order ``[00,01,10,11]``
(index ``2*tS + tL``), and the deployed confidence is the final-correctness
marginal ``softmax(Q)[01] + softmax(Q)[11]`` — never "either depth correct".

Objective (warmup inside the epoch budget)::

    L = CE_final + beta * CE_probe
        + lambda(epoch) * CE_four_state(2*tS + tL)
        + gamma * L_KD

``L_KD`` is the (1,0)-conditioned reverse distillation: only samples where the
probe is right on both views and the final classifier is wrong on the primary
distill the probe's softmax into the final classifier (``T²·KL``, harmonic-RC
weighted, ``/ (Σ d + eps)``). No KD on (0,1). Everything the spec calls "must
be detached" is detached; the student path may still move shared backbone
features, which is intended, not a teacher-gradient leak.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Method, MethodPrediction
from .rc_training.calibration import TransitionCalibrator
from .rc_training.losses import conditional_reverse_kd
from .rc_training.rc_weights import harmonic_weights, normalize_clip
from .rc_training.schedules import meta_weight_cosine_decay, temperature_schedule
from .scores import compute_scores


class DTRProbe(nn.Module):
    """One linear probe: GAP -> Linear(C) over the probe tap features."""

    def __init__(self, channels: int, num_classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(int(channels), int(num_classes)),
        )


class DTRSCSFMethod(Method):
    method_name = "dtr_scsf"
    needs_two_views = True

    def default_score(self) -> str:
        return "dtr_conf"

    def default_scores(self):
        return ("msp", "entropy", "energy", "logit_margin", "dtr_conf")

    def __init__(self, train_cfg: dict):
        super().__init__(train_cfg)
        m = train_cfg["method"]
        self.tap_roles = list(m.get("taps", ["top_l2", "top_l1"]))
        self.probe_role = str(m.get("probe_role", "top_l2"))
        if self.probe_role not in self.tap_roles:
            raise ValueError(f"dtr.probe_role {self.probe_role!r} not in taps {self.tap_roles}")
        self.warmup_epochs = int(m.get("warmup_epochs", 0))
        self.beta = float(m.get("beta", 1.0))
        self.gamma = float(m.get("gamma", 1.0))
        self.init_meta_weight = float(m.get("init_meta_weight", 1.0))
        self.min_meta_weight = float(m.get("min_meta_weight", 1e-4))
        self.meta_lr = float(m.get("meta_lr", 1e-4))
        self.hidden_dims = tuple(int(d) for d in m.get("calibrator_hidden_dims", (1024, 512, 256, 128)))
        self.dropout = float(m.get("calibrator_dropout", 0.3))
        self.kd_T0 = float(m.get("kd_T0", 2.0))
        self.kd_T1 = float(m.get("kd_T1", 2.0))
        self.use_four_state = bool(m.get("use_four_state", True))
        self.use_kd = bool(m.get("use_kd", True))
        self.use_rc_weights = bool(m.get("use_rc_weights", True))
        self.kd_all = bool(m.get("kd_all", False))
        self.kd_correct_probe_only = bool(m.get("kd_correct_probe_only", False))
        self.random_aux_states = bool(m.get("random_aux_states", False))
        self.random_aux_seed = int(m.get("random_aux_seed", 13))
        self.wbar_clip_max = m.get("wbar_clip_max")
        self._calib = None
        self._probe = None
        self._probe_dims()
        self._calib = TransitionCalibrator(
            feature_dims=self._feature_dims,
            logit_dim=self.num_classes,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        )
        self._probe = DTRProbe(self._probe_channels, self.num_classes)

    def _probe_dims(self):
        """Learn tap dims for the calibrator + probe channels from a probe forward."""
        _, shapes = self.backbone.probe_tap_shapes(batch=1)
        feature_dims = []
        for role in self.tap_roles:
            name = self.backbone.roles[role]
            shape = torch.Size(shapes[name])
            h, w = shape[-2], shape[-1]
            spatial = 4 if (h > 1 or w > 1) else max(h * w, 1)
            feature_dims.append(int(shape[1] * spatial))
        self._feature_dims = feature_dims
        probe_shape = torch.Size(shapes[self.backbone.roles[self.probe_role]])
        self._probe_channels = int(probe_shape[1])

    def _taps_role(self, bo, role):
        return bo.role(self.backbone, role)

    def _extract(self, bo):
        return [bo.role(self.backbone, role) for role in self.tap_roles]

    def predict_batch(self, x):
        bo = self.backbone(x)
        taps = self._extract(bo)
        with torch.no_grad():
            conf = self._calib.final_correctness_confidence(taps, bo.logits).detach()
        scores = compute_scores(bo.logits, self.default_scores())
        scores["dtr_conf"] = conf
        pred = bo.logits.argmax(dim=1)
        return MethodPrediction(bo.logits, pred, conf, scores)

    def _aux_bit(self, idx, salt) -> torch.Tensor:
        z = (idx.to(torch.int64) * 2654435761 + salt) & 0xFFFFFFFF
        return (z % 2).float()

    def train_loss(self, batch, state) -> dict:
        x, v, y, idx = batch[0], batch[1], batch[2], batch[3]
        bo_u = self.backbone(x)
        bo_v = self.backbone(v)
        taps_u = self._extract(bo_u)
        pu = self._probe(self._taps_role(bo_u, self.probe_role))
        pu_v = self._probe(self._taps_role(bo_v, self.probe_role))

        u_corr = (bo_u.logits.argmax(dim=1) == y)
        v_corr = (bo_v.logits.argmax(dim=1) == y)
        pc_u = (pu.argmax(dim=1) == y)
        pc_v = (pu_v.argmax(dim=1) == y)

        out = {"ce": F.cross_entropy(bo_u.logits, y)}
        if self.beta:
            out["beta_ce_probe"] = self.beta * F.cross_entropy(pu, y)

        joint = state.epoch >= self.warmup_epochs
        if not joint:
            return out

        Q_u = self._calib(taps_u, bo_u.logits)                    # (B, 4)
        soft_u = Q_u.softmax(dim=1).detach()
        conf = soft_u[:, 1] + soft_u[:, 3]                        # final-correctness

        if self.random_aux_states:
            ts = self._aux_bit(idx, 101)
            tl = self._aux_bit(idx, 202)
        else:
            ts = (pc_u & pc_v).float()
            tl = u_corr.float()
        target_idx = (2 * ts + tl).to(torch.long)

        if self.use_four_state:
            w = meta_weight_cosine_decay(
                state.epoch, self.warmup_epochs,
                int(self.cfg["train"]["epochs"]),
                self.init_meta_weight, self.min_meta_weight)
            out["four_state"] = w * F.cross_entropy(Q_u, target_idx)
            out["meta_weight"] = torch.tensor(w, device=bo_u.logits.device)

        if self.use_kd:
            kd_T = temperature_schedule(state.epoch, int(self.cfg["train"]["epochs"]),
                                        self.kd_T0, self.kd_T1)
            if self.random_aux_states:
                mask01 = torch.zeros_like(u_corr).float()
            elif self.kd_all:
                mask01 = torch.ones_like(u_corr).float()
            elif self.kd_correct_probe_only:
                mask01 = (pc_u & pc_v).float()
            else:
                mask01 = (pc_u & pc_v & (~u_corr)).float()
            if self.use_rc_weights:
                w_hat = harmonic_weights(conf, idx)
                if self.wbar_clip_max is not None:
                    w_bar = normalize_clip(
                        harmonic_weights(conf, idx), ~u_corr,
                        float(self.wbar_clip_max))
                else:
                    w_bar = normalize_clip(harmonic_weights(conf, idx), ~u_corr)
            else:
                w_bar = torch.ones_like(conf)
            out["kd"] = self.gamma * conditional_reverse_kd(
                pu, bo_u.logits, mask01, w_bar, temperature=kd_T)
            out["diag_mask_frac"] = mask01.mean().detach()
            out["diag_probe_corr_frac"] = (pc_u & pc_v).float().mean().detach()
            out["diag_state11_frac"] = target_idx.eq(3).float().mean().detach()
        return out

    def optimizer_specs(self):
        t = self.cfg["train"]
        shared = [p for p in self.backbone.parameters() if p.requires_grad]
        shared += [p for p in self._probe.parameters() if p.requires_grad]
        calib = [p for p in self._calib.parameters() if p.requires_grad]
        specs = [
            {
                "params": shared,
                "kind": t.get("optimizer", "sgd"),
                "lr": float(t["lr"]),
                "momentum": float(t.get("momentum", 0.9)),
                "weight_decay": float(t.get("weight_decay", 5e-4)),
            },
            {
                "params": calib,
                "kind": "adam",
                "lr": self.meta_lr,
                "momentum": 0.0,
                "weight_decay": 0.0,
            },
        ]
        return [s for s in specs if s["params"]]

    def inference_modules(self):
        return [self.backbone, self._calib]


__all__ = ["DTRProbe", "DTRSCSFMethod"]