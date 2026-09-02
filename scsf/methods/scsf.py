"""SCSF: backbone MetaCalibrator predicting True Class Probability (TCP).

This is the **scientific-integrity core**. v1 ``train_scsf.py`` exposed an
``end_to_end`` boolean; the default ``end_to_end=False`` path claimed
"post-hoc: block gradients" but only detached the **logits** — the tapped
pool4/pool5 features still carried gradients into the backbone, so the
meta-loss could move the classifier despite the README's claim that the
calibrator is post-hoc. We audit and fix that here.

Gradient semantics are now an explicit three-value ``mode`` (locked by tests):

* ``posthoc``               — detach **every** tapped feature **and** the
  logits; the meta-loss can only update the MetaCalibrator itself (default).
* ``e2e``                   — no detach anywhere; the meta-loss may also
  update the backbone (an explicit, disclosed joint-training mode).
* ``legacy_partial_detach``  — reproduces v1's ``end_to_end=False`` exactly
  (tapped features flow, logits detached). Deprecated; never the default.

The deprecated ``end_to_end`` constructor kwarg maps ``True -> e2e``,
``False -> legacy_partial_detach`` with a ``DeprecationWarning``.

Meta-loss: MSE(beta-hat(x), TCP) where TCP = softmax(logits)[true class] is
detached; a cosine meta-weight schedule (``init_meta_weight`` -> ``min_meta_weight``)
over the joint phase gates its magnitude; ``pretrain`` epochs train CE only.

Architecture & dims follow v1: taps are pooled to at most 2x2 then flattened;
the MLP is Linear->ReLU->Dropout(0.3) repeated through 1024/512/256/128 and a
sigmoid output, Xavier-initialized.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones import adaptive_flatten
from .base import Method, MethodPrediction
from .scores import compute_scores, msp, tcp

MODES = ("posthoc", "e2e", "legacy_partial_detach")


def meta_weight_cosine(epoch: int, pretrain: int, total_epochs: int,
                       start_weight: float = 1.0, min_weight: float = 1e-4) -> float:
    """Cosine-decayed meta loss weight across the joint phase (v1 default)."""
    if total_epochs <= pretrain or epoch < pretrain:
        return 0.0
    progress = (epoch - pretrain) / (total_epochs - pretrain)
    progress = min(max(progress, 0.0), 1.0)
    return min_weight + 0.5 * (start_weight - min_weight) * (1.0 - math.cos(math.pi * progress))


class MetaCalibrator(nn.Module):
    """The 5-layer MetaCalibrator with explicit gradient semantics."""

    def __init__(
        self,
        feature_dims: Sequence[int],
        logit_dim: int,
        hidden_dim: int = 256,
        mode: str = "posthoc",
        logits_only: bool = False,
        end_to_end: Optional[bool] = None,
    ):
        super().__init__()
        if end_to_end is not None:
            warnings.warn(
                "MetaCalibrator(end_to_end=...) is deprecated; pass "
                "mode='posthoc' | 'e2e' | 'legacy_partial_detach' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            mode = "e2e" if end_to_end else "legacy_partial_detach"
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.logits_only = bool(logits_only)
        self.feature_dims = [int(d) for d in feature_dims]
        self.logit_dim = int(logit_dim)
        input_dim = (self.logit_dim if logits_only
                     else sum(self.feature_dims) + self.logit_dim)

        layers = []
        for in_d, out_d in [(input_dim, 1024), (1024, 512), (512, 256), (256, 128)]:
            layers += [nn.Linear(in_d, out_d), nn.ReLU(inplace=False), nn.Dropout(0.3)]
        layers += [nn.Linear(128, 1), nn.Sigmoid()]
        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _feats(self, tapped_features):
        """(B, D) of concatenated pooled-then-flattened tap features."""
        parts = [adaptive_flatten(f, out=2) for f in tapped_features]
        return torch.cat(parts, dim=1)

    def forward(self, tapped_features: Sequence[torch.Tensor], logits: torch.Tensor):
        if self.logits_only:
            if self.mode == "e2e":
                combined = logits
            else:
                combined = logits.detach()
        else:
            feats = self._feats(tapped_features)
            if self.mode == "posthoc":
                feats = feats.detach()
            if self.mode in ("posthoc", "legacy_partial_detach"):
                combined = torch.cat([feats, logits.detach()], dim=1)
            else:
                combined = torch.cat([feats, logits], dim=1)
        return self.network(combined).squeeze(-1)  # (B,)


class SCSFMethod(Method):
    method_name = "scsf"

    def default_score(self) -> str:
        return "scsf_conf"

    def default_scores(self):
        return ("msp", "entropy", "energy", "logit_margin", "scsf_conf")

    def __init__(self, train_cfg: dict):
        super().__init__(train_cfg)
        m = train_cfg["method"]
        self.mode = str(m.get("mode", "posthoc"))
        if self.mode not in MODES:
            raise ValueError(f"scsf.mode must be one of {MODES}")
        #: semantic role names of the tap pair (v1 order: older tap first = top_l2)
        self.tap_roles = list(m.get("taps", ["top_l2", "top_l1"]))
        self.pretrain = int(m.get("pretrain", 0))
        self.init_meta_weight = float(m.get("init_meta_weight", 1.0))
        self.min_meta_weight = float(m.get("min_meta_weight", 1e-4))
        self.error_weight = float(m.get("error_weight", 1.0))
        self.meta_lr = float(m.get("meta_lr", 1e-4))
        self.hidden_dim = int(m.get("hidden_dim", 256))
        self.logits_only = bool(m.get("logits_only", False))
        self._calib = None
        self._probe()

    def _probe(self):
        """Learn tap dims from a deterministic probe forward (v1 dims stay exact)."""
        _, shapes = self.backbone.probe_tap_shapes(batch=1)
        feature_dims = []
        for role in self.tap_roles:
            name = self.backbone.roles[role]
            shape = torch.Size(shapes[name])
            # pooled to at most 2x2 then flattened (matches v1)
            h, w = shape[-2], shape[-1]
            spatial = 4 if (h > 1 or w > 1) else max(h * w, 1)
            feature_dims.append(int(shape[1] * spatial))
        self._calib = MetaCalibrator(
            feature_dims=feature_dims,
            logit_dim=self.num_classes,
            hidden_dim=self.hidden_dim,
            mode=self.mode,
            logits_only=self.logits_only,
        )

    def _taps_and_logits(self, x):
        bo = self.backbone(x)
        taps = [bo.role(self.backbone, role) for role in self.tap_roles]
        return taps, bo.logits

    def predict_batch(self, x):
        taps, logits = self._taps_and_logits(x)
        with torch.no_grad():
            conf = self._calib(taps, logits)
        scores = compute_scores(logits, self.default_scores())
        scores["scsf_conf"] = conf
        return MethodPrediction(logits, logits.argmax(dim=1), conf, scores)

    def train_loss(self, batch, state) -> dict:
        x, y = batch[0], batch[1]
        taps, logits = self._taps_and_logits(x)
        ce = F.cross_entropy(logits, y)
        out = {"ce": ce}
        if state.epoch >= self.pretrain:
            conf = self._calib(taps, logits)
            target = tcp(logits, y)                       # detached TCP
            meta = F.mse_loss(conf, target) * self.error_weight
            w = meta_weight_cosine(
                state.epoch, self.pretrain, int(self.cfg["train"]["epochs"]),
                self.init_meta_weight, self.min_meta_weight,
            )
            out["meta"] = w * meta
            out["meta_weight"] = torch.tensor(w, device=logits.device)
            out["meta_raw"] = meta.detach()
        return out

    def optimizer_specs(self):
        t = self.cfg["train"]
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        calib_params = [p for p in self._calib.parameters() if p.requires_grad]
        specs = [
            {
                "params": backbone_params,
                "kind": t.get("optimizer", "sgd"),
                "lr": float(t["lr"]),
                "momentum": float(t.get("momentum", 0.9)),
                "weight_decay": float(t.get("weight_decay", 5e-4)),
            },
            {
                "params": calib_params,
                "kind": "adam",
                "lr": self.meta_lr,
                "momentum": 0.0,
                "weight_decay": 0.0,
            },
        ]
        return [s for s in specs if s["params"]]


__all__ = ["SCSFMethod", "MetaCalibrator", "meta_weight_cosine", "MODES"]