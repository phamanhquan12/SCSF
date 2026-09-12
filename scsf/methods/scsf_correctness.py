"""SCSF review-aligned correctness baseline: softmax-CE + correctness-BCE.

This is the **review-aligned** family baseline. It shares the exact structure
of legacy ``SCSFMethod`` (tapped pair == ``top_l2, top_l1``, 5-layer
MetaCalibrator MLP, cosine meta-weight schedule) but replaces the target and
the gradient rule to match the review:

* **Target is correctness** ``1[argmax(z) == y]``, **not** the softmax TCP.
  The BCE-with-logits head predicts ``P(final classifier correct)`` directly.
* **Gradient rule is fixed**: tapped features flow into the head (attached),
  classifier logits are stop-gradient inside the head. This is the review
  rule for all new methods and is intentionally *not* configurable here (see
  the audit for why legacy modes stay unchanged).
* **Schedule is correctly oriented**: ``meta_weight_cosine_decay`` starts at
  ``init_meta_weight = 1`` and decays to ``min_meta_weight = 1e-4`` (legacy
  ``meta_weight_cosine`` is inverted; audit ``docs/audits/meta_weight_cosine_audit.md``).

The raw logit ``s`` and the head output ``sigmoid(s)`` are both exposed as
confidence scores. Per the review's precision note, the (1,0) "either-depth
correct" is **not** used here; ``dtr_scsf`` is the only method whose head has
a four-way transition output.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import Method, MethodPrediction
from .rc_training.calibration import RawScoreCalibrator
from .rc_training.losses import weighted_bce_correctness
from .rc_training.schedules import meta_weight_cosine_decay
from .scores import compute_scores


class SCSFCorrectnessMethod(Method):
    method_name = "scsf_correctness"

    def default_score(self) -> str:
        return "scsf_corr"

    def default_scores(self):
        return ("msp", "entropy", "energy", "logit_margin", "scsf_corr_raw", "scsf_corr")

    def __init__(self, train_cfg: dict):
        super().__init__(train_cfg)
        m = train_cfg["method"]
        #: semantic role names of the tap pair (older tap first = top_l2)
        self.tap_roles = list(m.get("taps", ["top_l2", "top_l1"]))
        self.pretrain = int(m.get("pretrain", 0))
        self.init_meta_weight = float(m.get("init_meta_weight", 1.0))
        self.min_meta_weight = float(m.get("min_meta_weight", 1e-4))
        self.error_weight = float(m.get("error_weight", 1.0))
        self.meta_lr = float(m.get("meta_lr", 1e-4))
        self.hidden_dims = tuple(int(d) for d in m.get("calibrator_hidden_dims", (1024, 512, 256, 128)))
        self.dropout = float(m.get("calibrator_dropout", 0.3))
        self._calib = None
        self._probe()

    def _probe(self):
        """Learn tap dims from a deterministic probe forward (matches SCSF v1 dims)."""
        _, shapes = self.backbone.probe_tap_shapes(batch=1)
        feature_dims = []
        for role in self.tap_roles:
            name = self.backbone.roles[role]
            shape = torch.Size(shapes[name])
            h, w = shape[-2], shape[-1]
            spatial = 4 if (h > 1 or w > 1) else max(h * w, 1)
            feature_dims.append(int(shape[1] * spatial))
        self._calib = RawScoreCalibrator(
            feature_dims=feature_dims,
            logit_dim=self.num_classes,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        )

    def _taps_and_logits(self, x):
        bo = self.backbone(x)
        taps = [bo.role(self.backbone, role) for role in self.tap_roles]
        return taps, bo.logits

    def predict_batch(self, x):
        taps, logits = self._taps_and_logits(x)
        with torch.no_grad():
            raw = self._calib(taps, logits).detach()
            conf = torch.sigmoid(raw)
        scores = compute_scores(logits, self.default_scores())
        scores["scsf_corr_raw"] = raw
        scores["scsf_corr"] = conf
        pred = logits.argmax(dim=1)
        return MethodPrediction(logits, pred, conf, scores)

    def train_loss(self, batch, state) -> dict:
        x, y = batch[0], batch[1]
        taps, logits = self._taps_and_logits(x)
        ce = F.cross_entropy(logits, y)
        out = {"ce": ce}
        if state.epoch >= self.pretrain:
            s = self._calib(taps, logits)                     # raw logit (B,)
            target01 = (logits.argmax(dim=1) == y).float()    # correctness, not TCP
            meta = weighted_bce_correctness(s, target01, self.error_weight)
            w = meta_weight_cosine_decay(
                state.epoch, self.pretrain,
                int(self.cfg["train"]["epochs"]),
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

    def inference_modules(self):
        return [self.backbone, self._calib]


__all__ = ["SCSFCorrectnessMethod"]