"""Method base class + MethodPrediction contract.

A callable model owns its backbone and any auxiliary machinery. The training
engine only talks to this interface:

* ``train_loss(batch, state) -> dict[str, Tensor]`` — named loss components;
  the engine sums them for the backward pass.
* ``predict_batch(x) -> MethodPrediction`` — logits + confidence + scores.
* ``on_epoch_start/on_epoch_end`` — method-native epoch hooks (momentum copy,
  pretraining transitions, LR-related state).
* ``optimizer_specs`` / ``scheduler_spec`` — describe the optimizer layout.
* ``inference_modules`` — exactly the modules used at deployment
  (used for parameter counts; training-only momentum encoders are excluded).

``MethodPrediction.scores`` holds every named confidence score of the method;
``confidence`` is the config-selected primary score (CE defaults to ``msp``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn

from ..backbones import Backbone, BackboneOutput, build_backbone
from .scores import compute_scores


@dataclass
class MethodPrediction:
    logits: torch.Tensor          # (B, C) main-class logits
    prediction: torch.Tensor      # (B,) int argmax over main classes
    confidence: torch.Tensor      # (B,) primary confidence
    scores: Dict[str, torch.Tensor] = field(default_factory=dict)
    aux: Dict[str, torch.Tensor] = field(default_factory=dict)


class Method(nn.Module):
    """Base selective-classification method."""

    method_name = "base"
    #: number of extra output neurons beyond C (DG/SAT add one reservation).
    output_offset = 0
    #: does the method require the global training-set index in each batch?
    needs_indices = False

    def __init__(self, train_cfg: dict):
        super().__init__()
        self.cfg = train_cfg
        self.num_classes = int(train_cfg["data"]["num_classes"])
        self.num_outputs = self.num_classes + self.output_offset
        self.score = str(train_cfg["method"].get("score", self.default_score()))
        self.available_scores = list(self.default_scores())
        backbone_cfg = train_cfg
        self.backbone = build_backbone(
            train_cfg["backbone"], self.num_outputs, backbone_cfg
        )
        assert isinstance(self.backbone, Backbone)

    # -- prediction ----------------------------------------------------------
    def predict_batch(self, x) -> MethodPrediction:
        bo = self.backbone(x)
        scores = self._scores(bo)
        conf = self._pick_confidence(bo, scores)
        pred = bo.logits[:, : self.num_classes].argmax(dim=1)
        return MethodPrediction(
            logits=bo.logits[:, : self.num_classes],
            prediction=pred,
            confidence=conf,
            scores=scores,
            aux={},
        )

    def _scores(self, bo: BackboneOutput) -> Dict[str, torch.Tensor]:
        return compute_scores(bo.logits[:, : self.num_classes], self.available_scores)

    def _pick_confidence(self, bo, scores):
        if self.score in scores:
            return scores[self.score]
        if self.score == "default" and len(scores):
            return scores[list(scores.keys())[0]]
        if "msp" in scores:
            return scores["msp"]
        raise ValueError(f"score {self.score!r} unavailable in {type(self).__name__}")

    def default_score(self) -> str:
        return "msp"

    def default_scores(self) -> Tuple[str, ...]:
        return ("msp", "entropy", "energy", "logit_margin")

    # -- training ------------------------------------------------------------
    def train_loss(self, batch, state) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def on_epoch_start(self, epoch: int) -> None:
        return None

    def on_epoch_end(self, epoch: int, val_metrics: dict) -> None:
        return None

    # -- optimizer / scheduler -----------------------------------------------
    def optimizer_specs(self) -> List[dict]:
        """Return optimizer spec dicts (first spec receives the scheduler)."""
        t = self.cfg["train"]
        return [
            {
                "params": self.parameters(),
                "kind": "sgd",
                "lr": float(t["lr"]),
                "momentum": float(t["momentum"]),
                "weight_decay": float(t["weight_decay"]),
            }
        ]

    def scheduler_spec(self) -> dict:
        t = self.cfg["train"]
        return {
            "kind": t.get("scheduler", "cosine"),
            "epochs": int(t["epochs"]),
            "milestones": list(t.get("milestones", [])),
            "gamma": float(t.get("gamma", 0.1)),
        }

    def inference_modules(self) -> Iterable[nn.Module]:
        return [self]

    def probe_mode(self):
        """Context for batch-1 shape probes: disables train-mode BN/Dropout."""
        import contextlib

        @contextlib.contextmanager
        def _probe():
            self.backbone.eval()
            try:
                yield
            finally:
                self.backbone.train(True)

        return _probe()

    def forward(self, x):
        return self.backbone(x)