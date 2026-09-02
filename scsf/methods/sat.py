"""Self-Adaptive Training with selective motivation (Ziyin et al., NeurIPS 2020).

Faithful port of ``SAT-selective-cls/loss.py`` (MIT). The class-local momentum
history ``prob_history[index]`` mixes each sample's running soft label with its
current prediction::

    cond    = updated[index] == 1
    prior   = where(cond, prob_history[index], onehot(y))
    prob    = mom * prior + (1 - mom) * softmax(logits)[:main_classes]
    …
    soft_label[:, y] = prob[:, y]
    soft_label[:, C] = 1 - prob[:, y]      # motivation is 1 - true-class prob
    loss = mean(-sum(log_softmax(logits) * l1_normalize(soft_label), 1))

The history buffer is a registered buffer so checkpoints carry it; it is keyed
by the **global official-training-fold index** returned by our dataset wrapper
(``batch[2]``), matching the official per-example accounting.

Inference uses the official selective score ``confidence = 1 - softmax(logits)[C]``
(higher motivation = more confidence = keep; ``eval.py`` ``eval_.py`` line 541).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Method
from .ce import _reservation
from .scores import compute_scores


class SelfAdaptiveTrainingLoss(nn.Module):
    """Official SAT loss with a device-agnostic momentum history buffer."""

    def __init__(self, num_examples: int, num_classes: int, mom: float = 0.9):
        super().__init__()
        self.register_buffer("prob_history", torch.zeros(int(num_examples), int(num_classes)))
        self.register_buffer("updated", torch.zeros(int(num_examples), dtype=torch.int64))
        self.mom = float(mom)
        self.num_classes = int(num_classes)

    def _update_prob(self, prob, index, y):
        onehot = torch.zeros_like(prob)
        onehot[torch.arange(y.shape[0]), y] = 1
        prob_history = self.prob_history[index].to(prob.device)
        cond = (self.updated[index] == 1).to(prob.device).unsqueeze(-1).expand_as(prob)
        prob_mom = torch.where(cond, prob_history, onehot)
        prob_mom = self.mom * prob_mom + (1 - self.mom) * prob
        self.updated[index] = 1
        self.prob_history[index] = prob_mom.detach().to(self.prob_history.device)
        return prob_mom

    def forward(self, logits, y, index):
        prob = F.softmax(logits.detach()[:, : self.num_classes], dim=1)
        prob = self._update_prob(prob, index, y)

        soft_label = torch.zeros_like(logits)
        soft_label[torch.arange(y.shape[0]), y] = prob[torch.arange(y.shape[0]), y]
        soft_label[:, -1] = 1 - prob[torch.arange(y.shape[0]), y]
        soft_label = F.normalize(soft_label, dim=1, p=1)
        loss = torch.sum(-F.log_softmax(logits, dim=1) * soft_label, dim=1)
        return torch.mean(loss)


class SATMethod(Method):
    method_name = "sat"
    output_offset = 1
    needs_indices = True

    def default_score(self) -> str:
        return "sat_conf"

    def default_scores(self):
        return ("msp", "entropy", "energy", "logit_margin", "sat_conf")

    def __init__(self, train_cfg: dict):
        super().__init__(train_cfg)
        self.pretrain = int(train_cfg["method"].get("pretrain", 0))
        # history buffer spans the full official training fold (global indices)
        self.sat = SelfAdaptiveTrainingLoss(
            num_examples=int(train_cfg["data"].get("official_train_size", 50000)),
            num_classes=self.num_classes,
            mom=float(train_cfg["method"].get("sat_mom", 0.9)),
        )

    def _scores(self, bo):
        scores = compute_scores(bo.logits[:, : self.num_classes], self.default_scores())
        scores["sat_conf"] = 1.0 - _reservation(bo.logits, self.num_classes)
        scores["reservation"] = _reservation(bo.logits, self.num_classes)
        return scores

    def train_loss(self, batch, state) -> dict:
        x, y, index = batch[0], batch[1], batch[2]
        raw = self.backbone(x).logits
        if state is not None and state.epoch < self.pretrain:
            return {"ce": F.cross_entropy(raw[:, : self.num_classes], y), "phase": 0}
        sat_loss = self.sat(raw, y, index.to(raw.device))
        return {"sat": sat_loss, "phase": 1}


__all__ = ["SATMethod", "SelfAdaptiveTrainingLoss"]
