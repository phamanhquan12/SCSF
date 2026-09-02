"""SelectiveNet (Geifman & El-Yaniv, ICML 2019) — faithful port.

The core module is a faithful port of the official open-source implementation
``pytorch-SelectiveNet`` (MIT). The backbone here is the transformer (the
repository's `SelectorNet<backbone>`-training is task: the SC head is small)
with three heads on the representation:

* ``classifier`` f(x) — main prediction (cross-entropy ``L_h``)
* ``selector``   g(x) — selection score in [0, 1] (sigmoid head)
* ``aux_classifier`` h(x) — auxiliary prediction head

Loss (official ``Ly = L_(f,g) + alpha_L`` form) with ``lm`` the coverage
penalty multiplier and ``alpha`` the coverage target::

    empirical_coverage = mean(g)
    empirical_risk     = mean(CE(f, y) * g) / empirical_coverage
    penalty            = lm * max(coverage - empirical_coverage, 0)^2
    L_(f,g)            = empirical_risk + penalty
    L_h                = CE(h, y)
    L                  = alpha * L_(f,g) + (1 - alpha) * L_h        (alpha=0.5)

Prediction = argmax f over the main classes; confidence = selection score g.
The native classification head of the backbone is **not part of the inference
graph** here; parameter counts therefore use only the three heads plus the
feature extractor (see empirical contract note on deployment parameter
counting for head-replacing methods).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Method, MethodPrediction
from .scores import compute_scores


class SelectiveNetHeads(nn.Module):
    """The three SelectiveNet heads over a 1-D representation."""

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(in_dim, num_classes)
        self.selector = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, 1),
            nn.Sigmoid(),
        )
        self.aux_classifier = nn.Linear(in_dim, num_classes)
        self.alpha = 0.5
        self.lm = 32.0

    def forward(self, embedding):
        return self.classifier(embedding), self.selector(embedding), self.aux_classifier(embedding)


class SelectiveNetMethod(Method):
    method_name = "selectivenet"
    needs_indices = False

    def default_score(self) -> str:
        return "selection"

    def default_scores(self):
        return ("msp", "entropy", "energy", "logit_margin", "selection")

    def __init__(self, train_cfg: dict):
        super().__init__(train_cfg)
        # build the three heads on the backbone representation
        probe = torch.zeros(1, 3, self.backbone.input_size, self.backbone.input_size)
        with torch.no_grad(), self.probe_mode():
            _, embedding = self.backbone.forward_embedding(probe)
        self.heads = SelectiveNetHeads(in_dim=int(embedding.shape[1]), num_classes=self.num_classes)
        self.heads.alpha = float(train_cfg["method"].get("alpha", 0.5))
        self.heads.lm = float(train_cfg["method"].get("lm", 32.0))

    def _forward(self, x):
        _, embedding = self.backbone.forward_embedding(x)
        return self.heads(embedding)

    def _scores(self, bo):
        # bo is unused here: scores come from the own f/g heads.
        raise NotImplementedError  # pragma: no cover - predict_batch overrides everything

    def predict_batch(self, x):
        f, g, _ = self._forward(x)
        gv = g.view(-1)
        scores = compute_scores(f, self.default_scores())
        scores["selection"] = gv
        scores["msp"] = F.softmax(f, dim=1).max(dim=1).values
        conf = scores[self.score]
        return MethodPrediction(f, f.argmax(dim=1), conf, scores)

    def train_loss(self, batch, state) -> dict:
        x, y = batch[0], batch[1]
        f, g, h = self._forward(x)
        gv = g.view(-1)
        emp_coverage = gv.mean()
        per_sample_ce = F.cross_entropy(f, y, reduction="none")
        emp_risk = (per_sample_ce * gv).mean() / emp_coverage
        penalty = self.heads.lm * torch.clamp(self.heads.alpha - emp_coverage, min=0.0) ** 2
        l_fg = emp_risk + penalty
        l_h = F.cross_entropy(h, y)
        loss = self.heads.alpha * l_fg + (1 - self.heads.alpha) * l_h
        return {"selective": loss}

__all__ = ["SelectiveNetMethod", "SelectiveNetHeads"]