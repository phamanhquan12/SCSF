"""Deep Gamblers (Liu, Li, Wang & Qiao, NeurIPS 2019).

C+1 logits: the first C are the main classes, the last is the **reservation
neuron**. Training uses the official doubling-rate objective::

    p  = softmax(logits)                       # over C+1
    gain = p[:, target]
    doubling_rate = log(gain + p[:, C] / reward)
    loss = -mean(doubling_rate)

Inference is selective-classification: the model must keep working like a
plain classifier over the first C classes (argmax confidence ∈ [0, C)), while
``R = logsumexp(main logits)`` is the paper's reject statistic used to rank
abstention (higher = keep). We report the standard CE scores over the main
classes as auxiliary scores too.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import Method
from .ce import _dg_r, _reservation
from .scores import compute_scores


class DeepGamblersMethod(Method):
    method_name = "dg"
    output_offset = 1

    #: auxiliaries tracked in Method() with the reported primary score.
    AVAILABLE = ("msp", "entropy", "energy", "logit_margin", "dg_r")

    def default_score(self) -> str:
        return "dg_r"

    def default_scores(self):
        return self.AVAILABLE

    def _scores(self, bo):
        scores = compute_scores(bo.logits[:, : self.num_classes], self.AVAILABLE)
        scores["dg_r"] = _dg_r(bo.logits, self.num_classes)
        scores["reservation"] = _reservation(bo.logits, self.num_classes)
        return scores

    def _reward(self):
        return float(self.cfg["method"].get("reward", 2.2))

    def train_loss(self, batch, state) -> dict:
        x, y = batch[0], batch[1]
        raw = self.backbone(x).logits
        p = F.softmax(raw, dim=1)
        gain = p.gather(1, y.view(-1, 1)).squeeze(1)
        reservation = p[:, self.num_classes]
        doubling = torch.log(gain + reservation / self._reward())
        dg = -doubling.mean()
        return {"dg": dg}