"""Deep Gamblers (Liu, Li, Wang & Qiao, NeurIPS 2019).

C+1 logits: the first C are the main classes, the last is the **reservation
neuron**. Training uses the official doubling-rate objective::

    p  = softmax(logits)                       # over C+1
    gain = p[:, target]
    doubling_rate = log(gain + p[:, C] / reward)
    loss = -mean(doubling_rate)

Inference is selective-classification: the model predicts over the first C
classes and ranks examples by the official reservation neuron.  The reference
code sorts by ascending ``p_reservation``; under our higher-is-keep convention
the primary score is therefore ``dg_conf = 1 - p_reservation``.  The former
``logsumexp(main logits)`` score remains available only as a diagnostic.

Pretraining phase
-----------------
The paper reports that with a low reward the gambler objective can converge
to the trivial always-abstain point, especially on harder datasets such as
CIFAR-10, and its reference code therefore *pretrains the backbone with plain
cross-entropy for a number of epochs* before switching to the gambler loss
(``--pretrain``; defaults to 100 epochs on CIFAR-10 when ``reward < 6.1``).
This is a method-defining requirement from the original paper, so we expose it
as ``method.pretrain`` (in full epochs) and honour the paper's low-reward
default when the field is unset. During the pretrain epochs, only the first C
main-class logits are supervised; the reservation neuron is trained solely by
the gambler phase. See `docs/EMPIRICAL_CONTRACT.md` hyperparameter section.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import Method
from .ce import _dg_conf, _dg_r, _reservation
from .scores import compute_scores


class DeepGamblersMethod(Method):
    method_name = "dg"
    output_offset = 1

    #: auxiliaries tracked in Method() with the reported primary score.
    AVAILABLE = ("msp", "entropy", "energy", "logit_margin", "dg_conf", "dg_r")

    def __init__(self, train_cfg: dict):
        super().__init__(train_cfg)
        m = train_cfg["method"]
        configured = int(m["pretrain"]) if m.get("pretrain") is not None else None
        if configured is None and self._reward() < 6.1:
            # Paper's CIFAR-10 default: pretrain 100 CE epochs at low reward.
            configured = 100
        self.pretrain = configured or 0

    def default_score(self) -> str:
        return "dg_conf"

    def default_scores(self):
        return self.AVAILABLE

    def _scores(self, bo):
        scores = compute_scores(bo.logits[:, : self.num_classes], self.AVAILABLE)
        scores["dg_conf"] = _dg_conf(bo.logits, self.num_classes)
        scores["dg_r"] = _dg_r(bo.logits, self.num_classes)
        scores["reservation"] = _reservation(bo.logits, self.num_classes)
        return scores

    def _reward(self):
        return float(self.cfg["method"].get("reward", 2.2))

    def train_loss(self, batch, state) -> dict:
        x, y = batch[0], batch[1]
        raw = self.backbone(x).logits
        # Pretrain phase: plain CE over the C main classes (reservation neuron
        # is not supervised yet), matching the paper's --pretrain semantics.
        if state is not None and state.epoch < self.pretrain:
            ce = F.cross_entropy(raw[:, : self.num_classes], y)
            return {"ce": ce, "phase": 0}
        p = F.softmax(raw, dim=1)
        gain = p.gather(1, y.view(-1, 1)).squeeze(1)
        reservation = p[:, self.num_classes]
        doubling = torch.log(gain + reservation / self._reward())
        dg = -doubling.mean()
        return {"dg": dg, "phase": 1}
