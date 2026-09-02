"""Cross-entropy baseline: one native classifier, tuneable confidence score.

The ``score`` setting selects which of the CE scores is reported as the
method's primary confidence. ``{msp, entropy, energy, logit_margin}`` are all
computed and stored in ``MethodPrediction.scores`` regardless, so the
evaluator can report every score's metrics side by side.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import Method


class CEMethod(Method):
    """Standard softmax cross-entropy model with 4 confidence scores."""

    method_name = "ce"

    def train_loss(self, batch, state) -> dict:
        x, y = batch[0], batch[1]
        bo = self.backbone(x)
        ce = F.cross_entropy(bo.logits, y)
        return {"ce": ce}


def _reservation(raw_logits, num_classes):
    """Softmax probability mass on the reserved (abstention) neuron."""
    return torch.softmax(raw_logits, dim=1)[:, num_classes]


def _dg_conf(raw_logits, num_classes):
    """Official Deep Gamblers keep score: inverse reservation probability.

    The authors rank examples by ascending reservation probability.  Our
    metrics contract expects larger values to mean "keep", so ``1-p_res`` is
    the exactly rank-equivalent confidence.
    """
    return 1.0 - _reservation(raw_logits, num_classes)


def _dg_r(raw_logits, num_classes):
    """Legacy main-logit mass score retained only as a diagnostic ablation."""
    main = raw_logits[:, : num_classes]
    return torch.logsumexp(main, dim=1)


# Make these small helpers public so SAT/DG reuse them without duplication.
__all__ = ["CEMethod"]
