"""Inference score primitives shared by every method.

All scores are **confidence-style**: higher = more confident = keep. The
selective evaluator converts to uncertainty ``u = -confidence``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def msp(logits):
    return F.softmax(logits, dim=1).max(dim=1).values


def entropy(logits):
    p = F.softmax(logits, dim=1)
    return -(p * torch.log(p.clamp_min(1e-12))).sum(dim=1)


def negative_entropy(logits):
    """Confidence-style entropy score (higher = more confident)."""
    return -entropy(logits)


def energy(logits, temperature: float = 1.0):
    """``T * logsumexp(logits/T)`` — the standard energy score."""
    return temperature * torch.logsumexp(logits / temperature, dim=1)


def logit_margin(logits):
    top2 = torch.topk(logits, k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def tcp(logits, targets):
    """True-class probability (softmax of the true class), detached."""
    with torch.no_grad():
        p = F.softmax(logits, dim=1)
        return p.gather(1, targets.view(-1, 1)).squeeze(1)


SCORE_FUNCS = {
    "msp": msp,
    "entropy": negative_entropy,
    "energy": energy,
    "logit_margin": logit_margin,
}


def compute_scores(logits, score_names) -> dict:
    out = {}
    for name in score_names:
        fn = SCORE_FUNCS.get(name)
        if fn is not None:
            out[name] = fn(logits)
    return out