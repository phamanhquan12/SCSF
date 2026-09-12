"""Differentiable loss terms for the review-aligned family.

Callers detach labels/ranks/masks/weights before calling these; each loss is
differentiable only in the intended direction (score, CE, soft-acceptance).
Every method guarantees: empty masks/pair sets produce finite zero losses;
no detached diagnostic is ever optimized by accident; no optimizer step is
hidden inside the loss.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

_EPS = 1e-8


def weighted_bce_correctness(s, targets_01, error_weight: float = 1.0):
    """``mean(w * BCEWithLogits(s, t))``; ``w`` weights incorrect examples only.

    ``targets_01`` is the detached correctness label ``1[argmax(z)==y]``.
    With the default ``error_weight=1`` this is exactly the ordinary BCE mean
    and matches the review's reference run.
    """
    t = targets_01.detach()
    w = torch.where(t <= 0.5, torch.full_like(s, float(error_weight)), torch.ones_like(s))
    bce = F.binary_cross_entropy_with_logits(s, t, reduction="none")
    return (w * bce).mean()


def rank_pair_loss(scores, errors, rho, w_diff, margin: float = 0.0,
                   temperature: float = 1.0, eps_rank: float = 1e-3,
                   eps: float = _EPS):
    """R3 pairwise ranking term.

    For incorrect ``i`` and correct ``j``::

        d_ij     = w_diff[i,j]              (stopgrad |W(r_i) - W(r_j)|)
        gate_ij  = eps_rank + 1 - rho_i
        pair_ij  = T * softplus((margin + s_i - s_j) / T)
        L_rank   = sum d_ij * gate_ij * pair_ij / (sum d_ij + eps)

    The denominator is the **sum of d**, not the sum of rho or its
    complement, so routing strength survives a constant rho. Missing either
    class in the batch yields a finite zero. The pair set is capped by config
    batch size (O(n_e * n_c), never a full O(B^2) unless the caller passes
    one).
    """
    device = scores.device
    err = errors.detach().bool()
    corr = ~err
    ei = err.nonzero(as_tuple=True)[0]
    cj = corr.nonzero(as_tuple=True)[0]
    if ei.numel() == 0 or cj.numel() == 0:
        return torch.zeros((), dtype=scores.dtype, device=device)
    d = w_diff.detach()[ei][:, cj]                      # (n_e, n_c) >= 0
    gate = (eps_rank + 1.0 - rho.detach())[ei].unsqueeze(1)
    u = scores[ei].unsqueeze(1) - scores[cj].unsqueeze(0)   # s_i - s_j
    pair = temperature * F.softplus((margin + u) / temperature)
    num = (d * gate * pair).sum()
    den = d.sum() + eps
    return num / den


def conditional_reverse_kd(probe_logits, final_logits, mask01, w_bar,
                           temperature: float = 1.0, eps: float = _EPS):
    """DTR (1,0)-conditioned reversal.

    Only samples where the probe is correct on both views and the final
    classifier is wrong on the primary view (``mask01 == 1``) are distilled.
    Teacher = ``softmax(probe/T)`` (detached); student = ``log_softmax(final
    /T)``; the KL is weighted by ``w_bar`` (normalized RC weights) and scaled
    by ``T^2``, normalized by the mask support count. Zero support -> finite
    zero.
    """
    sel = mask01.detach().bool()
    if not sel.any():
        return torch.zeros((), dtype=probe_logits.dtype, device=probe_logits.device)
    zs = probe_logits[sel] / temperature
    zf = final_logits[sel] / temperature
    teacher = torch.softmax(zs, dim=1).detach()
    log_student = F.log_softmax(zf, dim=1)
    kl = F.kl_div(log_student, teacher, reduction="none").sum(1)
    num = (w_bar.detach()[sel] * kl).sum() * (temperature ** 2)
    den = sel.sum().to(kl.dtype) + eps
    return num / den


def logsumexp_confusion(U, tau: float):
    """``tau * [logsumexp(U/tau) - log(#supported edges)]`` over a coverage.

    Empty edge set -> finite zero (never zero-risk inference).
    """
    n = U.numel()
    if n == 0:
        return torch.zeros((), dtype=U.dtype, device=U.device)
    return tau * (torch.logsumexp(U / tau, dim=0) - math.log(n))


def class_counts(y, num_classes: int):
    cnt = torch.zeros(int(num_classes), dtype=torch.long, device=y.device)
    return cnt.scatter_add_(0, y.detach().to(torch.long), torch.ones_like(y))


def class_coverage_fractions(soft_masks, y, num_classes: int):
    """``phi_a(c) = sum_{y_i=a} a_ic / n_a`` across coverages -> ``(C, n_c)``.

    Absent classes are represented by NaN, not zero risk: the caller must
    skip them (and skip their dual updates), never report them as 0.
    """
    B, nc = soft_masks.shape
    cnt = class_counts(y, num_classes).to(soft_masks.dtype)
    onehot = F.one_hot(y.detach().to(torch.long), int(num_classes)).float()
    num = onehot.t() @ soft_masks                              # (C, nc)
    phi = num / cnt.unsqueeze(1).clamp_min(1.0)
    phi = torch.where(cnt.unsqueeze(1) > 0, phi, torch.full_like(phi, math.nan))
    return phi


def confusion_utilities(probs, soft_masks, y, num_classes: int, eps: float = 1e-8):
    """``U_ab(c)`` for all class pairs -> ``(C, C, n_c)``.

    ``U_ab(c) = sum_{y_i=a} a_ic p_i(b) / (sum_{y_i=a} a_ic + eps)``. This is
    a soft proxy for the accepted ``a -> b`` confusion mass, not a hard
    selective error. The caller restricts to supported edges and accounts for
    soft accepted denominators.
    """
    onehot = F.one_hot(y.detach().to(torch.long), int(num_classes)).float()
    num = torch.einsum("bi,bj,bk->ijk", onehot, probs, soft_masks)      # (C, C, n_c)
    denom = torch.einsum("bi,bk->ik", onehot, soft_masks)               # (C, n_c)
    return num / (denom[:, None, :] + eps)


__all__ = [
    "class_coverage_fractions",
    "class_counts",
    "conditional_reverse_kd",
    "confusion_utilities",
    "logsumexp_confusion",
    "rank_pair_loss",
    "weighted_bce_correctness",
]