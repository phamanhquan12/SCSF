"""Small shared heads and ranking helpers for the NEXT5 methods."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_extra_module(module: nn.Module, base_seed: int, offset: int) -> None:
    """Initialize a training-only head from a dedicated RNG stream."""
    g = torch.Generator()
    g.manual_seed(int(base_seed) + int(offset))
    for mod in module.modules():
        if isinstance(mod, nn.Linear):
            nn.init.xavier_uniform_(mod.weight, generator=g)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
        elif isinstance(mod, nn.Embedding):
            nn.init.normal_(mod.weight, mean=0.0, std=0.02, generator=g)


class ScalarHead(nn.Module):
    """Linear(D, H) → ReLU → Linear(H, 1) producing a raw logit."""

    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden)),
            nn.ReLU(inplace=False),
            nn.Linear(int(hidden), 1),
        )

    def forward(self, h):
        return self.net(h).squeeze(-1)


class LinearClassHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(int(in_dim), int(num_classes))

    def forward(self, h):
        return self.fc(h)


class ClassVerifier(nn.Module):
    """q(h, embedding(c)) with concat [h, e, |h-e|, h⊙e]."""

    def __init__(self, in_dim: int, num_classes: int, hidden: int = 128):
        super().__init__()
        self.hidden = int(hidden)
        self.feat_proj = nn.Linear(int(in_dim), self.hidden)
        self.class_emb = nn.Embedding(int(num_classes), self.hidden)
        self.mlp = nn.Sequential(
            nn.Linear(4 * self.hidden, self.hidden),
            nn.ReLU(inplace=False),
            nn.Linear(self.hidden, 1),
        )

    def forward(self, h, class_ids):
        z = self.feat_proj(h)
        e = self.class_emb(class_ids)
        cat = torch.cat([z, e, (z - e).abs(), z * e], dim=1)
        return self.mlp(cat).squeeze(-1)

    def param_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))


def final_repr(bo) -> torch.Tensor:
    """Prefer ``final_embedding``; fall back to flattened last tap."""
    h = bo.final_embedding
    if h is None:
        raise RuntimeError("backbone did not provide final_embedding")
    if h.dim() > 2:
        h = h.reshape(h.size(0), -1)
    return h


def window_rank_weights(scores, ids, lo: float = 0.8, hi: float = 1.0):
    """Per-example prefix-AURC window weights on coverages [lo, hi].

    Rank 1 = most confident (ascending id tie-break). An error at rank ``r``
    is counted in every prefix ``k`` in the window with ``k >= r``.
    """
    s = scores.detach().reshape(-1)
    b = int(s.numel())
    device = s.device
    if b == 0:
        return s
    idv = ids.detach().reshape(-1).to(torch.int64)
    # lexsort (id asc, score desc) analogue: sort by id, then stable sort by -score
    order = torch.argsort(idv, stable=True)
    order = order[torch.argsort(s[order], descending=True, stable=True)]
    ranks = torch.empty(b, dtype=torch.long, device=device)
    ranks[order] = torch.arange(1, b + 1, device=device)
    k_lo = max(1, int(math.ceil(float(lo) * b)))
    k_hi = min(b, max(k_lo, int(math.floor(float(hi) * b))))
    denom = float(k_hi - k_lo + 1)
    r = ranks.to(torch.float32)
    lower = torch.clamp(r, min=float(k_lo))
    count = (float(k_hi) - lower + 1.0).clamp_min(0.0)
    return (count / denom).to(dtype=scores.dtype)


def sampled_rank_pair_loss(scores, errors, weights, margin: float = 0.0,
                           max_err: int = 32, max_corr: int = 32):
    """Bounded correct/error ranking: mean_valid softplus(s_wrong - s_corr + m).

    Pair identities must be detached by the caller. Empty → finite zero.
    """
    err = errors.detach().bool()
    corr = ~err
    ei = err.nonzero(as_tuple=True)[0]
    cj = corr.nonzero(as_tuple=True)[0]
    if ei.numel() == 0 or cj.numel() == 0:
        return scores.new_zeros(())
    if ei.numel() > max_err:
        ei = ei[:max_err]
    if cj.numel() > max_corr:
        cj = cj[:max_corr]
    s_e = scores[ei].unsqueeze(1)
    s_c = scores[cj].unsqueeze(0)
    pair = F.softplus(s_e - s_c + float(margin))
    w = weights.detach()[ei].unsqueeze(1).expand_as(pair)
    den = w.sum().clamp_min(1e-8)
    return (w * pair).sum() / den


__all__ = [
    "ClassVerifier",
    "LinearClassHead",
    "ScalarHead",
    "final_repr",
    "sampled_rank_pair_loss",
    "seed_extra_module",
    "window_rank_weights",
]
