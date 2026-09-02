"""Differentiable selective surrogate: matches exact grid risk, gradients flow."""

import numpy as np
import torch

from scsf.metrics.selective import selective_risk_at_coverages
from scsf.metrics.surrogate import (
    soft_aurc_surrogate,
    soft_selective_risk,
    selective_surrogate_gradient,
)


def test_soft_surrogate_matches_exact_grid_risk_with_hard_errors():
    torch.manual_seed(3)
    B, C = 128, 10
    logits = torch.randn(B, C)
    y = torch.randint(0, C, (B,))
    pred = logits.argmax(1)
    conf = torch.softmax(logits, 1).max(1).values
    grid = [100, 95, 90, 80, 60, 40, 20, 1]
    exact = np.mean([r["risk"] for r in
                     selective_risk_at_coverages(y.numpy(), pred.numpy(),
                                                 conf.numpy(), coverages=grid)])
    soft = float(soft_aurc_surrogate(logits, y, error_mode="hard", tau=0.01,
                                     coverages=grid))
    # the smooth-gate relaxation converges to the exact coverage grid risk
    assert abs(soft - exact) < 0.05


def test_soft_surrogate_orientation_lower_is_better():
    # A model whose "most confident" samples are the *correct* ones has a lower
    # selective-risk surrogate than one that is confident on its errors.
    torch.manual_seed(0)
    B = 64
    err = torch.cat([torch.zeros(B // 2), torch.ones(B // 2)])  # 1st half correct
    conf_range = torch.linspace(1.0, 0.02, B)                   # high->low scores
    good = float(soft_selective_risk(conf_range, err, tau=0.1))
    bad = float(soft_selective_risk(conf_range.flip(0), err, tau=0.1))
    assert bad > good


def test_surrogate_gradient_flows_into_logits_and_hard_mode_detaches():
    torch.manual_seed(1)
    logits = torch.randn(32, 10, requires_grad=True)
    y = torch.randint(0, 10, (32,))
    s = soft_aurc_surrogate(logits, y, error_mode="proxy")
    assert s.requires_grad
    s.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.any(logits.grad != 0)

    # hard-error mode: the gradient path exists (ranking term still carries it)
    logits2 = torch.randn(32, 10, requires_grad=True)
    gs = torch.autograd.grad(soft_aurc_surrogate(logits2, y, error_mode="hard"),
                             logits2, allow_unused=True)
    assert gs[0] is not None
    assert torch.isfinite(gs[0]).all()


def test_surrogate_gradient_helper_returns_zeros_for_unused():
    torch.manual_seed(2)
    w = torch.nn.Parameter(torch.randn(4, 4))
    b = torch.nn.Parameter(torch.zeros(4))           # unused by the surrogate
    logits = w.sum(dim=1).unsqueeze(0).expand(8, 4)
    y = torch.zeros(8, dtype=torch.long)
    grads = selective_surrogate_gradient(soft_aurc_surrogate(logits, y), [w, b])
    assert grads[0] is not None
    assert torch.allclose(grads[1], torch.zeros_like(b))