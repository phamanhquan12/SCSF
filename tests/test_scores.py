"""Score primitives: everything higher = more confident (keep)."""

import math

import torch

from scsf.methods.scores import compute_scores, energy, entropy, logit_margin, msp, negative_entropy


def test_msp_peaks_at_the_top_class():
    logits = torch.tensor([[1.0, 0.0, -1.0], [0.0, 2.0, 0.0], [-3.0, -3.0, 0.0]])
    s = msp(logits)
    assert torch.allclose(s, torch.softmax(logits, dim=1).max(dim=1).values)


def test_entropy_primitive_and_negative_entropy_wrapper():
    flat = torch.tensor([[1.0, 1.0, 1.0]])  # H = ln 3, maximally uncertain
    peak = torch.tensor([[5.0, 0.0, 0.0]])  # near-onehot, near-zero H
    assert math.isclose(float(entropy(flat)), math.log(3.0), abs_tol=1e-6)
    assert float(entropy(peak)) < 0.1
    assert entropy(peak) < entropy(flat)
    # raw `entropy` is the uncertainty primitive; the confidence-style wrapper
    # is negative_entropy, which SCORE_FUNCS["entropy"] points at.
    assert torch.allclose(negative_entropy(flat), -entropy(flat))
    assert negative_entropy(peak) > negative_entropy(flat)
    from scsf.methods.scores import SCORE_FUNCS

    assert SCORE_FUNCS["entropy"] is negative_entropy
    out = compute_scores(peak, ("entropy",))
    assert torch.allclose(out["entropy"], -entropy(peak))


def test_energy_is_temperature_scaled_logsumexp():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.allclose(energy(logits), torch.logsumexp(logits, dim=1))
    assert torch.allclose(energy(logits, temperature=2.0), 2.0 * torch.logsumexp(logits / 2.0, dim=1))


def test_logit_margin_matches_explicit_top2_gap():
    logits = torch.tensor([[0.5, 2.0, 1.0], [1.0, 1.0, 0.0]])
    top2, _ = torch.topk(logits, k=2, dim=1)
    assert torch.allclose(logit_margin(logits), top2[:, 0] - top2[:, 1])


def test_compute_scores_returns_exactly_the_requested_names():
    logits = torch.randn(4, 5)
    out = compute_scores(logits, ("msp", "energy"))
    assert set(out) == {"msp", "energy"}
    assert out["msp"].shape == (4,)