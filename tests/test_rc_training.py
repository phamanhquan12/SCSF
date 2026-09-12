"""Shared RC-training primitive tests (authored for the review-aligned family).

**STATUS: NOT RUN.** These exercises only make sense with torch + a GPU-free
CPU install; the delivery machine (python3.14 stdlib-only) has neither. They
are committed as authored tests to be executed on the experiment server, and
the doc/docs/VALIDATION_STATUS.md records that verification is still pending.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from scsf.methods import meta_weight_cosine
from scsf.methods.rc_training.calibration import (
    RawScoreCalibrator,
    TransitionCalibrator,
)
from scsf.methods.rc_training.losses import (
    class_coverage_fractions,
    conditional_reverse_kd,
    confusion_utilities,
    logsumexp_confusion,
    rank_pair_loss,
    weighted_bce_correctness,
)
from scsf.methods.rc_training.rc_weights import (
    harmonic_weights,
    harmonic_weights_at_ranks,
    normalize_clip,
    normalize_mean1,
    rank_of,
)
from scsf.methods.rc_training.schedules import (
    meta_weight_cosine_decay,
    temperature_schedule,
)
from scsf.methods.rc_training.softquantile import (
    soft_coverage_threshold,
    solve_soft_thresholds,
)
from scsf.methods.rc_training.state import DualState, EdgeSupport, PhaseState, RhoCache
from scsf.methods.rc_training.views import apply_transform_gated, view_seed


# --------------------------------------------------------------------------
# schedules
# --------------------------------------------------------------------------


def test_meta_weight_cosine_decay_endpoints_and_phases():
    assert meta_weight_cosine_decay(0, pretrain=5, total_epochs=10) == 0.0
    assert meta_weight_cosine_decay(4, pretrain=5, total_epochs=10) == 0.0
    # joint-phase start: full weight, decaying to min at the end
    assert math.isclose(meta_weight_cosine_decay(5, 5, 10), 1.0, rel_tol=1e-12)
    assert math.isclose(meta_weight_cosine_decay(9, 5, 10), 1e-4, rel_tol=1e-12)
    # degenerate span: joint phase never starts
    assert meta_weight_cosine_decay(5, 5, 5) == 0.0


def test_meta_weight_cosine_decay_is_non_increasing():
    joint = [meta_weight_cosine_decay(e, 3, 10) for e in range(3, 10)]
    assert all(b <= a for a, b in zip(joint, joint[1:]))
    assert joint[0] > joint[-1]


def test_meta_weight_cosine_decay_is_inverted_legacy():
    """New schedule starts high and decays; legacy scsf starts at min and climbs."""
    start = meta_weight_cosine_decay(5, 5, 10)
    end = meta_weight_cosine_decay(9, 5, 10)
    legacy_start = meta_weight_cosine(5, 5, 10)
    legacy_end = meta_weight_cosine(9, 5, 10)
    assert start == 1.0 and legacy_start == 1e-4
    assert end == 1e-4 and legacy_end > 0.9


def test_temperature_schedule_linear_endpoints():
    sched = [temperature_schedule(e, 10, 1.0, 4.0) for e in range(10)]
    assert sched[0] == 1.0 and sched[9] == 4.0
    assert all(b >= a for a, b in zip(sched, sched[1:]))
    assert math.isclose(sched[1], 1.0 + 1.0 / 9.0 * 3.0, rel_tol=1e-12)


# --------------------------------------------------------------------------
# calibrators
# --------------------------------------------------------------------------


def test_raw_calibrator_gradient_rule_attached_feats_detached_logits():
    cal = RawScoreCalibrator(feature_dims=[4, 4], logit_dim=3)
    f0 = torch.randn(5, 4, requires_grad=True)
    f1 = torch.randn(5, 4, requires_grad=True)
    logits = torch.randn(5, 3, requires_grad=True)
    loss = cal([f0, f1], logits).square().mean()
    loss.backward()
    assert f0.grad is not None and f1.grad is not None  # features attached
    assert logits.grad is None                          # logits stop-gradient


def test_raw_calibrator_output_and_sigmoid_route():
    cal = RawScoreCalibrator(feature_dims=[3, 3], logit_dim=5)
    feats = [torch.randn(7, 3) for _ in range(2)]
    logits = torch.randn(7, 5)
    s = cal(feats, logits)
    assert tuple(s.shape) == (7,)
    assert torch.allclose(cal.confidence(feats, logits), torch.sigmoid(s))


def test_transition_calibrator_four_way_and_marginal():
    cal = TransitionCalibrator(feature_dims=[2, 2], logit_dim=3)
    feats = [torch.randn(6, 2), torch.randn(6, 2)]
    logits = torch.randn(6, 3)
    q = cal(feats, logits)
    assert tuple(q.shape) == (6, 4)
    soft = torch.softmax(q, dim=1)
    conf = cal.final_correctness_confidence(feats, logits)
    assert torch.allclose(conf, soft[:, 1] + soft[:, 3])
    # (1,0) "either-depth correct" is NOT the marginal: mass only on 00 gives 0
    q0 = torch.zeros(1, 4)
    q0[:, 0] = 5.0
    assert float(final_correctness_for(q0)) == 0.0


def final_correctness_for(q_row):
    soft = torch.softmax(q_row, dim=1)
    return soft[0, 1] + soft[0, 3]


# --------------------------------------------------------------------------
# rc_weights
# --------------------------------------------------------------------------


def test_rank_of_desc_tie_by_ascending_id():
    conf = torch.tensor([0.5, 0.9, 0.5, 0.7])
    ids = torch.tensor([3, 2, 1, 0])
    r = rank_of(conf, ids)
    # sample 2 (id 1) is the lower-id of the two 0.5s, so it is rank 4 over id 3
    assert r.tolist() == [4, 1, 3, 2]
    assert rank_of(conf).tolist() == [3, 1, 4, 2]  # arc ids when none given


def test_rank_of_detached():
    conf = torch.tensor([0.5], requires_grad=True)
    r = rank_of(conf)
    assert r.requires_grad is False


def test_harmonic_identity_full_prefix_formula():
    torch.manual_seed(3)
    conf = torch.rand(50) * 4 - 2
    err = torch.randint(0, 2, (50,)).float()
    w = harmonic_weights(conf, torch.arange(50))
    lhs = (err * w).sum()
    order = torch.argsort(-conf.detach(), stable=True)
    cum = torch.cumsum(err[order], dim=0)
    k = torch.arange(1, 51, dtype=torch.float32)
    rhs = (cum / k).mean()
    assert torch.allclose(lhs, rhs, rtol=1e-4, atol=1e-4)


def test_harmonic_weights_decrease_with_rank():
    n = 8
    ranks = torch.arange(1, n + 1)
    w = harmonic_weights_at_ranks(n, ranks)
    assert (w[:-1] > w[1:]).all()
    exact = (1.0 / torch.arange(1, n + 1, dtype=torch.float64)).sum() / n
    assert math.isclose(w[0].item(), exact.item(), rel_tol=1e-6)


def test_swap_identity_adjacent_equal_errors():
    """Swapping confidences of adjacent equal-error samples leaves AURC intact."""
    conf = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
    err = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0])
    w = harmonic_weights(conf, torch.arange(5))
    base = (err * w).sum()
    conf2 = conf.clone()
    conf2[0], conf2[1] = conf2[1].item(), conf2[0].item()  # samples 0,1 both err=1
    w2 = harmonic_weights(conf2, torch.arange(5))
    assert torch.allclose((err * w2).sum(), base)


def test_harmonic_weights_empty_and_singleton_finite():
    c = torch.empty(0)
    w = harmonic_weights(c)
    assert tuple(w.shape) == (0,) and torch.all(torch.isfinite(w))
    w1 = harmonic_weights(torch.tensor([0.5]))
    assert w1.tolist() == [1.0]


def test_normalize_mean1_over_mask_and_empty():
    w = torch.tensor([1.0, 2.0, 3.0, 100.0])
    mask = torch.tensor([True, True, False, False])
    out, applied = normalize_mean1(w, mask)
    assert applied is True
    # whole vector re-scaled by the masked-subset mean (0.5 * (1+2) = 1.5)
    assert torch.allclose(out, w / 1.5)
    assert out.requires_grad is False
    out2, applied2 = normalize_mean1(w, torch.zeros(4, dtype=torch.bool))
    assert applied2 is False and torch.allclose(out2, w)


def test_normalize_clip_order_mean_then_clip():
    w = torch.tensor([1.0, 2.0, 6.0])
    mask = torch.tensor([True, False, True])
    rough = normalize_clip(w, mask, clip_max=1.0)
    # masked mean = (1+6)/2 = 3.5 -> [1/3.5, 2/3.5, 6/3.5]; clip at 1.0
    assert torch.allclose(rough, torch.tensor([1.0 / 3.5, 2.0 / 3.5, 1.0]), atol=1e-6)


# --------------------------------------------------------------------------
# losses
# --------------------------------------------------------------------------


def test_weighted_bce_only_incorrect_and_detached_target():
    s = torch.tensor([0.0, -1.0, 2.0, 1.0], requires_grad=True)
    t = torch.tensor([0.0, 0.0, 1.0, 1.0])
    loss = weighted_bce_correctness(s, t, error_weight=3.0)
    bce0 = F.binary_cross_entropy_with_logits(s[None], t[None], reduction="none")[0]
    expected = (torch.tensor([3.0, 3.0, 1.0, 1.0]) * bce0).mean()
    assert torch.allclose(loss, expected)
    loss.backward()
    assert s.grad is not None and t.grad is None      # target detached
    assert s.grad.isclose(torch.zeros(4)).any() is False


def test_rank_pair_loss_matches_hand_formula_and_grads_only_scores():
    scores = torch.tensor([0.5, 2.0, -1.0], requires_grad=True)
    errors = torch.tensor([1.0, 0.0, 1.0])
    rho = torch.tensor([0.5, 0.5, 0.5])
    w_diff = torch.tensor([[0.1, 0.2, 0.0], [0.3, 0.0, 0.4], [0.0, 0.5, 0.0]])
    T, margin, eps_r, eps = 1.0, 0.0, 1e-3, 1e-8
    loss = rank_pair_loss(scores, errors, rho, w_diff, margin, T, eps_r, eps)
    d = w_diff[[0, 2]][:, [1]]                      # incorrect i -> correct j=1
    gate = (eps_r + 1.0 - rho[[0, 2]]).unsqueeze(1)
    u = scores[[0, 2]].unsqueeze(1) - scores[1]
    pair = T * F.softplus((margin + u) / T)
    expected = (d * gate * pair).sum(0).sum() / (d.sum() + eps)
    assert torch.allclose(loss, expected)
    loss.backward()
    assert scores.grad is not None
    assert scores.grad.numel() == 3


def test_rank_pair_constant_rho_scales_but_denominator_excludes_it():
    """Routing strength survives a constant rho (denominator is sum of d)."""
    scores = torch.randn(6, requires_grad=True)
    errors = torch.tensor([1, 1, 1, 0, 0, 0.0])
    w_diff = torch.randint(1, 5, (3, 3)).float() / 10.0
    lo = rank_pair_loss(scores, errors, torch.full((6,), 0.3), w_diff).item()
    hi = rank_pair_loss(scores, errors, torch.full((6,), 0.8), w_diff).item()
    ratio = (1.001 + 1.0 - 0.8) / (1.001 + 1.0 - 0.3)
    assert math.isclose(lo / hi, 1.0 / ratio, rel_tol=1e-4)


def test_rank_pair_empty_side_finite_zero():
    scores = torch.randn(4, requires_grad=True)
    err_all_wrong = torch.ones(4)
    err_all_right = torch.zeros(4)
    w = torch.full((4, 4), 0.1)
    assert rank_pair_loss(scores, err_all_wrong, torch.ones(4), w) == 0.0
    assert rank_pair_loss(scores, err_all_right, torch.ones(4), w) == 0.0


def test_conditional_reverse_kd_orientation_rule():
    """Teacher is detached probe softmax; student is the final classifier."""
    probe = torch.randn(8, 10, requires_grad=True)
    final = torch.randn(8, 10, requires_grad=True)
    mask01 = torch.zeros(8)
    mask01[0] = mask01[1] = 1.0
    w_bar = torch.ones(8)
    loss = conditional_reverse_kd(probe, final, mask01, w_bar, temperature=2.0)
    exp = F.kl_div(F.log_softmax(final[:2] / 2.0, 1),
                   torch.softmax(probe[:2] / 2.0, 1).detach(), reduction="none"
                   ).sum(1).mean() * 4.0
    assert torch.allclose(loss, exp, atol=1e-5)
    loss.backward()
    assert probe.grad is None and final.grad is not None  # teacher detached


def test_conditional_reverse_kd_zero_mask_finite_zero():
    probe = torch.randn(4, 3, requires_grad=True)
    final = torch.randn(4, 3, requires_grad=True)
    loss = conditional_reverse_kd(probe, final, torch.zeros(4), torch.ones(4))
    assert loss == 0.0


def test_logsumexp_confusion_formula_and_empty():
    U = torch.tensor([0.0, 1.0])
    v = float(logsumexp_confusion(U, tau=1.0))
    assert math.isclose(v, math.log(math.e + 1) - math.log(2), rel_tol=1e-6)
    assert logsumexp_confusion(torch.empty(0), tau=2.0) == 0.0
    # all-equal U at tau: value is -tau*log(n) (never inherent zero-risk)
    assert math.isclose(logsumexp_confusion(torch.tensor([0.0, 0.0, 0.0]), 2.0),
                        -2.0 * math.log(3), rel_tol=1e-6)


def test_class_coverage_fractions_nan_for_absent_classes():
    soft_masks = torch.tensor([[0.2, 0.8], [0.5, 0.5], [1.0, 0.0]])
    y = torch.tensor([0, 0, 0])        # class 1 absent
    phi = class_coverage_fractions(soft_masks, y, num_classes=2)
    assert torch.isfinite(phi[0]).all()
    assert torch.isnan(phi[1]).all()
    assert math.isclose(phi[0, 0].item(), (0.2 + 0.5 + 1.0) / 3, rel_tol=1e-6)


def test_confusion_utilities_hand_case():
    probs = torch.tensor([[0.8, 0.2], [0.9, 0.1], [0.6, 0.4]])
    soft_masks = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    y = torch.tensor([0, 1, 1])
    U = confusion_utilities(probs, soft_masks, y, num_classes=2)
    # a=0,b=0,c=1: numerator y=0 sample (b=0) * probs[0,0] * mask[0,1] = 1*0.8*1
    # denominator = accepted mass of y=0 at coverage 1 = mask[0,1] = 1.0
    assert math.isclose(U[0, 0, 1].item(), 0.8, rel_tol=1e-6)


# --------------------------------------------------------------------------
# softquantile
# --------------------------------------------------------------------------


def test_soft_thresholds_residual_near_zero():
    torch.manual_seed(0)
    s = torch.randn(200)
    coverages = torch.tensor([0.3, 0.5, 0.8])
    t = solve_soft_thresholds(s, coverages, Ts=1.0)
    res = soft_threshold_residual(s, coverages, 1.0, t)
    assert torch.all(res.abs() < 1e-3)


def test_soft_threshold_shift_invariance():
    s = torch.randn(60)
    coverages = torch.tensor([0.25, 0.75])
    t = solve_soft_thresholds(s, coverages, 1.0)
    s2 = s + 5.0
    t2 = solve_soft_thresholds(s2, coverages, 1.0)
    # thresholds shift together with the scores (acceptance is relative)
    assert torch.allclose(t2, t + 5.0, atol=1e-3)


def test_soft_coverage_threshold_gradcheck_loose():
    torch.manual_seed(1)
    s = torch.randn(6, dtype=torch.float64, requires_grad=True)
    coverages = torch.tensor([0.4, 0.6], dtype=torch.float64)
    result = torch.autograd.gradcheck(
        lambda z: soft_coverage_threshold(z, coverages, torch.tensor(1.0, dtype=z.dtype)),
        (s,), eps=1e-3, atol=1e-2, rtol=1e-2)
    assert result is True


def test_soft_coverage_threshold_nesting_consistency():
    """Rounding one coverage to a solve against itself returns the same root."""
    torch.manual_seed(2)
    s = torch.randn(40)
    c = torch.tensor([0.66])
    t = solve_soft_thresholds(s, c, 1.0)
    a = (1.0 - (t - s) / 1.0).sigmoid().mean().item()
    assert math.isclose(a, 0.66, abs_tol=1e-3)


# --------------------------------------------------------------------------
# views
# --------------------------------------------------------------------------


def test_view_seed_deterministic_and_stream_separated():
    a = view_seed(13, 5, 1)
    assert a == view_seed(13, 5, 1)
    assert a != view_seed(13, 5, 0)    # view index separates
    assert a != view_seed(13, 6, 1)    # sample id separates
    assert 0 <= a < 2 ** 31


def test_apply_transform_gated_restores_rng():
    before = torch.random.get_rng_state()
    im = torch.randn(3, 32, 32)
    out, seed = apply_transform_gated(lambda t: t + torch.randn_like(t), im, 9)
    assert torch.random.get_rng_state().equal(before)   # torch stream restored
    out2, _ = apply_transform_gated(lambda t: t + torch.randn_like(t), im, 9)
    assert torch.allclose(out, out2)                     # seed reproduces view


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------


def test_rho_cache_persist_expiry_and_collision():
    cache = RhoCache(capacity=8)
    cache.update(3, torch.tensor(0.7), epoch=5)
    assert torch.allclose(cache.lookup(3, epoch=5), torch.tensor(0.7))
    assert cache.lookup(3, epoch=6) is not None          # within expiry window
    assert cache.lookup(3, epoch=7) is None              # stale
    with pytest.raises(RuntimeError):
        cache.update(3 + 8, torch.tensor(0.1), epoch=5)  # direct-mapped collision


def test_dual_state_clamp_and_detached_sample():
    ds = DualState(num_edges=3, dual_max=2.0)
    ds.add_dual_grad(torch.tensor([0.1, 9.0, -1.0]), lr=1.0)
    assert float(ds.nu()[1]) == 2.0 and ds.nu()[1].requires_grad is False
    sam = ds.sample()
    assert torch.allclose(sam, ds.nu()) and sam is not ds.nu()


def test_edge_support_counts_and_mask():
    es = EdgeSupport(num_classes=3, device="cpu")
    es.update_from_batch(torch.tensor([0, 1, 2, 0]),
                         torch.tensor([1, 2, 0, 1]))
    counts = es.counts()
    assert counts[0, 1] == 2 and counts[1, 2] == 1 and counts[2, 0] == 1
    assert bool(es.mask(min_support=2)[0, 1]) and not bool(es.mask(min_support=2)[1, 2])


def test_phase_state_scalars():
    ph = PhaseState(epoch=3, step=11)
    assert ph.epoch == 3 and ph.step == 11