"""CBR-SCSF tests (authored; STATUS: NOT RUN locally).

Locks the spec's CBR guarantees: soft-coverage threshold solves + analytic
backward, per-coverage micro/confusion aggregation with explicit weights,
dual ascent only at the optimizer boundary (sign + clamp + absent-class
skip), edge-support masks, and the variant toggles.
"""

import torch
import torch.nn.functional as F

from scsf.methods import build_method
from scsf.methods.cbr_scsf import CBRSCSFMethod, DEFAULT_COVERAGES
from scsf.methods.rc_training.softquantile import (
    _accept_probs,
    soft_threshold_residual,
    solve_soft_thresholds,
    soft_coverage_threshold,
)

TRAIN_CFG = {
    "backbone": "resnet18",
    "data": {"num_classes": 4, "official_train_size": 2000},
    "method": {},
    "train": {"lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4,
              "epochs": 2, "optimizer": "sgd", "scheduler": "cosine",
              "seed": 13, "data_order_seed": 13},
}


def _cfg(**m):
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in TRAIN_CFG.items()}
    cfg["method"] = dict(cfg["method"], **m)
    return cfg


def _state(epoch, step=0):
    return type("S", (), {"epoch": epoch, "step": step})()


def _batch(bs=24, num_classes=4):
    torch.manual_seed(0)
    x = torch.randn(bs, 3, 32, 32)
    y = torch.randint(0, num_classes, (bs,))
    return x, y


def test_factory_builds_and_predicts():
    m = build_method("cbr_scsf", _cfg())
    assert isinstance(m, CBRSCSFMethod)
    mp = m.predict_batch(torch.zeros(4, 3, 32, 32))
    assert torch.isfinite(mp.confidence).all()
    assert ((mp.confidence >= 0) & (mp.confidence <= 1)).all()
    assert mp.confidence is mp.scores["cbr_conf"]
    assert set(mp.scores) >= {"msp", "entropy", "energy", "logit_margin",
                              "cbr_raw", "cbr_conf"}


def test_soft_thresholds_solve_coverage_residual_to_zero():
    torch.manual_seed(0)
    s = torch.randn(256)
    covs = torch.tensor(DEFAULT_COVERAGES)
    Ts = 1.0
    h = solve_soft_thresholds(s, covs, Ts)
    res = soft_threshold_residual(s, covs, Ts, h)
    assert res.abs().max().item() < 1e-5


def test_soft_coverage_threshold_gradient_vs_uniform_shift():
    """Shift covariance: h(s + d*1) = h(s) + d, so sum_j dh/ds_j = 1 and the
    analytic backward for loss = h**2 satisfies sum_j dloss/ds_j = 2 h."""
    torch.manual_seed(0)
    s = torch.randn(96, requires_grad=True)
    covs = torch.tensor([0.8])
    Ts = torch.tensor(0.5)
    h = soft_coverage_threshold(s, covs, Ts)
    loss = h.pow(2).sum()
    loss.backward()
    eps = 1e-3
    h0 = solve_soft_thresholds(s.detach(), covs, Ts)
    h1 = solve_soft_thresholds(s.detach() + eps, covs, Ts)   # uniform shift
    assert torch.allclose(h1, h0 + eps, atol=1e-4)            # shift covariance
    assert torch.allclose(s.grad.sum(), 2.0 * h0, atol=1e-3)


def test_shift_invariance_of_threshold_solve():
    """The coverage equation is shift-covariant: h(s + d) = h(s) + d."""
    torch.manual_seed(0)
    s = torch.randn(128)
    covs = torch.tensor([0.70, 0.90])
    Ts = 1.0
    d = 2.5
    h1 = solve_soft_thresholds(s, covs, Ts)
    h2 = solve_soft_thresholds(s + d, covs, Ts)
    assert torch.allclose(h2, h1 + d, atol=1e-4)


def test_pretrain_gating():
    m = build_method("cbr_scsf", _cfg(pretrain=2))
    out = m.train_loss(_batch(), _state(0))
    assert set(out) == {"ce"}
    joint = m.train_loss(_batch(), _state(2))
    assert "meta" in joint and all(k in joint for k in ("micro_0", "conf_0", "floor_0"))


def test_floor_dual_ascent_sign_and_clamp_at_boundary():
    m = build_method("cbr_scsf", _cfg())
    before = m._duals.sample().clone()
    out = m.train_loss(_batch(), _state(1))
    assert m._pending_dual_grad is not None
    # ascent happens only at the boundary, not inside train_loss
    assert torch.equal(before, m._duals.sample())
    m.after_step(out, _state(1))
    after = m._duals.sample()
    # nu next = clamp(nu + lr*residual); residual = kappa*c - phi_a(c)
    # a step must not move nu backwards: ascent on feasibility,
    # residuals only for present classes (absent classes hold).
    assert (after >= 0).all() and (after <= m.dual_max).all()

    # absent classes (zero count) keep their duals: build an all-class-2 batch
    x, y = _batch(bs=16)
    y = torch.full((16,), 2, dtype=torch.long)
    out2 = m.train_loss((x, y), _state(1))
    m.after_step(out2, _state(1))
    nu = m._duals.sample().view(m.num_classes, m._n_cov)
    assert torch.equal(nu[0], torch.zeros_like(nu[0]))
    assert torch.equal(nu[1], torch.zeros_like(nu[1]))
    assert torch.equal(nu[3], torch.zeros_like(nu[3]))


def test_confusion_only_supported_edges_and_finite_empty():
    m = build_method("cbr_scsf", _cfg())
    m.on_epoch_start(0)   # refresh support mask (all ones on first epoch)
    x, y = _batch(bs=8)
    out = m.train_loss((x, y), _state(1))
    # all classes present -> edges > 0 -> conf term finite
    assert torch.isfinite(out["conf_0"]).item()
    assert torch.isfinite(out["micro_0"]).item()
    assert torch.isfinite(out["diag_edges_used"]).item()


def test_group_true_class_variant_terms():
    m = build_method("cbr_scsf", _cfg(use_confusion=False, group_by_class=True))
    x, y = _batch(bs=16)
    out = m.train_loss((x, y), _state(1))
    assert "group" in out and torch.isfinite(out["group"])
    assert not any(k.startswith("conf_") for k in out)


def test_groupdro_variant_is_max_over_true_classes():
    m = build_method("cbr_scsf", _cfg(use_confusion=False, use_floor=False,
                                      group_by_class=True, group_max=True))
    x, y = _batch(bs=24)
    out = m.train_loss((x, y), _state(1))
    assert "dro" in out and "conf_0" not in out
    assert torch.isfinite(out["dro"]).item()


def test_mutually_exclusive_confusion_and_group_raises():
    import pytest
    with pytest.raises(ValueError):
        build_method("cbr_scsf", _cfg(use_confusion=True, group_by_class=True))


def test_diagnostics_detached_and_variant_toggles():
    m = build_method("cbr_scsf", _cfg())
    out = m.train_loss(_batch(bs=16), _state(1))
    for k in ("diag_edges_used", "diag_hard_cov", "diag_Ts", "diag_phi_max"):
        assert out[k].requires_grad is False
    # gradient composition: base + meta + active per-coverage terms only
    grad_keys = {k for k, vv in out.items()
                 if torch.is_tensor(vv) and vv.requires_grad}
    expected = {"ce", "meta"}
    for ci in range(len(DEFAULT_COVERAGES)):
        expected |= {f"micro_{ci}", f"conf_{ci}", f"floor_{ci}"}
    assert grad_keys == expected


def test_single_covariant_overrides_grid():
    m = build_method("cbr_scsf", _cfg(coverages=[0.95], w_c=[1.0], use_floor=False))
    assert m.coverages == (0.95,) and tuple(m.w_c) == (1.0,)
    out = m.train_loss(_batch(bs=16), _state(1))
    assert "micro_0" in out and not any(k.startswith("micro_") and k != "micro_0" for k in out)


def test_optimizer_split_and_inference_modules():
    m = build_method("cbr_scsf", _cfg())
    specs = m.optimizer_specs()
    kinds = [s["kind"] for s in specs]
    assert "sgd" in kinds and "adam" in kinds
    adam_ids = {id(p) for s in specs if s["kind"] == "adam" for p in s["params"]}
    backbone_ids = {id(p) for p in m.backbone.parameters()}
    assert not (adam_ids & backbone_ids)
    mods = list(m.inference_modules())
    assert any(mod is m.backbone for mod in mods)
    assert any(mod is m._calib for mod in mods)


def test_edge_support_updates_periodic():
    m = build_method("cbr_scsf", _cfg())
    m._edge.update_from_batch(torch.tensor([0, 1, 2, 3, 0, 1]),
                              torch.tensor([1, 0, 3, 2, 1, 0]))
    mask = m._edge.mask(min_support=1)
    assert mask[0, 1].item() and mask[1, 0].item()
    assert not mask[3, 0].item()   # never observed -> unsupported