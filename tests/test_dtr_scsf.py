"""DTR-SCSF tests (authored; STATUS: NOT RUN locally).

Locks: two-view batch routing, warmup phase gating, the exact four-state index
``2*tS + tL``, detached diagnostics vs. optimized losses, and deployment using
backbone + transition head only (probe excluded).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from scsf.methods import build_method
from scsf.methods.rc_training.schedules import meta_weight_cosine_decay

TRAIN_CFG = {
    "backbone": "resnet18",
    "data": {"num_classes": 10, "official_train_size": 2000},
    "method": {},
    "train": {"lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4,
              "epochs": 2, "optimizer": "sgd", "scheduler": "cosine",
              "seed": 13, "data_order_seed": 13},
}


def _cfg(**m):
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in TRAIN_CFG.items()}
    cfg["method"] = dict(cfg["method"], **m)
    return cfg


def _state(epoch):
    return type("S", (), {"epoch": epoch})()


def _batch(bs=6, num_classes=10):
    torch.manual_seed(0)
    x = torch.randn(bs, 3, 32, 32)
    v = torch.randn(bs, 3, 32, 32)
    y = torch.arange(bs) % num_classes
    idx = torch.arange(bs, dtype=torch.long)
    return x, v, y, idx


def test_factory_builds_and_predicts_single_view():
    m = build_method("dtr_scsf", _cfg())
    assert m.needs_two_views is True
    assert m.probe_role in m.tap_roles
    mp = m.predict_batch(torch.zeros(4, 3, 32, 32))
    assert tuple(mp.confidence.shape) == (4,)
    assert torch.isfinite(mp.confidence).all()
    assert ((mp.confidence >= 0) & (mp.confidence <= 1)).all()
    assert mp.confidence is mp.scores["dtr_conf"]
    assert set(mp.scores) >= {"msp", "entropy", "energy", "logit_margin", "dtr_conf"}


def test_warmup_gating_and_meta_weight_at_joint_start():
    c = _cfg(warmup_epochs=2)
    c["train"] = dict(c["train"], epochs=5)
    m = build_method("dtr_scsf", c)
    x, v, y, idx = _batch()
    pre = m.train_loss((x, v, y, idx), _state(0))
    assert set(pre) == {"ce", "beta_ce_probe"}
    ons = m.train_loss((x, v, y, idx), _state(2))
    assert "four_state" in ons
    assert ons["meta_weight"].item() == 1.0              # joint phase starts at 1
    assert abs(ons["four_state"].item()) > 0 and torch.isfinite(ons["four_state"])


def test_four_state_index_is_2_tS_plus_tL():
    torch.manual_seed(0)
    m = build_method("dtr_scsf", _cfg(warmup_epochs=0))
    x, v, y, idx = _batch(bs=6)
    out = m.train_loss((x, v, y, idx), _state(0))
    bo_u = m.backbone(x)
    bo_v = m.backbone(v)
    pu = m._probe(m._taps_role(bo_u, m.probe_role))
    pv = m._probe(m._taps_role(bo_v, m.probe_role))
    ts = ((pu.argmax(1) == y) & (pv.argmax(1) == y)).long()
    tl = (bo_u.logits.argmax(1) == y).long()
    target = 2 * ts + tl
    q = m._calib(m._extract(bo_u), bo_u.logits)
    w = meta_weight_cosine_decay(0, 0, int(m.cfg["train"]["epochs"]), 1.0, 1e-4)
    assert w == 1.0
    expected = w * F.cross_entropy(q, target)
    assert torch.allclose(out["four_state"], expected)
    assert out["meta_weight"].item() == 1.0


def test_only_active_losses_carry_gradients():
    torch.manual_seed(0)
    m = build_method("dtr_scsf", _cfg(warmup_epochs=0))
    out = m.train_loss(_batch(), _state(0))
    grad_keys = sorted(k for k, v in out.items() if v.requires_grad)
    assert grad_keys == ["beta_ce_probe", "ce", "four_state", "kd"]
    assert out["meta_weight"].requires_grad is False
    for dk in ("diag_mask_frac", "diag_probe_corr_frac", "diag_state11_frac"):
        assert out[dk].requires_grad is False


def test_gradients_reach_backbone_probe_and_calibrator():
    torch.manual_seed(0)
    m = build_method("dtr_scsf", _cfg(warmup_epochs=0))
    x, v, y, idx = _batch()
    out = m.train_loss((x, v, y, idx), _state(0))
    total = sum(v for k, v in out.items() if k in ("ce", "beta_ce_probe", "four_state", "kd"))
    total.backward()
    bb = next(p for p in m.backbone.parameters() if p.ndim >= 4 and p.requires_grad)
    assert bb.grad is not None and abs(bb.grad.sum()) > 0
    probe_params = [p for p in m._probe.parameters() if p.requires_grad]
    assert all(p.grad is not None for p in probe_params)
    calib = [p for p in m._calib.parameters() if p.requires_grad]
    assert all(p.grad is not None for p in calib)


def test_aux_ce_variant_removes_head_and_kd():
    m = build_method("dtr_scsf", _cfg(warmup_epochs=0, use_four_state=False,
                                      use_kd=False))
    out = m.train_loss(_batch(), _state(0))
    assert "four_state" not in out and "kd" not in out
    assert set(out) >= {"ce", "beta_ce_probe"}


def test_optimizer_specs_shared_sgd_additive_head_adam():
    m = build_method("dtr_scsf", _cfg())
    specs = m.optimizer_specs()
    kinds = [s["kind"] for s in specs]
    assert "sgd" in kinds and "adam" in kinds
    sgd_ids = {id(p) for s in specs if s["kind"] == "sgd" for p in s["params"]}
    adam_ids = {id(p) for s in specs if s["kind"] == "adam" for p in s["params"]}
    probe_ids = {id(p) for p in m._probe.parameters()}      # shares recipe LR
    calib_ids = {id(p) for p in m._calib.parameters()}
    assert probe_ids <= sgd_ids
    assert calib_ids & adam_ids == calib_ids
    assert not (calib_ids & sgd_ids)


def test_inference_modules_exclude_probe():
    m = build_method("dtr_scsf", _cfg())
    mods = list(m.inference_modules())
    assert all(isinstance(mod, nn.Module) for mod in mods)
    assert any(mod is m.backbone for mod in mods)
    assert any(mod is m._calib for mod in mods)
    assert not any(mod is m._probe for mod in mods)