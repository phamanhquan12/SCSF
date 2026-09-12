"""SCSF review-aligned correctness-baseline tests.

**STATUS: NOT RUN** (delivery machine is stdlib-only; executed on the
experiment server — see docs/VALIDATION_STATUS.md).

The integrity locks: the meta-target is the hard correctness label
``1[argmax(z) == y]`` (never the soft TCP used by legacy ``scsf``), the
schedule starts at weight 1 (unlike the inverted legacy cosine), and the
CE + BCE-with-logits head is the entire supervised objective.
"""

import torch
import torch.nn as nn

from scsf.methods import build_method
from scsf.methods.base import MethodPrediction
from scsf.methods.rc_training.losses import weighted_bce_correctness

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


def test_factory_builds_and_predicts_label_free():
    m = build_method("scsf_correctness", _cfg())
    x = torch.zeros(4, 3, 32, 32)
    mp = m.predict_batch(x)
    assert isinstance(mp, MethodPrediction)
    assert tuple(mp.scores["scsf_corr_raw"].shape) == (4,)
    assert mp.confidence is mp.scores["scsf_corr"]
    assert torch.allclose(mp.confidence, torch.sigmoid(mp.scores["scsf_corr_raw"]))
    assert not torch.isnan(mp.confidence).any()
    for s in m.default_scores():
        assert s in mp.scores


def test_calibrator_head_feature_dims_learned_by_probe():
    m = build_method("scsf_correctness", _cfg())
    assert m._calib is not None and isinstance(m._calib, nn.Module)
    assert len(m.tap_roles) == 2 and m.tap_roles == ["top_l2", "top_l1"]


def test_meta_target_is_hard_correctness_not_tcp():
    torch.manual_seed(0)
    m = build_method("scsf_correctness", _cfg(pretrain=0))
    x = torch.randn(8, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    taps, logits = m._taps_and_logits(x)
    s = m._calib(taps, logits)
    t_obs = (logits.argmax(dim=1) == y).float()
    out = m.train_loss((x, y), _state(0))
    assert torch.allclose(out["meta_raw"],
                          weighted_bce_correctness(s, t_obs, m.error_weight))
    # flipping y to an always-wrong class flips the target to all-zeros even
    # when softmax TCP of that class is NOT tiny (a TCP-MSE head would then
    # return ~(sigmoid(s) - tcp)^2 instead of the hard-labeled BCE).
    y_wrong = (logits.argmax(dim=1) + 1) % m.num_classes
    out_w = m.train_loss((x, y_wrong), _state(0))
    tabs, logits2 = m._taps_and_logits(x)
    s2 = m._calib(tabs, logits2)                       # identical -> identical s
    exp_w = weighted_bce_correctness(s2, torch.zeros_like(t_obs.float()),
                                     m.error_weight)
    assert torch.allclose(out_w["meta_raw"], exp_w)


def test_pretrain_gating_and_meta_weight_at_joint_start():
    m = build_method("scsf_correctness", _cfg(pretrain=2))
    x = torch.randn(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3])
    pre = m.train_loss((x, y), _state(0))
    assert "ce" in pre and "meta" not in pre
    ons = m.train_loss((x, y), _state(2))
    assert ons["meta_weight"].item() == 1.0           # joint phase starts at 1
    assert ons["meta"].item() > 0 and torch.isfinite(ons["meta"])


def test_only_ce_and_meta_carry_gradients():
    torch.manual_seed(0)
    m = build_method("scsf_correctness", _cfg(pretrain=0))
    x = torch.randn(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3])
    out = m.train_loss((x, y), _state(0))
    grad_keys = [k for k, v in out.items() if v.requires_grad]
    assert sorted(grad_keys) == ["ce", "meta"]
    assert out["meta_weight"].requires_grad is False
    assert out["meta_raw"].requires_grad is False


def test_gradients_reach_backbone_and_calibrator():
    torch.manual_seed(0)
    m = build_method("scsf_correctness", _cfg(pretrain=0))
    x = torch.randn(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3])
    out = m.train_loss((x, y), _state(0))
    loss = out["ce"] + out["meta"]
    loss.backward()
    bb_conv = next(p for p in m.backbone.parameters()
                   if p.ndim >= 4 and p.requires_grad)
    assert bb_conv.grad is not None and abs(bb_conv.grad.sum()) > 0
    calib_params = [p for p in m._calib.parameters() if p.requires_grad]
    assert calib_params and all(p.grad is not None for p in calib_params)


def test_inference_modules_are_nn_modules():
    m = build_method("scsf_correctness", _cfg())
    mods = list(m.inference_modules())
    assert mods and all(isinstance(mod, nn.Module) for mod in mods)
    assert any(mod is m.backbone for mod in mods)
    assert any(mod is m._calib for mod in mods)


def test_optimizer_specs_dual_clean_split():
    m = build_method("scsf_correctness", _cfg())
    specs = m.optimizer_specs()
    kinds = [s["kind"] for s in specs]
    assert "sgd" in kinds and "adam" in kinds
    sgd_params = [s["params"] for s in specs if s["kind"] == "sgd"][0]
    adam_params = [s["params"] for s in specs if s["kind"] == "adam"][0]
    sgd_ids = set(map(id, sgd_params))
    adam_ids = set(map(id, adam_params))
    calib_ids = set(map(id, m._calib.parameters()))
    assert len(calib_ids & adam_ids) == len(calib_ids)
    assert not (calib_ids & sgd_ids)
    adam_lr = [s["lr"] for s in specs if s["kind"] == "adam"][0]
    assert m.meta_lr == adam_lr