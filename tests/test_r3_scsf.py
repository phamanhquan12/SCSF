"""R3-SCSF tests (authored; STATUS: NOT RUN locally).

Locks the spec's R3 guarantees: virtual-step isolation (params / buffers /
optimizer grad / RNG / module mode), no retained second-order measurement
graph, rho bounds + default + cache expiry, pretrain gating, and the
routing-strength invariance (rank term normalizes by ``sum d``, never by
``sum rho``).
"""

import torch

from scsf.methods import build_method
from scsf.methods.r3_scsf import OMEGA_PREFIXES, R3SCSFMethod
from scsf.methods.rc_training.losses import rank_pair_loss
from scsf.methods.rc_training.rc_weights import harmonic_weights

TRAIN_CFG = {
    "backbone": "resnet18",
    "data": {"num_classes": 2, "official_train_size": 2000},
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


def _batch(bs=8):
    torch.manual_seed(0)
    x = torch.randn(bs, 3, 32, 32)
    v = torch.randn(bs, 3, 32, 32)
    y = torch.randint(0, 2, (bs,))
    idx = torch.arange(bs, dtype=torch.long)
    return x, v, y, idx


def test_factory_builds_and_predicts():
    m = build_method("r3_scsf", _cfg())
    assert isinstance(m, R3SCSFMethod)
    assert m.needs_two_views is True
    assert m._omega_names  # validated final-block+classifier subset nonempty
    mp = m.predict_batch(torch.zeros(4, 3, 32, 32))
    assert torch.isfinite(mp.confidence).all()
    assert ((mp.confidence >= 0) & (mp.confidence <= 1)).all()
    assert mp.confidence is mp.scores["r3_conf"]
    assert set(mp.scores) >= {"msp", "entropy", "energy", "logit_margin",
                              "r3_rank_raw", "r3_conf"}


def test_omega_prefixes_match_backbones():
    for bn, pref in OMEGA_PREFIXES.items():
        cfg = _cfg() if bn == "resnet18" else _cfg()
        cfg["backbone"] = bn
        m = build_method("r3_scsf", cfg)
        names = {n for n, _ in m.backbone.named_parameters()}
        assert any(any(n.startswith(p) for n in names) for p in pref)


def test_pretrain_gating():
    m = build_method("r3_scsf", _cfg(pretrain=2, rho={"max_errors": 8}))
    out = m.train_loss(_batch(), _state(0))
    assert set(out) == {"ce"}
    joint = m.train_loss(_batch(), _state(2))
    assert set(joint) >= {"ce", "meta", "rank", "repair"}


def test_rho_default_bounds_and_zero_measured():
    m = build_method("r3_scsf", _cfg(rho={"max_errors": 0, "default": 0.5}))
    out = m.train_loss(_batch(), _state(1))
    assert out["diag_rho_measured"].item() == 0.0
    assert torch.isfinite(out["repair"]).item()


def test_rho_detached_no_second_order_retention():
    torch.manual_seed(0)
    m = build_method("r3_scsf", _cfg())
    x, v, y, idx = _batch()
    err = (~(m.backbone(x).logits.argmax(1) == y).bool())
    rho, n_meas, _ = m._rho_for_batch(x, v, y, idx, 0, corr_u=(m.backbone(x).logits.argmax(1) == y))
    assert rho.requires_grad is False
    assert (rho <= 1.0).all() and (rho >= 0.0).all()
    assert (rho[err] == 0.5).any() or n_meas.item() > 0  # default or measured


def test_measurement_isolation():
    torch.manual_seed(0)
    m = build_method("r3_scsf", _cfg())
    x, v, y, idx = _batch(bs=4)
    params_before = [p.detach().clone() for p in m.backbone.parameters()]
    bufs_before = {n: b.detach().clone() for n, b in m.backbone.named_buffers()}
    rng_before = torch.random.get_rng_state()
    corr = (m.backbone(x).logits.argmax(1) == y)
    err = (~corr).nonzero(as_tuple=True)[0]
    if err.numel():
        m._measure_rho(x[err], v[err], y[err])
    # live params / buffers / global RNG untouched, no grads parked
    params_after = [p.detach().clone() for p in m.backbone.parameters()]
    for a, b in zip(params_before, params_after):
        assert torch.allclose(a, b)
    for n, b in bufs_before.items():
        assert torch.allclose(b, dict(m.backbone.named_buffers())[n])
    assert torch.random.get_rng_state().equal(rng_before)
    assert all(p.grad is None for p in m.backbone.parameters())
    assert m.backbone.training is True


def test_rank_routing_strength_invariant_to_scaling_w_diff():
    """Normalization is by ``sum d``, so scaling every pair weight d -> 2d
    leaves the rank term identical (a /(sum rho) normalization would change it)."""
    torch.manual_seed(1)
    s = torch.randn(10, requires_grad=True)
    err = torch.tensor([0, 1, 2, 3, 4., 5., 6., 7.])
    corr = torch.tensor([1., 1., 1., 0., 0., 0., 0., 0.])
    rho = torch.tensor([0.9, 0.9, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5])
    d = torch.rand(8, 8) + 1e-3
    a = rank_pair_loss(s, corr, rho, d, margin=0.0, temperature=1.0)
    b = rank_pair_loss(s, corr, rho, 2.0 * d, margin=0.0, temperature=1.0)
    assert torch.allclose(a, b)


def test_repair_and_rank_match_manual_formula():
    torch.manual_seed(0)
    m = build_method("r3_scsf", _cfg(rho={"max_errors": 0, "default": 0.5}))
    x, v, y, idx = _batch(bs=6)
    out = m.train_loss((x, v, y, idx), _state(1))
    bo = m.backbone(x)
    logits = bo.logits
    taps = [bo.role(m.backbone, r) for r in m.tap_roles]
    s = m._calib(taps, logits)
    target01 = (logits.argmax(1) == y).float()
    err = target01 <= 0.5
    w_raw = harmonic_weights(s.detach(), idx)
    w_bar = w_raw / w_raw[err].mean()
    if err.any():
        ce_i = torch.nn.functional.cross_entropy(logits[err], y[err], reduction="none")
        rho = torch.full((6,), 0.5)
        manual = (w_bar[err] * rho[err] * ce_i).sum() / (err.sum().float() + 1e-8)
        assert torch.allclose(out["repair"], m.gamma * manual)
    assert torch.isfinite(out["rank"]).item()
    # diag values are detached (never optimized by accident)
    for k in ("diag_rho_mean", "diag_rho_measured", "diag_w_bar_mean"):
        assert out[k].requires_grad is False


def test_only_active_losses_carry_gradients():
    torch.manual_seed(0)
    m = build_method("r3_scsf", _cfg())
    out = m.train_loss(_batch(), _state(1))
    grad_keys = sorted(k for k, vv in out.items() if vv.requires_grad)
    assert grad_keys == ["ce", "meta", "rank", "repair"]


def test_optimizer_split_and_inference_modules():
    m = build_method("r3_scsf", _cfg())
    specs = m.optimizer_specs()
    kinds = [s["kind"] for s in specs]
    assert "sgd" in kinds and "adam" in kinds
    adam_ids = {id(p) for s in specs if s["kind"] == "adam" for p in s["params"]}
    backbone_ids = {id(p) for p in m.backbone.parameters()}
    assert not (adam_ids & backbone_ids)
    calib_ids = {id(p) for p in m._calib.parameters()}
    assert calib_ids & adam_ids == calib_ids
    mods = list(m.inference_modules())
    assert any(mod is m.backbone for mod in mods)
    assert any(mod is m._calib for mod in mods)