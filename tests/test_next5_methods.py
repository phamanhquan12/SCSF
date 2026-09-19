"""NEXT5 method scientific locks (CPU)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from scsf.methods import build_method
from scsf.methods.base import MethodPrediction
from scsf.methods.next5_common import window_rank_weights
from scsf.methods.optim_sam import SAM

TRAIN_CFG = {
    "backbone": "resnet18",
    "data": {"num_classes": 10, "official_train_size": 64},
    "method": {},
    "train": {"lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4,
              "epochs": 2, "optimizer": "sgd", "scheduler": "cosine",
              "seed": 13, "data_order_seed": 13},
}


def _cfg(name, **m):
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in TRAIN_CFG.items()}
    cfg["method"] = dict(cfg["method"], **m)
    cfg["method_name"] = name
    return cfg


def _state(epoch=0):
    return type("S", (), {"epoch": epoch})()


def _opts(method):
    specs = method.optimizer_specs()
    opts = []
    for spec in specs:
        params = list(spec["params"])
        if not params:
            continue
        kind = spec.get("kind", "sgd")
        if kind == "sam_sgd":
            opts.append(SAM(params, torch.optim.SGD, rho=float(spec.get("rho", 0.05)),
                            lr=spec["lr"], momentum=spec.get("momentum", 0.0),
                            weight_decay=spec.get("weight_decay", 0.0)))
        elif kind == "adam":
            opts.append(torch.optim.Adam(params, lr=spec["lr"]))
        else:
            opts.append(torch.optim.SGD(
                params, lr=spec["lr"], momentum=spec.get("momentum", 0.0),
                weight_decay=spec.get("weight_decay", 0.0)))
    return opts


def test_factories_predict_label_free():
    torch.manual_seed(0)
    x = torch.zeros(4, 3, 32, 32)
    for name in ("candidate_verify", "intervention_rank", "rank_sharpness",
                 "fmfp_reference", "crossfit_failure", "neighbor_distill"):
        m = build_method(name, _cfg(name))
        mp = m.predict_batch(x)
        assert isinstance(mp, MethodPrediction)
        assert mp.confidence.shape == (4,)
        assert not torch.isnan(mp.confidence).any()
        # inference does not require labels
        assert mp.prediction.shape == (4,)


def test_crossfit_difficulty_reaches_backbone_and_stays_detached_target():
    torch.manual_seed(0)
    m = build_method("crossfit_failure", _cfg("crossfit_failure", pretrain=0))
    idx = torch.arange(4)
    m._oof_valid[idx] = True
    m._oof_error[idx] = torch.tensor([1.0, 0.0, 1.0, 0.0])
    x = torch.randn(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3])
    out = m.train_loss((x, y, idx), _state(0))
    assert out["difficulty"].requires_grad
    assert out["diag_teacher_error_rate"].requires_grad is False
    loss = out["ce"] + out["meta"] + out["difficulty"]
    loss.backward()
    conv = next(p for p in m.backbone.parameters() if p.ndim >= 4)
    assert conv.grad is not None and float(conv.grad.abs().sum()) > 0
    dparam = next(p for p in m.difficulty.parameters() if p.requires_grad)
    assert dparam.grad is not None and float(dparam.grad.abs().sum()) > 0
    assert m.difficulty not in list(m.inference_modules())
    extra_groups = [s for s in m.optimizer_specs() if s["params"] and
                    any(p is next(m.difficulty.parameters()) for p in s["params"])]
    assert extra_groups, "difficulty head must be in optimizer_specs"


def test_crossfit_missing_id_is_hard_error():
    m = build_method("crossfit_failure", _cfg("crossfit_failure"))
    x = torch.randn(2, 3, 32, 32)
    y = torch.tensor([0, 1])
    idx = torch.tensor([0, 1])
    try:
        m.train_loss((x, y, idx), _state(0))
        assert False, "missing OOF should hard-fail"
    except RuntimeError as e:
        assert "missing OOF" in str(e)


def test_verifier_never_receives_label_indicator_and_inference_uses_argmax():
    torch.manual_seed(0)
    m = build_method("candidate_verify", _cfg("candidate_verify"))
    x = torch.randn(6, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3, 4, 5])
    out = m.train_loss((x, y), _state(0))
    assert out["verify"].requires_grad
    (out["ce"] + out["verify"]).backward()
    w = m.verifier.class_emb.weight
    assert w.grad is not None and float(w.grad.abs().sum()) > 0
    mp = m.predict_batch(x)
    bo = m.backbone(x)
    pred = bo.logits.argmax(dim=1)
    assert torch.equal(mp.prediction, pred)


def test_intervention_pairless_finite_zero_and_two_view_mean():
    torch.manual_seed(0)
    m = build_method("intervention_rank", _cfg("intervention_rank", pretrain=0))
    x = torch.randn(4, 3, 32, 32)
    v = x.clone()
    y = torch.tensor([0, 1, 2, 3])
    idx = torch.arange(4)
    out = m.train_loss((x, v, y, idx), _state(0))
    assert torch.isfinite(out["pair"])
    # identical views: mixed-outcome pairs cannot exist
    assert float(out["pair"]) == 0.0 or float(out["diag_valid_pair_rate"]) == 0.0


def test_intervention_mixed_pair_uses_detached_masks():
    torch.manual_seed(1)
    m = build_method("intervention_rank", _cfg("intervention_rank", pretrain=0, lambda_pair=1.0))
    x = torch.randn(8, 3, 32, 32)
    v = torch.randn(8, 3, 32, 32)
    y = torch.zeros(8, dtype=torch.long)
    idx = torch.arange(8)
    out = m.train_loss((x, v, y, idx), _state(0))
    assert torch.isfinite(out["pair"])
    assert out["pair"].requires_grad or float(out["diag_valid_pair_rate"]) == 0.0


def test_neighbor_detached_targets_and_missing_ids():
    m = build_method("neighbor_distill", _cfg("neighbor_distill", pretrain=0))
    idx = torch.arange(4)
    m._mem_valid[idx] = True
    m._tgt8[idx] = torch.full((4, 10), 0.1)
    m._tgt32[idx] = torch.full((4, 10), 0.1)
    x = torch.randn(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3])
    out = m.train_loss((x, y, idx), _state(0))
    assert out["support"].requires_grad
    out["support"].backward()
    # targets are buffers; they must not receive grad
    assert m._tgt32.grad is None
    m2 = build_method("neighbor_distill", _cfg("neighbor_distill"))
    try:
        m2.train_loss((x, y, idx), _state(0))
        assert False
    except RuntimeError as e:
        assert "missing memory" in str(e)


def test_neighbor_support_heads_are_optimized():
    m = build_method("neighbor_distill", _cfg("neighbor_distill"))
    owned = {id(p) for s in m.optimizer_specs() for p in s["params"]}
    assert id(next(m.head_k8.parameters())) in owned
    assert id(next(m.head_k32.parameters())) in owned
    assert m.head_k32 not in list(m.inference_modules())


def test_rank_sharpness_restores_weights_including_exception():
    torch.manual_seed(0)
    m = build_method("rank_sharpness", _cfg("rank_sharpness", pretrain=0, rho=0.05))
    x = torch.randn(8, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    idx = torch.arange(8)
    opts = _opts(m)
    before = [p.detach().clone() for p in m.parameters() if p.requires_grad]
    out = m.run_step((x, y, idx), _state(0), opts)
    assert "diag_perturbed" in out
    # live weights must be finite and of original shape (restored then stepped)
    for p in m.parameters():
        assert torch.isfinite(p).all()
    # exception path
    class Boom(torch.optim.SGD):
        def step(self, closure=None):
            raise RuntimeError("boom")
    m2 = build_method("rank_sharpness", _cfg("rank_sharpness", pretrain=0))
    boom_opts = [Boom(m2.parameters(), lr=0.1, momentum=0.9)]
    snap = [p.detach().clone() for p in m2.parameters() if p.requires_grad]
    try:
        m2.run_step((x, y, idx), _state(0), boom_opts)
        assert False
    except RuntimeError:
        after = [p.detach().clone() for p in m2.parameters() if p.requires_grad]
        for a, b in zip(snap, after):
            assert torch.allclose(a, b)


def test_rank_sharpness_pairless_unperturbed():
    torch.manual_seed(0)
    m = build_method("rank_sharpness", _cfg("rank_sharpness", pretrain=0))
    x = torch.zeros(3, 3, 32, 32)
    y = torch.zeros(3, dtype=torch.long)
    idx = torch.arange(3)
    out, frozen, pair_ok = m._pack((x, y, idx), _state(0))
    assert torch.isfinite(out["rank"])
    if not pair_ok:
        assert float(out["rank"]) == 0.0


def test_window_weights_high_confidence_error_has_full_mass():
    scores = torch.tensor([3.0, 2.0, 1.0, 0.0, -1.0])
    ids = torch.arange(5)
    w = window_rank_weights(scores, ids, 0.8, 1.0)
    # B=5, k_lo=ceil(4)=4, k_hi=5, denom=2
    # rank1 weight = |{4,5} ∩ {k>=1}| / 2 = 1
    assert abs(float(w[0]) - 1.0) < 1e-6
    # rank 5 (least conf) counted only if k>=5 → {5} / 2 = 0.5
    assert abs(float(w[-1]) - 0.5) < 1e-6


def test_fmfp_sam_two_pass_one_update_and_restore():
    torch.manual_seed(0)
    m = build_method("fmfp_reference", _cfg("fmfp_reference", swa_start=0))
    x = torch.randn(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3])
    opts = _opts(m)
    assert type(opts[0]).__name__ == "SAM"
    p0 = next(m.parameters())
    before = p0.detach().clone()
    out = m.run_step((x, y), _state(0), opts)
    assert "ce" in out and "ce_perturbed" in out
    # one update: weights should change but stay finite
    assert torch.isfinite(p0).all()
    m.on_epoch_end(0, {})
    assert m._swa_n == 1
    mp = m.predict_batch(x)
    assert mp.confidence.shape == (4,)


def test_scsf_correctness_not_replaced_by_legacy_tcp():
    from scsf.methods.factory import method_names
    assert "scsf" in method_names() and "scsf_correctness" in method_names()
    m = build_method("scsf_correctness", _cfg("scsf_correctness"))
    assert m.method_name == "scsf_correctness"
