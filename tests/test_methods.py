"""Method-level scientific-integrity tests (loss math + gradient semantics)."""

import math
import warnings

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from scsf.methods import build_method
from scsf.methods.base import MethodPrediction
from scsf.methods.ccl_sc import CSCMoCo
from scsf.methods.sat import SelfAdaptiveTrainingLoss
from scsf.methods.scsf import MODES, MetaCalibrator, meta_weight_cosine

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


def test_factory_builds_every_method_and_predicts():
    for name in ("ce", "dg", "selectivenet", "sat", "scsf", "ccl_sc"):
        method = build_method(name, _cfg())
        x = torch.zeros(4, 3, 32, 32)
        mp = method.predict_batch(x)
        assert isinstance(mp, MethodPrediction)
        assert tuple(mp.logits.shape) == (4, 10)
        assert tuple(mp.confidence.shape) == (4,)
        assert not torch.isnan(mp.confidence).any()
        for s in method.default_scores():
            assert s in mp.scores, name


def test_factory_rejects_unknown_method():
    with pytest.raises(KeyError):
        build_method("does_not_exist", _cfg())


def test_scsf_meta_weight_cosine_schedule():
    assert meta_weight_cosine(epoch=0, pretrain=5, total_epochs=10) == 0.0
    assert meta_weight_cosine(epoch=4, pretrain=5, total_epochs=10) == 0.0
    assert meta_weight_cosine(epoch=5, pretrain=5, total_epochs=10) == 1e-4
    expected_9 = 1e-4 + 0.5 * (1.0 - 1e-4) * (1.0 - math.cos(math.pi * 0.8))
    assert math.isclose(meta_weight_cosine(epoch=9, pretrain=5, total_epochs=10),
                        expected_9, rel_tol=1e-9)
    assert 1e-4 < meta_weight_cosine(epoch=7, pretrain=5, total_epochs=10) < 1.0
    # degenerate span: joint phase never starts
    assert meta_weight_cosine(epoch=5, pretrain=5, total_epochs=5) == 0.0


def test_metacalibrator_posthoc_detaches_every_input():
    cal = MetaCalibrator(feature_dims=[4, 4], logit_dim=3, mode="posthoc")
    f0 = torch.randn(6, 4, requires_grad=True)
    f1 = torch.randn(6, 4, requires_grad=True)
    logits = torch.randn(6, 3, requires_grad=True)
    loss = cal([f0, f1], logits).square().mean()
    loss.backward()
    assert f0.grad is None and f1.grad is None and logits.grad is None
    assert all(p.grad is not None for p in cal.network.parameters() if p.requires_grad)


def test_metacalibrator_e2e_allows_gradients_into_backbone_inputs():
    cal = MetaCalibrator(feature_dims=[4, 4], logit_dim=3, mode="e2e")
    f0 = torch.randn(6, 4, requires_grad=True)
    f1 = torch.randn(6, 4, requires_grad=True)
    logits = torch.randn(6, 3, requires_grad=True)
    loss = cal([f0, f1], logits).square().mean()
    loss.backward()
    assert f0.grad is not None and f1.grad is not None and logits.grad is not None


def test_metacalibrator_legacy_partial_detach_keeps_features():
    cal = MetaCalibrator(feature_dims=[4, 4], logit_dim=3, mode="legacy_partial_detach")
    f0 = torch.randn(6, 4, requires_grad=True)
    logits = torch.randn(6, 3, requires_grad=True)
    loss = cal([f0, torch.randn(6, 4, requires_grad=True)], logits).square().mean()
    loss.backward()
    assert f0.grad is not None          # v1 leak reproduced deliberately
    assert logits.grad is None          # v1 did detach logits


def test_metacalibrator_end_to_end_deprecated_alias():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cal = MetaCalibrator([4, 4], 3, end_to_end=False)
        assert any(issubclass(x.category, DeprecationWarning) for x in w)
        assert cal.mode == "legacy_partial_detach"


def test_meta_calibrator_logits_only_concat():
    # logits_only drops the feature taps entirely; the MLP input dim is the
    # logit dim (scsf.py computes input_dim accordingly).
    cal = MetaCalibrator([4, 4], logit_dim=3, mode="posthoc", logits_only=True)
    x = torch.zeros(5, 4)
    out = cal([x, x], torch.randn(5, 3))
    assert tuple(out.shape) == (5,)
    assert ((out >= 0) & (out <= 1)).all()


def test_selectivenet_loss_matches_explicit_formula():
    m = build_method("selectivenet", _cfg(alpha=0.5, lm=32.0))
    x = torch.zeros(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3])
    f, g, h = m._forward(x)
    gv = g.view(-1)
    per_sample_ce = F.cross_entropy(f, y, reduction="none")
    cov = gv.mean().item()
    emp_risk = ((per_sample_ce * gv).mean() / gv.mean()).item()
    penalty = 32.0 * max(0.5 - cov, 0.0) ** 2
    l_ce = F.cross_entropy(h, y).item()
    expected = 0.5 * (emp_risk + penalty) + 0.5 * l_ce
    got = m.train_loss((x, y), None)["selective"].item()
    assert math.isclose(got, expected, rel_tol=1e-5)


def test_selectivenet_selector_output_is_conf_and_scores_present():
    m = build_method("selectivenet", _cfg())
    x = torch.zeros(6, 3, 32, 32)
    mp = m.predict_batch(x)
    assert mp.scores["selection"] is mp.confidence
    assert set(mp.scores) >= {"msp", "entropy", "energy", "logit_margin", "selection"}
    assert ((mp.confidence >= 0) & (mp.confidence <= 1)).all()


def test_dg_doubling_rate_and_reservation_helpers():
    from scsf.methods.ce import _dg_r, _reservation

    raw = torch.tensor([[3.0, 1.0, 0.0, 0.0], [0.0, 2.0, 1.0, 0.5]])
    res = _reservation(raw, 3)
    assert torch.allclose(res, torch.softmax(raw, dim=1)[:, 3])
    dg = _dg_r(raw, 3)
    assert dg[0] > dg[1]  # higher main-class logsumexp -> higher reject stat
    m = build_method("dg", _cfg())
    x = torch.zeros(4, 3, 32, 32)
    out = m.train_loss((x, torch.zeros(4, dtype=torch.long)), None)
    assert "dg" in out and torch.isfinite(out["dg"]) and out["dg"] > 0
    assert m.num_outputs == 11  # C + 1 reservation neuron


def test_sat_history_buffer_mixes_and_persists():
    sat = SelfAdaptiveTrainingLoss(num_examples=6, num_classes=3, mom=0.9)
    x = torch.randn(2, 4)  # C=3 main logits + 1 reservation column
    y = torch.tensor([0, 2])
    idx = torch.tensor([1, 4])
    loss1 = sat(x, y, idx)  # forward(logits, y, index)
    assert sat.updated[1] == 1 and sat.updated[4] == 1
    # first pass uses the onehot prior: prob_hist = 0.9*onehot + 0.1*softmax
    expected = 0.9 + 0.1 * float(torch.softmax(x[0, :3], 0)[0])
    assert math.isclose(float(sat.prob_history[1, 0]), expected, rel_tol=1e-5)
    # second pass reuses history (not reset to onehot): value changes smoothly
    before = float(sat.prob_history[1, 0])
    loss2 = sat(x, y, idx)
    after = float(sat.prob_history[1, 0])
    assert before > 0.8  # dominated by the mixing that started at 0.9
    assert abs(after - before) < 0.2
    assert torch.isfinite(loss1) and torch.isfinite(loss2)


def test_sat_soft_label_normalized_and_motivation_correct():
    sat = SelfAdaptiveTrainingLoss(num_examples=8, num_classes=2, mom=0.0)
    logits = torch.randn(8, 3)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    idx = torch.arange(8)
    loss = sat(logits, y, idx)
    assert torch.isfinite(loss)
    # mom=0: history equals the (detached) softmax over the main columns
    p = torch.softmax(logits[:, :2], dim=1)
    assert torch.allclose(sat.prob_history[idx, y].float(), p.gather(1, y.view(-1, 1)).squeeze(1),
                          atol=1e-6)


def test_ccl_sc_queues_fill_and_cycle():
    moco = CSCMoCo(dim=4, K=8, m=0.999, T=0.1, num_class=3)
    labels = torch.arange(8) % 3
    # batch 1: every key predicted wrong -> error queue wraps (full_k1)
    preds_err = (labels + 1) % 3
    with torch.no_grad():
        moco._dequeue_and_enqueue(torch.randn(8, 4), torch.randn(0, 4),
                                  torch.empty(0, dtype=torch.long), preds_err)
    assert bool(moco.full_k1) and not bool(moco.full_k2)
    assert int(moco.queue_ptr) == 0  # wrapped back to start
    # batch 2: every key predicted right -> correct queue wraps (full_k2)
    with torch.no_grad():
        moco._dequeue_and_enqueue(torch.randn(0, 4), torch.randn(8, 4),
                                  labels, torch.empty(0, dtype=torch.long))
    assert bool(moco.full_k2)
    assert int(moco.correct_queue_ptr) == 0  # cyclic pointer returns to 0
    assert int(moco.queue_ptr) == 0


def test_ccl_sc_forward_backprop_builds_graph_and_never_nans():
    moco = CSCMoCo(dim=4, K=8, m=0.999, T=0.1, num_class=3)
    labels = torch.arange(8) % 3
    with torch.no_grad():
        moco._dequeue_and_enqueue(
            torch.randn(8, 4), torch.randn(0, 4),
            torch.empty(0, dtype=torch.long), (labels + 1) % 3)
        moco._dequeue_and_enqueue(
            torch.randn(0, 4), torch.randn(8, 4),
            labels, torch.empty(0, dtype=torch.long))
    assert bool(moco.full_k1) and bool(moco.full_k2)
    q = torch.randn(16, 4, requires_grad=True)
    targets = torch.arange(16) % 3
    outputs = torch.randn(16, 3)
    outputs_k = torch.randn(16, 3)
    loss = moco(q, targets, outputs, outputs_k)
    assert torch.isfinite(loss)  # official info-nce can be slightly negative
    loss.backward()
    assert q.grad is not None


def test_ccl_sc_momentum_and_queue_excluded_from_optimizer():
    m = build_method("ccl_sc", _cfg(queue_size=64, pretrain=1))
    specs = m.optimizer_specs()
    assert len(specs) == 1
    momentum_params = set(map(id, m.model_k.parameters()))
    for p in specs[0]["params"]:
        assert id(p) not in momentum_params
    assert all(p.requires_grad is False for p in m.model_k.parameters())


def test_scsf_optimizer_specs_dual_optimizer_clean_split():
    m = build_method("scsf", _cfg(mode="posthoc"))
    specs = m.optimizer_specs()
    kinds = [s["kind"] for s in specs]
    assert "sgd" in kinds and "adam" in kinds
    calib_ids = set(map(id, m._calib.parameters()))
    sgd_params = [s["params"] for s in specs if s["kind"] == "sgd"][0]
    adam_params = [s["params"] for s in specs if s["kind"] == "adam"][0]
    adam_ids = set(map(id, adam_params))
    assert len(calib_ids & adam_ids) == len(calib_ids)   # every calib param in adam
    assert not (calib_ids & set(map(id, sgd_params)))    # none leaked into sgd