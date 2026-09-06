"""RiskFlow-V2 protocol test suite (protocol section 10).

Covers: previous-state stop-gradient, intended backbone-gradient flow, both
innovation signs reachable, innovation bounds, EMA target stability + exact
resume, zero-error batches, label-free inference, confidence direction on a
synthetic ordered population, architecture-neutral taps, and (in the full
suite run) the unchanged existing RiskFlow tests.
"""

import json
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from scsf.engine import config
from scsf.engine.trainer import _build_optimizers
from scsf.methods import build_method
from scsf.methods.riskflow_v2 import (
    RiskFlowV2Method,
    _soft_target,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rfv2_cfg(results_root="/tmp/opencode/riskflow_v2_tests", seed=0):
    return config.resolve({
        "dataset": "cifar10",
        "backbone": "resnet18",
        "method_name": "riskflow_v2",
        "results_root": results_root,
        "train": {"device": "cpu", "seed": seed, "epochs": 1,
                  "batch_size": 8, "lr": 0.01},
    })


def _rfv2_method(results_root="/tmp/opencode/riskflow_v2_tests", seed=0, **m):
    cfg = _rfv2_cfg(results_root, seed)
    cfg["method"].update(m)
    return build_method("riskflow_v2", cfg)


def _rand_batch(m, B=6, seed=0):
    torch.manual_seed(seed)
    return (torch.randn(B, m.backbone.channels, m.backbone.input_size,
                        m.backbone.input_size),
            torch.randint(0, m.num_classes, (B,)))


def _train_step(m, epoch=0, batch_index=1, seed=1):
    """One engine-style step: record EMA update, loss, backward, optimizer."""
    cfg = _rfv2_cfg(seed=seed)
    x, y = _rand_batch(m, B=6, seed=seed)
    from types import SimpleNamespace
    st = SimpleNamespace(epoch=epoch, batch_index=batch_index)
    optimizer = _build_optimizers(m, cfg)[0]
    loss_dict = m.train_loss((x, y, torch.arange(6)), st)
    for p in m.parameters():
        p.grad = None
    total = sum(v for v in loss_dict.values() if torch.is_tensor(v)
                and v.requires_grad)
    total.backward()
    optimizer.step()
    return loss_dict


# ---------------------------------------------------------------------------
# 1. previous-state stop-gradient: L_l ignores risk modules of stages < l
# ---------------------------------------------------------------------------
def test_rfv2_previous_stage_stop_gradient():
    m = _rfv2_method(seed=1)
    m.train()
    x, y = _rand_batch(m, B=6, seed=2)
    with torch.enable_grad():
        bo = m.backbone(x)
        tl = m._teacher_logits(x)
        flow = m._flow(bo, y=y, teacher_logits=tl)
        for i in range(1, len(m.site_names)):       # site-index stage: s_hard[i+1]
            L_i = F.binary_cross_entropy_with_logits(flow.s_hard[i + 1],
                                                     flow.hard_error)
            for j in range(i):                      # cell/adapter of earlier stage
                g_cell = torch.autograd.grad(
                    L_i, list(m.cells[m.site_names[j]].parameters()),
                    allow_unused=True, retain_graph=True)
                g_ada = torch.autograd.grad(
                    L_i, list(m.adapters[m.site_names[j]].parameters()),
                    allow_unused=True, retain_graph=True)
                for g in g_cell + g_ada:
                    assert g is None or bool(torch.all(g == 0))
            # current stage is reachable
            g_cur = torch.autograd.grad(
                L_i, list(m.cells[m.site_names[i]].parameters()),
                allow_unused=True, retain_graph=True)
            assert any(g is not None and bool(torch.any(g != 0))
                       for g in g_cur)


# ---------------------------------------------------------------------------
# 2. intended current-feature / backbone-prefix gradients flow
# ---------------------------------------------------------------------------
def test_rfv2_current_backbone_prefix_gradflows():
    m = _rfv2_method(seed=3)
    m.train()
    x, y = _rand_batch(m, B=6, seed=4)
    with torch.enable_grad():
        bo = m.backbone(x)
        tl = m._teacher_logits(x)
        flow = m._flow(bo, y=y, teacher_logits=tl)
        L_1 = F.binary_cross_entropy_with_logits(flow.s_hard[1],
                                                 flow.hard_error)
        # cells/adapters of the current stage receive gradient
        g_state = torch.autograd.grad(
            L_1, list(m.cells[m.site_names[0]].parameters())
            + list(m.adapters[m.site_names[0]].parameters()),
            allow_unused=True, retain_graph=True)
        assert any(g is not None and bool(torch.any(g != 0)) for g in g_state)
        # the backbone prefix feeding the first site receives gradient
        g_pre = torch.autograd.grad(
            L_1, m.backbone.base_model.conv1.weight, retain_graph=True)
        assert bool(torch.any(g_pre[0] != 0))


# ---------------------------------------------------------------------------
# 3. positive and negative corrections both reachable
# ---------------------------------------------------------------------------
def test_rfv2_both_innovation_signs_reachable():
    m = _rfv2_method(seed=5)
    m.train()
    x, y = _rand_batch(m, B=6, seed=6)
    bo = m.backbone(x)
    # Teacher that is NEVER wrong -> BCE drives every stage's innovation down.
    tl_easy = torch.zeros(bo.logits[:, : m.num_classes].shape)
    tl_easy.scatter_(1, y.view(-1, 1), 10.0)
    tl_easy = tl_easy.detach()
    with torch.enable_grad():
        flow = m._flow(bo, y=y, teacher_logits=tl_easy)
        assert torch.all(flow.hard_error == 0)
        L = sum(F.binary_cross_entropy_with_logits(flow.s_hard[l], flow.hard_error)
                for l in range(len(m.site_names) + 1))
        g_neg = torch.autograd.grad(
            L, list(m.cells[m.site_names[0]].parameters()), allow_unused=True)
        assert any(g is not None and bool(torch.any(g != 0)) for g in g_neg)
    # Push the same first-stage innovation negative with optimizer steps.
    m2 = _rfv2_method(seed=7)
    m2.train()
    cfg = _rfv2_cfg(seed=7)
    cfg["train"]["lr"] = 0.1
    opt = _build_optimizers(m2, cfg)[0]
    for _ in range(40):
        opt.zero_grad(set_to_none=True)
        with torch.enable_grad():
            bo = m2.backbone(x)
            flow = m2._flow(bo, y=y, teacher_logits=tl_easy)
            loss = sum(F.binary_cross_entropy_with_logits(
                flow.s_hard[l], flow.hard_error)
                for l in range(0, 2))            # base + first site only
            loss.backward()
        opt.step()
    with torch.enable_grad():
        bo = m2.backbone(x)
        flow = m2._flow(bo, y=y, teacher_logits=tl_easy)
    innov = flow.innov_hard[0]
    assert float(innov.mean()) < -1e-3
    # And the opposite sign is reachable: teacher always wrong (fresh
    # optimizer so the negative-phase momentum cannot fight the reversal).
    tl_hard = torch.zeros_like(tl_easy)
    tl_hard.scatter_(1, ((y + 1) % m2.num_classes).view(-1, 1), 10.0)
    opt = _build_optimizers(m2, cfg)[0]
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        with torch.enable_grad():
            bo = m2.backbone(x)
            flow = m2._flow(bo, y=y, teacher_logits=tl_hard)
            loss = sum(F.binary_cross_entropy_with_logits(
                flow.s_hard[l], flow.hard_error)
                for l in range(0, 2))
            loss.backward()
        opt.step()
    with torch.enable_grad():
        bo = m2.backbone(x)
        flow = m2._flow(bo, y=y, teacher_logits=tl_hard)
    innov = flow.innov_hard[0]
    assert float(innov.mean()) > 1e-3


# ---------------------------------------------------------------------------
# 4. innovation bounds
# ---------------------------------------------------------------------------
def test_rfv2_innovation_bounds():
    m = _rfv2_method(seed=9)
    m.eval()
    for s in range(4):
        x, y = _rand_batch(m, B=5, seed=10 + s)
        _, flow = m.predict_with_trace(x, y=y)
        assert torch.all(flow.innov_hard.abs() <= m.delta_max + 1e-6)
        if flow.innov_soft is not None:
            assert torch.all(flow.innov_soft.abs() <= m.delta_max + 1e-6)


# ---------------------------------------------------------------------------
# 5. EMA target stability + exact resume
# ---------------------------------------------------------------------------
def test_rfv2_ema_moves_toward_student_and_buffers_track():
    m = _rfv2_method(seed=11)
    tw = m.teacher.base_model.conv1.weight
    tb = m.teacher.base_model.bn1.running_mean
    t0_w = tw.clone()
    t0_b = tb.clone()
    with torch.no_grad():
        m.backbone.base_model.conv1.weight.add_(0.05)
        m.backbone.base_model.bn1.running_mean.add_(0.25)
        m._ema_update_teacher()
    # opaque tensor copies must match the closed-form buffered EMA exactly
    assert torch.allclose(tw, m.nu * t0_w
                          + (1.0 - m.nu) * m.backbone.base_model.conv1.weight)
    assert torch.allclose(tb, m.nu * t0_b
                          + (1.0 - m.nu) * m.backbone.base_model.bn1.running_mean)
    # a real training step leaves the teacher a strict EMA (never equal)
    s_before = m.backbone.base_model.conv1.weight.clone()
    _train_step(m, epoch=1, batch_index=1, seed=1)
    t_now = m.teacher.base_model.conv1.weight.clone()
    assert not torch.equal(t_now, s_before)
    assert torch.isfinite(t_now).all()


def test_rfv2_exact_resume(tmp_path):
    m = _rfv2_method(results_root=str(tmp_path), seed=13)
    _train_step(m, epoch=1, batch_index=1, seed=3)
    _train_step(m, epoch=1, batch_index=2, seed=4)
    ckpt = os.path.join(tmp_path, "rfv2.pt")
    torch.save(m.state_dict(), ckpt)
    m2 = _rfv2_method(results_root=str(tmp_path), seed=13)
    m2.load_state_dict(torch.load(ckpt, weights_only=True))
    sd = m2.state_dict()
    for n, p in m.named_parameters():
        assert torch.allclose(p, sd[n]), n
    for n, b in m.named_buffers():
        assert torch.allclose(b, sd[n]), n
    m.eval()
    m2.eval()
    x, y = _rand_batch(m, B=4, seed=14)
    _, f1 = m.predict_with_trace(x, y=y)
    _, f2 = m2.predict_with_trace(x, y=y)
    assert torch.allclose(f2.s_hard, f1.s_hard)
    assert torch.allclose(f2.final_s_hard, f1.final_s_hard)
    assert torch.allclose(f2.innov_hard, f1.innov_hard)


# ---------------------------------------------------------------------------
# 6. zero-error minibatch is finite; states relax downward
# ---------------------------------------------------------------------------
def test_rfv2_zero_errors_batch():
    m = _rfv2_method(seed=15)
    m.train()
    x, y = _rand_batch(m, B=6, seed=16)
    loss_dict = _train_step(m, epoch=1, batch_index=1, seed=16)
    assert torch.isfinite(loss_dict["rfv2_stage_bce"])
    # force all e_T = 0 targets and observe monotonically decreasing logits
    tl_easy = torch.zeros(6, m.num_classes)
    tl_easy.scatter_(1, y.view(-1, 1), 10.0)
    opt = _build_optimizers(m, _rfv2_cfg(seed=16))[0]
    s_before = None
    for _ in range(10):
        opt.zero_grad(set_to_none=True)
        with torch.enable_grad():
            bo = m.backbone(x)
            flow = m._flow(bo, y=y, teacher_logits=tl_easy)
            assert torch.all(flow.hard_error == 0)
            loss = sum(F.binary_cross_entropy_with_logits(flow.s_hard[l], flow.hard_error)
                       for l in range(len(m.site_names) + 1))
            assert torch.isfinite(loss)
            loss.backward()
        opt.step()
        with torch.enable_grad():
            bo = m.backbone(x)
            flow = m._flow(bo, y=y, teacher_logits=tl_easy)
        s_now = flow.final_s_hard.detach()
        assert torch.isfinite(s_now).all()
        if s_before is not None:
            assert torch.all(s_now <= s_before + 1e-4)
        s_before = s_now


# ---------------------------------------------------------------------------
# 7. label-free inference
# ---------------------------------------------------------------------------
def test_rfv2_label_free_inference():
    m = _rfv2_method(seed=17)
    m.eval()
    x, _ = _rand_batch(m, B=4, seed=18)
    mp = m.predict_batch(x)
    conf = mp.scores["riskflow_v2"]
    assert tuple(conf.shape) == (4,)
    assert torch.isfinite(conf).all()
    assert torch.all(conf > 0.0) and torch.all(conf < 1.0)
    # consistent with the primary confidence field
    assert torch.equal(conf, mp.confidence)
    # exact 1 - sigmoid(s_L) relation
    with torch.no_grad():
        flow = m._flow(m.backbone(x))
    assert torch.allclose(conf, 1.0 - torch.sigmoid(flow.final_s_hard), atol=1e-6)


# ---------------------------------------------------------------------------
# 8. confidence direction: higher 1 - sigmoid(s_L) <=> lower error rate
# ---------------------------------------------------------------------------
def test_rfv2_confidence_direction():
    torch.manual_seed(19)
    m = _rfv2_method(seed=19)
    m.train()
    for p in m.backbone.parameters():
        p.requires_grad_(False)               # freeze: isolate the risk stack
    cfg = _rfv2_cfg(seed=19)
    opt = _build_optimizers(m, cfg)[0]
    # synthetic population: teacher errors on the corner-marker examples
    B = 64
    x = torch.randn(B, 3, 32, 32)
    y = torch.randint(0, m.num_classes, (B,))
    marker = 1 - (torch.arange(B) % 2)         # 1 on the first half-group
    x[::2] += 1.0                              # marker: +1 field on group A
    tl = torch.zeros(B, m.num_classes)
    tl.scatter_(1, ((y + marker) % m.num_classes).view(-1, 1), 10.0)
    e = (tl.argmax(dim=1) != y).float()
    assert bool(e[::2].all()) and not bool(e[1::2].any())
    for _ in range(120):
        opt.zero_grad(set_to_none=True)
        with torch.enable_grad():
            bo = m.backbone(x)
            flow = m._flow(bo, y=y, teacher_logits=tl)
            loss = sum(F.binary_cross_entropy_with_logits(flow.s_hard[l], e)
                       for l in range(len(m.site_names) + 1))
            if m.use_soft:
                loss = loss + sum(
                    F.huber_loss(flow.s_soft[l], flow.soft_target,
                                 delta=m.huber_delta)
                    for l in range(len(m.site_names) + 1))
            loss.backward()
        opt.step()
    # validation: on held-out marker populations, confidence anti-correlates
    # with the teacher error frequency
    xs = torch.randn(96, 3, 32, 32)
    xs[::2] += 1.0                            # held-out A group (errors)
    ys = torch.randint(0, m.num_classes, (96,))
    tls = torch.zeros(96, m.num_classes)
    tls.scatter_(1, ((ys + (1 - (torch.arange(96) % 2))) % m.num_classes).view(-1, 1), 10.0)
    es = (tls.argmax(dim=1) != ys).float()
    with torch.no_grad():
        flow = m._flow(m.backbone(xs), y=ys, teacher_logits=tls)
    conf = 1.0 - torch.sigmoid(flow.final_s_hard)
    # group precision: correct group's average confidence strictly exceeds
    # the erroneous group's, by the fitted risk
    assert float(conf[es == 0].mean()) > float(conf[es == 1].mean())
    # rank direction: Spearman(conf, 1 - e) > 0
    ra = conf.argsort().argsort().float()
    rb = (1.0 - es).argsort().argsort().float()
    rho = float(torch.corrcoef(torch.stack([ra, rb]))[0, 1])
    assert rho > 0.2


# ---------------------------------------------------------------------------
# 9. architecture-neutral taps
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backbone", ["resnet18", "vgg16_bn", "deit_s"])
def test_rfv2_arch_neutral_taps(tmp_path, backbone):
    cfg = config.resolve({
        "dataset": "cifar10",
        "backbone": backbone,
        "method_name": "riskflow_v2",
        "results_root": str(tmp_path),
        "train": {"device": "cpu", "seed": 0, "epochs": 1, "batch_size": 8,
                  "lr": 0.01},
    })
    m = build_method("riskflow_v2", cfg)
    assert list(m.site_names) == list(m.backbone.taps.keys())
    m.eval()
    x, y = _rand_batch(m, B=2, seed=20)
    mp = m.predict_batch(x)
    assert tuple(mp.confidence.shape) == (2,)
    _, flow = m.predict_with_trace(x, y=y)
    assert flow.s_hard.shape[0] == len(m.site_names) + 1
    assert torch.isfinite(flow.s_hard).all()


# ---------------------------------------------------------------------------
# primary config contract + logging channel
# ---------------------------------------------------------------------------
def test_rfv2_default_config_contract(tmp_path):
    cfg = config.resolve({
        "dataset": "cifar10",
        "backbone": "resnet18",
        "method_name": "riskflow_v2",
        "results_root": str(tmp_path),
        "train": {"device": "cpu", "seed": 0, "epochs": 1, "batch_size": 8,
                  "lr": 0.01},
    })
    m = build_method("riskflow_v2", cfg)
    assert m.variant == "riskflow_v2"
    assert m.remove_gate is True
    assert m.delta_max == 2.0
    assert m.state_dim == 64 and m.cell_hidden == 64
    # soft-target normalization contract
    tl = torch.randn(4, m.num_classes)
    d = _soft_target(tl, torch.arange(4), m.num_classes)
    assert torch.all(d >= 0.0) and torch.all(d <= 1.0 + 1e-6)
    # primary method excludes the multiplicative gate; gate ablation restores it
    cfg["method"]["remove_gate"] = False
    mg = build_method("riskflow_v2", cfg)
    assert mg.variant == "riskflow_v2_gate"
    assert mg.cells[mg.site_names[0]].gate is True
    assert m.cells[m.site_names[0]].gate is False
    # inference confidence for gate ablation is still 1 - sigmoid(s_L)
    mg.eval()
    mp = mg.predict_batch(torch.randn(1, 3, 32, 32))
    assert torch.isfinite(mp.confidence).all()
    assert m.default_score() == "riskflow_v2"


def test_rfv2_logging_channel(tmp_path):
    m = _rfv2_method(results_root=str(tmp_path), seed=21)
    _train_step(m, epoch=1, batch_index=1, seed=5)
    m.on_epoch_end(1, {})
    path = os.path.join(str(tmp_path), m.cfg["run_name"], "riskflow_v2.jsonl")
    rows = [json.loads(l) for l in open(path)]
    row = rows[0]
    for s in m.site_names:
        assert f"bce_{s}" in row
        assert f"innov_mean_{s}" in row
        assert f"pos_frac_{s}" in row
        assert f"grad_cell_{s}" in row
        assert f"grad_adapter_{s}" in row
    assert "teacher_err" in row
    assert "soft_target_mean" in row
    assert "deployment_macs_per_example" in row
    assert row["deployment_macs_per_example"] > 0
    assert isinstance(m.deployment_overhead["macs_per_example"], int)
    assert set(m.deployment_overhead["per_site"].keys()) == set(m.site_names)


# ---------------------------------------------------------------------------
# engine-style full step + complete training loss keys
# ---------------------------------------------------------------------------
def test_rfv2_train_step_optimizes_whole_stack():
    cfg = _rfv2_cfg()
    cfg["train"]["seed"] = 23
    m = build_method("riskflow_v2", cfg)
    m.train()
    x, y = _rand_batch(m, B=6, seed=24)
    opt = _build_optimizers(m, cfg)[0]
    loss_dict = m.train_loss((x, y, torch.arange(6)), None)
    assert {"ce", "rfv2_stage_bce", "rfv2_soft_huber"} <= set(loss_dict)
    total = sum(v for v in loss_dict.values() if torch.is_tensor(v) and v.requires_grad)
    assert torch.isfinite(total)
    for p in m.parameters():
        p.grad = None
    total.backward()
    before = {n: p.clone() for n, p in m.named_parameters()}
    opt.step()
    moved = [n for n, p in m.named_parameters() if not torch.equal(p, before[n])]
    assert any(n.startswith("backbone.") for n in moved)
    assert any("adapters" in n for n in moved)
    assert any("cells" in n for n in moved)
    # the EMA teacher is never moved by the optimizer
    assert not any(n.startswith("teacher.") for n in moved)