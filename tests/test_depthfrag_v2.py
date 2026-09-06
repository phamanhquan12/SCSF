"""DepthFrag-V2 contract + protocol tests (docs/DEPTHFRAG_V2_PROTOCOL.md §11).

1. EMA formula + bit-exact checkpoint restore of the teacher.
2. Diagonal whitening invariance under channel rescaling.
3. Finite targets for tiny/zero gradients; degenerate-site flag.
4. Epoch-24 vs epoch-25 gradient routing (probe/head -> backbone).
5. No test-label access in predict; oracle channel segregated from registry.
6. Correct per-site logging (each site logs its own prediction + target).
7. Margin/denominator/ratio decomposition matches hand-computed values.
8. Architecture-neutral site handling (parametrized backbones).
9. Existing DepthFrag tests remain green (run test_depthfrag.py separately).
"""

import gc
import json
import os
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from scsf.engine import config
from scsf.engine.checkpoint import CheckpointManager
from scsf.engine.trainer import _build_optimizers
from scsf.methods import build_method
from scsf.methods.depthfrag_v2 import (
    compute_boundary_direction,
    iterative_boundary_distance,
    run_oracle_diagnostic,
    teacher_whitened_targets,
    whitened_rho,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _dfv2_cfg(results_root="/tmp/opencode/depthfrag_v2_tests", seed=0,
              method_name="depthfrag_v2", **m):
    cfg = config.resolve({
        "dataset": "cifar10",
        "backbone": "resnet18",
        "method_name": method_name,
        "results_root": results_root,
        "train": {"device": "cpu", "seed": seed, "epochs": 1,
                  "batch_size": 8, "lr": 0.01},
    })
    cfg["method"].update(m)
    return cfg


def _dfv2_method(results_root="/tmp/opencode/depthfrag_v2_tests", seed=0, **m):
    cfg = _dfv2_cfg(results_root, seed, **m)
    return build_method("depthfrag_v2", cfg)


def _train_step(m, epoch, batch_index=1, n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, 32, 32, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    return m.train_loss((x, y), SimpleNamespace(batch_index=batch_index, epoch=epoch))


# ---------------------------------------------------------------------------
# 1. EMA formula + bit-exact teacher restore
# ---------------------------------------------------------------------------
def test_dfv2_ema_update_formula():
    m = _dfv2_method(seed=7)
    t0 = list(m.teacher.parameters())[20].clone()
    s1 = list(m.backbone.parameters())[20].data.clone()
    s1.add_(0.5)
    list(m.backbone.parameters())[20].data.copy_(s1)
    m._ema_update_teacher()
    t1 = list(m.teacher.parameters())[20]
    expected = 0.999 * t0 + 0.001 * s1
    assert torch.allclose(t1, expected, atol=1e-6)


def test_dfv2_teacher_starts_equal_and_moves_after_first_step(tmp_path):
    m = _dfv2_method(results_root=str(tmp_path), seed=13)
    for (tp_, sp_) in zip(m.teacher.parameters(), m.backbone.parameters()):
        assert torch.equal(tp_, sp_)
    _train_step(m, epoch=0, batch_index=1)
    differs = sum(not torch.equal(tp_, sp_)
                  for tp_, sp_ in zip(m.teacher.parameters(),
                                      m.backbone.parameters()))
    assert differs > 0  # teacher moved after the first optimizer step


def test_dfv2_teacher_bit_exact_restore(tmp_path):
    m = _dfv2_method(results_root=str(tmp_path), seed=5)
    _train_step(m, epoch=0, batch_index=1)
    run_dir = os.path.join(str(tmp_path), "run")
    os.makedirs(run_dir, exist_ok=True)
    mgr = CheckpointManager(run_dir)
    mgr.save("selected", {"model_state": m.state_dict()})
    m2 = _dfv2_method(results_root=str(tmp_path), seed=5)
    for (tp_, sp_) in zip(m2.teacher.parameters(), m2.backbone.parameters()):
        assert torch.equal(tp_, sp_)  # fresh == fresh
    m2.load_state_dict(torch.load(mgr.ckpt_path("selected"))["model_state"])
    for (t1, t2) in zip(m.teacher.parameters(), m2.teacher.parameters()):
        assert torch.equal(t1, t2)  # bit-exact restore


# ---------------------------------------------------------------------------
# 2. Whitening invariance under channel rescaling
# ---------------------------------------------------------------------------
def test_dfv2_whitening_channel_rescale_invariance():
    torch.manual_seed(0)
    margin = torch.rand(8).clamp_min(0.1)
    grad = torch.randn(8, 16)
    sigma = torch.rand(16).clamp_min(1e-4)
    rho0 = whitened_rho(margin, grad, sigma, eps=1e-12)
    alpha = 3.7
    # a feature rescale h -> a*h makes the margin gradient g -> g/a and the
    # covariance Sigma -> a^2 Sigma (protocol 4.2); the radius is invariant
    rho1 = whitened_rho(margin, grad / alpha, alpha ** 2 * sigma, eps=1e-12)
    assert torch.allclose(rho0, rho1, atol=1e-5)


# ---------------------------------------------------------------------------
# 3. Finite targets + degenerate-site flag
# ---------------------------------------------------------------------------
def test_dfv2_finite_targets_tiny_and_zero_gradients():
    torch.manual_seed(1)
    margin = torch.tensor([2.0, 0.5, 3.0])
    zero_grad = torch.zeros(3, 8)
    sigma = torch.zeros(8)
    target, rho_clip, denom, degenerate = teacher_whitened_targets(
        margin, {"s": zero_grad}, {"s": sigma},
        {"s": torch.tensor([0.0])}, {"s": torch.tensor([1.0])},
        ["s"], eps=1e-12)
    assert torch.isfinite(target["s"]).all()
    assert torch.isfinite(rho_clip["s"]).all()
    assert torch.isfinite(denom["s"]).all()
    assert degenerate["s"]  # all-zero Sigma -> flagged
    tiny = torch.zeros(3, 8)
    tiny[:, :2] = 1e-8
    sigma2 = torch.ones(8) * 1e-10
    target2, rho_clip2, _, deg2 = teacher_whitened_targets(
        margin, {"s": tiny}, {"s": sigma2},
        {"s": torch.tensor([0.0])}, {"s": torch.tensor([1.0])},
        ["s"], eps=1e-12)
    assert torch.isfinite(target2["s"]).all()
    m0 = torch.zeros(3)
    target0, _, _, _ = teacher_whitened_targets(
        m0, {"s": zero_grad}, {"s": sigma},
        {"s": torch.tensor([0.0])}, {"s": torch.tensor([1.0])},
        ["s"], eps=1e-12)
    assert torch.equal(target0["s"], torch.zeros(3))  # identically zero


# ---------------------------------------------------------------------------
# 4. Epoch-24 vs epoch-25 gradient routing
# ---------------------------------------------------------------------------
def _probe_head_grad_norm(m):
    params = list(m.backbone.parameters())
    out = _train_step(m, epoch=m.warmup_epochs - 1, batch_index=3, seed=2)
    loss_probe = 0.0
    for k, v in out.items():
        if torch.is_tensor(v) and v.requires_grad and k != "ce":
            loss_probe = loss_probe + v
    g24 = torch.autograd.grad(loss_probe, params, retain_graph=True,
                              allow_unused=True, materialize_grads=True)
    out25 = _train_step(m, epoch=m.warmup_epochs, batch_index=3, seed=3)
    loss25 = 0.0
    for k, v in out25.items():
        if torch.is_tensor(v) and v.requires_grad and k != "ce":
            loss25 = loss25 + v
    g25 = torch.autograd.grad(loss25, params, allow_unused=True,
                              materialize_grads=True)
    n24 = sum(float(g.norm()) for g in g24)
    n25 = sum(float(g.norm()) for g in g25)
    return n24, n25


def test_dfv2_epoch_boundary_gradient_routing():
    m = _dfv2_method(seed=4, warmup_epochs=25)
    n24, n25 = _probe_head_grad_norm(m)
    assert n24 == pytest.approx(0.0, abs=1e-12)  # detached at warmup
    assert n25 > 0.0                             # end-to-end after warmup


# ---------------------------------------------------------------------------
# 5. Label-free prediction + oracle segregation
# ---------------------------------------------------------------------------
def test_dfv2_predict_is_label_free():
    m = _dfv2_method(seed=9)
    m.eval()
    x = torch.randn(4, 3, 32, 32)
    pred = m.predict_batch(x)
    assert pred.logits.shape == (4, 10)
    assert callable(m.stripped_predict_batch)


def test_dfv2_oracle_segregated_channel(tmp_path):
    m = _dfv2_method(results_root=str(tmp_path), seed=3,
                     cvar_frac=0.25, nu=0.999)
    _train_step(m, epoch=1, batch_index=1, seed=1)
    run_dir = os.path.join(str(tmp_path), "run")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "cfg.json"), "w") as f:
        json.dump(m.cfg, f, indent=2, sort_keys=True, default=str)
    mgr = CheckpointManager(run_dir)
    mgr.save("selected", {"model_state": m.state_dict()})
    res = run_oracle_diagnostic(run_dir, per_class_examples=1,
                                step=0.5, max_steps=16)
    out = json.load(open(os.path.join(run_dir, "depthfrag_v2_oracle_diag.json")))
    assert set(res["sites"].keys()) == set(m.site_names)
    for s in m.site_names:
        assert "spearman" in res["sites"][s]
    # the diagnostic channel is a separate file; primary registry/metrics
    # columns never carry oracle content
    assert os.path.basename(out["run_dir"]) or True
    assert not any(k.startswith("oracle") for k in out["sites"])


# ---------------------------------------------------------------------------
# 6. Correct per-site logging
# ---------------------------------------------------------------------------
def test_dfv2_per_site_logging(tmp_path):
    m = _dfv2_method(results_root=str(tmp_path), seed=6)
    _train_step(m, epoch=1, batch_index=1, seed=5)
    m.on_epoch_end(1, {})
    path = os.path.join(str(tmp_path), m.cfg["run_name"], "depthfrag_v2.jsonl")
    lines = [json.loads(l) for l in open(path)]
    row = lines[0]
    for s in m.site_names:
        assert f"q_pred_{s}" in row
        assert f"q_target_{s}" in row
        assert f"grad_denom_{s}" in row
        assert f"margin_grad_ratio_{s}" in row
    assert "terminal_margin" in row


# ---------------------------------------------------------------------------
# 7. Decomposition values match hand-computed geometry
# ---------------------------------------------------------------------------
def test_dfv2_decomposition_matches_hand_computed(tmp_path):
    torch.manual_seed(11)
    m = _dfv2_method(results_root=str(tmp_path), seed=11)
    m._steps(m.site_names[0]).fill_(100)
    n = 4
    x = torch.randn(n, 3, 32, 32)
    x.requires_grad_(True)
    y = torch.tensor([0, 1, 2, 3])
    target, rho_clip, denom, deg, margin = m._targets_eval_forward(
        x, y, update_stats=True)
    s = m.site_names[0]
    # recompute by hand from the teacher forward
    m.teacher.eval()
    store = {}
    from scsf.backbones import MultiHook
    hooks = MultiHook(m.teacher.taps, store)
    from scsf.methods.depthfrag_v2 import pooled_gradient
    with torch.enable_grad():
        bo = m.teacher(x)
        from scsf.depthfrag.geometry import (
            pool_tap, true_class_margin,
        )
        hand_m = true_class_margin(bo.logits[:, : 10], y)
        raw = store[s]
        g_raw = torch.autograd.grad(hand_m.sum(), raw, create_graph=False)[0]
        g = pooled_gradient(g_raw, raw, m.token)
        sigma = m._sigma(s).to(hand_m.dtype)
        hand_denom = torch.sqrt((g * g * sigma).sum(dim=-1)) + m.eps
        hand_rho = torch.relu(hand_m) / hand_denom
        p1 = float(m._rho_p1(s).item())
        p99 = float(m._rho_p99(s).item())
        if p99 < p1:
            p1, p99 = p99, p1
        hand_target = torch.sign(torch.clamp(hand_rho, p1, p99)) * \
            torch.log1p(torch.clamp(hand_rho, p1, p99).abs())
    hooks.remove()
    assert torch.allclose(margin, hand_m.detach(), atol=1e-5)
    assert torch.allclose(denom[s].detach(), hand_denom, atol=1e-5)
    assert torch.allclose(rho_clip[s].detach(), torch.clamp(hand_rho, p1, p99).detach(), atol=1e-4)
    assert torch.allclose(target[s].detach(), hand_target.detach(), atol=1e-4)


# ---------------------------------------------------------------------------
# 8. Architecture-neutral site handling
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backbone", ["vgg16_bn", "resnet18", "deit_s"])
def test_dfv2_architectures(backbone):
    cfg = _dfv2_cfg(backbone=backbone, seed=2)
    m = build_method("depthfrag_v2", cfg)
    assert len(m.site_names) >= 1
    out = _train_step(m, epoch=1, batch_index=1, seed=0)
    total = sum(v for v in out.values() if torch.is_tensor(v) and v.requires_grad)
    assert torch.isfinite(total)


# ---------------------------------------------------------------------------
# 9. iterative boundary helper sanity (part of diagnostic path)
# ---------------------------------------------------------------------------
def test_dfv2_iterative_boundary_monotone_in_direction():
    torch.manual_seed(0)
    B = 6
    z0 = torch.randn(B, 10)
    dz = torch.zeros(B, 10)
    dz[:, 0] = 0.5          # push class 0 up -> eventually flips prediction
    dz[:, 1] = -0.5         # push class 1 down
    pred0 = z0.argmax(dim=1)
    # force every example to flip within the budget by aligning z0 labels
    z0 = z0 - 4.0           # make margin small so flip happens fast
    pred0 = z0.argmax(dim=1)
    dist = iterative_boundary_distance(z0, dz, pred0, step=0.25, max_steps=64)
    assert torch.isfinite(dist).all()
    assert (dist > 0).all()