"""SAGE-DS scientific-integrity tests (utility sign lock + controller math)."""

import math
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from scsf.engine import config
from scsf.engine.trainer import _build_optimizers
from scsf.methods import build_method
from scsf.methods.sage_ds import (
    Controller,
    params_reached_by_aux,
    project_aux,
    selective_utility,
)
from scsf.metrics.surrogate import soft_aurc_surrogate


def _sg_cfg(results_root="/tmp/opencode/sage_ds_tests", seed=0):
    cfg = config.resolve({
        "dataset": "cifar10",
        "backbone": "resnet18",
        "method_name": "sage_ds",
        "results_root": results_root,
        "train": {"device": "cpu", "seed": seed, "epochs": 1,
                  "batch_size": 8, "lr": 0.01},
    })
    return cfg


def _sg_method(results_root="/tmp/opencode/sage_ds_tests", seed=0, **method_overrides):
    cfg = _sg_cfg(results_root, seed)
    cfg["method"].update(method_overrides)
    return build_method("sage_ds", cfg)


# ---------------------------------------------------------------------------
# 1. finite-difference unit test: the sign of `-eta * U` must match the real
#    change of the selective surrogate along `-g_l` (fails if the sign is
#    reversed inside selective_utility / the controller move).
# ---------------------------------------------------------------------------
def _fd_case(seed, D=16, C=10, B=64, tau=0.5, eta=0.2):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, C))
    aux = nn.Linear(D, C)
    xb = torch.randn(B, D)
    yb = torch.randint(0, C, (B,))

    def J_of():
        h = net[0](xb).relu()
        return soft_aurc_surrogate(net[2](h), yb, error_mode="proxy", tau=tau)

    J0 = float(J_of().detach())
    g_sel = torch.autograd.grad(J_of(), list(net.parameters()), retain_graph=True,
                                allow_unused=True, materialize_grads=True)
    h = net[0](xb).relu()
    g_l = torch.autograd.grad(F.cross_entropy(aux(h), yb), list(net.parameters()),
                              retain_graph=True, allow_unused=True,
                              materialize_grads=True)
    assert any(g is not None for g in g_l)
    U = float(selective_utility(g_sel, g_l))
    with torch.no_grad():
        for p, g in zip(net.parameters(), g_l):
            if g is not None:
                p.data.sub_(eta * g)
    fd = float(J_of().detach()) - J0
    return U, fd


def test_sage_ds_finite_difference_utility_sign_lock():
    for seed in range(4):
        for eta in (0.02, 0.2):
            U, fd = _fd_case(seed, eta=eta)
            assert U != 0.0
            assert fd != 0.0
            # theta'' = theta - eta*g_l  ==>  fd ~= -eta * U, same sign
            assert (fd < 0) == (-eta * U < 0), (seed, eta, U, fd)


# ---------------------------------------------------------------------------
# 2. controller: positive selective utility should raise the gate probability,
#    negative utility should lower it.
# ---------------------------------------------------------------------------
def test_sage_ds_controller_helpful_suppresses_harmful():
    ctl = Controller(["a", "b"], tau=0.3, beta=0.9)
    p0a = float(ctl.gate_prob("a").detach())
    p0b = float(ctl.gate_prob("b").detach())
    for _ in range(5):
        ctl.update_utility_ema([1.0, -1.0])
        ctl.step_from_utility(controller_lr=0.1, sparsity_cost=0.0, cap=0.3)
    assert float(ctl.gate_prob("a").detach()) > p0a
    assert float(ctl.gate_prob("b").detach()) < p0b
    assert float(ctl.utility_ema_dict()["a"]) > 0.0
    assert float(ctl.utility_ema_dict()["b"]) < 0.0


# ---------------------------------------------------------------------------
# 3. classification-safety projection invariant: never push against g0_ema
# ---------------------------------------------------------------------------
def test_sage_ds_projection_safety_invariant():
    torch.manual_seed(0)
    g0 = torch.randn(50)
    g_conflict = -0.9 * g0 + 1e-3 * torch.randn(50)
    assert torch.sum(g_conflict * g0) < 0.0
    g_safe, align_before = project_aux(g_conflict, g0)
    assert align_before < 0.0
    # the conflicting component is removed up to float32 cancellation round-off
    assert float(torch.sum(g_safe * g0)) > -1e-4
    assert float(torch.sum(g_safe * g0)) < 1e-4

    g_agree = 0.3 * g0
    g_safe2, align2 = project_aux(g_agree, g0)
    assert align2 > 0.0
    assert torch.allclose(g_safe2, g_agree, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. aux-only backward reaches only the backbone prefix up to the tap
# ---------------------------------------------------------------------------
def test_sage_ds_aux_reached_params_are_prefix_bounded():
    m = _sg_method(seed=1)
    layer_idx = {"layer1": 1, "layer2": 2, "layer3": 3, "layer4": 4}
    for site, submod in m.backbone.taps.items():
        reached = params_reached_by_aux(m, site, num_examples=2)
        assert reached, site
        k = layer_idx[site]
        for name in reached:
            assert not name.startswith("fc"), (site, name)
            for j, idx in layer_idx.items():
                assert not (idx > k and name.startswith(j)), (site, name)
        assert m.backbone.roles["top_l1"] == "layer4"
        assert m.backbone.roles["top_l2"] == "layer3"


# ---------------------------------------------------------------------------
# 5. inference graph is stripped of aux heads + controller (MSP only)
# ---------------------------------------------------------------------------
def test_sage_ds_msp_inference_excludes_aux_and_controller():
    m = _sg_method(seed=2)
    x = torch.randn(6, 3, 32, 32)
    mp1 = m.predict_batch(x)
    assert torch.equal(mp1.confidence, mp1.scores["sage_conf"])
    assert set(mp1.scores) >= {"msp", "entropy", "energy", "logit_margin", "sage_conf"}

    # deployment modules = backbone only (risk head off); aux heads/controller are
    # training-only instruments absent from the deployment graph
    infer_params = set()
    for mod in m.inference_modules():
        infer_params |= {id(p) for p in mod.parameters()}
    all_params = set(id(p) for p in m.parameters())
    assert infer_params == all_params - set(map(id, m.aux_heads.parameters())) \
        - set(map(id, m.controller.parameters()))

    # mutating aux/controller state must leave predictions bit-identical
    with torch.no_grad():
        for p in m.aux_heads.parameters():
            p.uniform_(-1.0, 1.0)
        for g in m.controller.gates.values():
            g.log_alpha.fill_(7.0)
    mp2 = m.predict_batch(x)
    assert torch.allclose(mp2.logits, mp1.logits)
    assert torch.equal(mp2.prediction, mp1.prediction)
    assert torch.allclose(mp2.confidence, mp1.confidence)


# ---------------------------------------------------------------------------
# 6. checkpoint resume preserves controller EMA + gates exactly
# ---------------------------------------------------------------------------
def test_sage_ds_checkpoint_resume_preserves_controller_gates_and_ema(tmp_path):
    m = _sg_method(results_root=str(tmp_path), seed=3)
    utilities = [0.9, -0.4, 0.2, -0.1]
    for _ in range(3):
        m.controller.update_utility_ema(utilities)
        m.controller.step_from_utility(controller_lr=0.05,
                                       sparsity_cost=0.1, cap=0.5)
    with torch.no_grad():
        m.controller.gates["layer3"].log_alpha.data.copy_(torch.tensor(2.3))
    ckpt = os.path.join(tmp_path, "ckpt.pt")
    torch.save(m.state_dict(), ckpt)

    m2 = _sg_method(results_root=str(tmp_path), seed=3)
    m2.load_state_dict(torch.load(ckpt, weights_only=True))
    assert m2.controller.utility_ema_dict() == m.controller.utility_ema_dict()
    assert int(m2.controller.ui_step) == int(m.controller.ui_step)
    for s in m.site_names:
        assert float(m2.controller.gate_prob(s)) == pytest.approx(
            float(m.controller.gate_prob(s)))


# ---------------------------------------------------------------------------
# 7. every registered backbone exposes taps/roles and builds a sage_ds method
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backbone",
                         ["resnet18", "vgg16_bn", "wideresnet28_10", "convnext_tiny", "deit_s"])
def test_sage_ds_config_loads_across_backbones(tmp_path, backbone):
    cfg = config.resolve({
        "dataset": "cifar10",
        "backbone": backbone,
        "method_name": "sage_ds",
        "results_root": str(tmp_path),
        "train": {"device": "cpu", "seed": 0},
    })
    m = build_method("sage_ds", cfg)
    assert set(m.backbone.roles) >= {"top_l1", "top_l2"}
    assert set(m.backbone.taps) >= set(m.site_names)
    m.eval()  # BN running statistics: batch size 1 is fine under eval
    x = torch.randn(1, 3, 32, 32)
    mp = m.predict_batch(x)
    assert tuple(mp.confidence.shape) == (1,)


# ---------------------------------------------------------------------------
# 8. one engine-style training step: routed gradients land on backbone + aux
#    heads while the controller gates stay manual (no optimizer gradient)
# ---------------------------------------------------------------------------
def test_sage_ds_train_step_routes_gradients_and_leaves_gates_manual():
    cfg = _sg_cfg()
    cfg["train"]["seed"] = 7
    m = build_method("sage_ds", cfg)
    m.train()
    opt = _build_optimizers(m, cfg)[0]

    x = torch.randn(6, 3, 32, 32)
    y = torch.randint(0, 10, (6,))
    state = SimpleNamespace(batch_index=5)   # not a multiple of 50 -> no utility call
    loss_dict = m.train_loss((x, y, torch.arange(6)), state)
    total = sum(v for v in loss_dict.values() if torch.is_tensor(v) and v.requires_grad)
    assert torch.isfinite(total)
    for p in m.backbone.parameters():
        p.grad = None
    total.backward()
    got_backbone = any(p.grad is not None and bool(torch.any(p.grad != 0))
                       for p in m.backbone.parameters())
    got_aux = any(p.grad is not None and bool(torch.any(p.grad != 0))
                  for p in m.aux_heads.parameters())
    gate_grads = {s: None if g.log_alpha.grad is None else g.log_alpha.grad.item()
                  for s, g in m.controller.gates.items()}
    assert got_backbone and got_aux
    assert all(gg is None or gg == 0.0 for gg in gate_grads.values())

    before = {n: p.clone() for n, p in m.named_parameters()}
    opt.step()
    moved = [n for n, p in m.named_parameters() if not torch.equal(p, before[n])]
    assert any(n.startswith("backbone.") for n in moved)
    assert any("aux_heads" in n for n in moved)
    assert all(n.startswith("backbone.") or "aux_heads" in n for n in moved)


def test_sage_ds_utility_interval_passes_training_device(monkeypatch):
    """Exercise the utility branch that the ordinary one-step smoke skips."""
    cfg = _sg_cfg()
    cfg["method"]["utility_interval"] = 1
    m = build_method("sage_ds", cfg)
    m.train()
    seen = []
    monkeypatch.setattr(m, "_estimate_utilities", lambda device: seen.append(device))

    x = torch.randn(2, 3, 32, 32)
    y = torch.randint(0, 10, (2,))
    loss_dict = m.train_loss(
        (x, y, torch.arange(2)), SimpleNamespace(batch_index=1))

    assert seen == [next(m.backbone.parameters()).device]
    assert all(torch.isfinite(value).all() for value in loss_dict.values()
               if torch.is_tensor(value))
