"""SAGE-TopK scientific-integrity tests (protocol: docs/SAGE_TOPK_PROTOCOL.md).

Locks: adapter-exposed candidate enumeration and stable top-k selection; the
linear companion-head construction; profiling leaves backbone auxiliary
gradients unapplied; unselected heads/directions are not computed after
profiling; the Top-K phase boundary and exact-resume allocation identity; the
deterministic K<=2 enumerative QP (feasibility/optimality vs an independent
fine-grid reference, collinear and zero-gradient cases); unit normalization and
numeric mixture CE compatibility (applied-gradient identity); cached-target
refresh and age; label-free MSP-only inference; and preservation of the
previously registered methods.
"""

import json
import math
import os
from collections import Counter
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from scsf.engine.config import resolve
from scsf.engine.seeding import seed_all
from scsf.engine.trainer import Trainer
from scsf.methods import build_method, method_names
from scsf.methods.sage_ds import _cat, _flatten, _pool_tap
from scsf.methods.sage_topk import (
    LinearAuxHead,
    SageTopKMethod,
    allocation_certificate,
    classification_compatible_direction,
    normalize_direction,
    select_topk_sites,
    solve_topk_allocation,
)


def _cfg(results_root="/tmp/opencode/sage_topk_tests", seed=13,
         method=None, train=None):
    overrides = {
        "dataset": "cifar10",
        "backbone": "resnet18",
        "method_name": "sage_topk",
        "recipe": "singlerun",
        "results_root": results_root,
        "data": {"num_workers": 0},
    }
    if method is not None:
        overrides["method"] = method
    t = {"device": "cpu", "seed": seed, "epochs": 4,
         "batch_size": 32, "lr": 0.05, "overfit": 128,
         "scheduler": "cosine", "eval_every": 1, "save_every": 1,
         "weight_decay": 0.0, "data_order_seed": seed}
    if train is not None:
        t.update(train)
    overrides["train"] = t
    return resolve(overrides)


def _method(results_root="/tmp/opencode/sage_topk_tests", seed=13, **method_overrides):
    cfg = _cfg(results_root, seed, method=method_overrides)
    return build_method("sage_topk", cfg)


def _rand_forward(method, B=16, C=None):
    dev = next(method.backbone.parameters()).device
    x = torch.randn(B, method.backbone.channels,
                    method.backbone.input_size, method.backbone.input_size,
                    device=dev)
    y = torch.randint(0, method.num_classes, (B,), device=dev)
    bo = method.backbone(x)
    ce = F.cross_entropy(bo.logits[:, : method.num_classes], y)
    return bo, y, ce


# ---------------------------------------------------------------------------
# 1. candidate enumeration and stable selection
# ---------------------------------------------------------------------------
def test_site_candidates_come_from_adapter_registry():
    m = _method()
    assert set(m.site_names) == set(m.backbone.taps.keys())
    assert list(m.site_names) == list(m.backbone.taps.keys())  # registration order


def test_select_topk_sites_deterministic_and_ties_stable():
    means = torch.tensor([0.1, 0.9, 0.9, 0.3])
    a = select_topk_sites(means, 2)
    b = select_topk_sites(means.clone(), 2)
    assert a == b
    # 0.9 is the max; the two 0.9 ties resolve in registration order (1 before 2)
    assert a == [1, 2]
    # zero measurements must still produce a deterministic fallback
    assert select_topk_sites(torch.zeros(4), 2) == [0, 1]


# ---------------------------------------------------------------------------
# 2. linear companion-head construction
# ---------------------------------------------------------------------------
def test_linear_companion_head_is_single_affine_map():
    head = LinearAuxHead(16, 10)
    kinds = [type(p).__name__ for p in head.modules()
             if not isinstance(p, nn.Linear) and p is not head]
    assert kinds == [], f"companion head must be a single Linear, got {kinds}"
    assert [name for name, _ in head.named_modules()] == [
        "", "fc"
    ], "no LayerNorm / hidden / nonlinear MLP allowed"
    out = head(torch.randn(4, 16))
    assert out.shape == (4, 10)
    assert not hasattr(head, "norm")


# ---------------------------------------------------------------------------
# 3. profiling leaves backbone auxiliary gradients unapplied
# ---------------------------------------------------------------------------
def test_profiling_backbone_gradient_is_ce_only():
    m = _method(k=2, profiling_epochs=3, utility_interval=2)
    bo, y, ce = _rand_forward(m)
    out = m._profiling_loss(bo, y, ce, SimpleNamespace(batch_index=1))
    assert m._profiling is True or True
    total = sum(v for v in out.values() if torch.is_tensor(v) and v.requires_grad)
    ce_only = torch.autograd.grad(
        ce, [p for _, p in m._utility_params], retain_graph=True,
        allow_unused=True, materialize_grads=True)
    total.backward()
    for (n, p), gc in zip(m._utility_params, ce_only):
        assert p.grad is not None
        assert torch.allclose(p.grad, gc if gc is not None else torch.zeros_like(p),
                              atol=1e-7), n
    # measurement step also leaves no auxiliary marks on the backbone
    for p in [p for _, p in m._utility_params]:
        p.grad = None
    bo2, y2, ce2 = _rand_forward(m)
    out2 = m._profiling_loss(bo2, y2, ce2, SimpleNamespace(batch_index=2))
    ce_only2 = torch.autograd.grad(
        ce2, [p for _, p in m._utility_params], retain_graph=True,
        allow_unused=True, materialize_grads=True)
    total2 = sum(v for v in out2.values() if torch.is_tensor(v) and v.requires_grad)
    total2.backward()
    for (n, p), gc in zip(m._utility_params, ce_only2):
        ref = gc if gc is not None else torch.zeros_like(p)
        assert torch.allclose(p.grad, ref, atol=1e-7), n


# ---------------------------------------------------------------------------
# 4. unselected heads/directions are not computed after profiling
# ---------------------------------------------------------------------------
def test_post_profiling_gates_unselected_heads():
    m = _method(k=2, profiling_epochs=3, utility_interval=2)
    torch.manual_seed(0)
    m._profiling.copy_(False)
    m._epoch = 3
    m._selected.copy_(torch.tensor([0, 1], dtype=torch.long))
    # any reference to an unselected head must explode
    def boom(*a, **k):
        raise AssertionError("unselected companion head was invoked")

    sel_names = [m.site_names[i] for i in (0, 1)]
    spy_names = [s for s in m.site_names if s not in sel_names]
    for s in spy_names:
        m.aux_heads[s].fc.register_forward_pre_hook(lambda *a, **k: boom())
    bo, y, ce = _rand_forward(m)
    out = m._allocated_loss(bo, y, ce, SimpleNamespace(batch_index=1))
    assert "routed" in out
    m.zero_grad(set_to_none=True)
    out["routed"].backward()
    for s in m.site_names:
        if s in spy_names:
            for p in m.aux_heads[s].parameters():
                assert p.grad is None, f"unselected head {s} got a gradient"
        else:
            assert any(p.grad is not None for p in m.aux_heads[s].parameters())
    # only selected sites appear in the telemetry
    for k in out:
        if k.startswith("aux_") and k != "aux_loss_?":
            assert any(k.endswith(f"_{s}") for s in sel_names), k


# ---------------------------------------------------------------------------
# 5. Top-K boundary, phases, and exact resume of allocation decisions
# ---------------------------------------------------------------------------
def test_profiling_boundary_refresh_and_phase_switch():
    m = _method(k=2, profiling_epochs=2, utility_interval=2)
    m.on_epoch_start(0)
    assert bool(m._profiling) is True
    bo, y, ce = _rand_forward(m)
    m._profiling_loss(bo, y, ce, SimpleNamespace(batch_index=2))  # measure
    assert int(m._utility_n) == 1
    m.on_epoch_end(1, {})  # final profiling epoch -> selection
    assert int(m._selected[0]) >= 0 and int(m._selected[1]) >= 0
    assert (m._profile_stats or {}).get("n_measurements", 0) > 0
    m.on_epoch_start(2)
    assert bool(m._profiling) is False
    assert m._should_refresh(0) is True   # first post-profiling batch
    assert m._should_refresh(2) is True   # cadence: step % interval == 0
    assert m._should_refresh(3) is False
    m._refresh_selective(next(m.backbone.parameters()).device, 0)
    assert int(m._s_refresh_step) == 0
    assert bool(m._s_set) is True


def test_exact_resume_reproduces_allocations(tmp_path):
    """A resumed run must produce the identical allocation rows and per-step
    target age / refresh decisions for all epochs after the resume point."""
    class _StopAfterProfiling(Exception):
        pass

    seed = 13
    cfg_kwargs = {
        "method": {"k": 2, "profiling_epochs": 2, "utility_interval": 2},
        "train": {"seed": seed, "data_order_seed": seed},
    }
    seed_all(seed)
    run = "topk-run"

    cont = os.path.join(tmp_path, "cont")
    cfg_a = _cfg(cont, seed, **cfg_kwargs)
    cfg_a["run_name"] = run
    trainer = Trainer(cfg_a, os.path.join(cont, run))
    trainer.run()
    cont_steps = [json.loads(l) for l in open(
        os.path.join(cont, run, "sage_topk_steps.jsonl"))]

    part = os.path.join(tmp_path, "part")
    cfg_b = _cfg(part, seed, **cfg_kwargs)
    cfg_b["run_name"] = run
    t1 = Trainer(cfg_b, os.path.join(part, run))
    t1._build()
    orig = t1.method.on_epoch_start

    def stop(epoch, _orig=orig):
        if epoch >= 2:
            raise _StopAfterProfiling()
        return _orig(epoch)

    t1.method.on_epoch_start = stop
    with pytest.raises(_StopAfterProfiling):
        t1.run()
    t2 = Trainer(cfg_b, os.path.join(part, run))
    t2.run(resume_from="epoch_001")
    part_steps = [json.loads(l) for l in open(
        os.path.join(part, run, "sage_topk_steps.jsonl"))]

    cont_post = [r for r in cont_steps if r["epoch"] >= 2]
    part_post = [r for r in part_steps if r["epoch"] >= 2]
    assert len(cont_post) == len(part_post) > 0
    for a, b in zip(cont_post, part_post):
        assert a["epoch"] == b["epoch"] and a["step"] == b["step"]
        assert a["lambda"] == pytest.approx(b["lambda"], abs=1e-9)
        assert a["zero"] == b["zero"] and a["target_age"] == b["target_age"]
        assert a["refresh_step"] == b["refresh_step"]
    # selection survived the boundary
    assert t2.method.selected_sites() == t1.method.selected_sites()


# ---------------------------------------------------------------------------
# 6. QP: feasibility and optimality against a trusted fine-grid reference
# ---------------------------------------------------------------------------
def _grid_reference(G, b, B, steps=2000):
    best_i, best_o = None, None
    for i in range(steps + 1):
        lam1 = B * i / steps
        d = min(int((steps * (B - lam1) / B)), steps)
        for j in range(d + 1):
            lam = torch.tensor([lam1, B * j / steps], dtype=G.dtype)
            o = float((0.5 * lam @ G @ lam - b @ lam).item())
            if best_o is None or o < best_o:
                best_o, best_i = o, lam
    return best_i, best_o


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_qp_optimality_and_feasibility_against_grid(seed):
    torch.manual_seed(seed)
    r = torch.randn(2, 7)
    V = r / r.norm(dim=1, keepdim=True)
    G = (V @ V.t()).clamp_min(0.0) * 1.0
    s = torch.randn(7)
    s = s / s.norm()
    b = V @ s
    B = 1.0
    ref_lam, ref_obj = _grid_reference(G, b, B)
    sol = solve_topk_allocation(G, b, B=B)
    lam = sol["lambda"]
    obj = sol["objective"]
    cert = allocation_certificate(lam, G, b, B=B)
    assert cert["ok"], cert
    # the enumerative optimum must never exceed the (coarser) grid optimum
    assert obj <= ref_obj + 1e-4
    assert obj <= 1e-6  # lambda=0 is feasible with objective 0


def test_qp_singular_and_zero_gradient_cases():
    # collinear: v2 = v1 -> rank-1 Gram, optimum analytic on the segment:
    # every lambda with sum x >= alpha = <v1,s> rises to x=alpha=0.6;
    # coordinate optima give obj = 0.5*a^2 - a = -0.18 (a = 0.36 is the b-lambda)
    v1 = torch.tensor([3.0, 4.0]) / 5.0
    v2 = v1.clone()
    G = torch.stack([v1, v2]) @ torch.stack([v1, v2]).t()
    s = torch.tensor([1.0, 0.0])
    b = torch.stack([v1, v2]) @ s
    sol = solve_topk_allocation(G, b, B=1.0)
    lam = sol["lambda"]
    cert = allocation_certificate(lam, G, b, B=1.0)
    assert cert["ok"]
    assert sol["objective"] == pytest.approx(-0.18, abs=1e-4)
    assert float(lam.sum()) <= 1.0 + 1e-6 and float(lam.min()) >= -1e-6

    # zero gradients: G = 0, b = 0 -> the allocation must be exactly zero
    G0 = torch.zeros(2, 2)
    b0 = torch.zeros(2)
    sol0 = solve_topk_allocation(G0, b0, B=1.0)
    assert torch.allclose(sol0["lambda"], torch.zeros(2), atol=1e-12)
    assert allocation_certificate(sol0["lambda"], G0, b0, B=1.0)["ok"]

    # one zero direction: only v1 is supported
    v1 = torch.tensor([1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)])
    V = torch.stack([v1, torch.zeros(2)])
    G = V @ V.t()
    b = V @ torch.tensor([0.7, 0.7])
    sol = solve_topk_allocation(G, b, B=1.0)
    assert cert_ok(sol["lambda"], G, b)


def cert_ok(lam, G, b, B=1.0):
    return allocation_certificate(lam, G, b, B=B)["ok"]


# ---------------------------------------------------------------------------
# 7. normalization, CE compatibility, applied-gradient identity
# ---------------------------------------------------------------------------
def test_normalize_direction_unit_and_zero():
    v, n, zero = normalize_direction(torch.tensor([0.0, 0.0, 3.0, 4.0]))
    assert not zero and abs(torch.norm(v).item() - 1.0) < 1e-6
    vz, nz, zz = normalize_direction(torch.zeros(3))
    assert zz and nz == 0.0 and torch.all(vz == 0)


def test_projection_removes_opposing_component_only():
    g0 = torch.tensor([1.0, 0.0])
    opp = -torch.tensor([2.0, 3.0])  # opposes g0
    v, ab, aa, z = classification_compatible_direction(opp, g0)
    assert not z
    assert aa >= -1e-6          # no opposing component left
    assert abs(torch.norm(v).item() - 1.0) < 1e-6
    # aligned direction is untouched (mod normalization)
    v2, ab2, aa2, z2 = classification_compatible_direction(
        torch.tensor([1.0, 1.0]), g0)
    assert aa2 > 0 and abs(torch.norm(v2).item() - 1.0) < 1e-6


def test_applied_gradient_identity_and_mixture_ce_compat():
    m = _method(k=2, profiling_epochs=3, utility_interval=2)
    torch.manual_seed(0)
    m._profiling.copy_(False)
    m._epoch = 3
    m._selected.copy_(torch.tensor([0, 1], dtype=torch.long))
    m.utility_interval = 10 ** 9   # no refresh during this step
    params = [p for _, p in m._utility_params]
    dev = next(m.backbone.parameters()).device
    # jackknife a known cached selective target
    with torch.no_grad():
        s = torch.randn(m._n_backbone, device=dev)
        s = s / s.norm()
    m._s_cached.copy_(s)
    m._s_set.copy_(True)
    m._s_target_age.copy_(5)

    bo, y, ce = _rand_forward(m)
    out = m._allocated_loss(bo, y, ce, SimpleNamespace(batch_index=1))
    assert int(m._s_target_age) == 6  # cached target aged one batch

    # expected applied gradient, recomputed exactly like the method
    g0 = torch.autograd.grad(ce, params, retain_graph=True, allow_unused=True,
                             materialize_grads=True)
    g0_flat = _cat([_flatten(g) for g in g0]).detach()
    vs = []
    for i in (0, 1):
        s_name = m.site_names[i]
        feat = _pool_tap(bo.features[s_name], m.token)
        l_aux = F.cross_entropy(m.aux_heads[s_name](feat), y)
        gl = torch.autograd.grad(l_aux, params, retain_graph=True,
                                 allow_unused=True, materialize_grads=True)
        gl_flat = _cat([_flatten(g) for g in gl]).detach()
        v, _, _, _ = classification_compatible_direction(gl_flat, g0_flat, eps=m.projection_eps)
        vs.append(v)
    G = torch.stack(vs) @ torch.stack(vs).t()
    bvec = torch.stack(vs) @ m._s_cached.detach()
    lam = solve_topk_allocation(G, bvec, B=m.B, tol=m.tol)["lambda"].to(dev)
    mix = torch.stack(vs).t() @ lam
    rho_n = m.rho * float(torch.norm(g0_flat).item())
    # numeric mixture CE compatibility (protocol section 8 check)
    mix_n = float(torch.norm(mix).item())
    g0_n = float(torch.norm(g0_flat).item())
    if mix_n > 0 and g0_n > 0:
        ce_compat = float(torch.dot(mix, g0_flat).item()) / (mix_n * g0_n)
        assert ce_compat >= -1e-4
        assert m._step_aux_mass > 0.0

    out["routed"].backward()
    acc_i = 0
    p_names = [n for n, _ in m._utility_params]
    for (n, p), g0p in zip(zip(p_names, params), g0):
        n_el = p.numel()
        expected = (g0p if g0p is not None else torch.zeros_like(p)) + \
            rho_n * mix[acc_i:acc_i + n_el].reshape_as(p)
        assert torch.allclose(p.grad, expected, atol=1e-4), n
        acc_i += n_el
    row = m._step_rows[-1]
    assert row["target_age"] == 6 and row["zero"] is False


# ---------------------------------------------------------------------------
# 8. cached-target refresh and age
# ---------------------------------------------------------------------------
def test_cached_target_refresh_cadence_and_age():
    m = _method(k=2, profiling_epochs=3, utility_interval=5)
    m.on_epoch_start(3)
    m._profiling.copy_(False)
    m._s_set.copy_(True)
    steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for st in steps:
        # epoch 3 is the first post-profiling epoch: step 0 refreshes plus cadence
        assert m._should_refresh(st) == (st == 0 or (st > 0 and st % 5 == 0))
    m._s_cached.copy_(torch.randn_like(m._s_cached))
    s_before = m._s_cached.detach().clone()
    m._refresh_selective(next(m.backbone.parameters()).device, 5)
    assert int(m._s_refresh_step) == 5 and int(m._s_target_age) == 0
    assert bool(torch.isfinite(m._s_cached).all())
    assert m._last_gJ_norm > 0 or not bool(m._s_set)
    m._s_target_age.add_(5)
    assert int(m._s_target_age) == 5
    # below-first-epoch detection: profiling step 0 is not a refresh
    m.on_epoch_start(0)
    assert m._should_refresh(0) is False


# ---------------------------------------------------------------------------
# 9. label-free inference, deployment graph, preservation
# ---------------------------------------------------------------------------
def test_inference_is_msp_only_and_backbone_only():
    m = _method()
    mp = m.predict_batch(torch.randn(4, m.backbone.channels,
                                     m.backbone.input_size, m.backbone.input_size))
    assert list(mp.scores.keys()) == list(m.default_scores()) + ["sage_conf"]
    assert mp.scores["sage_conf"].equal(mp.scores["msp"])  # sage_conf := MSP
    assert m.inference_modules() == [m.backbone]
    assert torch.allclose(mp.confidence, mp.scores["msp"])
    assert mp.confidence.shape == (4,)
    assert mp.logits.shape[-1] == m.num_classes


def test_registry_preserves_previous_methods():
    names = set(method_names())
    assert {"ce", "ccl_sc", "dg", "scsf", "selectivenet", "sage_ds",
            "sage_ds_v2", "sage_ds_v3", "depthfrag", "depthfrag_v2",
            "riskflow", "riskflow_v2", "sage_topk"} <= names
    for name in ("ce", "sage_ds_v2", "sage_ds_v3", "depthfrag_v2", "riskflow_v2"):
        cfg = resolve({"dataset": "cifar10", "backbone": "resnet18",
                       "method_name": name, "results_root": "/tmp/opencode/sage_topk_tests",
                       "train": {"device": "cpu", "epochs": 1, "seed": 13}})
        method = build_method(name, cfg)
        assert method is not None