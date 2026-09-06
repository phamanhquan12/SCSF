"""SAGE-V3 scientific-integrity tests (protocol: docs/SAGE_V3_PROTOCOL.md).

Locks the robust training-aware gradient allocation: class-conditioned robust
selective target (logsumexp over per-class J_c, tau = 10), the deterministic
class-balanced meta batch, the exact active-set QP solver with KKT-verified
optimality, the certificate-before-application fallback (lambda = 0), the
applied-gradient identity ``g_CE + rho*||g_CE||*A*lambda``, exact-resume
buffers, and the MSP-only, backbone-only inference contract.
"""

import json
import math
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from scsf.engine import config
from scsf.engine.trainer import _build_optimizers
from scsf.data import get_split
from scsf.methods import build_method
from scsf.methods.sage_ds import _cat, _flatten
from scsf import methods as methods_pkg
from scsf.methods import sage_ds_v3 as v3
from scsf.methods.sage_ds_v3 import (
    AmortizedAllocationSolver,
    SageDSV3Method,
    class_balanced_meta_batch,
    qp_certificate,
    qp_kkt_residual,
    robust_selective_target,
    solve_sage_v3_qp,
)
from scsf.metrics.surrogate import soft_aurc_surrogate


def _sg_cfg(results_root="/tmp/opencode/sage_ds_v3_tests", seed=0):
    cfg = config.resolve({
        "dataset": "cifar10",
        "backbone": "resnet18",
        "method_name": "sage_ds_v3",
        "results_root": results_root,
        "train": {"device": "cpu", "seed": seed, "epochs": 1,
                  "batch_size": 8, "lr": 0.01},
    })
    return cfg


def _sg_method(results_root="/tmp/opencode/sage_ds_v3_tests", seed=0, **method_overrides):
    cfg = _sg_cfg(results_root, seed)
    cfg["method"].update(method_overrides)
    return build_method("sage_ds_v3", cfg)


def _classful_labels(C, reps):
    # every class present (the robust target requires all C classes in batch)
    return torch.arange(C).repeat(max(1, reps))


def _constant_meta_batch(xm, ym, gids):
    def fake(cfg, num_classes, k_meta, offset, device, store):
        return (xm.to(torch.device(device)),
                ym.to(torch.device(device)), list(gids))
    return fake


# ---------------------------------------------------------------------------
# 1. robust target: sign/magnitude finite-difference lock and boundedness.
# ---------------------------------------------------------------------------
def _robust_fd_case(seed, D=16, C=6, B=32, tau=10.0, eta=0.05):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, C))
    xm, ym = torch.randn(B, D), torch.randint(0, C, (B,))
    gp = list(net.parameters())

    def J_rob():
        with torch.enable_grad():
            h = net[0](xm).relu()
            logits = net[2](h)
            per_class = [
                soft_aurc_surrogate(logits[ym == c], ym[ym == c], tau=0.3)
                for c in range(C)
            ]
        return robust_selective_target(per_class, tau)

    J0 = float(J_rob().detach())
    g_r = torch.autograd.grad(J_rob(), gp, allow_unused=True,
                              materialize_grads=True)
    r = _cat([_flatten(g) for g in g_r])
    rn = float(r.norm().item())
    with torch.no_grad():
        if rn > 0:
            acc = 0
            for p in gp:
                n = p.numel()
                p.data.sub_(eta * r[acc:acc + n].reshape_as(p))
                acc += n
    fd = float(J_rob().detach()) - J0
    return rn, fd, J0


def test_sage_ds_v3_robust_target_finite_difference_sign_lock():
    for seed in range(4):
        for eta in (0.02, 0.2):
            rn, fd, J0 = _robust_fd_case(seed, eta=eta)
            assert math.isfinite(J0)
            assert fd != 0.0
            # theta' = theta - eta*g_r  ==>  fd ~= -eta*||g_r||^2 < 0
            assert fd < 0.0, (seed, eta, rn, fd)
            assert abs(fd) <= 5.0 * eta * rn * rn + 1e-3, (seed, eta, rn, fd)


def test_sage_ds_v3_robust_mean_between_mean_and_max():
    torch.manual_seed(0)
    for C in (5, 10):
        J = torch.randn(C)
        robust = robust_selective_target(list(J), 10.0)
        assert float(robust) >= float(J.mean()) - 1e-6
        assert float(robust) <= float(J.max()) + 1e-6
    # equal classes -> robust == arithmetic mean exactly
    J = torch.full((8,), 0.37)
    assert float(robust_selective_target(list(J), 10.0)) == pytest.approx(0.37)


def test_sage_ds_v3_robust_target_monotone_in_tau():
    J = [0.1, 0.2, 0.8, 1.5]
    vals = [float(robust_selective_target(list(J), tau)) for tau in (1e-3, 10.0, 50.0, 1e4)]
    # tau -> 0 approaches the arithmetic mean (l'Hopital); monotone increasing
    # toward the max as tau grows; large tau approaches max.
    assert vals[0] == pytest.approx(sum(J) / len(J), abs=1e-2)
    assert vals[0] < vals[1] < vals[2]
    assert vals[-1] == pytest.approx(max(J), abs=5e-4)


# ---------------------------------------------------------------------------
# 2. QP solver: feasibility, KKT optimality, degeneracies, determinism.
# ---------------------------------------------------------------------------
def _random_qp(seed, L, B=1.0):
    torch.manual_seed(seed)
    A = torch.randn(24, L)
    G = A.T @ A
    b = torch.randn(L) * 0.5
    q = torch.randn(L)
    return G, b, q


@pytest.mark.parametrize("L", [2, 3, 5])
def test_sage_ds_v3_qp_primal_feasible_and_kkt_optimal(L):
    for seed in range(5):
        G, b, q = _random_qp(seed, L)
        lambd, info = solve_sage_v3_qp(G, b, q)
        assert tuple(lambd.shape) == (L,)
        # primal feasibility
        assert float(lambd.min()) >= -1e-9
        assert float(lambd.sum()) <= 1.0 + 1e-9
        assert float(torch.dot(lambd, q.double())) >= -1e-9
        # independent KKT/optimality verification
        assert qp_kkt_residual(lambd, G, b, q, 1.0, 1e-4) <= 1e-5, (seed, info)
        # info carries the certificate-worthy statistics
        assert set(info) >= {"obj", "active_sum", "active_q", "free", "kkt_residual"}


def test_sage_ds_v3_qp_matches_brute_force_grid():
    G, b, q = _random_qp(7, L=2, B=1.0)
    lambd, info = solve_sage_v3_qp(G, b, q)
    obj_exact = float(0.5 * lambd @ G.double() @ lambd - torch.dot(lambd, b.double()))
    best = float("inf")
    for i in range(1001):
        for j in range(1001):
            la = torch.tensor([i / 1000.0, j / 1000.0])
            if la.sum() > 1.0 + 1e-9 or float(torch.dot(la, q)) < -1e-9:
                continue
            o = float(0.5 * la @ G @ la - torch.dot(la, b))
            if o < best:
                best = o
    assert obj_exact <= best + 1e-3


def test_sage_ds_v3_qp_zero_b_touches_zero_and_determinism():
    G, b, q = _random_qp(11, L=4, B=1.0)
    b0 = torch.zeros_like(b)
    lambd0, _ = solve_sage_v3_qp(G, b0, q)
    assert float(lambd0.abs().max()) <= 1e-9  # 0 is the unique optimum for b=0
    l1, i1 = solve_sage_v3_qp(G, b, q)
    l2, i2 = solve_sage_v3_qp(G, b, q)
    assert torch.equal(l1, l2)
    assert i1["obj"] == i2["obj"]
    assert i1["kkt_residual"] <= 1e-5


def test_sage_ds_v3_qp_certificate_predicates():
    B, tol = 1.0, 1e-6
    b = torch.tensor([1.0, 0.0])
    q = torch.tensor([0.5, -0.5])
    for lambd, expected_ok in [
        (torch.tensor([0.4, 0.2]), True),   # b·l > 0, q·l >= 0, sum <= B
        (torch.tensor([0.6, 0.6]), False),  # sum > B
        (torch.tensor([-0.1, 0.5]), False),  # nonnegative violated
        (torch.tensor([0.0, 1.0]), False),  # b·l = 0 fails the > 0 predicate
    ]:
        c = qp_certificate(lambd, b, q, B, tol)
        assert c["ok"] == expected_ok
        assert math.isfinite(c["b_lambda"]) and math.isfinite(c["q_lambda"])


def test_sage_ds_v3_qp_budget_b_respected():
    for B in (0.1, 0.5, 2.0):
        G, b, q = _random_qp(3, L=5)
        lambd, _ = solve_sage_v3_qp(G, b, q, B=B)
        assert float(lambd.sum()) <= B + 1e-9


# ---------------------------------------------------------------------------
# 3. class-balanced meta batch: exact size, per-class counts, determinism,
#    disjointness by construction.
# ---------------------------------------------------------------------------
def test_sage_ds_v3_class_balanced_meta_batch_deterministic(tmp_path):
    cfg = _sg_cfg(results_root=str(tmp_path), seed=1)
    split = get_split(cfg)
    C = 10
    k = 3
    store = SimpleNamespace()
    x1, y1, g1 = class_balanced_meta_batch(cfg, C, k, offset=0, device="cpu", store=store)
    x2, y2, g2 = class_balanced_meta_batch(cfg, C, k, offset=0, device="cpu", store=store)
    assert torch.equal(x1, x2) and torch.equal(y1, y2) and g1 == g2
    assert int(y1.numel()) == C * k
    counted = torch.bincount(y1.long(), minlength=C)
    assert torch.all(counted == k), counted  # exactly k_meta per class
    # meta samples come from the val split (disjoint from train by construction)
    assert set(g1).issubset(set(split.val_indices))
    assert set(g1).isdisjoint(set(split.train_indices))
    # rotation changes the sample window across rounds
    x3, y3, g3 = class_balanced_meta_batch(cfg, C, k, offset=1, device="cpu", store=store)
    assert g1 != g3


# ---------------------------------------------------------------------------
# 4. locked constants + factory registration + needs_indices.
# ---------------------------------------------------------------------------
def test_sage_ds_v3_factory_and_locked_constants(tmp_path):
    for name in ("sage_ds_v3", "sage_ds_v3_amortized"):
        m = build_method(name, _sg_cfg(results_root=str(tmp_path), seed=0))
        assert isinstance(m, SageDSV3Method)
        assert m.needs_indices is True
    m = _sg_method(seed=6)
    assert m.robust_tau == 10.0
    assert m.budget_B == 1.0
    assert m.rho == 1.0
    assert m.qp_ridge == 1e-4
    assert m.meta_k == 8
    assert m.cert_tol == 1e-6
    assert m.amortized is False
    a = _sg_method(seed=6, amortized=True, amortized_steps=4)
    assert a.amortized is True
    assert a.solver.steps == 4


def test_sage_ds_v3_method_name():
    m = _sg_method(seed=6)
    assert m.method_name == "sage_ds_v3"


# ---------------------------------------------------------------------------
# 5. applied-gradient identity (audit): grad(routed) == g0 + rho*||g0||*A lambda;
#    aux heads keep their own CE gradients; budget honored; fallback on cert fail.
# ---------------------------------------------------------------------------
def test_sage_ds_v3_applied_gradient_identity(tmp_path, monkeypatch):
    m = _sg_method(results_root=str(tmp_path), seed=7, utility_interval=1)
    B = int(m.budget_B)
    xm = torch.randn(80, 3, 32, 32)
    ym = _classful_labels(10, 8)
    monkeypatch.setattr(
        v3, "class_balanced_meta_batch",
        _constant_meta_batch(xm, ym, [45000 + i for i in range(80)]))
    m.train()
    m._audit_applied = True
    x = torch.randn(6, 3, 32, 32)
    y = torch.randint(0, 10, (6,))
    idx = torch.arange(6)
    loss_dict = m.train_loss((x, y, idx), SimpleNamespace(batch_index=1))
    audit = m._audit
    assert audit is not None
    sites = list(m.site_names)

    params = [p for _, p in m._utility_params]
    g_back = torch.autograd.grad(loss_dict["routed"], params, retain_graph=True,
                                 allow_unused=True, materialize_grads=True)
    for i, (gb, g0p, add) in enumerate(zip(g_back, audit["g0"], audit["add"])):
        assert torch.allclose(gb, g0p + add, atol=1e-6), i

    # reconstructed auxiliary part equals rho*||g0|| * sum_j lambda_j * a_j
    lambda_j = [audit["lambda"][s] for s in sites]
    assert float(sum(lambda_j)) <= B + 1e-6
    aux_params = list(m.aux_heads.parameters())
    g_aux_back = torch.autograd.grad(loss_dict["routed"], aux_params,
                                     retain_graph=True, allow_unused=True,
                                     materialize_grads=True)
    for gb, g_aud in zip(g_aux_back, audit["aux"]):
        assert torch.allclose(gb, g_aud, atol=1e-6)


def test_sage_ds_v3_fallback_zero_on_failed_certificate(tmp_path):
    # utility_interval=1e9 so NO refresh runs (which would overwrite r)
    m = _sg_method(results_root=str(tmp_path), seed=7, utility_interval=10**9)
    # zero robust gradient => b = A^T r = 0 => b_lambda > 0 fails => lambda = 0
    m._v3_cached_r.zero_()
    m._v3_round.fill_(0)
    m.train()
    x, y, idx = torch.randn(6, 3, 32, 32), torch.randint(0, 10, (6,)), torch.arange(6)
    loss_dict = m.train_loss((x, y, idx), SimpleNamespace(batch_index=2))
    assert float(loss_dict["fallback_zero"]) == 1.0
    assert float(loss_dict["lambda_sum"]) == 0.0
    assert float(loss_dict["cert_ok"]) == 0.0
    assert float(loss_dict["lambda_zero"]) == 1.0
    total = sum(v for v in loss_dict.values() if torch.is_tensor(v) and v.requires_grad)
    assert torch.isfinite(total)


def test_sage_ds_v3_pre_refresh_no_r_branch(tmp_path):
    # Before the first robust-gradient refresh (_v3_round == -1) the cached r
    # is unavailable; the no-QP branch must still certify with zero b/q inputs
    # (regression: kwargs bvec=/q= were passed to qp_certificate(lambd, b, q, ...)).
    m = _sg_method(results_root=str(tmp_path), seed=11, utility_interval=10**9)
    m.train()
    x, y, idx = torch.randn(6, 3, 32, 32), torch.randint(0, 10, (6,)), torch.arange(6)
    loss_dict = m.train_loss((x, y, idx), SimpleNamespace(batch_index=1))
    assert float(loss_dict["fallback_zero"]) == 1.0
    assert float(loss_dict["lambda_sum"]) == 0.0
    total = sum(v for v in loss_dict.values() if torch.is_tensor(v) and v.requires_grad)
    assert torch.isfinite(total)


# ---------------------------------------------------------------------------
# 6. train/meta disjointness enforced (lock), and single-backward refresh.
# ---------------------------------------------------------------------------
def test_sage_ds_v3_meta_train_disjointness_enforced(tmp_path, monkeypatch):
    m = _sg_method(results_root=str(tmp_path), seed=4, utility_interval=1)
    m.train()
    train_ids = [0, 1, 2, 3]
    xm, ym = torch.randn(80, 3, 32, 32), _classful_labels(10, 8)
    monkeypatch.setattr(v3, "class_balanced_meta_batch",
                        _constant_meta_batch(xm, ym, train_ids))
    x, y = torch.randn(6, 3, 32, 32), torch.randint(0, 10, (6,))
    with pytest.raises(RuntimeError, match="robustness violation"):
        m.train_loss((x, y, torch.tensor(train_ids)), SimpleNamespace(batch_index=1))


def test_sage_ds_v3_refresh_uses_single_backward(tmp_path, monkeypatch):
    m = _sg_method(results_root=str(tmp_path), seed=4)
    xm = torch.randn(80, 3, 32, 32)
    ym = _classful_labels(10, 8)
    monkeypatch.setattr(v3, "class_balanced_meta_batch",
                        _constant_meta_batch(xm, ym, [45000 + i for i in range(80)]))
    calls = []

    orig_grad = torch.autograd.grad

    def counting_grad(*args, **kwargs):
        calls.append(1)
        return orig_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", counting_grad)
    m._refresh_robust_gradient("cpu", step=50)
    # exactly one backward for the robust estimate (the single-backward lock)
    assert len(calls) == 1
    # robust gradient normalized and cached in the persistent buffer
    assert int(m._v3_round.item()) == 1
    r = m._v3_cached_r
    assert float(r.norm()) == pytest.approx(1.0, abs=1e-3)
    assert len(m._per_class_J) == 10


# ---------------------------------------------------------------------------
# 7. exact-resume: cached robust gradient + counters live in state_dict.
# ---------------------------------------------------------------------------
def test_sage_ds_v3_checkpoint_resume_preserves_robust_state(tmp_path):
    m = _sg_method(results_root=str(tmp_path), seed=3)
    r = torch.randn(m._v3_cached_r.shape)
    m._v3_cached_r.copy_(r)
    m._v3_round.fill_(7)
    m._v3_zero_ct.fill_(3)
    m._v3_qp_calls.fill_(123)
    ckpt = os.path.join(str(tmp_path), "ckpt.pt")
    torch.save(m.state_dict(), ckpt)

    m2 = _sg_method(results_root=str(tmp_path), seed=3)
    m2.load_state_dict(torch.load(ckpt, weights_only=True))
    assert torch.equal(m2._v3_cached_r, r)
    assert int(m2._v3_round.item()) == 7
    assert int(m2._v3_zero_ct.item()) == 3
    assert int(m2._v3_qp_calls.item()) == 123
    assert torch.equal(m2._r_or_none(), r)


# ---------------------------------------------------------------------------
# 8. inference is plain MSP; aux heads + controller are training-only.
# ---------------------------------------------------------------------------
def test_sage_ds_v3_msp_inference_excludes_aux():
    m = _sg_method(seed=2)
    x = torch.randn(6, 3, 32, 32)
    mp1 = m.predict_batch(x)
    assert torch.equal(mp1.confidence, mp1.scores["msp"])
    assert torch.equal(mp1.scores["sage_conf"], mp1.scores["msp"])
    assert set(mp1.scores) >= {"msp", "entropy", "energy", "logit_margin", "sage_conf"}

    infer_params = {id(p) for mod in m.inference_modules() for p in mod.parameters()}
    all_params = set(id(p) for p in m.parameters())
    assert infer_params == all_params - set(map(id, m.aux_heads.parameters())) \
        - set(map(id, m.controller.parameters()))
    with torch.no_grad():
        for p in m.aux_heads.parameters():
            p.uniform_(-1.0, 1.0)
    mp2 = m.predict_batch(x)
    assert torch.allclose(mp2.logits, mp1.logits)
    assert torch.equal(mp2.prediction, mp1.prediction)


def test_sage_ds_v3_aux_heads_add_no_deployment_overhead():
    m = _sg_method(seed=0)
    depl = sum(p.numel() for mod in m.inference_modules() for p in mod.parameters())
    bb = sum(p.numel() for p in m.backbone.parameters())
    assert depl == bb
    assert m.use_risk_head is False


# ---------------------------------------------------------------------------
# 9. optimizer routing: backbone + aux heads move; buffers untouched by SGD.
# ---------------------------------------------------------------------------
def test_sage_ds_v3_train_step_routes_gradients(tmp_path, monkeypatch):
    cfg = _sg_cfg(results_root=str(tmp_path), seed=7)
    cfg["method"]["utility_interval"] = 1
    m = build_method("sage_ds_v3", cfg)
    m.train()
    monkeypatch.setattr(
        v3, "class_balanced_meta_batch",
        _constant_meta_batch(torch.randn(80, 3, 32, 32),
                             _classful_labels(10, 8),
                             [45000 + i for i in range(80)]))
    opt = _build_optimizers(m, cfg)[0]

    x, y = torch.randn(6, 3, 32, 32), torch.randint(0, 10, (6,))
    loss_dict = m.train_loss((x, y, torch.arange(6)), SimpleNamespace(batch_index=1))
    total = sum(v for v in loss_dict.values() if torch.is_tensor(v) and v.requires_grad)
    assert torch.isfinite(total)
    for p in m.backbone.parameters():
        p.grad = None
    total.backward()
    assert any(p.grad is not None and bool(torch.any(p.grad != 0))
               for p in m.backbone.parameters())
    assert any(p.grad is not None and bool(torch.any(p.grad != 0))
               for p in m.aux_heads.parameters())
    before_buf = {n: b.clone() for n, b in m._buffers.items()
                  if n.startswith("_v3_") and b.dtype.is_floating_point}
    before = {n: p.clone() for n, p in m.named_parameters()}
    opt.step()
    moved = [n for n, p in m.named_parameters() if not torch.equal(p, before[n])]
    assert any(n.startswith("backbone.") for n in moved)
    assert all(n.startswith("backbone.") or "aux_heads" in n for n in moved)
    for n, b in m._buffers.items():
        if n.startswith("_v3_") and b.dtype.is_floating_point:
            assert torch.equal(b, before_buf[n]), n  # SGD never moves the buffers


# ---------------------------------------------------------------------------
# 10. refresh + solve + certificate writes the locked telemetry schema.
# ---------------------------------------------------------------------------
def test_sage_ds_v3_utility_log_schema(tmp_path, monkeypatch):
    m = _sg_method(results_root=str(tmp_path), seed=5, utility_interval=1)
    monkeypatch.setattr(
        v3, "class_balanced_meta_batch",
        _constant_meta_batch(torch.randn(80, 3, 32, 32),
                             _classful_labels(10, 8),
                             [45000 + i for i in range(80)]))
    m.train()
    m.train_loss((torch.randn(6, 3, 32, 32), torch.randint(0, 10, (6,)),
                  torch.arange(6)), SimpleNamespace(batch_index=1))
    log_path = os.path.join(str(tmp_path), m.cfg["run_name"],
                            "sage_ds_v3_utility.jsonl")
    assert os.path.exists(log_path)
    with open(log_path) as f:
        row = json.loads(f.readline())
    for key in ("step", "refresh_round", "class_surrogate", "G", "b", "q",
                "lambda", "qp_obj", "qp_kkt", "cert_ok", "cert_b_lambda",
                "cert_q_lambda", "cert_min_lambda", "cert_sum_lambda",
                "lambda_zero_count", "qp_calls", "utility_ms", "tau", "B", "rho"):
        assert key in row, key
    L = len(m.site_names)
    assert len(row["lambda"]) == L
    assert len(row["class_surrogate"]) == 10
    assert row["tau"] == 10.0 and row["rho"] == 1.0
    assert math.isfinite(row["qp_obj"]) and math.isfinite(row["qp_kkt"])

    m.on_epoch_end(0, {})
    steps_path = os.path.join(str(tmp_path), m.cfg["run_name"],
                              "sage_ds_v3_steps.jsonl")
    assert os.path.exists(steps_path)
    with open(steps_path) as f:
        srow = json.loads(f.readline())
    for key in ("step", "epoch", "lambda_sum", "cert_ok", "fallback_zero",
                "qp_obj", "applied_aux_mass", "gJ_norm", "lambda_zero_count",
                "qp_calls"):
        assert key in srow, key
    assert os.path.exists(os.path.join(str(tmp_path), m.cfg["run_name"],
                                       "sage_ds_v3.jsonl"))


# ---------------------------------------------------------------------------
# 11. amortized secondary variant: registered, feasible, deterministic.
# ---------------------------------------------------------------------------
def test_sage_ds_v3_amortized_solver_feasible_and_deterministic():
    torch.manual_seed(9)
    sol = AmortizedAllocationSolver(steps=4, B=1.0, ridge=1e-4)
    for _ in range(5):
        G, b, q = _random_qp(torch.randint(0, 999, ()).item(), L=5)
        l1 = sol(G, b, q)
        l2 = sol(G, b, q)
        assert torch.equal(l1, l2)
        assert float(l1.min()) >= -1e-9
        assert float(l1.sum()) <= 1.0 + 1e-9
        assert float(torch.dot(l1, q.double())) >= -1e-9
        assert torch.isfinite(l1).all()
    assert sol.steps in (3, 4, 5)


# ---------------------------------------------------------------------------
# every registered backbone builds a sage_ds_v3 method end to end
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backbone",
                         ["resnet18", "vgg16_bn", "wideresnet28_10", "convnext_tiny", "deit_s"])
def test_sage_ds_v3_config_loads_across_backbones(tmp_path, backbone):
    cfg = config.resolve({
        "dataset": "cifar10",
        "backbone": backbone,
        "method_name": "sage_ds_v3",
        "results_root": str(tmp_path),
        "train": {"device": "cpu", "seed": 0},
    })
    m = build_method("sage_ds_v3", cfg)
    assert m.needs_indices is True
    assert set(m.backbone.taps) >= set(m.site_names)
    L = len(m.site_names)
    assert L >= 2
    G, b, q = _random_qp(1, L)
    lam, _ = solve_sage_v3_qp(G, b, q)
    assert float(lam.sum()) <= 1.0 + 1e-9
    m.eval()
    mp = m.predict_batch(torch.randn(1, 3, 32, 32))
    assert tuple(mp.confidence.shape) == (1,)