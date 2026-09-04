"""SAGE-V2 scientific-integrity tests (protocol: docs/SAGE_V2_PROTOCOL.md).

Locks the bilevel utility correction: train-side auxiliary gradient + disjoint
meta selective gradient + per-site CE-safety projection + cosine-controlled
gates, plus the required per-estimate logging schema and the no-overhead
inference contract.
"""

import math
import os
import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from scsf.engine import config
from scsf.engine.trainer import _build_optimizers
from scsf.data import get_split
from scsf.methods import build_method
from scsf.methods.sage_ds import Controller, project_aux
from scsf.methods.sage_ds_v2 import (
    SageDSV2Method,
    bilevel_utilities,
    cosine_utility,
    support_fraction,
)
from scsf.metrics.surrogate import soft_aurc_surrogate


def _sg_cfg(results_root="/tmp/opencode/sage_ds_v2_tests", seed=0):
    cfg = config.resolve({
        "dataset": "cifar10",
        "backbone": "resnet18",
        "method_name": "sage_ds_v2",
        "results_root": results_root,
        "train": {"device": "cpu", "seed": seed, "epochs": 1,
                  "batch_size": 8, "lr": 0.01},
    })
    return cfg


def _sg_method(results_root="/tmp/opencode/sage_ds_v2_tests", seed=0, **method_overrides):
    cfg = _sg_cfg(results_root, seed)
    cfg["method"].update(method_overrides)
    return build_method("sage_ds_v2", cfg)


def _flat(grads):
    return torch.cat([g.reshape(-1) for g in grads])


# ---------------------------------------------------------------------------
# 1. finite-difference bilevel utility sign + magnitude: pulling theta along
#    `-eta * tilde_g_l` must move the meta AURC surrogate by `-eta * U_raw`.
# ---------------------------------------------------------------------------
def _fd_case(seed, D=16, C=10, B=64, tau=0.5, eta=0.02):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, C))
    aux = nn.Linear(D, C)
    xt, yt = torch.randn(B, D), torch.randint(0, C, (B,))   # B_TRAIN
    xm, ym = torch.randn(B, D), torch.randint(0, C, (B,))   # disjoint B_META
    gp = list(net.parameters())

    def J_meta():
        h = net[0](xm).relu()
        return soft_aurc_surrogate(net[2](h), ym, error_mode="proxy", tau=tau)

    J0 = float(J_meta().detach())

    # TRAIN side (gradient we actually pull with): aux grad + same-batch CE ref
    h_t = net[0](xt).relu()
    g0 = torch.autograd.grad(F.cross_entropy(net[2](h_t), yt), gp,
                             retain_graph=True, allow_unused=True,
                             materialize_grads=True)
    g_l = torch.autograd.grad(F.cross_entropy(aux(h_t), yt), gp,
                              retain_graph=True, allow_unused=True,
                              materialize_grads=True)
    g0_flat = _flat(g0)
    gl_flat = _flat(g_l)
    assert any(g is not None for g in g_l)
    tilde_flat, _ = project_aux(gl_flat.clone(), g0_flat)

    # META side (the selective objective, on the disjoint meta batch)
    g_sel = torch.autograd.grad(J_meta(), gp, retain_graph=True,
                                allow_unused=True, materialize_grads=True)
    gsel_flat = _flat(g_sel)
    U = float(bilevel_utilities(gsel_flat, gl_flat, tilde_flat)["raw"])

    with torch.no_grad():
        acc = 0
        for p in gp:
            n = p.numel()
            p.data.sub_(eta * tilde_flat[acc:acc + n].reshape_as(p))
            acc += n
    fd = float(J_meta().detach()) - J0
    return U, fd


def test_sage_ds_v2_finite_difference_bilevel_utility_sign_lock():
    for seed in range(4):
        for eta in (0.02, 0.2):
            U, fd = _fd_case(seed, eta=eta)
            assert U != 0.0
            assert fd != 0.0
            # theta' = theta - eta * tilde_g_l  ==>  fd ~= -eta * U, same sign
            assert (fd < 0) == (-eta * U < 0), (seed, eta, U, fd)
            assert abs(fd) <= 5.0 * abs(eta * U) + 1e-3, (seed, eta, U, fd)


# ---------------------------------------------------------------------------
# 2. cosine utility is invariant to positive rescaling; U_raw scales linearly.
# ---------------------------------------------------------------------------
def test_sage_ds_v2_cosine_invariance_and_raw_linearity():
    torch.manual_seed(3)
    for _ in range(5):
        gJ = torch.randn(30)
        gl = torch.randn(30)
        g0 = torch.randn(30)
        til, _ = project_aux(gl.clone(), g0)
        c = 5.0
        u1 = bilevel_utilities(gJ, gl, til)
        u2 = bilevel_utilities(gJ, c * gl, c * til)
        # invariance up to the eps-blocking factor (~ 1e-7 here)
        assert u1["cos"] == pytest.approx(u2["cos"], abs=1e-6)
        assert u1["raw"] != 0.0
        assert u2["raw"] == pytest.approx(c * u1["raw"], rel=1e-5)
        # projection commutes with positive rescaling (exact)
        til_c, align_c = project_aux(c * gl.clone(), g0)
        assert torch.allclose(til_c, c * til, atol=1e-6)
        assert align_c == pytest.approx(float(torch.dot(c * gl, g0).item()), rel=1e-5)


# ---------------------------------------------------------------------------
# 3. per-site CE-safety inequality: <tilde_g_l, g0_train> >= -tol for every site.
# ---------------------------------------------------------------------------
def test_sage_ds_v2_per_site_ce_safety_inequality():
    m = _sg_method(seed=4)
    m.train()
    x = torch.randn(6, 3, 32, 32)
    y = torch.randint(0, 10, (6,))
    idx = torch.arange(6)
    loss_dict = m.train_loss((x, y, idx), SimpleNamespace(batch_index=1))
    assert set(m.site_names)
    for s in m.site_names:
        before = float(loss_dict[f"align_before_{s}"])
        after = float(loss_dict[f"align_after_{s}"])
        g0n2 = float(loss_dict["g0_norm2"])
        # projection identity in real arithmetic: a conflicted direction is
        # reduced to the epsilon-blocking residue, exactly
        #   align_after = align_before * eps / (||g0_train||^2 + eps)
        # (and align_after == align_before when there is no conflict).
        expected = (before * m.eps / (g0n2 + m.eps) if before < 0.0 else before)
        assert abs(after - expected) <= 1e-5 * max(abs(before), 1.0), \
            (s, before, after, g0n2)
        # CE-safety: alignment with the training CE gradient is ~0 after the
        # per-site projection (epsilon-blocking residue is << any gate signal).
        assert after >= -1e-2, (s, before, after)
        # a conflict must have been removed (after ~= 0 whenever before < 0)
        if before < 0.0:
            assert abs(after) < 1e-2, (s, before, after)


# ---------------------------------------------------------------------------
# 4. applied gradient identity: grad(routed) == g0 + sum_l (z_l * s) * tilde_g_l
#    on backbone params; aux heads keep their own unweighted CE gradients.
# ---------------------------------------------------------------------------
def test_sage_ds_v2_applied_gradient_equals_ce_plus_weighted_projected_sites():
    m = _sg_method(seed=7)
    m.train()
    m._audit_applied = True
    x = torch.randn(6, 3, 32, 32)
    y = torch.randint(0, 10, (6,))
    idx = torch.arange(6)
    loss_dict = m.train_loss((x, y, idx), SimpleNamespace(batch_index=2))
    audit = m._audit
    assert audit is not None
    assert set(m.site_names) == set(audit["z"].keys())

    params = [p for _, p in m._utility_params]
    g_back = torch.autograd.grad(loss_dict["routed"], params, retain_graph=True,
                                 allow_unused=True, materialize_grads=True)
    for i, (gb, g0p, add) in enumerate(zip(g_back, audit["g0"], audit["add"])):
        assert torch.allclose(gb, g0p + add, atol=1e-6), i

    # per-param per-site reconstruction of the auxiliary part: each param's
    # segment is its flat offset within every site's concatenated gradient
    # (the offset accumulates across params, not across sites).
    offsets = []
    off = 0
    for p in params:
        offsets.append(off)
        off += p.numel()
    for i, p in enumerate(params):
        n = p.numel()
        expected_add = torch.zeros_like(p)
        for s in m.site_names:
            expected_add += (audit["z"][s] * audit["scale"]
                             * audit["tilde"][s][offsets[i]:offsets[i] + n]
                             .reshape_as(p))
        assert torch.allclose(audit["add"][i], expected_add, atol=1e-6), i

    # auxiliary-head parameters receive the raw (unweighted) own-CE gradient
    aux_params = list(m.aux_heads.parameters())
    g_aux_back = torch.autograd.grad(loss_dict["routed"], aux_params,
                                     retain_graph=True, allow_unused=True,
                                     materialize_grads=True)
    for gb, g_aud in zip(g_aux_back, audit["aux"]):
        assert torch.allclose(gb, g_aud, atol=1e-6)


# ---------------------------------------------------------------------------
# 5. train/meta batch discipline: disjoint in the true pipeline, and enforced
#    (the estimate raises RuntimeError on any overlap).
# ---------------------------------------------------------------------------
def test_sage_ds_v2_train_meta_batches_disjoint_and_enforced():
    cfg = _sg_cfg()
    cfg["method"]["utility_interval"] = 1
    m = build_method("sage_ds_v2", cfg)
    assert m.needs_indices is True
    m.train()

    split = get_split(cfg)  # deterministic split; no dataset files required
    train_ids = list(split.train_indices[:6])
    val_ids = list(split.val_indices[:6])
    assert set(train_ids) & set(val_ids) == set()

    x, y = torch.randn(6, 3, 32, 32), torch.randint(0, 10, (6,))
    xm, ym = torch.randn(6, 3, 32, 32), torch.randint(0, 10, (6,))
    state = SimpleNamespace(batch_index=1)

    m._meta_batch = lambda device: (xm.to(device), ym.to(device), val_ids)
    m.train_loss((x, y, torch.tensor(train_ids)), state)
    assert set(m._last_train_ids) < set(split.train_indices)
    assert set(val_ids) <= set(split.val_indices)
    assert set(m._last_train_ids).isdisjoint(set(val_ids))

    # an overlapping meta batch must be rejected inside the estimate
    def bad_meta(device):
        return xm.to(device), ym.to(device), train_ids[:3]

    m._meta_batch = bad_meta
    with pytest.raises(RuntimeError, match="bilevel violation"):
        m.train_loss((x, y, torch.tensor(train_ids)), state)


# ---------------------------------------------------------------------------
# 6. zero-gradient behaviour stays finite (epsilon-guarded utilities/EMA).
# ---------------------------------------------------------------------------
def test_sage_ds_v2_zero_gradient_utilities_finite():
    gJ = torch.zeros(20)
    gl = torch.zeros(20)
    til = torch.zeros(20)
    u = bilevel_utilities(gJ, gl, til)
    for k, v in u.items():
        assert math.isfinite(v), (k, v)
    assert u["raw"] == 0.0 and u["cos"] == 0.0
    assert u["gJ_norm"] == 0.0 and u["tilde_norm"] == 0.0

    # zero reference gradient leaves the site gradient unchanged
    gl_r = torch.randn(20)
    safe, align = project_aux(gl_r.clone(), torch.zeros(20))
    assert torch.allclose(safe, gl_r, atol=1e-12)
    assert align == 0.0

    # zero meta gradient with nonzero projected gradient stays finite
    til2, _ = project_aux(gl_r.clone(), gJ)
    u2 = bilevel_utilities(gJ, gl_r, til2)
    assert u2["raw"] == 0.0 and u2["cos"] == 0.0
    assert all(math.isfinite(v) for v in u2.values())

    # controller consuming zero utilities stays finite and moves sanely
    ctl = Controller(["a", "b"], tau=0.3, beta=0.99)
    for _ in range(50):
        ctl.update_utility_ema([0.0, 0.0])
        ctl.step_from_utility(0.1, 0.5, cap=0.5)
    assert torch.isfinite(ctl.utility_ema).all()
    assert all(math.isfinite(float(ctl.gate_prob(s))) for s in ctl.site_names)
    assert all(math.isfinite(v) for v in ctl.utility_ema_dict().values())


def test_sage_ds_v2_utility_primitives():
    assert support_fraction(torch.zeros(50)) == 0.0
    assert support_fraction(torch.ones(50)) == 1.0
    g = torch.zeros(40)
    g[:10] = 1.0
    assert support_fraction(g) == pytest.approx(0.25)
    assert support_fraction(torch.tensor([])) == 0.0
    assert cosine_utility(3.0, 2.0, 4.0) == pytest.approx(3.0 / (8.0 + 1e-8))
    assert cosine_utility(0.0, 0.0, 0.0) == 0.0
    assert cosine_utility(0.0, 0.0, 5.0) == 0.0


# ---------------------------------------------------------------------------
# 7. exact checkpoint/resume of the controller state.
# ---------------------------------------------------------------------------
def test_sage_ds_v2_checkpoint_resume_preserves_controller_state(tmp_path):
    m = _sg_method(results_root=str(tmp_path), seed=3)
    m.controller.update_utility_ema([0.9, -0.4, 0.2, -0.1])
    m.controller.step_from_utility(0.05, 0.1, 0.5)
    with torch.no_grad():
        m.controller.gates["layer3"].log_alpha.data.copy_(torch.tensor(2.3))
    ckpt = os.path.join(str(tmp_path), "ckpt.pt")
    torch.save(m.state_dict(), ckpt)

    m2 = _sg_method(results_root=str(tmp_path), seed=3)
    m2.load_state_dict(torch.load(ckpt, weights_only=True))
    assert m2.controller.utility_ema_dict() == m.controller.utility_ema_dict()
    assert int(m2.controller.ui_step) == int(m.controller.ui_step)
    for s in m.site_names:
        assert float(m2.controller.gate_prob(s)) == pytest.approx(
            float(m.controller.gate_prob(s)))


# ---------------------------------------------------------------------------
# 8. inference is plain MSP; aux heads + controller are training-only.
# ---------------------------------------------------------------------------
def test_sage_ds_v2_msp_inference_excludes_aux_and_controller():
    m = _sg_method(seed=2)
    x = torch.randn(6, 3, 32, 32)
    mp1 = m.predict_batch(x)
    assert torch.equal(mp1.confidence, mp1.scores["msp"])
    assert torch.equal(mp1.scores["sage_conf"], mp1.scores["msp"])
    assert set(mp1.scores) >= {"msp", "entropy", "energy", "logit_margin", "sage_conf"}

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
# 9. auxiliary heads add no deployment overhead (backbone-only inference graph).
# ---------------------------------------------------------------------------
def test_sage_ds_v2_aux_heads_add_no_deployment_overhead():
    m = _sg_method(seed=0)
    depl = sum(p.numel() for mod in m.inference_modules() for p in mod.parameters())
    bb = sum(p.numel() for p in m.backbone.parameters())
    assert depl == bb
    total = sum(p.numel() for p in m.parameters())
    assert total > bb  # training-only aux heads + controller exist module-side
    assert m.use_risk_head is False
    assert isinstance(m, SageDSV2Method)


# ---------------------------------------------------------------------------
# engine-pattern smoke: gradients route onto backbone + aux heads, controller
# gates stay manual (no optimizer gradient).
# ---------------------------------------------------------------------------
def test_sage_ds_v2_train_step_routes_gradients_and_leaves_gates_manual():
    cfg = _sg_cfg()
    cfg["train"]["seed"] = 7
    m = build_method("sage_ds_v2", cfg)
    m.train()
    opt = _build_optimizers(m, cfg)[0]

    x = torch.randn(6, 3, 32, 32)
    y = torch.randint(0, 10, (6,))
    idx = torch.arange(6)
    state = SimpleNamespace(batch_index=5)  # not a utility step (interval 50)
    loss_dict = m.train_loss((x, y, idx), state)
    total = sum(v for v in loss_dict.values() if torch.is_tensor(v)
                and v.requires_grad)
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


def test_sage_ds_v2_utility_estimate_runs_and_writes_log_schema(tmp_path):
    cfg = _sg_cfg(results_root=str(tmp_path), seed=5)
    cfg["method"]["utility_interval"] = 1
    m = build_method("sage_ds_v2", cfg)
    m.train()

    xm, ym = torch.randn(6, 3, 32, 32), torch.randint(0, 10, (6,))
    meta_ids = [45000 + i for i in range(6)]
    m._meta_batch = lambda device: (xm.to(device), ym.to(device), meta_ids)

    x, y = torch.randn(6, 3, 32, 32), torch.randint(0, 10, (6,))
    idx = torch.arange(6)
    loss_dict = m.train_loss((x, y, idx), SimpleNamespace(batch_index=1))
    total = sum(v for v in loss_dict.values() if torch.is_tensor(v)
                and v.requires_grad)
    assert torch.isfinite(total)

    log_path = os.path.join(str(tmp_path), m.cfg["run_name"],
                            "sage_ds_v2_utility.jsonl")
    assert os.path.exists(log_path)
    with open(log_path) as f:
        row = json.loads(f.readline())
    for s in m.site_names:
        for key in (f"raw_unprojected_utility_{s}", f"raw_utility_{s}",
                    f"cos_utility_{s}", f"gl_norm_{s}", f"tilde_gl_norm_{s}",
                    f"support_frac_{s}", f"align_before_{s}", f"align_after_{s}",
                    f"gatep_{s}", f"sampled_gate_{s}", f"eff_aux_w_{s}",
                    f"uema_{s}"):
            assert key in row, key
            assert math.isfinite(float(row[key])), key
            assert 0.0 <= float(row[f"support_frac_{s}"]) <= 1.0
    assert math.isfinite(row["gJ_norm"]) and math.isfinite(row["g0_norm"])
    assert row["meta_bs"] == 6
    assert row["train_ids"] == sorted(row["train_ids"])
    assert set(row["train_ids"]).isdisjoint(set(row["meta_ids"]))


# ---------------------------------------------------------------------------
# every registered backbone builds a sage_ds_v2 method end to end
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("backbone",
                         ["resnet18", "vgg16_bn", "wideresnet28_10", "convnext_tiny", "deit_s"])
def test_sage_ds_v2_config_loads_across_backbones(tmp_path, backbone):
    cfg = config.resolve({
        "dataset": "cifar10",
        "backbone": backbone,
        "method_name": "sage_ds_v2",
        "results_root": str(tmp_path),
        "train": {"device": "cpu", "seed": 0},
    })
    m = build_method("sage_ds_v2", cfg)
    assert m.needs_indices is True
    assert set(m.backbone.roles) >= {"top_l1", "top_l2"}
    assert set(m.backbone.taps) >= set(m.site_names)
    m.eval()
    x = torch.randn(1, 3, 32, 32)
    mp = m.predict_batch(x)
    assert tuple(mp.confidence.shape) == (1,)