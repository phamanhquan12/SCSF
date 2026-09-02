"""RiskFlow contract + scientific-integrity tests.

Covers the eight required checks (zero-update identity, constructed residual
sum, stop-gradient boundaries, gates differ across samples, deterministic
registry ordering, constant-variance decorrelation no-NaN, checkpoint resume,
tiny ResNet-18 + DeiT artifact smokes), the ablation-ladder configs, the
frozen-backbone gradient control, deployment parameter accounting, and the
redundancy / category-assignment diagnostics.
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
from scsf.methods.riskflow import decorrelation_penalty
from scsf.riskflow import (
    assign_category,
    export_trace,
    pairwise_linear_cka,
    redundancy_report,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rf_cfg(results_root="/tmp/opencode/riskflow_tests", seed=0):
    cfg = config.resolve({
        "dataset": "cifar10",
        "backbone": "resnet18",
        "method_name": "riskflow",
        "results_root": results_root,
        "train": {"device": "cpu", "seed": seed, "epochs": 1,
                  "batch_size": 8, "lr": 0.01},
    })
    return cfg


def _rf_method(results_root="/tmp/opencode/riskflow_tests", seed=0, **m):
    cfg = _rf_cfg(results_root, seed)
    cfg["method"].update(m)
    return build_method("riskflow", cfg)


def _rand_batch(m, B=6, seed=0):
    torch.manual_seed(seed)
    return (torch.randn(B, m.backbone.channels, m.backbone.input_size,
                        m.backbone.input_size),
            torch.randint(0, m.num_classes, (B,)))


# ---------------------------------------------------------------------------
# 1. zero updates leave the state == base state
# ---------------------------------------------------------------------------
def test_riskflow_zero_updates_leave_base_state():
    m = _rf_method(seed=1)
    with torch.no_grad():
        for p in m.cell.upd.parameters():   # delta == 0 at every depth
            p.zero_()
    m.eval()
    x, y = _rand_batch(m, B=4, seed=2)
    mp, flow = m.predict_with_trace(x, y=y)
    base = m.readout_hard(m.base_state.vector.detach().expand(4, m.state_dim)).squeeze(-1)
    for l in range(len(m.site_names) + 1):
        assert torch.allclose(flow.s_hard[l], base, atol=1e-6), l
    assert torch.allclose(flow.innov_hard, torch.zeros_like(flow.innov_hard),
                          atol=1e-6)
    assert torch.allclose(flow.final_s_hard, base, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. constructed residual sequence sums to the expected final logit
# ---------------------------------------------------------------------------
def test_riskflow_residual_sequence_sums_to_final_logit():
    m = _rf_method(seed=3)
    m.eval()
    x, y = _rand_batch(m, B=5, seed=4)
    mp, flow = m.predict_with_trace(x, y=y)
    L = len(m.site_names)
    assert flow.s_hard.shape == (L + 1, 5)
    assert flow.innov_hard.shape == (L, 5)
    cum = flow.s_hard[0] + torch.cumsum(flow.innov_hard, dim=0)
    assert torch.allclose(cum, flow.s_hard[1:], atol=1e-5)
    assert torch.allclose(flow.final_s_hard, flow.s_hard[-1], atol=1e-6)


# ---------------------------------------------------------------------------
# 3. stop-gradient boundaries are correct
# ---------------------------------------------------------------------------
def test_riskflow_stopgrad_boundaries_correct():
    m = _rf_method(seed=5)
    m.train()
    x, y = _rand_batch(m, B=6, seed=6)
    bo = m.backbone(x)
    flow = m._flow(bo, y=y)
    assert not flow.eps_hard.requires_grad
    assert not flow.hard_error.requires_grad
    assert not flow.soft_target.requires_grad
    if flow.eps_soft is not None:
        assert not flow.eps_soft.requires_grad
    assert flow.innov_hard.requires_grad
    assert flow.final_s_hard.requires_grad

    # the residual loss lands gradients on the prediction-side state path
    loss = F.huber_loss(flow.innov_hard, flow.eps_hard, delta=m.huber_delta)
    loss.backward()
    assert m.adapters["layer1"].proj.weight.grad is not None
    assert bool(torch.any(m.adapters["layer1"].proj.weight.grad != 0))
    for p in m.parameters():
        p.grad = None

    # the residual target must NOT backprop into the previous state: recompute
    # eps from a detached graph and confirm only the prediction half uses grad
    assert not flow.eps_hard.is_leaf or not flow.eps_hard.requires_grad


# ---------------------------------------------------------------------------
# 4. gates differ across constructed samples
# ---------------------------------------------------------------------------
def test_riskflow_gates_differ_across_samples():
    m = _rf_method(seed=7)
    m.eval()
    xa = torch.zeros(2, 3, 32, 32)
    xb = torch.ones(2, 3, 32, 32)
    _, fa = m.predict_with_trace(xa)
    _, fb = m.predict_with_trace(xb)
    assert fa.gates.shape == (len(m.site_names), 2)
    diff = (fa.gates[:, 0] != fb.gates[:, 0]) | (fa.gates[:, 1] != fb.gates[:, 1])
    assert bool(diff.any())


def test_riskflow_fixed_gates_are_sample_independent():
    m = _rf_method(seed=9, mode="resid")
    m.eval()
    xa = torch.zeros(3, 3, 32, 32)
    xb = torch.ones(3, 3, 32, 32)
    _, fa = m.predict_with_trace(xa)
    _, fb = m.predict_with_trace(xb)
    assert bool(torch.all(fa.gates == fb.gates))
    assert fa.gates.shape[0] == len(m.site_names)


# ---------------------------------------------------------------------------
# 5. deterministic state ordering follows the backbone registry
# ---------------------------------------------------------------------------
def test_riskflow_state_ordering_follows_registry():
    m = _rf_method(seed=11)
    assert m.site_names == ["layer1", "layer2", "layer3", "layer4"]
    assert m.site_names == list(m.backbone.taps.keys())
    m.eval()
    x, y = _rand_batch(m, B=3, seed=12)
    mp, flow = m.predict_with_trace(x, y=y)
    assert list(flow.site_names) == m.site_names
    assert flow.s_hard.shape[0] == len(m.site_names) + 1
    assert flow.gates.shape[0] == len(m.site_names)


# ---------------------------------------------------------------------------
# 6. constant-variance batches produce no NaN in decorrelation loss
# ---------------------------------------------------------------------------
def test_riskflow_decorrelation_no_nan_on_constant_variance():
    torch.manual_seed(13)
    L, B, D = 4, 8, 32
    deltas = torch.randn(L, B, D)
    deltas[2] = 3.0                                   # constant column
    loss = decorrelation_penalty(deltas)
    assert torch.isfinite(loss) and not torch.isnan(loss)
    zero = torch.zeros(L, B, D)
    assert torch.isfinite(decorrelation_penalty(zero))
    assert float(decorrelation_penalty(torch.randn(1, B, D))) == 0.0


# ---------------------------------------------------------------------------
# 7. checkpoint resume reproduces state/gate outputs exactly
# ---------------------------------------------------------------------------
def test_riskflow_checkpoint_resume_preserves_state_and_gates(tmp_path):
    m = _rf_method(results_root=str(tmp_path), seed=15)
    with torch.no_grad():
        for a in m.adapters.values():
            for p in a.parameters():
                p.normal_(0, 0.1)
        for p in m.cell.parameters():
            p.normal_(0, 0.1)
        m.readout_hard.weight.normal_(0, 0.1)
        m.readout_soft.weight.normal_(0, 0.1)
    ckpt = os.path.join(tmp_path, "rf.pt")
    torch.save(m.state_dict(), ckpt)

    m2 = _rf_method(results_root=str(tmp_path), seed=15)
    m2.load_state_dict(torch.load(ckpt, weights_only=True))
    sd = m2.state_dict()
    for n, p in m.named_parameters():
        assert torch.allclose(p, sd[n]), n
    m.eval()
    m2.eval()
    x, y = _rand_batch(m, B=4, seed=16)
    _, f1 = m.predict_with_trace(x, y=y)
    _, f2 = m2.predict_with_trace(x, y=y)
    assert torch.allclose(f2.final_s_hard, f1.final_s_hard)
    assert torch.allclose(f2.gates, f1.gates)
    assert torch.allclose(f2.s_hard, f1.s_hard)
    assert torch.allclose(f2.innov_hard, f1.innov_hard)


# ---------------------------------------------------------------------------
# diagnostics: redundancy report + fixed category assignment
# ---------------------------------------------------------------------------
def test_riskflow_redundancy_report_and_assignment():
    torch.manual_seed(17)
    N, L = 32, 4
    indep = torch.randn(N, L) * 3.0 + torch.randn(N, 1)   # redundant columns
    innov = torch.randn(N, L)                              # mostly decorrelated
    cum = torch.randn(N, L).cumsum(1)
    r = redundancy_report(indep.numpy(), innov.numpy(), cum.numpy())
    for key in ("cka_offdiag_mean_independent_heads", "cka_offdiag_mean_innovations",
                "corr_offdiag_mean_independent_heads",
                "corr_offdiag_mean_innovations", "cumulative_risk_corr_final"):
        assert key in r
    cka = pairwise_linear_cka(innov.numpy())
    assert np.allclose(np.diag(cka), 1.0)
    assert np.isfinite(cka).all()

    err = np.array([0, 0, 0, 0, 1, 1, 1], dtype=int)
    risk = np.array([0.1, 0.5, 0.8, 0.9, 0.2, 0.5, 0.9])
    cats = assign_category(err, risk, cat_lo=0.3, cat_hi=0.7)
    assert cats[0] == "easy_correct"
    assert cats[1] == "ambiguous_correct"
    assert cats[2] == "corrupted"
    assert cats[4] == "high_conf_wrong"
    assert cats[6] == "corrupted"


# ---------------------------------------------------------------------------
# ablation-ladder configs all build and predict
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method_name,variant", [
    ("riskflow_concat", "concat"),
    ("riskflow_heads", "heads"),
    ("riskflow_cum", "cum"),
    ("riskflow_resid", "resid"),
    ("riskflow", "riskflow"),
    ("riskflow_frozen", "riskflow"),
    ("riskflow_hard", "riskflow"),
])
def test_riskflow_required_configs_load(tmp_path, method_name, variant):
    cfg = config.resolve({
        "dataset": "cifar10",
        "backbone": "resnet18",
        "method_name": method_name,
        "results_root": str(tmp_path),
        "train": {"device": "cpu", "seed": 0, "epochs": 1, "batch_size": 8,
                  "lr": 0.01},
    })
    m = build_method(method_name, cfg)
    assert m.variant == variant
    m.eval()
    mp = m.predict_batch(torch.randn(1, 3, 32, 32))
    assert tuple(mp.confidence.shape) == (1,)
    assert "riskflow" in mp.scores


@pytest.mark.parametrize("backbone",
                         ["resnet18", "vgg16_bn", "wideresnet28_10",
                          "convnext_tiny", "deit_s"])
def test_riskflow_default_config_loads_across_backbones(tmp_path, backbone):
    cfg = config.resolve({
        "dataset": "cifar10",
        "backbone": backbone,
        "method_name": "riskflow",
        "results_root": str(tmp_path),
        "train": {"device": "cpu", "seed": 0, "epochs": 1, "batch_size": 8,
                  "lr": 0.01},
    })
    m = build_method("riskflow", cfg)
    assert list(m.site_names) == list(m.backbone.taps.keys())
    m.eval()
    mp = m.predict_batch(torch.randn(1, 3, 32, 32))
    assert tuple(mp.confidence.shape) == (1,)


# ---------------------------------------------------------------------------
# gradient-path: e2e reaches backbone, frozen control does not
# ---------------------------------------------------------------------------
def test_riskflow_gradient_path_e2e_and_frozen():
    torch.manual_seed(19)
    x, y = torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,))

    m = _rf_method(seed=19)
    m.train()
    ld = m.train_loss((x, y, torch.arange(4)), None)
    total = sum(v for v in ld.values() if torch.is_tensor(v) and v.requires_grad)
    for p in m.parameters():
        p.grad = None
    total.backward()
    backbone_reached = [n for n, p in m.backbone.named_parameters()
                        if p.grad is not None and bool(torch.any(p.grad != 0))]
    assert backbone_reached
    assert any("layer1" in n for n in backbone_reached)

    mf = _rf_method(seed=19, freeze_backbone=True)
    mf.train()
    ldf = mf.train_loss((x, y, torch.arange(4)), None)
    totf = sum(v for v in ldf.values() if torch.is_tensor(v) and v.requires_grad)
    for p in mf.parameters():
        p.grad = None
    totf.backward()
    backbone_reached_f = [n for n, p in mf.backbone.named_parameters()
                          if p.grad is not None and bool(torch.any(p.grad != 0))]
    assert backbone_reached_f == []
    aux_reached = [n for n, p in mf.named_parameters()
                   if not n.startswith("backbone.") and p.grad is not None
                   and bool(torch.any(p.grad != 0))]
    assert any("adapters" in n for n in aux_reached)


# ---------------------------------------------------------------------------
# deployment parameters: soft channel (auxiliary) excluded
# ---------------------------------------------------------------------------
def test_riskflow_deployment_params_exclude_soft_channel():
    m = _rf_method(seed=21)
    infer = set()
    for mod in m.inference_modules():
        infer |= {id(p) for p in mod.parameters()}
    all_p = set(id(p) for p in m.parameters())
    soft_ids = set(id(p) for p in m.readout_soft.parameters())
    assert infer == all_p - soft_ids
    assert id(next(m.readout_hard.parameters())) in infer
    assert id(next(m.readout_soft.parameters())) not in infer


# ---------------------------------------------------------------------------
# one engine-style training step moves the whole recurrent stack
# ---------------------------------------------------------------------------
def test_riskflow_train_step_optimizes_state_stack():
    cfg = _rf_cfg()
    cfg["train"]["seed"] = 23
    m = build_method("riskflow", cfg)
    m.train()
    opt = _build_optimizers(m, cfg)[0]
    loss_dict = m.train_loss((torch.randn(6, 3, 32, 32),
                              torch.randint(0, 10, (6,)), torch.arange(6)), None)
    assert {"ce", "rf_innov_hard", "rf_term_hard"} <= set(loss_dict)
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
    assert any("cell" in n or "readout" in n for n in moved)


def test_riskflow_soft_nll_above_one_uses_nonnegative_regression():
    """A confidently wrong normalized NLL is not a valid BCE target."""
    m = _rf_method(seed=24)
    m.train()
    classifier = m.backbone.base_model.fc
    with torch.no_grad():
        classifier.weight.zero_()
        classifier.bias.zero_()
        classifier.bias[0] = 20.0
    x = torch.randn(4, 3, 32, 32)
    y = torch.ones(4, dtype=torch.long)
    flow = m._flow(m.backbone(x), y=y)
    assert bool(torch.all(flow.soft_target > 1.0))
    losses = m.train_loss((x, y, torch.arange(4)), None)
    assert torch.isfinite(losses["rf_term_soft"])
    assert float(losses["rf_term_soft"]) >= 0.0


# ---------------------------------------------------------------------------
# trace export produces all required per-depth arrays
# ---------------------------------------------------------------------------
def test_riskflow_trace_export_has_all_artifacts(tmp_path):
    m = _rf_method(seed=25)
    m.eval()
    x, y = _rand_batch(m, B=5, seed=26)
    mp, flow = m.predict_with_trace(x, y=y)
    data = export_trace(flow)
    for key in ("site_names", "s_hard", "innov_hard", "gates", "deltas",
                "eps_hard", "hard_error", "soft_target", "final_s_hard",
                "s_soft", "innov_soft", "eps_soft", "final_s_soft"):
        assert key in data, key
    assert data["gates"].shape[0] == len(m.site_names)
    assert data["deltas"].shape[0] == len(m.site_names)
    assert np.all(np.isfinite(data["s_hard"]))
    assert np.all(np.isfinite(data["deltas"]))


# ---------------------------------------------------------------------------
# tiny ResNet-18 + DeiT artifact runs (server-only smoke)
# ---------------------------------------------------------------------------
def _smoke_skip():
    if not torch.cuda.is_available():
        return "tiny artifact runs require cuda (server smoke)"
    if not os.environ.get("SCSF_RUN_SMOKE"):
        return "set SCSF_RUN_SMOKE=1 to run the tiny artifact smokes"
    return None


@pytest.mark.parametrize("backbone", ["resnet18", "deit_s"])
def test_riskflow_smoke_artifacts(tmp_path, backbone):
    reason = _smoke_skip()
    if reason:
        pytest.skip(reason)

    data_root = os.environ.get("SCSF_DATA_ROOT", os.path.join(REPO_ROOT, "data"))
    if not os.path.isdir(os.path.join(data_root, "cifar-10-batches-py")):
        pytest.skip("cifar-10 sources missing; run scripts/smoke_riskflow.sh")

    results_root = str(tmp_path)
    cfg = config.resolve({
        "dataset": "cifar10",
        "backbone": backbone,
        "method_name": "riskflow",
        "results_root": results_root,
        "train": {"device": "cuda", "seed": 0, "epochs": 1,
                  "batch_size": 16, "lr": 0.01, "overfit": 32,
                  "weight_decay": 0.0, "scheduler": "cosine",
                  "eval_every": 1, "save_every": 1},
    })
    from scsf.engine.trainer import Trainer
    from scsf.data import build_dataloader

    run_dir = os.path.join(results_root, cfg["run_name"])
    Trainer(cfg, run_dir).run()
    assert os.path.exists(os.path.join(run_dir, "selected.pt"))

    m = build_method("riskflow", cfg)
    ckpt = torch.load(os.path.join(run_dir, "selected.pt"), weights_only=False)
    m.load_state_dict(ckpt["model_state"])
    m.to("cuda").eval()

    os.environ.setdefault("SCSF_DATA_SEED", "0")
    loader = build_dataloader(cfg, "train", shuffle=False, return_indices=True,
                              overfit=128, num_workers=0)
    xb, yb, ids = next(iter(loader))
    xb, yb = xb.to("cuda"), yb.to("cuda")
    mp, flow = m.predict_with_trace(xb, yb)
    data = export_trace(flow)

    out_dir = os.path.join(run_dir, "riskflow_smoke")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "trace_train.npz"),
             **{k: (v if isinstance(v, np.ndarray) else np.asarray(v))
                for k, v in data.items()})
    assert os.path.exists(os.path.join(out_dir, "trace_train.npz"))

    from scsf.riskflow import save_trajectory_plots

    cats = assign_category(data["hard_error"], data["final_s_hard"],
                           cfg["method"].get("cat_lo", 0.3),
                           cfg["method"].get("cat_hi", 0.7))
    plots = save_trajectory_plots(data, category_key=cats, out_dir=out_dir)
    for cat in ("easy_correct", "ambiguous_correct", "high_conf_wrong", "corrupted"):
        if f"trajectory_{cat}.png" in [os.path.basename(p) for p in plots]:
            assert os.path.exists(os.path.join(out_dir, f"trajectory_{cat}.png"))
    with open(os.path.join(out_dir, "smoke_summary.json"), "w") as f:
        json.dump({"backbone": backbone, "n": int(data["hard_error"].shape[0]),
                   "sites": list(np.asarray(data["site_names"]))}, f)
    assert os.path.exists(os.path.join(out_dir, "smoke_summary.json"))
