"""DepthFrag contract + scientific-integrity tests.

Covers the geometry (exact linear case, scale invariance, finite-degenerate,
sign/aggregation primitives), the BatchNorm treatment (fast ~ exact on
block-diagonal networks, train-mode coupling demonstrated), the method-level
contracts (detached targets, stripped inference, gradient reach e2e vs frozen,
checkpoint resume, target-interval semantics, required configs), and server-only
smoke artifacts for tiny ResNet-18 / DeiT runs (SCSF_RUN_SMOKE=1).
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
import torch.nn.functional as F

try:
    import torch.func  # noqa: F401  torch >= 2.0

    _HAS_FUNC = True
except Exception:  # pragma: no cover
    _HAS_FUNC = False

from scsf.backbones import Backbone, BackboneOutput, adaptive_flatten
from scsf.depthfrag.geometry import (
    SiteRadiiComputer,
    aggregate_profile,
    radii_from_site,
    target_transform,
    true_class_margin,
)
from scsf.engine import config
from scsf.engine.trainer import Trainer, _build_optimizers
from scsf.methods import build_method

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _df_cfg(results_root="/tmp/opencode/depthfrag_tests", seed=0,
            method_name="depthfrag", **m):
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


def _df_method(results_root="/tmp/opencode/depthfrag_tests", seed=0, **m):
    cfg = _df_cfg(results_root, seed, **m)
    return build_method("depthfrag", cfg)


# ---------------------------------------------------------------------------
# tiny test backbone (registry-independent)
# ---------------------------------------------------------------------------
class TinyCNN(Backbone):
    def __init__(self, num_classes=6, bn=False, seed=7):
        super().__init__(num_classes, input_size=32, channels=3)
        self._bn = bool(bn)
        self._seed = int(seed)
        torch.manual_seed(seed)
        in_ch = 3
        blocks = OrderedDict()
        for i, out_ch in enumerate((8, 16, 32), start=1):
            layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1)]
            if bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            blocks[f"block{i}"] = nn.Sequential(*layers)
            in_ch = out_ch
        for name, mod in blocks.items():
            self.add_module(name, mod)
        self.taps = blocks
        self.roles = {"top_l1": "block3", "top_l2": "block2", "top_l3": "block1"}
        self.final_dim = 32
        self.fc = nn.Linear(32, num_classes)

    def forward_backbone(self, x):
        feats = OrderedDict()
        h = x
        for name in self.taps:
            h = self.taps[name](h)
            feats[name] = h
        emb = adaptive_flatten(h, out=1)
        return BackboneOutput(self.fc(emb), feats, emb)


def _rand_batch(net, B=8, seed=0, C=None):
    torch.manual_seed(seed)
    x = torch.randn(B, net.channels, net.input_size, net.input_size)
    y = torch.randint(0, C or net.num_classes, (B,))
    return x, y


def _rel_err(a, b):
    a, b = a.float(), b.float()
    return float(((a - b).abs() / (1 + a.abs() + b.abs())).mean())


# ---------------------------------------------------------------------------
# 1. geometry: exact analytic case on a linear classifier
# ---------------------------------------------------------------------------
def test_depthfrag_exact_linear_classifier_radii():
    torch.manual_seed(0)
    D, C, B = 6, 4, 64
    W = torch.randn(D, C) * 0.7
    b = torch.randn(C) * 0.2
    x = torch.randn(B, D)
    y = torch.randint(0, C, (B,))
    z = x @ W + b
    m = true_class_margin(z, y)
    assert tuple(m.shape) == (B,)

    top2 = torch.topk(z, 2, dim=1)
    is_argmax = z.argmax(dim=1) == y
    c_star = torch.where(is_argmax, top2.indices[:, 1], top2.indices[:, 0])
    g_analytic = (W[:, y] - W[:, c_star]).T          # d m / d h, h == x
    xr = x.detach().clone().requires_grad_(True)
    g_numeric = torch.autograd.grad(
        true_class_margin(xr @ W + b, y).sum(), xr)[0]
    assert torch.allclose(g_numeric, g_analytic, atol=1e-5)

    for (p, q) in ((2, 2), (float("inf"), 1)):
        rho, rel = radii_from_site(m, g_numeric, x, p=p, q=q, eps=1e-12)
        rho_a, rel_a = radii_from_site(m, g_analytic, x, p=p, q=q, eps=1e-12)
        assert torch.allclose(rho, rho_a, atol=1e-5)
        assert torch.allclose(rel, rel_a, atol=1e-5)
        if p == 2:
            assert torch.allclose(rho, m / (g_analytic.norm(dim=1) + 1e-12),
                                  atol=1e-5)
    t = target_transform(rel, "signed_log1p")
    assert torch.allclose(t, torch.sign(rel) * torch.log1p(rel.abs()))


# ---------------------------------------------------------------------------
# 2. scale invariance of the signed relative radius (eps = 0 exactly)
# ---------------------------------------------------------------------------
def test_depthfrag_scale_invariance_positive_logit_scalar():
    torch.manual_seed(0)
    m = torch.randn(32).abs() + 0.1
    g = torch.randn(32, 8)
    h = torch.randn(32, 8).abs() + 0.1
    c = 2.3
    for (p, q) in ((2, 2), (float("inf"), 1)):
        rho0, rel0 = radii_from_site(m, g, h, p=p, q=q, eps=0.0)
        rho1, rel1 = radii_from_site(c * m, c * g, h, p=p, q=q, eps=0.0)
        assert torch.allclose(rho0, rho1, atol=1e-5)
        assert torch.allclose(rel0, rel1, atol=1e-5)


# ---------------------------------------------------------------------------
# 3. finite when degenerate (zero margin, zero gradient, missing gradient)
# ---------------------------------------------------------------------------
def test_depthfrag_finite_when_margin_or_gradient_zero():
    B, D = 8, 16
    z = torch.zeros(B, D)
    o = torch.ones(B, D)
    m0 = torch.zeros(B)
    rho, rel = radii_from_site(m0, z, z, eps=1e-12)
    assert torch.all(torch.isfinite(rho)) and torch.all(torch.isfinite(rel))
    assert torch.all(rho == 0.0) and torch.all(rel == 0.0)

    m1 = torch.full((B,), 2.0)
    rho1, rel1 = radii_from_site(m1, z, o, eps=1e-12)
    assert torch.all(torch.isfinite(rho1)) and torch.all(rho1 == m1 / 1e-12)
    assert torch.all(torch.isfinite(rel1))

    rho2, rel2 = radii_from_site(m1, None, o, eps=1e-12)   # site off the path
    assert torch.all(rho2 == 0.0) and torch.all(rel2 == 0.0)

    rho3, rel3 = radii_from_site(m1, z, o, p=float("inf"), q=1, eps=1e-12)
    assert torch.all(torch.isfinite(rho3)) and torch.all(torch.isfinite(rel3))


# ---------------------------------------------------------------------------
# 4. target-transform kinds + aggregation variants
# ---------------------------------------------------------------------------
def test_depthfrag_target_kinds_sign_and_bounds():
    rel = torch.tensor([2.0, -3.0])
    t = target_transform(rel, "signed_log1p")
    assert torch.allclose(t, torch.sign(rel) * torch.log1p(rel.abs()), atol=1e-6)
    ta = target_transform(rel, "absolute")
    assert torch.all(ta >= 0)
    assert torch.allclose(ta, torch.log1p(rel.abs()), atol=1e-6)
    tc = target_transform(torch.tensor([50.0, -50.0]), "clipped", cap=1.0)
    assert torch.allclose(tc, torch.sign(tc) * torch.log1p(torch.tensor(1.0)))


def test_depthfrag_aggregate_profile_variants():
    torch.manual_seed(0)
    profile = torch.randn(20, 4)
    assert torch.allclose(aggregate_profile(profile, "mean"), profile.mean(-1))
    assert torch.allclose(aggregate_profile(profile, "min"),
                          profile.min(-1).values)
    assert torch.allclose(aggregate_profile(profile, "terminal"),
                          profile[..., -1])
    sm = aggregate_profile(profile, "soft_min", tau=2.0)
    w = F.softmax(-2.0 * profile, dim=-1)
    assert torch.allclose(sm, (w * profile).sum(-1))
    cv = aggregate_profile(profile, "cvar", frac=0.5)
    small = torch.topk(-profile, 2, dim=-1).values
    assert torch.allclose(cv, small.mean(-1))
    with pytest.raises(ValueError):
        aggregate_profile(profile, "no_such_agg")


# ---------------------------------------------------------------------------
# 5. fast ~ exact on a block-diagonal network; no 2nd-order graph retained
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _HAS_FUNC, reason="torch.func unavailable")
def test_depthfrag_fast_matches_exact_without_cross_example_ops():
    torch.manual_seed(0)
    net = TinyCNN(bn=False)
    sites = list(net.taps)
    x, y = _rand_batch(net, B=8)
    fast = SiteRadiiComputer(net, sites, mode="fast", eps=1e-6)
    exact = SiteRadiiComputer(net, sites, mode="exact", exact_microbatch=2,
                              eps=1e-6)
    rb_f = fast.compute(x, y, role="eval")
    rb_e = exact.compute(x, y, role="eval")
    for key in ("rho", "rel", "target"):
        for s in sites:
            err = _rel_err(getattr(rb_f, key)[s], getattr(rb_e, key)[s])
            assert err < 1e-3, (key, s, err)
    assert torch.equal(rb_f.prediction, rb_e.prediction)
    assert torch.allclose(rb_f.margin, rb_e.margin, atol=1e-5)


@pytest.mark.skipif(not _HAS_FUNC, reason="torch.func unavailable")
def test_depthfrag_batchnorm_training_role_couples_examples():
    torch.manual_seed(0)
    sites = ["block1", "block2", "block3"]
    x, y = _rand_batch(TinyCNN(), B=8)

    def _mk_and_run(role, mode):
        # fresh network per run so no BN buffer state leaks between modes
        fresh = TinyCNN(bn=True)
        rb = SiteRadiiComputer(fresh, sites, mode=mode, eps=1e-6)
        return rb.compute(x, y, role=role)

    def _err(rb_a, rb_b):
        return float(np.mean([_rel_err(getattr(rb_a, k)[s], getattr(rb_b, k)[s])
                              for k in ("rho", "rel", "target") for s in sites]))

    diff_eval = _err(_mk_and_run("eval", "fast"), _mk_and_run("eval", "exact"))
    diff_train = _err(_mk_and_run("train", "fast"), _mk_and_run("train", "exact"))
    # eval role: fast batch VJP is a valid per-example reading
    assert diff_eval < 1e-3, diff_eval
    # train role: batch statistics couple the examples, so the single-batch VJP
    # diverges from the per-example (true) reading by much more than float noise
    assert diff_train > max(1e-2, 10.0 * diff_eval), (diff_eval, diff_train)


def test_depthfrag_no_second_order_graph_and_bounded_memory():
    torch.manual_seed(0)
    net = TinyCNN(bn=False)
    sites = list(net.taps)
    computer = SiteRadiiComputer(net, sites, mode="fast", eps=1e-6)
    x, y = _rand_batch(net, B=8)

    def _live_tensors():
        gc.collect()
        return len([o for o in gc.get_objects() if isinstance(o, torch.Tensor)])

    base = _live_tensors()
    rb = computer.compute(x, y, role="eval", return_logits=True)
    # everything returned is detached / graph-free: no torch path can extend it
    for t in (rb.margin, rb.prediction, rb.logits):
        assert not t.requires_grad
    for d in (rb.rho, rb.rel, rb.target):
        assert all(not v.requires_grad for v in d.values())
    # the graph is freed after compute: no second-order backprop is possible
    p0 = next(net.parameters())
    with pytest.raises(RuntimeError):
        torch.autograd.grad(rb.margin.sum(), p0)
    # repeated batch-level compute keeps the live-tensor count bounded
    for _ in range(6):
        computer.compute(x, y)
    after = _live_tensors()
    assert after < base + 3000, (base, after)


# ---------------------------------------------------------------------------
# 6. method-level contract: stripped inference reproduces fragility scores
# ---------------------------------------------------------------------------
def test_depthfrag_stripped_inference_reproduces_fragility_scores():
    m = _df_method(seed=2)
    m.eval()
    x = torch.randn(6, 3, 32, 32)
    mp1 = m.predict_batch(x)
    assert torch.equal(mp1.confidence, mp1.scores["depthfrag"])
    assert set(mp1.scores) >= {"msp", "entropy", "energy", "logit_margin",
                               "depthfrag"}
    mp2 = m.stripped_predict_batch(x)
    assert torch.equal(mp2.confidence, mp1.confidence)

    infer = {id(p) for mod in m.inference_modules() for p in mod.parameters()}
    probe_params = {id(p) for pr in m.probes.values() for p in pr.parameters()}
    assert infer & probe_params == set()          # probes never in deployment
    assert id(next(m.head.parameters())) in infer  # the fragility head is

    with torch.no_grad():                         # probes are off the score path
        for pr in m.probes.values():
            for p in pr.parameters():
                p.uniform_(-2.0, 2.0)
    mp3 = m.predict_batch(x)
    assert torch.allclose(mp3.logits, mp1.logits)
    assert torch.equal(mp3.prediction, mp1.prediction)
    assert torch.allclose(mp3.confidence, mp1.confidence)
    # the head reads the embedding, not the logits
    assert not torch.allclose(mp1.scores["depthfrag"],
                              mp1.scores["logit_margin"], atol=1e-3)


# ---------------------------------------------------------------------------
# 7. gradient reach: e2e reaches the backbone, frozen control does not
# ---------------------------------------------------------------------------
def test_depthfrag_probe_gradients_reach_backbone_e2e_not_frozen():
    from scsf.methods.depthfrag import params_reached_by_probes, probe_gradient_report

    torch.manual_seed(3)
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))

    m = _df_method(seed=3)
    rep = probe_gradient_report(m, x, y)
    assert len(rep["backbone_reached"]) > 0
    assert rep["probe_grad_norm"] > 0.0
    assert rep["head_grad_norm"] > 0.0
    for site, submod in m.backbone.taps.items():
        reached = params_reached_by_probes(m, site, num_examples=2)
        assert reached, site  # every site's probe has a gradient path into the CNN

    mf = _df_method(seed=3, freeze_backbone=True)
    rep_f = probe_gradient_report(mf, x, y)
    assert rep_f["backbone_reached"] == []
    assert rep_f["backbone_grad_norm"] == 0.0
    assert rep_f["probe_grad_norm"] > 0.0
    assert rep_f["head_grad_norm"] > 0.0


# ---------------------------------------------------------------------------
# 8. checkpoint resume preserves probes + head exactly
# ---------------------------------------------------------------------------
def test_depthfrag_resume_preserves_probes_and_head(tmp_path):
    m = _df_method(results_root=str(tmp_path), seed=4)
    with torch.no_grad():
        for pr in m.probes.values():
            for p in pr.parameters():
                p.normal_(0.0, 0.1)
        for p in m.head.parameters():
            p.normal_(0.0, 0.1)
    ckpt = os.path.join(tmp_path, "df.pt")
    torch.save(m.state_dict(), ckpt)

    m2 = _df_method(results_root=str(tmp_path), seed=4)
    m2.load_state_dict(torch.load(ckpt, weights_only=True))
    sd = m2.state_dict()
    assert any(k.startswith("probes.") for k in sd)
    assert any(k.startswith("head.") for k in sd)
    assert any(k.startswith("backbone.") for k in sd)
    for n, p in m.named_parameters():
        assert torch.allclose(p, sd[n]), n

    m.eval()
    m2.eval()
    x = torch.randn(3, 3, 32, 32)
    mp, mp2 = m.predict_batch(x), m2.predict_batch(x)
    assert torch.equal(mp2.prediction, mp.prediction)
    assert torch.allclose(mp2.confidence, mp.confidence)


# ---------------------------------------------------------------------------
# 9. method targets are detached and agree with the batch-geometry computer
# ---------------------------------------------------------------------------
def test_depthfrag_targets_detached_and_consistent_with_geometry():
    torch.manual_seed(5)
    m = _df_method(seed=5)
    m.eval()
    x = torch.randn(6, 3, 32, 32)
    y = torch.randint(0, 10, (6,))
    targets, rel, margin = m._targets_eval_forward(x, y)
    for s in m.site_names:
        assert not targets[s].requires_grad
    assert not margin.requires_grad

    rb = SiteRadiiComputer(m.backbone, m.site_names, p=m.p, q=m.q, eps=m.eps,
                           target_kind=m.target_kind, clip_cap=m.clip_cap,
                           mode="fast").compute(x, y, role="eval")
    for s in m.site_names:
        assert torch.allclose(rel[s].detach(), rb.rel[s], atol=1e-5), s
        assert torch.allclose(targets[s].detach(), rb.target[s], atol=1e-5), s
    assert torch.allclose(margin, rb.margin, atol=1e-5)


# ---------------------------------------------------------------------------
# 10. one engine-style step: backbone + probes + head all move under the opt
# ---------------------------------------------------------------------------
def test_depthfrag_train_step_optimizes_backbone_probes_and_head():
    cfg = _df_cfg()
    cfg["train"]["seed"] = 7
    m = build_method("depthfrag", cfg)
    m.train()
    opt = _build_optimizers(m, cfg)[0]
    state = SimpleNamespace(batch_index=0)
    loss_dict = m.train_loss((torch.randn(6, 3, 32, 32),
                              torch.randint(0, 10, (6,)), torch.arange(6)), state)
    assert {"ce", "depthfrag_probe", "depthfrag_head"} <= set(loss_dict)
    total = sum(v for v in loss_dict.values()
                if torch.is_tensor(v) and v.requires_grad)
    assert torch.isfinite(total)
    for p in m.parameters():
        p.grad = None
    total.backward()
    before = {n: p.clone() for n, p in m.named_parameters()}
    opt.step()
    moved = [n for n, p in m.named_parameters() if not torch.equal(p, before[n])]
    assert any(n.startswith("backbone.") for n in moved)
    assert any("probes." in n for n in moved)
    assert any("head." in n for n in moved)


def test_depthfrag_target_interval_skips_aux_steps():
    cfg = _df_cfg()
    cfg["method"]["target_interval"] = 3
    m = build_method("depthfrag", cfg)
    m.train()
    x = torch.randn(6, 3, 32, 32)
    y = torch.randint(0, 10, (6,))
    off = m.train_loss((x, y, torch.arange(6)), SimpleNamespace(batch_index=1))
    assert set(off) == {"ce"}
    on = m.train_loss((x, y, torch.arange(6)), SimpleNamespace(batch_index=3))
    assert {"depthfrag_probe", "depthfrag_head"} <= set(on)


# ---------------------------------------------------------------------------
# 11. every required config loads and predicts
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method_name", [
    "depthfrag",                 # default e2e distilled score
    "depthfrag_terminal_margin", # ablation 1: terminal true-class margin only
    "depthfrag_terminal",        # ablation 2: terminal radius only (top_l1)
    "depthfrag_intermediate",    # ablation 3: single intermediate radius (top_l2)
    "depthfrag_raw",             # ablation 4: raw full-depth profile (mean agg)
    "depthfrag_frozen",          # ablation 6: distilled score, frozen backbone
    "depthfrag_clip",            # ablation 7: absolute/clipped radius control
])
def test_depthfrag_required_configs_load(tmp_path, method_name):
    cfg = config.resolve({
        "dataset": "cifar10",
        "backbone": "resnet18",
        "method_name": method_name,
        "results_root": str(tmp_path),
        "train": {"device": "cpu", "seed": 0, "epochs": 1, "batch_size": 8,
                  "lr": 0.01},
    })
    m = build_method(method_name, cfg)
    assert set(m.backbone.roles) >= {"top_l1", "top_l2"}
    assert set(m.backbone.taps) >= set(m.site_names)
    m.eval()
    mp = m.predict_batch(torch.randn(1, 3, 32, 32))
    assert tuple(mp.confidence.shape) == (1,)


@pytest.mark.parametrize("backbone",
                         ["resnet18", "vgg16_bn", "wideresnet28_10",
                          "convnext_tiny", "deit_s"])
def test_depthfrag_default_config_loads_across_backbones(tmp_path, backbone):
    cfg = config.resolve({
        "dataset": "cifar10",
        "backbone": backbone,
        "method_name": "depthfrag",
        "results_root": str(tmp_path),
        "train": {"device": "cpu", "seed": 0, "epochs": 1, "batch_size": 8,
                  "lr": 0.01},
    })
    m = build_method("depthfrag", cfg)
    assert set(m.backbone.roles) >= {"top_l1", "top_l2"}
    m.eval()
    mp = m.predict_batch(torch.randn(1, 3, 32, 32))
    assert tuple(mp.confidence.shape) == (1,)


# ---------------------------------------------------------------------------
# 12. extraction CLI rejects run dirs without cfg.json
# ---------------------------------------------------------------------------
def test_extract_depthfg_rejects_non_run_dir():
    from scsf import extract_depthfg as cli

    with pytest.raises(FileNotFoundError):
        cli.main(["run_dir=/tmp/opencode/not_a_run_dir_depthfrag", "split=val"])


# ---------------------------------------------------------------------------
# 13. oracle fits on validation only, applied to test (scipy-guarded)
# ---------------------------------------------------------------------------
def test_depthfrag_oracle_fit_then_apply_orientation():
    pytest.importorskip("scipy")
    from scsf.depthfrag.oracle import fit_and_apply

    rng = np.random.RandomState(0)
    n_fit, n_apply = 80, 40
    x0_fit = rng.randn(n_fit)
    x0_apply = rng.randn(n_apply)
    labels_fit = np.zeros(n_fit, dtype=int)
    labels_apply = np.zeros(n_apply, dtype=int)
    predictions_fit = (x0_fit >= 0.0).astype(int)   # small radius <=> error
    predictions_apply = (x0_apply >= 0.0).astype(int)
    profile_fit = np.column_stack([x0_fit, rng.randn(n_fit, 2)])
    profile_apply = np.column_stack([x0_apply, rng.randn(n_apply, 2)])

    for variant in ("lin", "logit"):
        conf, meta = fit_and_apply(profile_fit, labels_fit, predictions_fit,
                                   profile_apply, variant)
        assert conf.shape == (n_apply,)
        assert np.all(np.isfinite(conf))
        err_apply = (labels_apply != predictions_apply).astype(float)
        left, right = conf[err_apply == 1], conf[err_apply == 0]
        assert float(left.mean()) < float(right.mean()), variant  # lower risk
        if variant == "logit":
            assert np.all((conf >= -1.0) & (conf <= 0.0))


# ---------------------------------------------------------------------------
# 14. iterative boundary audit runs, reports the required numbers
# ---------------------------------------------------------------------------
def test_depthfrag_iterative_audit_contract():
    from scsf.depthfrag.iterative import (
        compare_analytic_iterative,
        iterative_boundary_audit,
    )

    torch.manual_seed(0)
    net = TinyCNN()
    x, y = _rand_batch(net, B=8)
    audit = iterative_boundary_audit(net, x, y, max_steps=3)
    assert set(audit) == {"per_sample", "summary"}
    assert audit["per_sample"].dtype.names == ("dist", "steps", "flipped")
    assert audit["summary"]["n"] == 8
    assert np.all(np.isfinite(audit["per_sample"]["dist"]))
    assert audit["summary"]["wall_ms"] >= 0.0
    cmp = compare_analytic_iterative(np.random.randn(8), audit["per_sample"]["dist"])
    assert set(cmp) == {"spearman", "relative_error", "n"}
    assert cmp["n"] == 8


# ---------------------------------------------------------------------------
# 15. tiny ResNet-18 / DeiT artifact runs (server-only smoke, ops + data)
# ---------------------------------------------------------------------------
def _smoke_skip():
    if not torch.cuda.is_available():
        return "tiny artifact runs require cuda (server smoke)"
    if not os.environ.get("SCSF_RUN_SMOKE"):
        return "set SCSF_RUN_SMOKE=1 to run the tiny artifact smokes"
    return None


@pytest.mark.parametrize("backbone", ["resnet18", "deit_s"])
def test_depthfrag_smoke_artifacts(tmp_path, backbone):
    reason = _smoke_skip()
    if reason:
        pytest.skip(reason)
    pytest.importorskip("scipy")
    from scsf.depthfrag.extract import DepthFragExtractor, evaluate_variants

    data_root = os.environ.get("SCSF_DATA_ROOT", os.path.join(REPO_ROOT, "data"))
    if not os.path.isdir(os.path.join(data_root, "cifar-10-batches-py")):
        pytest.skip("cifar-10 sources missing; run scripts/smoke_depthfrag.sh")

    results_root = str(tmp_path)
    cfg = config.resolve({
        "dataset": "cifar10",
        "backbone": backbone,
        "method_name": "depthfrag",
        "results_root": results_root,
        "train": {"device": "cuda", "seed": 0, "epochs": 1, "batch_size": 16,
                  "lr": 0.01, "overfit": 16, "weight_decay": 0.0,
                  "scheduler": "cosine", "eval_every": 1, "save_every": 1},
    })
    run_dir = os.path.join(results_root, cfg["run_name"])
    Trainer(cfg, run_dir).run()
    assert os.path.exists(os.path.join(run_dir, "selected.pt"))

    ext = DepthFragExtractor(cfg, run_dir, checkpoint="last", device="cuda",
                             p=cfg["method"].get("p", 2), q=cfg["method"].get("q", 2),
                             eps=cfg["method"].get("eps", 1e-12),
                             mid_roles=("top_l2",))
    prof_v = ext.profile_split("val", subset=128, num_workers=0)
    assert prof_v["n"] == 128
    assert prof_v["rel"].shape == (128, len(ext.site_names))
    assert np.all(np.isfinite(prof_v["rel"]))

    from scsf.data.cifar import TEST_SPLIT_DISABLED, set_test_allowed

    was = TEST_SPLIT_DISABLED   # flip the official-test guard like the CLI does
    set_test_allowed(True)
    try:
        prof_t = ext.profile_split("test", subset=256, num_workers=0)
    finally:
        set_test_allowed(was)
    assert prof_t["n"] == 256

    ev = evaluate_variants(prof_v, prof_t, ext.terminal_site, ext.mid_sites)
    json.dumps(ev)  # artifact must be json-serializable
    assert {"msp", "logit_margin", "term_radius", "softmin_radius",
            "oracle_lin", "oracle_logit"} <= set(ev["variants"])
    assert set(ev["variants"]["term_radius"]) == {"val", "test"}

    audit = ext.iterative_audit(prof_v, subset=16, max_steps=5)
    assert audit["summary"]["n"] == 16
    assert {"spearman", "relative_error", "n"} <= set(
        audit["analytic_vs_iter"]["terminal_radius"])

    out_dir = os.path.join(run_dir, "depthfrag_smoke")
    ext.save_artifacts(out_dir, prof_v, {"mode": "fast", "smoke": True})
    ext.save_artifacts(out_dir, prof_t, {"mode": "fast", "smoke": True})
    for fn in ("profiles_val.npz", "profiles_test.npz"):
        assert os.path.exists(os.path.join(out_dir, fn))