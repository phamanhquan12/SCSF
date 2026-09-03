"""DepthFrag warmup (warmup_epochs) correctness tests.

Warmup contract (locked protocol ``docs/protocol_depthfrag_warm25.md``):
- epochs 0..warmup_epochs-1 (warmup): the backbone receives classification CE
  gradients only. The probe and terminal fragility-head losses may only reach
  their own parameters; they are fed with detached features so no auxiliary
  gradient flows into the backbone.
- epoch warmup_epochs onward: ordinary end-to-end DepthFrag, i.e. auxiliary
  (probe/head) gradients reach the intended backbone prefixes.
- A ``warmup_epochs`` key absent is equivalent to ``warmup_epochs=0`` under
  identical initialization, RNG, batch, losses, predictions and gradients.
- Resume across the warmup boundary is exact (matches the continuous run).
- ``method_name=depthfrag_warm25`` must resolve the real YAML by that exact
  name and build a ``DepthFragMethod``.

These tests build the method directly and drive ``train_loss`` at controlled
epochs with an explicit state, so auxiliary-isolation is verified without
running full training loops.
"""

import os

import pytest
import torch
from types import SimpleNamespace

from scsf.engine.config import resolve
from scsf.engine.seeding import seed_all
from scsf.methods import build_method

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cfg(results_root, seed=0, method_name="depthfrag_warm25", epochs=5, **m):
    cfg = resolve({
        "dataset": "cifar10",
        "backbone": "resnet18",
        "method_name": method_name,
        "results_root": results_root,
        "train": {
            "device": "cpu",
            "seed": seed,
            "epochs": epochs,
            "overfit": 32,
            "batch_size": 16,
            "lr": 0.01,
            "scheduler": "cosine",
            "weight_decay": 0.0,
            "eval_every": 1,
            "save_every": 1,
            "data_order_seed": seed,
        },
        "data": {"num_workers": 0},
    })
    cfg["method"].update(m)
    return cfg


def _method(results_root, seed=0, method_name="depthfrag_warm25", **m):
    return build_method(method_name, _cfg(results_root, seed, method_name, **m))


def _batch(method, n=6):
    x = torch.randn(n, method.backbone.channels,
                    method.backbone.input_size, method.backbone.input_size)
    y = torch.randint(0, method.num_classes, (n,))
    return (x, y, torch.arange(n))


def _aux_loss(loss_dict):
    """Sum only the DepthFrag auxiliary losses (never CE)."""
    keys = [k for k in ("depthfrag_probe", "depthfrag_head") if k in loss_dict]
    assert keys, f"no auxiliary losses present, got {sorted(loss_dict)}"
    return sum(loss_dict[k] for k in keys)


def _backbone_reached(method, state, losses):
    """Backward only the given losses and report which backbone params moved."""
    for p in method.parameters():
        p.grad = None
    total = sum(v for v in losses.values() if torch.is_tensor(v) and v.requires_grad)
    total.backward()
    reached = [n for n, p in method.backbone.named_parameters()
               if p.grad is not None and bool(torch.any(p.grad != 0))]
    return reached


# ---------------------------------------------------------------------------
# 1. method_name=depthfrag_warm25 resolves the real YAML and builds
# ---------------------------------------------------------------------------
def test_depthfrag_warm25_alias_resolves_and_builds(tmp_path):
    method_name = "depthfrag_warm25"
    cfg = _cfg(str(tmp_path), method_name=method_name)
    # sanity: the resolved method doc came from the real YAML
    assert cfg["method"]["warmup_epochs"] == 25
    assert cfg["method"]["score"] == "depthfrag"
    m = build_method(method_name, cfg)
    from scsf.methods.depthfrag import DepthFragMethod
    assert isinstance(m, DepthFragMethod)
    assert m.warmup_epochs == 25


# ---------------------------------------------------------------------------
# 2. warmup (epoch 24 < 25): aux grads never reach backbone, probe/head do
# ---------------------------------------------------------------------------
def test_warmup_epoch24_aux_isolated_from_backbone(tmp_path):
    m = _method(str(tmp_path), warmup_epochs=25)
    m.train()
    state = SimpleNamespace(batch_index=0, epoch=24)
    loss_dict = m.train_loss(_batch(m), state)
    assert {"ce", "depthfrag_probe", "depthfrag_head"} <= set(loss_dict)

    # backward ONLY the auxiliary losses
    _aux_loss(loss_dict).backward()
    reached = [n for n, p in m.backbone.named_parameters()
               if p.grad is not None and bool(torch.any(p.grad != 0))]
    assert reached == [], f"aux grads leaked into backbone during warmup: {reached}"

    # probe/head parameters did receive gradients
    probe_grad = sum(
        p.grad.abs().sum() for pr in m.probes.values() for p in pr.parameters()
        if p.grad is not None
    )
    head_grad = sum(p.grad.abs().sum() for p in m.head.parameters()
                    if p.grad is not None)
    assert probe_grad > 0, "no probe gradient during warmup"
    assert head_grad > 0, "no head gradient during warmup"


# ---------------------------------------------------------------------------
# 3. warmup (epoch 24 < 25): CE alone reaches the backbone
# ---------------------------------------------------------------------------
def test_warmup_epoch24_ce_reaches_backbone(tmp_path):
    m = _method(str(tmp_path), warmup_epochs=25)
    m.train()
    state = SimpleNamespace(batch_index=0, epoch=24)
    loss_dict = m.train_loss(_batch(m), state)
    _backbone_reached(m, state, {"ce": loss_dict["ce"]})
    # CE alone must reach the backbone during warmup
    assert any(p.grad is not None and bool(torch.any(p.grad != 0))
               for p in m.backbone.parameters()), "CE did not reach backbone in warmup"


# ---------------------------------------------------------------------------
# 4. post-warmup (epoch 25 == 25): aux grads reach the backbone
# ---------------------------------------------------------------------------
def test_post_warmup_epoch25_aux_reaches_backbone(tmp_path):
    m = _method(str(tmp_path), warmup_epochs=25)
    m.train()
    state = SimpleNamespace(batch_index=0, epoch=25)
    loss_dict = m.train_loss(_batch(m), state)
    assert {"depthfrag_probe", "depthfrag_head"} <= set(loss_dict)

    # backward ONLY the auxiliary losses
    reached = _backbone_reached(m, state, {"aux": _aux_loss(loss_dict)})
    assert reached, "post-warmup aux grads should reach the backbone"


# ---------------------------------------------------------------------------
# 5. absent warmup_epochs == explicit warmup_epochs=0 (same init/RNG/batch)
# ---------------------------------------------------------------------------
def test_absent_warmup_key_equals_warmup_zero(tmp_path):
    def _stats(method_name, warmup_override):
        seed_all(13)
        m = build_method(method_name, _cfg(str(tmp_path), seed=13,
                                           method_name=method_name, **warmup_override))
        m.train()
        # identical RNG draw for the same synthetic batch in both runs
        torch.manual_seed(13)
        batch = _batch(m)
        state = SimpleNamespace(batch_index=0, epoch=25)
        loss_dict = m.train_loss(batch, state)
        preds = [float(v.item()) if torch.is_tensor(v) else float(v)
                 for v in loss_dict.values()]
        grads = {}
        for p in m.parameters():
            p.grad = None
        sum(v for v in loss_dict.values()
            if torch.is_tensor(v) and v.requires_grad).backward()
        grads = {n: p.grad.clone() for n, p in m.named_parameters()
                 if p.grad is not None}
        logits = m.predict_batch(batch[0])
        return preds, grads, logits.confidence

    p_no, g_no, conf_no = _stats("depthfrag", {})
    p_ex, g_ex, conf_ex = _stats("depthfrag", {"warmup_epochs": 0})

    assert p_no == pytest.approx(p_ex, abs=1e-12)
    assert set(g_no) == set(g_ex)
    for n in g_no:
        assert torch.equal(g_no[n], g_ex[n]), f"grad mismatch on {n}"
    assert torch.equal(conf_no, conf_ex)

# ---------------------------------------------------------------------------
# 6. resume across the warmup boundary is exact (matches continuous run)
# ---------------------------------------------------------------------------
def test_warmup_resume_across_boundary_exact(tmp_path, monkeypatch):
    """Resume must reproduce a continuous run across the warmup->e2e boundary.

    Uses the proven test_trainer_resume.py pattern: an interrupted run resumes
    from the on-disk boundary checkpoint (end of warmup) and must match the
    continuous run bit-for-bit on the post-boundary epochs. The boundary is set
    small (warmup_epochs=2, interrupt at epoch 3) to keep the test fast.
    """
    from scsf.engine.trainer import Trainer

    class _StopAfterEpochThree(Exception):
        pass

    class _RecordingTrainer(Trainer):
        def __init__(self, cfg, run_dir, record):
            super().__init__(cfg, run_dir)
            self._record = record

        def _eval_val(self) -> dict:
            mcur = super()._eval_val()
            self._record[int(self.epoch)] = {
                "acc": float(mcur["acc"]),
                "aurc": float(mcur["aurc"]),
            }
            return mcur

    seed = 13
    # warmup_epochs=2 => epochs 0,1 warmup; epoch 2 is the first e2e epoch.
    train_cfg = _cfg(str(tmp_path), seed=seed, warmup_epochs=2, epochs=5)

    # continuous run
    cont_rec = {}
    cont_run = os.path.join(tmp_path, "cont", "warm-resume")
    cfg_a = dict(train_cfg)
    cfg_a["results_root"] = str(os.path.join(tmp_path, "cont"))
    cfg_a["run_name"] = "warm-resume"
    seed_all(seed)
    ta = _RecordingTrainer(cfg_a, cont_run, cont_rec)
    ta.run()
    assert sorted(cont_rec) == [0, 1, 2, 3, 4]

    # interrupted run: record through the boundary epoch 2, stop before epoch 3
    part_rec1, part_rec2 = {}, {}
    part_run = os.path.join(tmp_path, "part", "warm-resume")
    cfg_b = dict(train_cfg)
    cfg_b["results_root"] = str(os.path.join(tmp_path, "part"))
    cfg_b["run_name"] = "warm-resume"
    seed_all(seed)
    tb = _RecordingTrainer(cfg_b, part_run, part_rec1)
    tb._build()
    orig_start = tb.method.on_epoch_start

    def _stop_before_epoch_3(epoch, _orig=orig_start):
        if epoch >= 3:
            raise _StopAfterEpochThree()
        return _orig(epoch)

    tb.method.on_epoch_start = _stop_before_epoch_3
    with pytest.raises(_StopAfterEpochThree):
        tb.run()
    assert sorted(part_rec1) == [0, 1, 2]

    # resume from the boundary snapshot saved at the end of epoch 2
    seed_all(seed)
    tb2 = _RecordingTrainer(cfg_b, part_run, part_rec2)
    out2 = tb2.run(resume_from="epoch_002")
    full_rec = {**part_rec1, **part_rec2}
    assert sorted(full_rec) == [0, 1, 2, 3, 4]
    for ep in (3, 4):
        assert cont_rec[ep]["acc"] == pytest.approx(full_rec[ep]["acc"], abs=1e-9)
        assert cont_rec[ep]["aurc"] == pytest.approx(full_rec[ep]["aurc"], abs=1e-9)
    assert out2["selection"]["selected_epoch"] is not None
    monkeypatch.delenv("SCSF_DATA_ROOT", raising=False)
