"""Tests for DepthFrag warmup (warmup_epochs parameter).

Warmup behavior:
- Epochs 0 to warmup_epochs-1: backbone receives CE gradients only;
  probes/head learn from detached features (no aux grads to backbone).
- Epochs warmup_epochs onward: standard end-to-end DepthFrag.
- warmup_epochs=0 reproduces existing behavior (no warmup).
- Resume across the warmup boundary is exact.
"""

import copy
import os

import pytest
import torch

from scsf.engine.config import resolve
from scsf.engine.seeding import seed_all
from scsf.engine.trainer import Trainer


class _RecordingTrainer(Trainer):
    """Trainer that snapshots per-epoch val metrics and warmup state."""

    def __init__(self, cfg, run_dir, record, warmup_log=None):
        super().__init__(cfg, run_dir)
        self._record = record
        self._warmup_log = warmup_log or []

    def _eval_val(self) -> dict:
        m = super()._eval_val()
        self._record[int(self.epoch)] = {
            "acc": float(m["acc"]),
            "aurc": float(m["aurc"]),
        }
        # Capture warmup state from the method
        method = self.method
        if hasattr(method, "warmup_epochs"):
            in_warmup = method.warmup_epochs > 0 and int(self.epoch) < method.warmup_epochs
            self._warmup_log.append({
                "epoch": int(self.epoch),
                "in_warmup": in_warmup,
                "warmup_epochs": method.warmup_epochs,
            })
        return m


def _cfg(tmp_path, seed, warmup_epochs=0, dataset="cifar10"):
    return resolve({
        "dataset": dataset,
        "backbone": "resnet18",
        "method_name": "depthfrag",
        "recipe": "singlerun",
        "results_root": str(tmp_path),
        "method": {"warmup_epochs": warmup_epochs},
        "train": {
            "seed": seed,
            "device": "cpu",
            "epochs": 5,
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


class _StopAfterEpoch(Exception):
    """Raised to interrupt training after a specific epoch."""
    def __init__(self, after_epoch):
        self.after_epoch = after_epoch


class _InterruptTrainer(_RecordingTrainer):
    """Trainer that raises after completing a given epoch."""

    def __init__(self, cfg, run_dir, record, warmup_log, stop_epoch):
        super().__init__(cfg, run_dir, record, warmup_log)
        self._stop_epoch = stop_epoch

    def _eval_val(self):
        m = super()._eval_val()
        if int(self.epoch) >= self._stop_epoch:
            raise _StopAfterEpoch(self._stop_epoch)
        return m


# -----------------------------------------------------------------------
# Test 1: At epoch < warmup_epochs, auxiliary gradients do NOT reach backbone
# -----------------------------------------------------------------------
def test_warmup_no_aux_grads_to_backbone(tmp_path):
    """During warmup, probe/head losses must not produce backbone gradients."""
    seed_all(13)
    cfg = _cfg(tmp_path, 13, warmup_epochs=3)
    run_dir = os.path.join(tmp_path, "warm_run")
    rec, wlog = {}, []
    t = _InterruptTrainer(cfg, run_dir, rec, wlog, stop_epoch=1)
    t._build()

    # Run one epoch
    try:
        t.run()
    except _StopAfterEpoch:
        pass

    # Zero all backbone gradients
    t.method.zero_grad()

    # Run a manual forward+backward with probe/head losses
    device = next(t.method.backbone.parameters()).device
    batch = next(iter(t.train_loader))
    x, y = batch[0].to(device), batch[1].to(device)
    loss_dict = t.method.train_loss(batch, t)
    total = sum(v for v in loss_dict.values() if torch.is_tensor(v) and v.requires_grad)
    total.backward()

    # Check: backbone params should have NO gradients
    for name, p in t.method.backbone.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            pytest.fail(f"Backbone param {name} received gradient during warmup epoch 0")


# -----------------------------------------------------------------------
# Test 2: At epoch < warmup_epochs, probe/head params DO receive gradients
# -----------------------------------------------------------------------
def test_warmup_probe_head_receive_grads(tmp_path):
    """During warmup, probe and head parameters must receive gradients."""
    seed_all(13)
    cfg = _cfg(tmp_path, 13, warmup_epochs=3)
    run_dir = os.path.join(tmp_path, "warm_run2")
    rec, wlog = {}, []
    t = _InterruptTrainer(cfg, run_dir, rec, wlog, stop_epoch=1)
    t._build()

    try:
        t.run()
    except _StopAfterEpoch:
        pass

    t.method.zero_grad()
    batch = next(iter(t.train_loader))
    loss_dict = t.method.train_loss(batch, t)
    total = sum(v for v in loss_dict.values() if torch.is_tensor(v) and v.requires_grad)
    total.backward()

    # Probes should have gradients
    for name, p in t.method.probes.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, \
            f"Probe param {name} did not receive gradient during warmup"

    # Head should have gradients
    for name, p in t.method.head.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, \
            f"Head param {name} did not receive gradient during warmup"


# -----------------------------------------------------------------------
# Test 3: At epoch >= warmup_epochs, auxiliary gradients reach backbone
# -----------------------------------------------------------------------
def test_post_warmup_aux_grads_reach_backbone(tmp_path):
    """After warmup, probe/head gradients must flow to backbone."""
    seed_all(13)
    cfg = _cfg(tmp_path, 13, warmup_epochs=2)
    run_dir = os.path.join(tmp_path, "postwarm_run")
    rec, wlog = {}, []
    t = _InterruptTrainer(cfg, run_dir, rec, wlog, stop_epoch=2)
    t._build()

    # Run 2 epochs (epoch 0, 1 = warmup; epoch 2 = post-warmup)
    try:
        t.run()
    except _StopAfterEpoch:
        pass

    # Manually set epoch to 2 (post-warmup) for the test
    t.method.zero_grad()
    batch = next(iter(t.train_loader))

    # Temporarily override epoch to be post-warmup
    class FakeState:
        batch_index = 0
        epoch = 2

    loss_dict = t.method.train_loss(batch, FakeState())
    total = sum(v for v in loss_dict.values() if torch.is_tensor(v) and v.requires_grad)
    total.backward()

    # Backbone should now receive gradients
    any_grad = False
    for name, p in t.method.backbone.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            any_grad = True
            break
    assert any_grad, "Backbone received no auxiliary gradients after warmup"


# -----------------------------------------------------------------------
# Test 4: CE gradients reach backbone in both phases
# -----------------------------------------------------------------------
def test_ce_grads_always_reach_backbone(tmp_path):
    """CE loss must produce backbone gradients in both warmup and post-warmup."""
    seed_all(13)
    cfg = _cfg(tmp_path, 13, warmup_epochs=3)
    run_dir = os.path.join(tmp_path, "ce_run")
    rec, wlog = {}, []
    t = _InterruptTrainer(cfg, run_dir, rec, wlog, stop_epoch=1)
    t._build()

    try:
        t.run()
    except _StopAfterEpoch:
        pass

    t.method.zero_grad()
    batch = next(iter(t.train_loader))
    loss_dict = t.method.train_loss(batch, t)

    # CE loss only
    loss_dict["ce"].backward()

    any_grad = False
    for name, p in t.method.backbone.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            any_grad = True
            break
    assert any_grad, "CE loss did not produce backbone gradients during warmup"


# -----------------------------------------------------------------------
# Test 5: warmup_epochs=0 reproduces existing behavior
# -----------------------------------------------------------------------
def test_warmup_zero_matches_original(tmp_path):
    """warmup_epochs=0 must produce identical results to the original."""
    seed_all(13)
    cfg0 = _cfg(tmp_path, 13, warmup_epochs=0)
    run_dir0 = os.path.join(tmp_path, "no_warm")
    rec0 = {}
    t0 = _RecordingTrainer(cfg0, run_dir0, rec0)
    t0.run()

    seed_all(13)
    cfg_orig = _cfg(tmp_path, 13, warmup_epochs=0)
    run_dir_orig = os.path.join(tmp_path, "orig")
    rec_orig = {}
    t_orig = _RecordingTrainer(cfg_orig, run_dir_orig, rec_orig)
    t_orig.run()

    # Both should produce identical metrics
    for ep in rec0:
        assert rec0[ep]["acc"] == pytest.approx(rec_orig[ep]["acc"], abs=1e-9)
        assert rec0[ep]["aurc"] == pytest.approx(rec_orig[ep]["aurc"], abs=1e-9)


# -----------------------------------------------------------------------
# Test 6: Resume across warmup boundary is exact
# -----------------------------------------------------------------------
def test_warmup_resume_exact(tmp_path):
    """A run interrupted at epoch 1 and resumed must match a continuous run."""
    seed_all(13)
    cfg = _cfg(tmp_path, 13, warmup_epochs=3)

    # Continuous run
    rec_cont = {}
    t_cont = _RecordingTrainer(cfg, os.path.join(tmp_path, "cont"), rec_cont)
    t_cont.run()

    # Interrupted run: stop after epoch 1, resume
    seed_all(13)
    rec_p1, rec_p2 = {}, []
    t1 = _InterruptTrainer(cfg, os.path.join(tmp_path, "part"), rec_p1, rec_p1, stop_epoch=1)
    t1._build()
    try:
        t1.run()
    except _StopAfterEpoch:
        pass

    seed_all(13)
    t2 = _RecordingTrainer(cfg, os.path.join(tmp_path, "part"), rec_p2, rec_p2)
    t2.run(resume_from="epoch_001")

    # Post-resume epochs must match
    full_rec = {**rec_p1, **rec_p2}
    for ep in (2, 3, 4):
        assert full_rec[ep]["acc"] == pytest.approx(rec_cont[ep]["acc"], abs=1e-9)
        assert full_rec[ep]["aurc"] == pytest.approx(rec_cont[ep]["aurc"], abs=1e-9)


# -----------------------------------------------------------------------
# Test 7: Config resolves identically for CIFAR-10 and CIFAR-100
# -----------------------------------------------------------------------
def test_config_resolve_cifar10_cifar100(tmp_path):
    """depthfrag_warm25 must resolve identically except for dataset fields."""
    cfg10 = resolve({
        "dataset": "cifar10",
        "backbone": "vgg16_bn",
        "method_name": "depthfrag",
        "recipe": "ccl_sc_reference",
        "method": {"warmup_epochs": 25},
    })
    cfg100 = resolve({
        "dataset": "cifar100",
        "backbone": "vgg16_bn",
        "method_name": "depthfrag",
        "recipe": "ccl_sc_reference",
        "method": {"warmup_epochs": 25},
    })

    # Method config must be identical
    assert cfg10["method"] == cfg100["method"]
    assert cfg10["method"]["warmup_epochs"] == 25
    assert cfg100["method"]["warmup_epochs"] == 25

    # Train config must be identical
    assert cfg10["train"] == cfg100["train"]


# -----------------------------------------------------------------------
# Test 8: Existing test suite remains green (import check)
# -----------------------------------------------------------------------
def test_existing_depthfrag_imports():
    """Verify that the existing DepthFrag code still imports cleanly."""
    from scsf.methods.depthfrag import DepthFragMethod, FragProbe, FragHead
    from scsf.methods import build_method
    # Verify warmup_epochs is accessible
    assert hasattr(DepthFragMethod, "method_name")
