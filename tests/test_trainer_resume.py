"""Exact-resume regression for the Trainer (AGENTS.md required check).

A continuous run and a run that is interrupted and resumes from ``last`` must
produce the same per-epoch validation metrics for the epochs after the resume
point. The test pins ``num_workers=0`` so augmentation noise cannot leak in from
worker processes; the persistent generator, torch/numpy/python RNGs and the
SelectionTracker state are what make the equivalence exact.
"""

import copy
import os

import pytest

from scsf.engine.config import resolve
from scsf.engine.seeding import seed_all
from scsf.engine.trainer import Trainer
from scsf.engine.checkpoint import SelectionTracker


class _StopAfterEpochOne(Exception):
    """Raised the instant a run tries to start epoch 2 (clean interruption)."""


class _RecordingTrainer(Trainer):
    """Trainer that snapshots the per-epoch val metrics."""

    def __init__(self, cfg, run_dir, record):
        super().__init__(cfg, run_dir)
        self._record = record

    def _eval_val(self) -> dict:
        m = super()._eval_val()
        self._record[int(self.epoch)] = {
            "acc": float(m["acc"]),
            "aurc": float(m["aurc"]),
        }
        return m


def _cfg(tmp_path, seed):
    return resolve({
        "dataset": "cifar10",
        "backbone": "resnet18",
        "method_name": "ce",
        "recipe": "singlerun",
        "results_root": str(tmp_path),
        "train": {
            "seed": seed,
            "device": "cpu",
            "epochs": 4,
            "overfit": 64,
            "batch_size": 32,
            "lr": 0.05,
            "scheduler": "cosine",
            "weight_decay": 0.0,
            "eval_every": 1,
            "save_every": 1,
            "data_order_seed": seed,
        },
        "data": {"num_workers": 0},
    })


def test_trainer_resume_reproduces_post_resume_epochs(tmp_path):
    seed = 13
    seed_all(seed)

    cont_rec = {}
    run_a = os.path.join(tmp_path, "cont")
    cfg_a = _cfg(tmp_path, seed)
    cfg_a["results_root"] = str(run_a)
    cfg_a["run_name"] = "cont-run"
    Trainer = _RecordingTrainer
    contra = Trainer(cfg_a, os.path.join(run_a, "cont-run"), cont_rec)
    contra.run()
    assert sorted(cont_rec.keys()) == [0, 1, 2, 3]

    part = os.path.join(tmp_path, "part")
    cfg_b = _cfg(tmp_path, seed)
    cfg_b["results_root"] = str(part)
    cfg_b["run_name"] = "part-run"
    # Interrupt a run of the IDENTICAL full config (same config hash, same
    # scheduler schedule) after epoch 1 completes, so its on-disk epoch_001
    # checkpoint is bit-comparable with the continuous run's.
    rec_p1, rec_p2 = {}, {}
    t1 = Trainer(cfg_b, os.path.join(part, "part-run"), rec_p1)
    t1._build()
    orig_start = t1.method.on_epoch_start

    def _stop_at_epoch_2(epoch, _orig=orig_start):
        if epoch >= 2:
            raise _StopAfterEpochOne()
        return _orig(epoch)

    t1.method.on_epoch_start = _stop_at_epoch_2
    with pytest.raises(_StopAfterEpochOne):
        t1.run()
    assert sorted(rec_p1.keys()) == [0, 1]
    # resume with the exact same config from the epoch-1 boundary save
    t2 = Trainer(cfg_b, os.path.join(part, "part-run"), rec_p2)
    out2 = t2.run(resume_from="epoch_001")
    full_rec = {**rec_p1, **rec_p2}
    assert sorted(full_rec.keys()) == [0, 1, 2, 3]
    for ep in (2, 3):
        assert cont_rec[ep]["acc"] == pytest.approx(full_rec[ep]["acc"], abs=1e-9)
        assert cont_rec[ep]["aurc"] == pytest.approx(full_rec[ep]["aurc"], abs=1e-9)
    # selection state carried across the boundary
    assert out2["selection"]["selected_epoch"] is not None


def test_selection_tracker_state_roundtrip():
    st = SelectionTracker(guard_delta_acc=1.0)
    st.update(0, {"acc": 0.5, "aurc": 0.4})
    st.update(1, {"acc": 0.9, "aurc": 0.3})
    state = st.state()
    st2 = SelectionTracker(guard_delta_acc=1.0)
    st2.restore(state)
    assert st2.summary() == st.summary()
    assert st2.state() == st.state()


def test_gpu_map_location_resume_rng_state_is_copied_to_cpu():
    """GPU resume regression: a checkpoint loaded with map_location=cuda holds
    the generator state as a CUDA tensor; restore_global_state must move it to
    a CPU tensor before set_state on the CPU generator. Skips on CPU-only hosts.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("cuda not available")
    from scsf.engine.seeding import capture_global_state, make_generator, restore_global_state

    g = make_generator(13)
    torch.randn(1, generator=g)  # advance it past its initial state
    state = capture_global_state(generator=g)
    # emulate torch.load(map_location='cuda'): all saved state tensors move to cuda
    from scsf.engine.checkpoint import CheckpointManager
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        m = CheckpointManager(td)
        m.save("t", {"rng": state})
        payload = m.load("t", map_location="cuda")
    st = payload["rng"]
    assert st["generator"][0].is_cuda, "test fixture: generator state must be on cuda"
    g2 = make_generator(13)
    restore_global_state(st, g2)  # must not raise and must stay CPU-stateful
    assert g2.get_state().clone().equal(state["generator"][0].cpu())