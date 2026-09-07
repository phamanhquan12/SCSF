"""GPU smoke for SAGE-TopK (run with the server venv on a CUDA node).

Three modes (SMOKE_MODE env):
  full    - train epochs=13 end-to-end in ROOT/seed13, then eval val+test.
  part    - identical cfg into ROOT/seed13_part, forcibly stopped after
            epoch 4 (so the epoch_004 snapshot survives pruning).
  replay  - resume ROOT/seed13_part from epoch_004, continue to epoch 12,
            and assert the appended allocation/epoch rows are IDENTICAL to
            the uninterrupted run in ROOT/seed13 for every epoch >= 5.

The exact-resume identity is also locked by a CPU test
(tests/test_sage_topk.py::test_exact_resume_reproduces_allocations).
"""
import json
import os
import sys
import torch
import pytest

from scsf.engine.config import resolve
from scsf.engine.trainer import Trainer
from scsf.engine.evaluator import evaluate_run
from scsf.engine.seeding import seed_all

ROOT = "/tmp/sage_topk_gpu_smoke"


def cfg(root, epochs, run_name, overrides=None):
    base = {
        "backbone": "vgg16_bn",
        "method_name": "sage_topk",
        "recipe": "ccl_sc_reference",
        "results_root": root,
        "run_name": run_name,
        "data": {"num_workers": 4},
        "train": {"device": "cuda", "seed": 13, "epochs": epochs,
                  "eval_every": 5, "save_every": 1},
    }
    if overrides:
        deep_update(base, overrides)
    return resolve(base)


def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v


def _steps(path):
    return [json.loads(l) for l in open(path)]


def _n_lines(path):
    with open(path) as f:
        return sum(1 for _ in f)


def main():
    mode = os.environ.get("SMOKE_MODE", "full")
    assert torch.cuda.is_available(), "smoke requires a CUDA device"
    print(f"[smoke] GPU: {torch.cuda.get_device_name(0)} mode={mode}")

    if mode == "full":
        run_dir = os.path.join(ROOT, "seed13")
        c = cfg(ROOT, epochs=13, run_name="seed13")
        seed_all(int(c["train"]["seed"]))
        Trainer(c, run_dir).run()
        _report(run_dir)
        rows = _steps(os.path.join(run_dir, "sage_topk_steps.jsonl"))
        n_zero = sum(1 for r in rows if r["zero"])
        print(f"[smoke] allocation steps={len(rows)} zero-updates={n_zero} "
              f"({100.0 * n_zero / max(1, len(rows)):.1f}%)")
        print(f"[smoke] peak GPU memory "
              f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        for split in ("val", "test"):
            ev = evaluate_run(run_dir, split=split, device="cuda")
            m = ev["metrics"]
            print(f"[smoke] {split}: acc={m['acc']:.4f} "
                  f"aurc={m['aurc']:.4f} excess_aurc={m['excess_aurc']:.4f}")

    elif mode == "part":
        class _StopAfterProfiling(Exception):
            pass
        run_dir = os.path.join(ROOT, "seed13_part")
        c = cfg(ROOT, epochs=13, run_name="seed13_part")
        seed_all(int(c["train"]["seed"]))
        t = Trainer(c, run_dir)
        t._build()
        orig = t.method.on_epoch_start

        def stop(epoch, _orig=orig):
            if epoch >= 5:
                raise _StopAfterProfiling()
            return _orig(epoch)

        t.method.on_epoch_start = stop
        try:
            t.run()
        except _StopAfterProfiling:
            pass
        part_sel = json.load(open(os.path.join(run_dir,
                                               "sage_topk_selection.json")))
        print(f"[smoke] part selection={part_sel['selected']} "
              f"n_measurements={part_sel['n_measurements']}")
        assert os.path.exists(os.path.join(run_dir, "epoch_004.pt"))

    elif mode == "replay":
        run_dir = os.path.join(ROOT, "seed13_part")
        n_a_steps = _n_lines(os.path.join(ROOT, "seed13",
                                          "sage_topk_steps.jsonl"))
        n_a_epochs = _n_lines(os.path.join(ROOT, "seed13",
                                           "sage_topk.jsonl"))
        n_p = _n_lines(os.path.join(run_dir, "sage_topk_steps.jsonl"))
        c = cfg(ROOT, epochs=13, run_name="seed13_part")
        seed_all(int(c["train"]["seed"]))
        t2 = Trainer(c, run_dir)
        t2.run(resume_from="epoch_004")
        full = _steps(os.path.join(ROOT, "seed13", "sage_topk_steps.jsonl"))
        part_all = _steps(os.path.join(run_dir, "sage_topk_steps.jsonl"))
        part_new = part_all[n_p:]
        ref_a = [r for r in full if r["epoch"] >= 5]
        assert len(part_new) == len(ref_a), (len(part_new), len(ref_a))
        for a, b in zip(ref_a, part_new):
            assert a["epoch"] == b["epoch"] and a["step"] == b["step"]
            assert a["lambda"] == pytest.approx(b["lambda"], abs=1e-12)
            assert a["zero"] == b["zero"]
            assert a["target_age"] == b["target_age"]
            assert a["refresh_step"] == b["refresh_step"]
            assert a["selected"] == b["selected"]
        full_ep = _steps(os.path.join(ROOT, "seed13", "sage_topk.jsonl"))
        part_ep_all = _steps(os.path.join(run_dir, "sage_topk.jsonl"))
        ref_ep = [r for r in full_ep if r["epoch"] >= 5]
        part_ep_new = part_ep_all[-len(ref_ep):]
        assert [r["epoch"] for r in part_ep_new] == [r["epoch"] for r in ref_ep]
        for a, b in zip(ref_ep, part_ep_new):
            assert a["epoch"] == b["epoch"] and a["phase"] == b["phase"]
            assert a["lambda"] == b["lambda"]
            assert a["zero_update_frac"] == b["zero_update_frac"]
        print(f"[smoke] exact-resume replay OK: {len(ref_a)} step rows, "
              f"{len(ref_ep)} epoch rows identical vs uninterrupted run")

    print("SMOKE_OK")


def _report(run_dir):
    ep = _steps(os.path.join(run_dir, "sage_topk.jsonl"))
    last = ep[-1]
    print(f"[smoke] last epoch row: {json.dumps(last, default=str)}")
    sel = json.load(open(os.path.join(run_dir, "sage_topk_selection.json")))
    print(f"[smoke] selection: {sel['selected']} "
          f"means={ {k: round(v, 4) for k, v in sel['means'].items()} }")
    mani = json.load(open(os.path.join(run_dir, "sage_topk_manifest.json")))
    print(f"[smoke] manifest: {json.dumps(mani, default=str)}")


if __name__ == "__main__":
    main()