"""GPU smoke for SAGE-TopK (run with the server venv on a CUDA node).

Covers: real-VGG16-BN profiling crossing, step telemetry, selected sites,
zero-update fraction, per-part timers, checkpoint resume, final eval, memory.
"""
import json
import os
import torch

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
                  "eval_every": epochs, "save_every": 4},
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


def main():
    assert torch.cuda.is_available(), "smoke requires a CUDA device"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    run_dir = os.path.join(ROOT, "seed13")

    c = cfg(ROOT, epochs=13, run_name="seed13")
    seed_all(int(c["train"]["seed"]))
    Trainer(c, run_dir).run()
    _report(run_dir)

    rows = [json.loads(l) for l in open(os.path.join(run_dir, "sage_topk_steps.jsonl"))]
    n_zero = sum(1 for r in rows if r["zero"])
    print(f"[smoke] allocation steps={len(rows)} zero-updates={n_zero} "
          f"({100.0 * n_zero / max(1, len(rows)):.1f}%)")
    print(f"[smoke] peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    for split in ("val", "test"):
        ev = evaluate_run(run_dir, split=split, device="cuda")
        m = ev["metrics"]
        print(f"[smoke] {split}: acc={m['acc']:.4f} aurc={m['aurc']:.4f} "
              f"excess_aurc={m['excess_aurc']:.4f}")

    run_dir2 = os.path.join(ROOT, "seed13_resume")
    c2 = cfg(ROOT, epochs=10, run_name="seed13_resume")
    seed_all(int(c2["train"]["seed"]))
    t2 = Trainer(c2, run_dir2)
    t2.run(resume_from="epoch_004")
    rows2 = [json.loads(l) for l in open(os.path.join(run_dir2, "sage_topk_steps.jsonl"))]
    print(f"[smoke] resumed epochs={sorted({r['epoch'] for r in rows2})} "
          f"steps={len(rows2)} selected={t2.method.selected_sites()}")
    print("SMOKE_OK")


def _report(run_dir):
    ep = [json.loads(l) for l in open(os.path.join(run_dir, "sage_topk.jsonl"))]
    last = ep[-1]
    print("[smoke] epoch log fields:", sorted(last.keys()))
    print(f"[smoke] last epoch row: {json.dumps(last, default=str)}")
    rows = [json.loads(l) for l in
            open(os.path.join(run_dir, "sage_topk_steps.jsonl"))]
    if rows:
        last_step = rows[-1]
        print("[smoke] step row fields:", sorted(last_step.keys()))
        print("[smoke] step sample:", json.dumps(last_step, default=str))
    util = [json.loads(l) for l in
            open(os.path.join(run_dir, "sage_topk_utility.jsonl"))]
    if util:
        u = util[-1]
        print("[smoke] utility row fields:", sorted(u.keys()))
        print("[smoke] last utility row:", json.dumps(u, default=str))


if __name__ == "__main__":
    main()