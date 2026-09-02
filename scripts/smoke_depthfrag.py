"""DepthFrag tiny smoke: overfit train -> extract ladder -> save artifacts.

Runs the full DepthFrag path on a tiny overfit run for one backbone, then the
frozen-checkpoint extractor (raw-profile npz, metrics.json, scores CSV, and
the analytic-vs-iterative audit JSON). Designed to run on the GPU training
host; pass ``--data-root`` pointing at torchvision CIFAR-10 sources.

Example::

    python scripts/smoke_depthfrag.py --backbone resnet18 --results-root /root/scsf_scratch
    python scripts/smoke_depthfrag.py --backbone deit_s    --results-root /root/scsf_scratch
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=("resnet18", "deit_s"),
                    default="resnet18")
    ap.add_argument("--results-root", default="/tmp/scsf_smoke_dfd")
    ap.add_argument("--data-root", default=os.environ.get("SCSF_DATA_ROOT", "data"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--overfit", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--subset", type=int, default=128,
                    help="samples profiled on val")
    ap.add_argument("--test-subset", type=int, default=256)
    ap.add_argument("--iterative-subset", type=int, default=16)
    ap.add_argument("--iterative-steps", type=int, default=5)
    ap.add_argument("--train-only", action="store_true",
                    help="stop after training (phase probe, no extraction)")
    args = ap.parse_args(argv)

    os.environ.setdefault("SCSF_DATA_ROOT", args.data_root)
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda unavailable on this host")

    from scsf.train import main as train_main
    from scsf.extract_depthfg import main as extract_main

    train_main([
        "dataset=cifar10", f"backbone={args.backbone}",
        "method_name=depthfrag", "train.seed=0", "recipe=singlerun",
        f"results_root={args.results_root}",
        f"train.device={args.device}", f"train.epochs={args.epochs}",
        f"train.overfit={args.overfit}", f"train.batch_size={args.batch_size}",
        "train.weight_decay=0.0", "train.lr=%.6g" % args.lr,
        "train.scheduler=cosine", "train.eval_every=1", "train.save_every=1",
        "data.num_workers=2", "data.download=1",
    ])

    candidates = sorted(
        glob.glob(os.path.join(args.results_root,
                               f"cifar10-{args.backbone}-depthfrag-*")),
        key=os.path.getmtime)
    if not candidates:
        raise SystemExit("no depthfrag run dir created under results-root")
    run_dir = candidates[-1]
    if args.train_only:
        print("phase probe: training complete only; skipping extraction")
        return {"run_dir": run_dir}

    summary = extract_main([
        f"run_dir={run_dir}", "split=val",
        "checkpoint=last", f"device={args.device}",
        f"subset={args.subset}", f"test_subset={args.test_subset}",
        f"iterative_subset={args.iterative_subset}",
        f"iterative_steps={args.iterative_steps}", "mode=fast",
    ])
    print(json.dumps(summary, indent=2, sort_keys=True, default=float))
    return summary


if __name__ == "__main__":
    main()