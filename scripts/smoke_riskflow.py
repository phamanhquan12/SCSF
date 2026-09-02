"""RiskFlow tiny smoke: overfit train -> trace export -> redundancy + plots.

Runs the full RiskFlow path on a tiny overfit run for one backbone, then
exports per-depth traces, the redundancy report, trajectory plots, and an
overhead (params / MACs / latency / memory) JSON. Designed to run on the GPU
training host; pass ``--data-root`` pointing at torchvision CIFAR-10 sources.

Example::

    python scripts/smoke_riskflow.py --backbone resnet18 --results-root /root/scsf_scratch
    python scripts/smoke_riskflow.py --backbone deit_s    --results-root /root/scsf_scratch
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
    ap.add_argument("--result-method", default="riskflow")
    ap.add_argument("--baseline-method", default="riskflow_concat")
    ap.add_argument("--results-root", default="/tmp/scsf_smoke_rf")
    ap.add_argument("--data-root", default=os.environ.get("SCSF_DATA_ROOT", "data"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--overfit", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--trace-subset", type=int, default=128)
    args = ap.parse_args(argv)

    os.environ.setdefault("SCSF_DATA_ROOT", args.data_root)
    import numpy as np
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda unavailable on this host")

    from scsf.train import main as train_main
    from scsf.engine import config
    from scsf.methods import build_method
    from scsf.data import build_dataloader
    from scsf.riskflow import (
        assign_category,
        export_trace,
        redundancy_report,
        save_trajectory_plots,
    )
    from scsf.riskflow.overhead import report_overhead, deployment_params, added_macs

    for method_name in (args.result_method, args.baseline_method, "riskflow_heads"):
        train_main([
            "dataset=cifar10", f"backbone={args.backbone}",
            f"method_name={method_name}", "train.seed=0", "recipe=singlerun",
            f"results_root={args.results_root}",
            f"train.device={args.device}", f"train.epochs={args.epochs}",
            f"train.overfit={args.overfit}", f"train.batch_size={args.batch_size}",
            "train.weight_decay=0.0", "train.lr=%.6g" % args.lr,
            "train.scheduler=cosine", "train.eval_every=1", "train.save_every=1",
            "data.num_workers=2", "data.download=1",
        ])

    # locate the two run dirs
    def _find(mid):
        cands = sorted(
            glob.glob(os.path.join(args.results_root, f"cifar10-{args.backbone}-{mid}-*")),
            key=os.path.getmtime)
        if not cands:
            raise SystemExit(f"no run dir for {mid} under {args.results_root}")
        return cands[-1]

    run_dir = _find(args.result_method)
    base_dir = _find(args.baseline_method)

    # --- load the RiskFlow result method and export trace / diagnostics
    cfg = config.resolve({
        "dataset": "cifar10", "backbone": args.backbone,
        "method_name": args.result_method, "results_root": args.results_root,
        "train": {"device": args.device, "seed": 0},
    })
    m = build_method(args.result_method, cfg)
    ckpt = torch.load(os.path.join(run_dir, "selected.pt"), weights_only=False)
    m.load_state_dict(ckpt["model_state"])
    m.to(args.device).eval()

    os.environ.setdefault("SCSF_DATA_SEED", "0")
    loader = build_dataloader(cfg, "train", shuffle=False, return_indices=True,
                              overfit=args.trace_subset, num_workers=0)
    xb, yb, ids = next(iter(loader))
    xb, yb = xb.to(args.device), yb.to(args.device)
    mp, flow = m.predict_with_trace(xb, yb)
    data = export_trace(flow)

    out_dir = os.path.join(run_dir, "riskflow_smoke")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "trace_train.npz"),
             **{k: (v if isinstance(v, np.ndarray) else np.asarray(v))
                for k, v in data.items()})

    cats = assign_category(data["hard_error"], data["final_s_hard"],
                           cfg["method"].get("cat_lo", 0.3),
                           cfg["method"].get("cat_hi", 0.7))
    plots = save_trajectory_plots(data, category_key=cats, out_dir=out_dir)

    # --- redundancy: RiskFlow innovations vs a heads-mode baseline
    heads_method = args.result_method.replace("riskflow", "riskflow_heads")
    heads_cfg = config.resolve({
        "dataset": "cifar10", "backbone": args.backbone,
        "method_name": heads_method, "results_root": args.results_root,
        "train": {"device": args.device, "seed": 0},
    })
    mh = build_method(heads_method, heads_cfg)
    h_ckpt = glob.glob(os.path.join(args.results_root,
                                    f"cifar10-{args.backbone}-{heads_method}-*",
                                    "selected.pt"))
    mh.load_state_dict(torch.load(os.path.join(
        os.path.dirname(sorted(h_ckpt, key=os.path.getmtime)[-1]),
        "selected.pt"), weights_only=False)["model_state"])
    mh.to(args.device).eval()
    with torch.no_grad():
        _, fh = mh.predict_with_trace(xb, yb)
        heads_cols = fh.s_hard.detach().cpu().numpy().T        # (N, L)
        innov_cols = flow.innov_hard.detach().cpu().numpy().T  # (N, L)
        cum_cols = flow.s_hard[1:].detach().cpu().numpy().T    # (N, L)

    red = redundancy_report(heads_cols, innov_cols, cum_cols)

    # --- overhead: RiskFlow default vs concat baseline
    mb = build_method(args.baseline_method, config.resolve({
        "dataset": "cifar10", "backbone": args.backbone,
        "method_name": args.baseline_method, "results_root": args.results_root,
        "train": {"device": args.device, "seed": 0},
    }))
    mb.to(args.device).eval()
    small = xb[: args.batch_size]
    overhead_rf = report_overhead(m, small)
    overhead_concat = report_overhead(mb, small)

    summary = {
        "backbone": args.backbone,
        "n_trace": int(data["hard_error"].shape[0]),
        "site_names": list(np.asarray(data["site_names"])),
        "categories_present": {c: int(np.sum(cats == c)) for c in
                               ("easy_correct", "ambiguous_correct",
                                "high_conf_wrong", "corrupted")},
        "trajectory_plots": plots,
        "redundancy": red,
        "overhead_riskflow": overhead_rf,
        "overhead_concat_baseline": overhead_concat,
        "overhead_ratio_params": float(
            overhead_rf["deployment_params"] / max(1, overhead_concat["deployment_params"])),
        "extra_params_abs": int(overhead_rf["deployment_params"]
                                - overhead_concat["deployment_params"]),
    }
    with open(os.path.join(out_dir, "smoke_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True, default=float)
    print(json.dumps(summary, indent=2, sort_keys=True, default=float))
    return summary


if __name__ == "__main__":
    main()
