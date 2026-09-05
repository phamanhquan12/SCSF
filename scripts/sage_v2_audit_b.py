"""SAGE-V2 CIFAR-100 per-class audit (read-only) -- seed-13, v1 vs v2.

Recomputes test-split predictions from the *selected* checkpoints of both the
frozen v1 run and the seed-13 v2 run, using the evaluator's exact load path
but writing *nothing* into the run directories.  All outputs go to --out.

Guarantees:
- no registry writes, no eval_*.json writes, no checkpoints touched;
- official-test access flipped on only for loader construction (same
  contract as ``scsf.evaluate split=test``), restored afterwards;
- no checkpoint/config selection is ever based on these test-class numbers.

Outputs (one per run + combined):
  per_class_{name}.csv    100 rows: n, n_err, acc, aurc, coverage@[50,70,90,95]
  global_{name}.csv       global metrics incl. mean/worst-class AURC
  auditB_summary.csv      worst-class identification + delta distribution

Usage::

    python scripts/sage_v2_audit_b.py <v1_run_dir> <v2_run_dir> --out <out_dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import torch

COVERAGE_GRID = [50, 70, 90, 95]


def _infer(run_dir: str) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    from scsf.data.cifar import (TEST_SPLIT_DISABLED, build_dataloader,
                                 set_test_allowed)
    from scsf.engine.checkpoint import CheckpointManager
    from scsf.methods import build_method

    with open(os.path.join(run_dir, "cfg.json")) as f:
        cfg = json.load(f)
    dev = torch.device("cpu")
    manager = CheckpointManager(run_dir)
    payload = manager.load("selected", map_location=dev)
    cfg["train"]["device"] = "cpu"
    method = build_method(cfg["method_name"], cfg)
    method.load_state_dict(payload["model_state"])
    method.to(dev)
    method.eval()

    labels = preds = confs = None
    if TEST_SPLIT_DISABLED:
        set_test_allowed(True)
    try:
        loader = build_dataloader(cfg, "test", shuffle=False, return_indices=False)
        lab, pre, cn = [], [], []
        with torch.no_grad():
            for batch in loader:
                x, y = batch[0], batch[1]
                mp = method.predict_batch(x.to(dev))
                lab.append(np.asarray(y))
                pre.append(mp.prediction.detach().cpu().numpy())
                cn.append(mp.confidence.detach().cpu().numpy())
        labels = np.concatenate(lab)
        preds = np.concatenate(pre)
        confs = np.concatenate(cn)
    finally:
        if TEST_SPLIT_DISABLED:
            set_test_allowed(False)
    if labels is None:
        raise RuntimeError(f"loader produced nothing for {run_dir}")
    return cfg, labels, preds, confs


def _class_table(labels, preds, confs, num_classes: int) -> dict:
    from scsf.metrics.selective import per_class_aurc

    pc = per_class_aurc(labels, preds, confs, None, num_classes)
    rows = {}
    for c in range(num_classes):
        m = labels == c
        n = int(m.sum())
        err = int((preds[m] != labels[m]).sum())
        cov = []
        for q in COVERAGE_GRID:
            cov.append(float((confs[m] >= q / 100.0).mean()))
        rows[c] = dict(
            class_idx=c, n=n, n_err=err,
            acc=float((labels[m] == preds[m]).mean()),
            aurc=float(pc.get(c, np.nan)),
            cov50=cov[0], cov70=cov[1], cov90=cov[2], cov95=cov[3],
        )
    return rows


def _write_csv(path: str, fieldnames: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("v1_run_dir")
    ap.add_argument("v2_run_dir")
    ap.add_argument("--out", default="/root/scsf_v2_auditB")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    from scsf.metrics import all_metrics

    results = {}
    for tag, run_dir in (("v1", a.v1_run_dir), ("v2", a.v2_run_dir)):
        if not os.path.isdir(run_dir):
            raise SystemExit(f"missing run dir: {run_dir}")
        cfg, labels, preds, confs = _infer(run_dir)
        num_classes = cfg["data"]["num_classes"]
        name = cfg["run_name"]
        results[tag] = dict(run_dir=run_dir, name=name,
                            labels=labels, preds=preds, confs=confs,
                            num_classes=num_classes)

    for tag, r in results.items():
        m = all_metrics(r["labels"], r["preds"], r["confs"], None, r["num_classes"])
        r["global"] = m
        r["classes"] = _class_table(r["labels"], r["preds"], r["confs"], r["num_classes"])

        fields = list(r["classes"][0].keys())
        _write_csv(os.path.join(a.out, f"per_class_{tag}.csv"), fields,
                   [r["classes"][c] for c in range(r["num_classes"])])
        g_fieldnames = ["run", "n", "acc", "err", "aurc", "auroc_error",
                        "aupr_error", "excess_aurc", "mean_class_aurc",
                        "worst_class_aurc", "worst_class"]
        g_worst = max(range(r["num_classes"]),
                      key=lambda c: r["classes"][c]["aurc"])
        _write_csv(os.path.join(a.out, f"global_{tag}.csv"), g_fieldnames, [dict(
            run=r["name"],
            n=r["global"]["n"], acc=r["global"]["acc"],
            err=r["global"]["err"], aurc=r["global"]["aurc"],
            auroc_error=r["global"]["auroc_error"],
            aupr_error=r["global"]["aupr_error"],
            excess_aurc=r["global"]["excess_aurc"],
            mean_class_aurc=r["global"]["mean_class_aurc"],
            worst_class_aurc=r["global"]["worst_class_aurc"],
            worst_class=g_worst)])

    v1, v2 = results["v1"], results["v2"]
    nc = v1["num_classes"]
    assert nc == v2["num_classes"]

    # ---- combined worst-class analysis ----
    v2_worst = max(range(nc), key=lambda c: v2["classes"][c]["aurc"])
    v1_worst = max(range(nc), key=lambda c: v1["classes"][c]["aurc"])
    deltas = {}
    for c in range(nc):
        a1 = v1["classes"][c]["aurc"]
        a2 = v2["classes"][c]["aurc"]
        deltas[c] = (a2 - a1) if (np.isfinite(a1) and np.isfinite(a2)) else np.nan
    dv = np.array([d for d in deltas.values() if np.isfinite(d)])

    summary = [
        dict(run="v1", name=v1["name"], acc=v1["global"]["acc"],
             aurc=v1["global"]["aurc"],
             mean_class_aurc=v1["global"]["mean_class_aurc"],
             worst_class_aurc=v1["global"]["worst_class_aurc"],
             worst_class=v1_worst, note=""),
        dict(run="v2", name=v2["name"], acc=v2["global"]["acc"],
             aurc=v2["global"]["aurc"],
             mean_class_aurc=v2["global"]["mean_class_aurc"],
             worst_class_aurc=v2["global"]["worst_class_aurc"],
             worst_class=v2_worst, note=""),
        dict(run="v2@v1-worst", name="",
             acc=float(v2["classes"][v1_worst]["acc"]),
             aurc=v2["classes"][v1_worst]["aurc"],
             mean_class_aurc=np.nan, worst_class_aurc=np.nan,
             worst_class=v1_worst,
             note=f"v2 value at v1 worst class "
                  f"(v1={v1['classes'][v1_worst]['aurc']:.4f})"),
        dict(run="v1@v2-worst", name="",
             acc=float(v1["classes"][v2_worst]["acc"]),
             aurc=v1["classes"][v2_worst]["aurc"],
             mean_class_aurc=np.nan, worst_class_aurc=np.nan,
             worst_class=v2_worst,
             note=f"v1 value at v2 worst class "
                  f"(v2={v2['classes'][v2_worst]['aurc']:.4f})"),
    ]
    summary.append(dict(run="", name="", acc=np.nan, aurc=np.nan,
                        mean_class_aurc=np.nan, worst_class_aurc=np.nan,
                        worst_class=-1,
                        note=f"delta(v2-v1) n={dv.size} "
                             f"median={np.nanmedian(dv):.4f} "
                             f"q25={np.percentile(dv,25):.4f} "
                             f"q75={np.percentile(dv,75):.4f} "
                             f"p90={np.percentile(dv,90):.4f} "
                             f"p95={np.percentile(dv,95):.4f} "
                             f"max={np.nanmax(dv):.4f} "
                             f"n_worse={(dv>0).sum()} "
                             f"n_worse_gt0.05={(dv>0.05).sum()}"))
    _write_csv(os.path.join(a.out, "auditB_summary.csv"),
               ["run", "name", "acc", "aurc", "mean_class_aurc",
                "worst_class_aurc", "worst_class", "note"], summary)

    # per-class delta CSV (compact)
    _write_csv(os.path.join(a.out, "per_class_delta.csv"),
               ["class_idx", "aurc_v1", "aurc_v2", "delta", "n", "acc_v1",
                "acc_v2"],
               [dict(class_idx=c, aurc_v1=v1["classes"][c]["aurc"],
                     aurc_v2=v2["classes"][c]["aurc"], delta=deltas[c],
                     n=v1["classes"][c]["n"], acc_v1=v1["classes"][c]["acc"],
                     acc_v2=v2["classes"][c]["acc"])
                for c in range(nc)])

    print(f"v1 {v1['name'][:44]:46} acc={v1['global']['acc']:.4f} "
          f"aurc={v1['global']['aurc']:.4f} worst={v1_worst} "
          f"({v1['global']['worst_class_aurc']:.4f})")
    print(f"v2 {v2['name'][:44]:46} acc={v2['global']['acc']:.4f} "
          f"aurc={v2['global']['aurc']:.4f} worst={v2_worst} "
          f"({v2['global']['worst_class_aurc']:.4f})")
    print(f"v2 worst class {v2_worst}: v1={v1['classes'][v2_worst]['aurc']:.4f} "
          f"v2={v2['classes'][v2_worst]['aurc']:.4f} "
          f"n_err v1={v1['classes'][v2_worst]['n_err']} "
          f"v2={v2['classes'][v2_worst]['n_err']}")
    print(f"v1 worst class {v1_worst}: v1={v1['classes'][v1_worst]['aurc']:.4f} "
          f"v2={v2['classes'][v1_worst]['aurc']:.4f}")
    print(f"delta median={np.nanmedian(dv):.4f} max={np.nanmax(dv):.4f} "
          f"n_worse={(dv>0).sum()}/{nc}")


if __name__ == "__main__":
    main()