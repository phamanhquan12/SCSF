"""Run evaluator: apply the selected checkpoint to train/val/test splits.

Only ``scsf.evaluate split=test`` may open the official test set (it flips
``TEST_SPLIT_DISABLED`` around that exact construction). Every evaluation
appends a registry row keyed by (run_dir, split).
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from ..data.cifar import TEST_SPLIT_DISABLED, build_dataloader, set_test_allowed
from ..methods import build_method
from ..methods.scores import logit_margin, msp, negative_entropy, normalized_logit
from ..metrics import (
    all_metrics,
    class_metrics_at_global_coverages,
    selective_risk_at_coverages,
)
from .checkpoint import CheckpointManager
from .registry import BASE_COLUMNS, append_rows


def _registry_row(cfg, manifest, split, metrics, created_at, run_dir):
    row = {c: "" for c in BASE_COLUMNS}
    row.update(
        run_dir=run_dir,
        dataset=cfg["dataset"],
        backbone=cfg["backbone"],
        method_name=cfg["method_name"],
        score=cfg.get("method", {}).get("score", ""),
        seed=cfg.get("train", {}).get("seed", ""),
        recipe=cfg.get("recipe", ""),
        split=split,
        commit=manifest.get("commit", ""),
        dirty=manifest.get("dirty", ""),
        config_hash=manifest.get("config_hash", ""),
        n=int(metrics["n"]),
        acc=f"{float(metrics['acc']):.6f}",
        err=f"{float(metrics['err']):.6f}",
        aurc=f"{float(metrics['aurc']):.6f}",
        auroc_error=f"{float(metrics['auroc_error']):.6f}",
        aupr_error=f"{float(metrics['aupr_error']):.6f}",
        excess_aurc=f"{float(metrics['excess_aurc']):.6f}",
        mean_class_aurc=f"{float(metrics['mean_class_aurc']):.6f}",
        worst_class_aurc=f"{float(metrics['worst_class_aurc']):.6f}",
        checkpoint_epoch=manifest.get("selection", {}).get("selected_epoch", ""),
        selection=str(manifest.get("selection", {}).get("selection_rule", "")),
        params_total=manifest.get("params_total", ""),
        created_at=created_at,
        complete="1",
    )
    split_key = {"train": "train_hash", "val": "val_hash"}.get(split)
    if split_key and manifest.get("split_hashes"):
        row["split_hash"] = manifest["split_hashes"].get(split_key, "")
    else:
        row["split_hash"] = "official" if split == "test" else ""
    for q in (100, 99, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 1):
        row[f"risk_at_cov_{q}"] = f"{float(metrics.get(f'risk_at_cov_{q}', float('nan'))):.6f}"
    return row


def evaluate_run(run_dir: str, split: str = "val", checkpoint: str = "selected",
                 append: bool = True, device: str | None = None) -> dict:
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be train/val/test, got {split!r}")
    if not os.path.exists(os.path.join(run_dir, "cfg.json")):
        raise FileNotFoundError(f"not a run dir: {run_dir} (missing cfg.json)")

    with open(os.path.join(run_dir, "cfg.json")) as f:
        cfg = json.load(f)
    manifest = {}
    if os.path.exists(os.path.join(run_dir, "manifest.json")):
        with open(os.path.join(run_dir, "manifest.json")) as f:
            manifest = json.load(f)

    dev = torch.device(device or cfg["train"].get("device", "cpu"))
    manager = CheckpointManager(run_dir)
    if not manager.exists(checkpoint):
        raise FileNotFoundError(f"checkpoint {checkpoint!r} missing in {run_dir}")
    payload = manager.load(checkpoint, map_location=dev)
    cfg["train"]["device"] = str(dev)
    method = build_method(cfg["method_name"], cfg)
    method.load_state_dict(payload["model_state"])
    method.to(dev)
    method.eval()

    if split == "test" and TEST_SPLIT_DISABLED:
        set_test_allowed(True)
        try:
            return _score_split(cfg, method, run_dir, split, dev, append, manifest, manager, checkpoint)
        finally:
            set_test_allowed(False)
    return _score_split(cfg, method, run_dir, split, dev, append, manifest, manager, checkpoint)


def _score_split(cfg, method, run_dir, split, dev, append, manifest, manager, checkpoint):
    import time
    labels, preds, confs, ids = [], [], [], []
    score_bank = {}
    loader = build_dataloader(cfg, split, shuffle=False, return_indices=split != "test")
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0], batch[1]
            mp = method.predict_batch(x.to(dev))
            labels.append(np.asarray(y))
            preds.append(mp.prediction.detach().cpu().numpy())
            confs.append(mp.confidence.detach().cpu().numpy())
            logits = mp.logits
            extras = {
                "msp": msp(logits),
                "logit_margin": logit_margin(logits),
                "entropy": negative_entropy(logits),
                "normalized_logit": normalized_logit(logits),
            }
            merged = dict(extras)
            merged.update(mp.scores)
            for name, tensor in merged.items():
                score_bank.setdefault(name, []).append(
                    tensor.detach().cpu().numpy().reshape(-1)
                )
            if split != "test":
                ids.append(np.asarray(batch[2]))
    labels = np.concatenate(labels)
    id_arr = np.concatenate(ids) if ids else np.arange(len(labels))
    pred_arr = np.concatenate(preds)
    conf_arr = np.concatenate(confs)
    metrics = all_metrics(labels, pred_arr, conf_arr, id_arr, cfg["data"]["num_classes"])
    for c in selective_risk_at_coverages(labels, pred_arr, conf_arr, id_arr):
        metrics[f"risk_at_cov_{int(c['coverage'])}"] = float(c["risk"])
    metrics["class_at_global"] = class_metrics_at_global_coverages(
        labels, pred_arr, conf_arr, id_arr, cfg["data"]["num_classes"]
    )

    per_score = {}
    stacked = {}
    for name, parts in score_bank.items():
        s = np.concatenate(parts)
        stacked[name] = s
        sm = all_metrics(labels, pred_arr, s, id_arr, cfg["data"]["num_classes"])
        for c in selective_risk_at_coverages(labels, pred_arr, s, id_arr):
            sm[f"risk_at_cov_{int(c['coverage'])}"] = float(c["risk"])
        per_score[name] = sm

    err = (pred_arr != labels).astype(int)
    order = np.lexsort((id_arr, -conf_arr))
    top_err = []
    for rank, idx in enumerate(order, start=1):
        if err[idx]:
            top_err.append({"rank": int(rank), "id": int(id_arr[idx]),
                            "confidence": float(conf_arr[idx])})
            if len(top_err) >= 50:
                break

    out = {
        "split": split,
        "checkpoint": checkpoint,
        "primary_score": cfg.get("method", {}).get("score", ""),
        "metrics": metrics,
        "per_score": per_score,
        "top_confidence_errors": top_err,
        "n_ties": int(len(conf_arr) - len(np.unique(np.round(conf_arr, 6)))),
    }
    with open(os.path.join(run_dir, f"eval_{split}.json"), "w") as f:
        json.dump(out, f, indent=2, sort_keys=True, default=float)
    with open(os.path.join(run_dir, f"eval_{split}_all_scores.json"), "w") as f:
        json.dump({"split": split, "checkpoint": checkpoint, "per_score": per_score},
                  f, indent=2, sort_keys=True, default=float)
    np.savez_compressed(
        os.path.join(run_dir, f"predictions_{split}.npz"),
        ids=id_arr, labels=labels, predictions=pred_arr, confidence=conf_arr,
        **{f"score_{k}": v for k, v in stacked.items()},
    )

    if append:
        row = _registry_row(cfg, manifest, split, metrics,
                            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), run_dir)
        registry = os.path.join(cfg.get("results_root", "results"), "registry.csv")
        append_rows(registry, [row])
    return out