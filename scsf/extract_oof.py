"""Extract excluded-fold teacher targets.

``python -m scsf.extract_oof teacher_run_dir=... out=... [device=cuda:0]``

Evaluates the selected teacher checkpoint on the fold it **excluded** from
training, using the eval transform (one deterministic view). Writes an OOF
JSON shard with sample IDs, predictions, correctness, logits, and hashes.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

from .data.cifar import get_split
from .data.splits import load_or_make_train_folds
from .engine.artifacts import OOF_SCHEMA, save_json
from .engine.checkpoint import CheckpointManager
from .engine.config import overrides_from_cli
from .methods import build_method


def main(argv=None) -> dict:
    ov = overrides_from_cli(argv)
    teacher_run = str(ov.pop("teacher_run_dir"))
    out = str(ov.pop("out"))
    checkpoint = str(ov.pop("checkpoint", "selected"))
    device = ov.pop("device", None)
    subset = int(ov.pop("subset", 0) or ov.pop("smoke_n", 0) or 0)
    smoke_only = bool(subset)

    cfg_path = os.path.join(teacher_run, "cfg.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    man = {}
    man_path = os.path.join(teacher_run, "manifest.json")
    if os.path.exists(man_path):
        with open(man_path) as f:
            man = json.load(f)
    exclude = cfg.get("data", {}).get("exclude_fold", cfg.get("method", {}).get("exclude_fold"))
    if exclude is None:
        raise SystemExit(f"{teacher_run} is not a fold teacher (no exclude_fold)")
    exclude = int(exclude)
    dev = torch.device(device or cfg["train"].get("device", "cpu"))
    method = build_method(cfg["method_name"], cfg)
    payload = CheckpointManager(teacher_run).load(checkpoint, map_location=dev)
    method.load_state_dict(payload["model_state"], strict=False)
    method.to(dev)
    method.eval()

    split = get_split(cfg)
    idx_dir = cfg["data"].get("split_index_dir") or cfg["data"].get("root") or "."
    folds = load_or_make_train_folds(
        idx_dir, cfg["dataset"], split.train_indices,
        n_folds=int(cfg["data"].get("n_folds", 2)),
        seed=int(cfg["data"].get("fold_seed", split.seed)),
    )
    held_out = [int(i) for i in folds["folds"][exclude]]
    if subset:
        held_out = list(split.train_indices)[: int(subset)]
        smoke_only = True
    # Build a dataloader over the held-out train IDs with eval transform.
    held_cfg = json.loads(json.dumps(cfg))
    # Temporarily pretend val indices are the held-out fold so we reuse eval transform.
    from .data.cifar import _IndexSubset, _IndexDataset, _open_train_fold, get_test_transform
    import torch.utils.data as tud

    base = _open_train_fold(cfg, "val")  # eval transform
    ds = _IndexDataset(_IndexSubset(base, held_out))
    loader = tud.DataLoader(ds, batch_size=int(cfg["train"]["batch_size"]),
                            shuffle=False, num_workers=0)

    rows = []
    with torch.no_grad():
        for batch in loader:
            x, y, idx = batch[0].to(dev), batch[1].to(dev), batch[2]
            mp = method.predict_batch(x)
            logits = mp.logits.detach().cpu().numpy()
            pred = mp.prediction.detach().cpu().numpy()
            conf = mp.confidence.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            ids = np.asarray(idx)
            for i in range(len(ids)):
                rows.append({
                    "id": int(ids[i]),
                    "label": int(y_np[i]),
                    "pred": int(pred[i]),
                    "correct": int(pred[i] == y_np[i]),
                    "confidence": float(conf[i]),
                    "logits": [float(v) for v in logits[i].tolist()],
                })
    rows.sort(key=lambda r: r["id"])
    got = [r["id"] for r in rows]
    if sorted(got) != sorted(held_out):
        missing = set(held_out) - set(got)
        extra = set(got) - set(held_out)
        raise RuntimeError(
            f"OOF extraction ID mismatch missing={len(missing)} extra={len(extra)}"
        )
    blob = {
        "schema": OOF_SCHEMA,
        "dataset": cfg["dataset"],
        "fold": exclude,
        "fold_hash": folds["hashes"][exclude],
        "fold_seed": folds["seed"],
        "teacher_run": os.path.abspath(teacher_run),
        "teacher_config_hash": man.get("config_hash", ""),
        "teacher_scientific_hash": man.get("scientific_hash", ""),
        "source_sha": man.get("commit", ""),
        "checkpoint": checkpoint,
        "checkpoint_epoch": man.get("selection", {}).get("selected_epoch"),
        "n": len(rows),
        "smoke_only": bool(subset),
        "label_hist": _hist([r["label"] for r in rows]),
        "teacher_error_rate": float(1.0 - np.mean([r["correct"] for r in rows])),
        "rows": rows,
    }
    digest = save_json(out, blob)
    print(f"extract_oof: wrote {out} n={len(rows)} fold={exclude} sha256={digest[:12]}")
    return blob


def _hist(labels):
    from collections import Counter
    c = Counter(int(x) for x in labels)
    return {str(k): int(v) for k, v in sorted(c.items())}


if __name__ == "__main__":
    main(sys.argv[1:])
