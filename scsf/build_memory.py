"""Build a train-only neighborhood memory from a frozen CE checkpoint.

``python -m scsf.build_memory reference_run_dir=... out=... [device=cuda:0]``

Uses the eval transform (deterministic view), excludes self/group from
neighbors, and writes chunked k={8,32} class-support targets. Never allocates
an N-by-N matrix.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from .data.cifar import build_dataloader
from .engine.artifacts import MEMORY_SCHEMA
from .engine.checkpoint import CheckpointManager
from .engine.config import overrides_from_cli
from .methods import build_method
from .methods.next5_common import final_repr


def main(argv=None) -> dict:
    ov = overrides_from_cli(argv)
    ref = str(ov.pop("reference_run_dir"))
    out = str(ov.pop("out"))
    checkpoint = str(ov.pop("checkpoint", "selected"))
    device = ov.pop("device", None)
    ks = [int(x) for x in str(ov.pop("ks", "8,32")).split(",") if x]
    tau = float(ov.pop("tau", 0.1))
    eps = float(ov.pop("smooth", 1e-3))
    chunk = int(ov.pop("chunk", 512))
    subset = int(ov.pop("subset", 0) or ov.pop("smoke_n", 0) or 0)

    with open(os.path.join(ref, "cfg.json")) as f:
        cfg = json.load(f)
    man = {}
    if os.path.exists(os.path.join(ref, "manifest.json")):
        with open(os.path.join(ref, "manifest.json")) as f:
            man = json.load(f)
    if cfg.get("method_name") not in ("ce",):
        # matched CE anchor is required; refuse silent substitution
        raise SystemExit(
            f"build_memory requires a CE reference run, got {cfg.get('method_name')!r}"
        )
    dev = torch.device(device or cfg["train"].get("device", "cpu"))
    method = build_method(cfg["method_name"], cfg)
    payload = CheckpointManager(ref).load(checkpoint, map_location=dev)
    method.load_state_dict(payload["model_state"], strict=False)
    method.to(dev)
    method.eval()

    loader = build_dataloader(cfg, "train", shuffle=False, return_indices=True,
                              num_workers=0, overfit=int(subset) if subset else 0)
    feats, labels, ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            x, y, idx = batch[0].to(dev), batch[1], batch[2]
            bo = method.backbone(x)
            h = F.normalize(final_repr(bo), dim=1, p=2)
            feats.append(h.cpu())
            labels.append(np.asarray(y))
            ids.append(np.asarray(idx))
    features = torch.cat(feats, dim=0).numpy().astype(np.float32)
    labels = np.concatenate(labels).astype(np.int64)
    ids = np.concatenate(ids).astype(np.int64)
    groups = ids.copy()  # CIFAR: group = sample id
    n, d = features.shape
    c = int(cfg["data"]["num_classes"])
    if len(set(ids.tolist())) != n:
        raise RuntimeError("duplicate IDs in memory construction")

    targets = {k: np.zeros((n, c), dtype=np.float32) for k in ks}
    id_to_row = {int(i): r for r, i in enumerate(ids.tolist())}
    feat_t = torch.from_numpy(features)
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        q = feat_t[start:end]                    # (B, D)
        sim = q @ feat_t.t()                     # (B, N) cosine (already L2)
        for local, row in enumerate(range(start, end)):
            sim[local, row] = -1e9               # exclude self
            # exclude same group (identical to self on CIFAR)
        dist = 1.0 - sim
        for k in ks:
            topv, topi = torch.topk(-dist, k=min(k, n - 1), dim=1)
            # topv is -dist of neighbors
            w = torch.softmax(topv / tau, dim=1)  # (B, k)
            neigh = topi.cpu().numpy()
            ww = w.cpu().numpy()
            for bi, row in enumerate(range(start, end)):
                hist = np.zeros((c,), dtype=np.float64)
                for j, weight in zip(neigh[bi], ww[bi]):
                    hist[int(labels[int(j)])] += float(weight)
                hist = hist + (eps / c)
                hist = hist / hist.sum()
                targets[k][row] = hist.astype(np.float32)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(
        out,
        schema=MEMORY_SCHEMA,
        dataset=cfg["dataset"],
        ids=ids,
        labels=labels,
        groups=groups,
        features=features,
        target_k8=targets.get(8, targets[ks[0]]),
        target_k32=targets.get(32, targets[ks[-1]]),
        ks=np.asarray(ks),
        tau=np.asarray(tau),
        smooth=np.asarray(eps),
        reference_run=np.asarray(os.path.abspath(ref)),
        reference_config_hash=np.asarray(man.get("config_hash", "")),
        source_sha=np.asarray(man.get("commit", "")),
        checkpoint=np.asarray(checkpoint),
        n=np.asarray(n),
        dim=np.asarray(d),
        disclosure=np.asarray(
            "self excluded from retrieval; reference encoder trained on the query image"
        ),
    )
    print(f"build_memory: wrote {out} n={n} d={d} ks={ks}")
    return {"n": n, "dim": d, "out": out}


if __name__ == "__main__":
    main(sys.argv[1:])
