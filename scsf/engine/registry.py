"""Run-registry CSV: one row per (run, split) evaluation.

The registry is append-only text; ``aggregate`` reads it, groups rows by the
experimental design (dataset x backbone x method x score), and averages over
seeds. Column set is locked (tests assert it).
"""

from __future__ import annotations

import csv
import os

BASE_COLUMNS = [
    "run_dir", "dataset", "backbone", "method_name", "score", "seed", "recipe",
    "split", "style", "split_hash", "config_hash", "commit", "dirty", "n", "acc", "err",
    "aurc", "auroc_error", "aupr_error", "excess_aurc", "mean_class_aurc",
    "worst_class_aurc",
] + [f"risk_at_cov_{q}" for q in (100, 99, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50,
                                  45, 40, 35, 30, 25, 20, 15, 10, 5, 1)] + [
    "checkpoint_epoch", "selection", "params_total", "created_at", "complete",
]


def _row_template():
    return {c: "" for c in BASE_COLUMNS}


def load_registry(path: str):
    """Return list of row dicts (missing file -> [])."""
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _dedupe(rows, key):
    seen = {}
    out = []
    for r in rows:
        k = key(r)
        if k in seen:
            out[seen[k]] = r
        else:
            seen[k] = len(out)
            out.append(r)
    return out


def append_rows(path: str, rows, dedupe_by=("run_dir", "split")) -> None:
    """Append rows, replacing any prior rows with the same dedupe key."""
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    old = load_registry(path)
    keep = [r for r in old if any(tuple(r.get(c) for c in dedupe_by) != tuple(x.get(c) for c in dedupe_by) for x in rows)]
    all_rows = keep + rows
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=BASE_COLUMNS)
        w.writeheader()
        for r in all_rows:
            row = _row_template()
            row.update({k: v for k, v in r.items() if k in BASE_COLUMNS})
            w.writerow(row)