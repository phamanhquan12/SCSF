"""Variant-aware, non-overwriting aggregation (spec §11; commit 6).

The legacy ``scsf/aggregate.py`` groups by
``(dataset, backbone, method_name, score, recipe, split)`` — a key that cannot
see the two `sage_topk` fixed-K=2 ids that share one class
(``v2_fixedk2_pool`` vs ``v2_fixedk2_pool_conv``), nor the r3/dtr/cbr ablation
ladders that map several ids onto one implementation.  Under that key the
aggregator would silently last-writer-win across those rows.

This module is the **variant-aware** counterpart (§11):

* grouping key is a strict superset of the legacy key, adding ``variant``,
  ``comparison_signature`` and ``intended_seed_set`` — so the two TopK ids of
  §10.2 never merge and ablation rows never collapse onto the primary id;
* output paths are **non-overwriting by default** (fresh ``*_v.csv``; an
  existing target is refused unless ``--force``), mirroring the launcher's
  plan-lock contract (§10.3 #5);
* it stays torch/numpy-free (pure stdlib + the engine registry CSV), so it can
  be invoked from the launcher's torch-free plan path.

The aggregation is *skip-merge, never last-writer-overwrite*: rows sharing a
legacy key but differing in variant/signature/seed-set stay in separate
groups; a completed score artifact is never clobbered.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys

BASE_COLUMNS = [
    "dataset", "backbone", "method_name", "variant", "score", "recipe",
    "split", "comparison_signature", "intended_seed_set",
]
LEGACY_COLUMNS = [
    "dataset", "backbone", "method_name", "score", "recipe", "split",
]
NUMERIC_COLUMNS = (
    ["acc", "aurc", "excess_aurc", "worst_class_aurc", "mean_class_aurc",
     "auroc_error", "aupr_error", "err"]
    + [f"risk_at_cov_{q}" for q in (100, 99, 95, 90, 85, 80, 75, 70, 65, 60,
                                    55, 50, 45, 40, 35, 30, 25, 20, 15, 10,
                                    5, 1)]
)


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def load_registry(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _group_key(row: dict, variant_aware: bool) -> tuple:
    cols = BASE_COLUMNS if variant_aware else LEGACY_COLUMNS
    return tuple(row.get(c, "") for c in cols)


def aggregate_v(path: str, out_path: str | None = None,
                variant_aware: bool = True, force: bool = False) -> list[dict]:
    rows = load_registry(path)
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        if str(r.get("complete", "")).strip() != "1":
            continue
        key = _group_key(r, variant_aware)
        groups.setdefault(key, []).append(r)

    summary = []
    for key, grp in sorted(groups.items()):
        if variant_aware:
            cols = BASE_COLUMNS
        else:
            cols = LEGACY_COLUMNS + ["variant", "comparison_signature",
                                     "intended_seed_set"]
        row = dict(zip(cols, key))
        if not variant_aware:
            row["variant"] = row["comparison_signature"] = ""
            row["intended_seed_set"] = ""
        row["runs"] = len(grp)
        for c in NUMERIC_COLUMNS:
            vals = [_f(g.get(c)) for g in grp if g.get(c) not in (None, "")]
            if not vals:
                continue
            row[c] = f"{statistics.mean(vals):.6f}"
            row[f"std_{c}"] = (f"{statistics.stdev(vals):.6f}"
                               if len(vals) > 1 else "0.000000")
        summary.append(row)

    if out_path:
        target = out_path
        if os.path.exists(target) and not force:
            raise FileExistsError(
                f"{target} exists; pass --force to overwrite (non-overwriting "
                "contract §11)")
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        fields = (list(BASE_COLUMNS) if variant_aware
                  else list(LEGACY_COLUMNS) + ["variant",
                                               "comparison_signature",
                                               "intended_seed_set"])
        fields.append("runs")
        fields += sorted({c for r in summary for c in r if c.startswith("std_")})
        with open(target, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in summary:
                w.writerow(r)
    return summary


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        prog="python -m scsf.aggregate_v",
        description="Variant-aware, non-overwriting aggregation (§11).")
    p.add_argument("path", help="registry CSV")
    p.add_argument("--out", help="output CSV (default: <stem>_v.csv)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--legacy-key", action="store_true",
                   help="variant-unaware legacy grouping (parity)")
    a = p.parse_args(argv)
    out = a.out or (os.path.splitext(a.path)[0] + "_v.csv")
    summary = aggregate_v(a.path, out, variant_aware=not a.legacy_key,
                          force=a.force)
    print(f"aggregate_v: {len(summary)} variant-aware groups -> {out}")
    for r in summary:
        print(f"  {r.get('dataset')}-{r.get('backbone')}-"
              f"{r.get('method_name')}.{r.get('variant') or '-'}-"
              f"r{r.get('recipe')}-s{r.get('split')} runs={r.get('runs')} "
              f"aurc={r.get('aurc', '')}")


if __name__ == "__main__":
    main()
