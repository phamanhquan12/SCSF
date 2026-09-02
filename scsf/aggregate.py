"""aggregate entrypoint: mean/std over seeds across registry.csv.

Grouping key: (dataset, backbone, method_name, score, recipe, split).
"""

from __future__ import annotations

import csv
import statistics

from .engine.registry import BASE_COLUMNS, load_registry

NUMERIC = [c for c in BASE_COLUMNS
           if c in {"n", "acc", "err", "aurc", "auroc_error", "aupr_error",
                    "excess_aurc", "mean_class_aurc", "worst_class_aurc"}
           or c.startswith("risk_at_cov_")]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def aggregate(path: str, out_path: str | None = None) -> list:
    rows = load_registry(path)
    groups = {}
    for r in rows:
        if r.get("complete") != "1":
            continue
        key = tuple(r.get(c, "") for c in ("dataset", "backbone", "method_name",
                                           "score", "recipe", "split"))
        groups.setdefault(key, []).append(r)
    summary = []
    for key, grp in groups.items():
        row = {c: "" for c in BASE_COLUMNS}
        row.update(zip(("dataset", "backbone", "method_name", "score", "recipe", "split"), key))
        row["runs"] = len(grp)
        for c in NUMERIC:
            vals = [_f(g[c]) for g in grp if g.get(c) not in (None, "")]
            if not vals:
                continue
            row[c] = f"{statistics.mean(vals):.6f}"
            row[f"std_{c}"] = f"{statistics.stdev(vals) if len(vals) > 1 else 0.0:.6f}"
        summary.append(row)
    if out_path:
        fields = list(BASE_COLUMNS)
        fields += ["runs"]
        fields += sorted({c for row in summary for c in row if c.startswith("std_")})
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in summary:
                w.writerow(row)
    return summary


def main(argv=None) -> None:
    import sys as _sys
    argv = list(sys.argv[1:] if argv is None else argv)
    path = argv[0] if argv else "results/registry.csv"
    out_path = argv[1] if len(argv) > 1 else path.replace(".csv", "_aggregate.csv")
    summary = aggregate(path, out_path)
    if not summary:
        print("no complete rows found")
        return
    print(f"{'design':<56} {'split':<6} {'acc':>8} {'aurc':>9} {'worst_aurc':>11} {'runs':>3}")
    for r in summary:
        design = f"{r['dataset']}-{r['backbone']}-{r['method_name']}"
        score = r.get("score")
        if score:
            design += f".{score}"
        design = f"{design}-r{r['recipe']}"[:56]
        print(f"{design:<56} {r['split']:<6} {r.get('acc', ''):>8} {r.get('aurc', ''):>9} "
              f"{r.get('worst_class_aurc', ''):>11} {r.get('runs', ''):>3}")
    print(f"aggregate written to {out_path}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])