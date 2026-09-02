"""Aggregate and decide the locked CIFAR selective-classification gate.

The gate compares each proposed method with the strongest *complete matched*
baseline on one backbone. Confidence scores are method-specific; forcing all
methods through MSP would silently discard DepthFrag and RiskFlow.

Usage::

    python scripts/analyze_gate.py --results-root results \
        --backbone vgg16_bn --source-commit <frozen-training-commit>

``--source-commit`` is only a provenance fallback for execution copies that
were intentionally deployed without ``.git``. A non-empty commit already
recorded by a run always takes precedence.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np


LOCKED_SEEDS = (13, 17, 23, 29, 31)
RECIPE = "backbone_transfer"

BASELINE_SCORES = {
    "ce": "msp",
    "dg": "dg_r",
    "selectivenet": "selection",
    "sat": "sat_conf",
    "scsf_posthoc": "scsf_conf",
    "scsf_e2e": "scsf_conf",
    "ccl_sc": "msp",
}

CANDIDATE_SCORES = {
    "sage_ds": "msp",
    "depthfrag": "depthfrag",
    "riskflow": "riskflow",
}


def load_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def f(x, default="nan"):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float(default)


def canonical_method(row):
    """Disambiguate SCSF modes without changing historical registry rows."""
    method = row.get("method_name", "")
    if method != "scsf":
        return method
    style = row.get("style", "")
    if style in ("posthoc", "e2e"):
        return f"scsf_{style}"
    run_dir = row.get("run_dir", "")
    return "scsf_e2e" if "-scsf.e2e-" in run_dir else "scsf_posthoc"


def test_rows(rows):
    return [row for row in rows
            if row.get("split") == "test" and row.get("complete") == "1"]


def aggregate_by_cell(rows):
    """Map (dataset, backbone, canonical method, score, recipe) to rows."""
    agg = {}
    for row in test_rows(rows):
        key = (row.get("dataset"), row.get("backbone"), canonical_method(row),
               row.get("score", ""), row.get("recipe", ""))
        agg.setdefault(key, []).append(row)
    return agg


def summarize(cell_rows):
    def arr(key):
        return np.array([f(row.get(key)) for row in cell_rows], dtype=float)

    keys = ["acc", "err", "aurc", "auroc_error", "excess_aurc",
            "mean_class_aurc", "worst_class_aurc"]
    out = {}
    for key in keys:
        values = arr(key)
        out[key] = float(np.nanmean(values))
        out[key + "_std"] = (float(np.nanstd(values, ddof=1))
                              if len(values) > 1 else 0.0)
    out["n_seeds"] = len(cell_rows)
    out["seeds"] = sorted(int(row.get("seed")) for row in cell_rows)
    return out


def _complete_cell(agg, dataset, backbone, method, score, recipe, source_commit):
    rows = agg.get((dataset, backbone, method, score, recipe), [])
    by_seed = {}
    duplicates = []
    for row in rows:
        try:
            seed = int(row.get("seed"))
        except (TypeError, ValueError):
            continue
        if seed in by_seed:
            duplicates.append(seed)
        by_seed[seed] = row

    expected = set(LOCKED_SEEDS)
    observed = set(by_seed)
    reasons = []
    if observed != expected:
        reasons.append({"seed_set": sorted(observed),
                        "missing_seeds": sorted(expected - observed),
                        "extra_seeds": sorted(observed - expected)})
    if duplicates:
        reasons.append({"duplicate_seeds": sorted(set(duplicates))})

    commits = set()
    missing_commit = []
    for seed, row in by_seed.items():
        commit = (row.get("commit") or source_commit or "").strip()
        if not commit:
            missing_commit.append(seed)
        else:
            commits.add(commit)
    if missing_commit:
        reasons.append({"missing_commit_seeds": sorted(missing_commit)})
    if len(commits) > 1:
        reasons.append({"mixed_commits": sorted(commits)})

    if reasons:
        return None, None, reasons
    ordered = [by_seed[seed] for seed in LOCKED_SEEDS]
    return summarize(ordered), by_seed, []


def _paired_delta(candidate_by_seed, baseline_by_seed, metric):
    return np.array([
        f(candidate_by_seed[seed].get(metric)) -
        f(baseline_by_seed[seed].get(metric))
        for seed in LOCKED_SEEDS
    ], dtype=float)


def paired_bootstrap_ci(values, n_boot=10000, seed=20260902):
    """Deterministic percentile CI for paired seed differences."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return [float(lo), float(hi)]


def strongest_baseline(agg, dataset, backbone, recipe=RECIPE,
                       source_commit=None):
    """Return strongest baseline only when every required baseline is complete."""
    complete = []
    incomplete = {}
    for method, score in BASELINE_SCORES.items():
        summary, by_seed, reasons = _complete_cell(
            agg, dataset, backbone, method, score, recipe, source_commit)
        if reasons:
            incomplete[method] = reasons
        else:
            complete.append((method, score, summary, by_seed))
    if incomplete:
        return None, incomplete
    complete.sort(key=lambda item: item[2]["aurc"])
    return complete[0], {}


def gate(agg, backbone="vgg16_bn", recipe=RECIPE, source_commit=None):
    """Apply the locked one-non-ResNet-backbone passing gate."""
    out = {
        "backbone": backbone,
        "recipe": recipe,
        "locked_seeds": list(LOCKED_SEEDS),
        "decisions": {},
        "pass": {},
    }

    for dataset in ("cifar10", "cifar100"):
        baseline, incomplete = strongest_baseline(
            agg, dataset, backbone, recipe, source_commit)
        decision = {"result": "PENDING"}
        out["decisions"][dataset] = decision
        if incomplete:
            decision["result"] = "INCOMPLETE_BASELINES"
            decision["incomplete_baselines"] = incomplete
            continue

        baseline_method, baseline_score, baseline_summary, baseline_by_seed = baseline
        decision["strongest_baseline"] = {
            "method": baseline_method,
            "score": baseline_score,
            **baseline_summary,
        }
        candidates = []
        for method, score in CANDIDATE_SCORES.items():
            summary, by_seed, reasons = _complete_cell(
                agg, dataset, backbone, method, score, recipe, source_commit)
            if reasons:
                candidates.append({"method": method, "score": score,
                                   "result": "INCOMPLETE", "reasons": reasons})
                continue

            aurc_delta = _paired_delta(by_seed, baseline_by_seed, "aurc")
            acc_drop = baseline_summary["acc"] - summary["acc"]
            candidates.append({
                "method": method,
                "score": score,
                "result": "COMPLETE",
                **summary,
                "baseline_method": baseline_method,
                "baseline_score": baseline_score,
                "baseline_aurc": baseline_summary["aurc"],
                "baseline_acc": baseline_summary["acc"],
                "aurc_delta": summary["aurc"] - baseline_summary["aurc"],
                "paired_aurc_delta_by_seed": aurc_delta.tolist(),
                "paired_aurc_delta_ci95": paired_bootstrap_ci(aurc_delta),
                "aurc_improves": summary["aurc"] < baseline_summary["aurc"],
                # Registry accuracy is a fraction: 0.005 == 0.5 percentage point.
                "acc_drop": acc_drop,
                "acc_ok": acc_drop <= 0.005,
            })
        decision["candidates"] = candidates
        decision["result"] = ("COMPLETE" if all(
            candidate["result"] == "COMPLETE" for candidate in candidates
        ) else "INCOMPLETE_CANDIDATES")

    # A method passes only after both five-seed dataset cells are complete,
    # improve mean AURC, lose <=0.5 pp accuracy in either cell, and lose
    # <=0.2 pp accuracy on average over the two cells.
    for method in CANDIDATE_SCORES:
        cells = []
        for dataset in ("cifar10", "cifar100"):
            candidates = out["decisions"][dataset].get("candidates", [])
            cells.append(next((candidate for candidate in candidates
                               if candidate.get("method") == method), None))
        if not all(cell and cell.get("result") == "COMPLETE" for cell in cells):
            continue
        mean_acc_drop = float(np.mean([cell["acc_drop"] for cell in cells]))
        passed = (all(cell["aurc_improves"] for cell in cells)
                  and all(cell["acc_ok"] for cell in cells)
                  and mean_acc_drop <= 0.002)
        out["pass"][method] = {
            "passed": bool(passed),
            "mean_acc_drop": mean_acc_drop,
            "mean_acc_drop_ok": mean_acc_drop <= 0.002,
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--backbone", default="vgg16_bn")
    ap.add_argument("--recipe", default=RECIPE)
    ap.add_argument("--source-commit", default=None,
                    help="frozen commit fallback when execution copy has no .git")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    rows = load_rows(os.path.join(args.results_root, "registry.csv"))
    result = gate(aggregate_by_cell(rows), backbone=args.backbone,
                  recipe=args.recipe, source_commit=args.source_commit)
    if args.json:
        with open(args.json, "w") as fobj:
            json.dump(result, fobj, indent=2, sort_keys=True)
        print(f"wrote {args.json}")

    for dataset in ("cifar10", "cifar100"):
        decision = result["decisions"][dataset]
        print(f"\n=== {dataset} / {args.backbone} ===")
        if decision["result"] == "INCOMPLETE_BASELINES":
            missing = ", ".join(sorted(decision.get("incomplete_baselines", {})))
            print(f"  {decision['result']}: {missing}")
            continue
        baseline = decision["strongest_baseline"]
        print(f"  strongest baseline: {baseline['method']}({baseline['score']}) "
              f"AURC={baseline['aurc']:.6f} acc={baseline['acc']:.4f}")
        for candidate in decision["candidates"]:
            if candidate["result"] != "COMPLETE":
                print(f"  {candidate['method']}({candidate['score']}): INCOMPLETE")
                continue
            print(f"  {candidate['method']}({candidate['score']}): "
                  f"AURC={candidate['aurc']:.6f} "
                  f"delta={candidate['aurc_delta']:+.6f} "
                  f"acc={candidate['acc']:.4f} acc_ok={candidate['acc_ok']}")

    passed = [method for method, status in result["pass"].items()
              if status["passed"]]
    print(f"\nPASS methods: {passed}")
    return 0 if all(decision["result"] == "COMPLETE"
                    for decision in result["decisions"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
