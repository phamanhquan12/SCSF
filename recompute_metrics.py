#!/usr/bin/env python
"""Recompute selective metrics of finished runs from their saved per_sample.csv, without retraining.

Applies the current metric definitions (tie-aware AURC and risk@coverage, NAURC without an upper clip,
NaN for undefined AUROC / FPR@95TPR) and the same-checkpoint softmax-response columns (msp_*).
Rewrites summary.csv and risk_coverage.csv of the last and val_last evaluations, then rebuilds tables/.

    python recompute_metrics.py --run-root results/paper/medical_suite/thesis
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np

from run_paper_medical_suite import aggregate_tables
from scsf.metrics import DEFAULT_COVERAGES, risk_coverage_from_correctness, summary_from_correctness
from scsf.reporting import coverage_rows, write_dict_csv
from scsf.run_status import complete_runs

ID_FIELDS = ("dataset", "method", "arch", "variant", "seed", "checkpoint")


def recompute(eval_dir: Path) -> tuple[float, float] | None:
    per_sample_path = eval_dir / "per_sample.csv"
    summary_path = eval_dir / "summary.csv"
    if not per_sample_path.is_file() or not summary_path.is_file():
        return None
    with per_sample_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    with summary_path.open(newline="") as f:
        old = next(csv.DictReader(f))
    confidence = np.array([float(r["confidence"]) for r in rows])
    msp = np.array([float(r["softmax_response"]) for r in rows])
    correct = np.array([float(r["correct"]) for r in rows])
    extra = {key: old[key] for key in ID_FIELDS if key in old}

    summary = summary_from_correctness(confidence, correct)
    msp_summary = summary_from_correctness(msp, correct)
    row = {**asdict(summary), **extra}
    row.update({"msp_aurc": msp_summary.aurc, "msp_eaurc": msp_summary.eaurc, "msp_naurc": msp_summary.naurc, "msp_auroc": msp_summary.auroc})
    write_dict_csv(summary_path, [row])
    curve = risk_coverage_from_correctness(confidence, correct, DEFAULT_COVERAGES)
    write_dict_csv(eval_dir / "risk_coverage.csv", coverage_rows(curve, extra))
    return float(old["aurc"]), summary.aurc


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()

    changed = []
    for dataset, arch, method, variant, seed in complete_runs(args.run_root):
        seed_dir = args.run_root / "metrics" / dataset / arch / method / variant / f"seed_{seed}"
        for checkpoint in ("last", "val_last"):
            result = recompute(seed_dir / checkpoint)
            if result is not None and checkpoint == "last" and abs(result[0] - result[1]) > 1e-9:
                changed.append((dataset, method, variant, seed, *result))
    aggregate_tables(args.run_root)
    print(f"Test AURC changed on {len(changed)} runs (ties in the confidence score):")
    for dataset, method, variant, seed, old, new in changed:
        print(f"  {dataset}/{method}/{variant}/seed_{seed}: {old * 1e3:.3f} -> {new * 1e3:.3f} (x1e-3)")
    print(f"Tables rebuilt: {args.run_root / 'tables'}")


if __name__ == "__main__":
    main()
