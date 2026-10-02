from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

from .metrics import CoverageResult, MetricSummary


def write_dict_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, summary: MetricSummary, extra: dict | None = None):
    row = asdict(summary)
    if extra:
        row.update(extra)
    write_dict_csv(path, [row])


def coverage_rows(curve: list[CoverageResult], extra: dict | None = None) -> list[dict]:
    rows = []
    for result in curve:
        row = {
            "coverage": result.coverage,
            "risk": result.error / 100.0,
            "error": result.error,
            "accuracy": result.accuracy,
            "selected": result.selected,
        }
        if extra:
            row.update(extra)
        rows.append(row)
    return rows


def write_plots(
    risk_curve_csv: Path,
    roc_curve_csv: Path,
    figure_dir: Path,
    dataset: str,
    method: str,
    variant: str,
    checkpoint: str,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    figure_dir.mkdir(parents=True, exist_ok=True)
    label = f"{method}/{variant}".replace("_", " ")

    risk_rows = _read_csv(risk_curve_csv)
    if risk_rows:
        fig, ax = plt.subplots(figsize=(5.0, 3.6))
        ax.plot([float(row["coverage"]) for row in risk_rows], [float(row["risk"]) for row in risk_rows], marker="o", label=label)
        ax.set_xlabel("Coverage (%)")
        ax.set_ylabel("Risk")
        ax.set_title(f"{dataset} risk-coverage")
        ax.invert_xaxis()
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(figure_dir / f"risk_coverage_{method}_{variant}_{checkpoint}.{ext}", dpi=200)
        plt.close(fig)

    roc_rows = _read_csv(roc_curve_csv)
    if roc_rows:
        fig, ax = plt.subplots(figsize=(5.0, 3.6))
        ax.plot([float(row["fpr"]) for row in roc_rows], [float(row["tpr"]) for row in roc_rows], label=label)
        ax.plot([0, 1], [0, 1], color="0.65", linestyle="--", linewidth=1)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(f"{dataset} ROC")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(figure_dir / f"roc_{method}_{variant}_{checkpoint}.{ext}", dpi=200)
        plt.close(fig)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))
