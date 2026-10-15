"""Completion status of paper-layout runs, shared by the suite runner, the aggregator and run_medical_day.sh.

A run is complete when run_experiment.py has written COMPLETE_MARKER into its checkpoint directory, which it
does only after every evaluation output exists. Runs from before the marker existed count as complete when
test_curve.csv (the last file the old runner wrote) is present.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

COMPLETE_MARKER = "complete.json"
LEGACY_LAST_OUTPUT = "test_curve.csv"
REQUIRED_EVAL_FILES = ("summary.csv", "risk_coverage.csv", "roc_curve.csv", "per_sample.csv")


def checkpoint_dir(run_root: Path, dataset: str, arch: str, method: str, variant: str, seed: int | str) -> Path:
    return Path(run_root) / "checkpoints" / dataset / arch / method / variant / str(seed)


def metrics_dir(run_root: Path, dataset: str, arch: str, method: str, variant: str, seed: int | str) -> Path:
    return Path(run_root) / "metrics" / dataset / arch / method / variant / f"seed_{seed}"


def _nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def is_complete(run_root: Path, dataset: str, arch: str, method: str, variant: str, seed: int | str) -> bool:
    ckpt = checkpoint_dir(run_root, dataset, arch, method, variant, seed)
    metrics = metrics_dir(run_root, dataset, arch, method, variant, seed)
    if not _nonempty(ckpt / "last.pt"):
        return False
    if not all(_nonempty(metrics / "last" / name) for name in REQUIRED_EVAL_FILES):
        return False
    marker = ckpt / COMPLETE_MARKER
    if marker.exists():
        try:
            json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return True
    return _nonempty(ckpt / LEGACY_LAST_OUTPUT)


def complete_runs(run_root: Path):
    """Yield (dataset, arch, method, variant, seed) for every complete run under run_root/metrics."""
    for seed_dir in sorted((Path(run_root) / "metrics").glob("*/*/*/*/seed_*")):
        dataset, arch, method, variant = seed_dir.relative_to(Path(run_root) / "metrics").parts[:4]
        seed = seed_dir.name.removeprefix("seed_")
        if is_complete(run_root, dataset, arch, method, variant, seed):
            yield dataset, arch, method, variant, seed


def count_complete(run_root: Path, method: str, datasets: list[str], seeds: list[str], variants: list[str] | None = None) -> int:
    """Complete runs of `method`: (dataset, seed) pairs with any complete variant, or, when `variants`
    is given, (dataset, seed, variant) triples restricted to those variants."""
    runs = [(d, s, v) for d, _, m, v, s in complete_runs(run_root) if m == method]
    if variants:
        done = set(runs)
        return sum((d, str(s), v) in done for d in datasets for s in seeds for v in variants)
    done_pairs = {(d, s) for d, s, _ in runs}
    return sum((d, str(s)) in done_pairs for d in datasets for s in seeds)


def main():
    parser = argparse.ArgumentParser(description="Count complete runs of one method")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", required=True)
    parser.add_argument("--variants", nargs="*", default=None)
    args = parser.parse_args()
    print(count_complete(Path(args.run_root), args.method, args.datasets, args.seeds, args.variants))


if __name__ == "__main__":
    main()
