"""Run the registered SAGE-TopK round-3 dynamic-K pilot.

This launcher is intentionally independent of the author's execution host. It
creates the four-row manifest under the requested results root, then delegates
training/evaluation to ``scripts/parallel_scheduler.py``.

Example::

    python scripts/run_sage_topk_dynamic.py \
        --data-root /datasets/cifar \
        --results-root /scratch/scsf/sage_topk_round3 \
        --python .venv/bin/python \
        --max-jobs 2

Use ``--dry-run`` to generate and print the manifest without starting jobs.
The protocol keeps ``data.download=false``; prepare the CIFAR data at
``--data-root`` before launching.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys


DATASETS = ("cifar10", "cifar100")
CANDIDATE_SETS = (("pool", "pool"), ("pc", "pool+conv"))


def _resolve_python(value: str) -> str:
    if os.path.sep in value or (os.path.altsep and os.path.altsep in value):
        path = Path(value).expanduser().resolve()
    else:
        found = shutil.which(value)
        if found is None:
            raise SystemExit(f"Python executable not found: {value}")
        path = Path(found).resolve()
    if not path.is_file():
        raise SystemExit(f"Python executable is not a file: {path}")
    return str(path)


def _check_no_whitespace(label: str, value: Path) -> None:
    if any(ch.isspace() for ch in str(value)):
        raise SystemExit(f"{label} must not contain whitespace: {value}")


def _source_commit(repo_root: Path) -> str:
    override = os.environ.get("SCSF_SOURCE_COMMIT")
    if override:
        return override
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except Exception:
        return "unknown"


def _rows(data_root: Path, results_root: Path, download: bool) -> list[dict[str, str]]:
    rows = []
    for dataset in DATASETS:
        for tag, candidates in CANDIDATE_SETS:
            run_name = (
                f"{dataset}-vgg16_bn-sage_topk-dynamic-{tag}-"
                "rccl_sc_reference-s13"
            )
            args = [
                f"dataset={dataset}",
                "backbone=vgg16_bn",
                "method_name=sage_topk",
                "seed=13",
                "recipe=ccl_sc_reference",
                f"results_root={results_root}",
                f"data.root={data_root}",
                "train.device=cuda",
                "train.seed=13",
                "data.num_workers=8",
                f"run_name={run_name}",
                f"method.candidates={candidates}",
                "method.k=2",
                "method.dynamic_k=true",
                "method.dk_z_crit=2.0",
                "method.dk_max_k=18",
            ]
            if download:
                args.append("data.download=true")
            rows.append({
                "priority": "0",
                "stage": "MAIN",
                "dataset": dataset,
                "backbone": "vgg16_bn",
                "method_name": "sage_topk",
                "mode": "",
                "seed": "13",
                "run_dir": str(results_root / run_name),
                "args": " ".join(args),
            })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True,
                        help="directory containing the CIFAR-10/100 files")
    parser.add_argument("--results-root", required=True,
                        help="new writable directory for runs and logs")
    parser.add_argument("--python", default=sys.executable,
                        help="Python executable with torch/torchvision installed")
    parser.add_argument("--max-jobs", type=int, default=2)
    parser.add_argument("--max-gpu-mib", type=int, default=22000)
    parser.add_argument("--download", action="store_true",
                        help="allow torchvision to download missing CIFAR files")
    parser.add_argument("--dry-run", action="store_true",
                        help="write and print the manifest without running jobs")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    source_commit = _source_commit(repo_root)
    data_root = Path(args.data_root).expanduser().resolve()
    results_root = Path(args.results_root).expanduser().resolve()
    if not data_root.is_dir():
        raise SystemExit(f"data root does not exist or is not a directory: {data_root}")
    _check_no_whitespace("data root", data_root)
    _check_no_whitespace("results root", results_root)
    if args.max_jobs < 1:
        raise SystemExit("--max-jobs must be at least 1")
    python = _resolve_python(args.python)

    manifest = results_root / "manifests" / "sage_topk_round3_dynamic.tsv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    rows = _rows(data_root, results_root, args.download)
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["priority", "stage", "dataset", "backbone",
                        "method_name", "mode", "seed", "run_dir", "args"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    command = [
        python,
        str(repo_root / "scripts" / "parallel_scheduler.py"),
        "--manifest", str(manifest),
        "--results-root", str(results_root),
        "--python", python,
        "--max-jobs", str(args.max_jobs),
        "--max-gpu-mib", str(args.max_gpu_mib),
    ]
    if args.dry_run:
        command.append("--dry-run")
    print(f"source commit: {source_commit}")
    print(f"manifest: {manifest}")
    for row in rows:
        print(f"  {row['run_dir']}")
    if args.dry_run:
        print("scheduler command:")
        print("  " + " ".join(command))
        return 0

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    env["SCSF_SOURCE_COMMIT"] = source_commit
    subprocess.run([python, "-c", "import torch; assert torch.cuda.is_available()"],
                   check=True, env=env)
    return subprocess.run(command, cwd=repo_root, check=False, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
