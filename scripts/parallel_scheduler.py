"""Parallel GPU scheduler for the CIFAR gate matrix.

Runs up to ``--max-jobs`` training+eval jobs concurrently on a single GPU.
PyTorch handles CUDA memory sharing; the scheduler monitors nvidia-smi and
throttles when memory exceeds ``--max-gpu-miB``.  Completed runs
(complete=1 in registry.csv) are never restarted.

Usage::

    python scripts/parallel_scheduler.py \\
        --manifest results/manifests/gate.tsv \\
        --results-root results \\
        --python /root/scsf_venv/bin/python \\
        --max-jobs 3
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scsf.engine.registry import load_registry  # noqa: E402


def _highest_epoch_ckpt(run_dir: str) -> str | None:
    if not os.path.isdir(run_dir):
        return None
    tags = [os.path.basename(p)[:-3] for p in glob.glob(os.path.join(run_dir, "epoch_*.pt"))]
    tags = [t for t in tags if t[6:].isdigit()]
    if not tags:
        return None
    return max(tags, key=lambda t: int(t[6:]))


def _training_finished(run_dir: str) -> bool:
    return os.path.exists(os.path.join(run_dir, "manifest.json"))


def _registry_test_complete(registry_path: str, run_dir: str) -> bool:
    for r in load_registry(registry_path):
        if (r.get("run_dir") == run_dir and r.get("split") == "test"
                and r.get("complete") == "1"):
            return True
    return False


def _gpu_used_mib() -> int:
    """Return current GPU memory used in MiB, or 0 if nvidia-smi unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=5, text=True,
        ).strip()
        return int(out.split("\n")[0])
    except Exception:
        return 0


def _run_one(row: dict, results_root: str, python: str, progress_path: str,
             registry_path: str, log_root: str, max_gpu_mib: int) -> dict:
    """Run one manifest row (train + val + test).  Returns a summary dict."""
    run_dir = row["run_dir"]
    short = run_dir.replace(results_root + "/", "")
    logfile = os.path.join(log_root, short.replace("/", "__") + ".log")
    result = {"run_dir": run_dir, "status": "OK", "train_s": 0.0,
              "eval_s": 0.0, "test_s": 0.0, "total_s": 0.0, "resumed": "-"}

    # Skip if already complete
    if _registry_test_complete(registry_path, run_dir):
        result["status"] = "SKIP"
        return result

    resume_from = None
    if not _training_finished(run_dir):
        ckpt = _highest_epoch_ckpt(run_dir)
        if ckpt:
            resume_from = ckpt
        elif os.path.exists(os.path.join(run_dir, "last.pt")):
            resume_from = "last"
    result["resumed"] = resume_from or "-"

    t0 = time.time()
    try:
        # Wait for GPU memory
        while True:
            used = _gpu_used_mib()
            if used < max_gpu_mib:
                break
            time.sleep(30)

        # Train
        train_cmd = [python, "-m", "scsf.train"] + row["args"].split()
        if resume_from:
            train_cmd += ["+resume_from=" + resume_from]
        with open(logfile, "ab") as f:
            p = subprocess.run(train_cmd, check=False, stdout=f,
                               stderr=subprocess.STDOUT)
        if p.returncode != 0:
            raise RuntimeError(f"train rc={p.returncode}")
        result["train_s"] = time.time() - t0

        # Evaluate val
        t_ev = time.time()
        ev_cmd = [python, "-m", "scsf.evaluate",
                  f"run_dir={run_dir}", "split=val"]
        with open(logfile, "ab") as f:
            p = subprocess.run(ev_cmd, check=False, stdout=f,
                               stderr=subprocess.STDOUT)
        if p.returncode != 0:
            raise RuntimeError(f"eval-val rc={p.returncode}")
        result["eval_s"] = time.time() - t_ev

        # Evaluate test
        t_te = time.time()
        ev_cmd[-1] = "split=test"
        with open(logfile, "ab") as f:
            p = subprocess.run(ev_cmd, check=False, stdout=f,
                               stderr=subprocess.STDOUT)
        if p.returncode != 0:
            raise RuntimeError(f"eval-test rc={p.returncode}")
        result["test_s"] = time.time() - t_te

    except RuntimeError as exc:
        result["status"] = "FAILED"
        result["error"] = str(exc)
        with open(logfile, "a") as f:
            f.write(f"\nSCHEDULER_CAUGHT: {exc}\n")

    result["total_s"] = time.time() - t0

    # Append progress row
    with open(progress_path, "a", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow([row.get("priority", ""), row.get("stage", ""),
                    row.get("dataset", ""), row.get("backbone", ""),
                    row.get("method_name", ""), row.get("mode", ""),
                    row.get("seed", ""), run_dir, result["resumed"],
                    f"{result['train_s']:.1f}", f"{result['eval_s']:.1f}",
                    f"{result['test_s']:.1f}", f"{result['total_s']:.1f}",
                    "1" if result["train_s"] else "0",
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    result["status"]])
    return result


def parallel_scheduler(manifest: str, results_root: str, python: str,
                       max_jobs: int = 3, max_gpu_mib: int = 20000,
                       dry_run: bool = False) -> list:
    registry_path = os.path.join(results_root, "registry.csv")
    progress_path = os.path.join(results_root, "progress.tsv")
    os.makedirs(results_root, exist_ok=True)
    log_root = os.path.join(results_root, "logs")
    os.makedirs(log_root, exist_ok=True)

    # Initialize progress header if needed
    if not os.path.exists(progress_path) or os.path.getsize(progress_path) == 0:
        with open(progress_path, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(["priority", "stage", "dataset", "backbone",
                        "method_name", "mode", "seed", "run_dir",
                        "resume_from", "train_s", "eval_s", "test_s",
                        "total_s", "rc_train_ok", "time", "status"])

    rows = list(csv.DictReader(open(manifest, newline=""), delimiter="\t"))

    # Filter out already-complete runs
    pending = [r for r in rows if not _registry_test_complete(registry_path, r["run_dir"])]
    skipped = len(rows) - len(pending)
    print(f"manifest: {len(rows)} rows, {skipped} already complete, "
          f"{len(pending)} pending", flush=True)

    if dry_run:
        for r in pending:
            print(f"  (dry-run) {r['run_dir']}", flush=True)
        return rows

    done = 0
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=max_jobs) as pool:
        futures = {}
        for row in pending:
            fut = pool.submit(_run_one, row, results_root, python,
                              progress_path, registry_path, log_root, max_gpu_mib)
            futures[fut] = row

        for fut in as_completed(futures):
            row = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:
                res = {"status": "CRASHED", "error": str(exc)}
            done += 1
            short = row["run_dir"].replace(results_root + "/", "")
            status = res.get("status", "?")
            elapsed = res.get("total_s", 0)
            print(f"  [{done}/{len(pending)}] {short} → {status} "
                  f"({elapsed/60:.1f} min)", flush=True)

    wall = time.time() - t_start
    print(f"\nfinished: {done} runs in {wall/60:.1f} min "
          f"({skipped} skipped)", flush=True)
    return rows


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--max-jobs", type=int, default=3)
    ap.add_argument("--max-gpu-mib", type=int, default=20000)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    parallel_scheduler(a.manifest, a.results_root, a.python,
                       max_jobs=a.max_jobs, max_gpu_mib=a.max_gpu_mib,
                       dry_run=a.dry_run)


if __name__ == "__main__":
    main()
