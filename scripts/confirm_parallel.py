"""Two-phase parallel executor for the SAGE-V2 confirmation manifest.

Phase A (parallel train):  up to ``--max-jobs`` ``scsf.train`` subprocesses run
concurrently on one GPU.  Each job is an independent process with its own seed
and RNG, so results are bit-identical to serial execution (verified timings
show ~25% GPU on a 4090 — the bottleneck is per-process python/launch latency,
so several frozen-code runs overlap without competing for GPU throughput).

Phase B (serial eval):    all evaluations (``scsf.evaluate split=val`` then
``split=test``) run strictly one at a time from the parent, because
``append_rows`` rewrites ``registry.csv`` without a lock (see
``scsf/engine/registry.py``); concurrent evaluates would race.

Resume / idempotence contracts match ``scripts/scheduler.py``:
  * a run with a complete ``split=test`` registry row is never touched again;
  * a run whose directory lacks ``manifest.json`` is resumed from the highest
    intact ``epoch_NNN`` checkpoint, or started fresh;
  * no two processes ever write the same run directory.

The ML code executed per run is unchanged vs the serial scheduler; the frozen
commit ``SCSF_SOURCE_COMMIT`` and config hashes therefore stay identical.

Usage::

    env SCSF_SOURCE_COMMIT=<commit> SCSF_DATA_ROOT=/root/scsf/data \\
        python scripts/confirm_parallel.py \\
        --manifest results/manifests/confirmation_sage_v2.tsv \\
        --results-root results --python /root/scsf_venv/bin/python \\
        --max-jobs 4
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scsf.engine.registry import load_registry  # noqa: E402
from parallel_scheduler import (  # noqa: E402
    _fit_resume_ckpt,
    _gpu_used_mib,
    _registry_test_complete,
    _training_finished,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(cmd, logfile):
    with open(logfile, "ab") as f:
        p = subprocess.run(cmd, check=False, stdout=f, stderr=subprocess.STDOUT)
    return p.returncode


def _highest_epoch_ckpt(run_dir: str) -> str | None:
    if not os.path.isdir(run_dir):
        return None
    tags = [os.path.basename(p)[:-3] for p in glob.glob(os.path.join(run_dir, "epoch_*.pt"))]
    tags = [t for t in tags if t[6:].isdigit()]
    return max(tags, key=lambda t: int(t[6:])) if tags else None


def _phase_a(rows, results_root, python, max_jobs, max_gpu_mib):
    log_root = os.path.join(results_root, "logs")
    os.makedirs(log_root, exist_ok=True)

    def _worker(row):
        run_dir = row["run_dir"]
        short = run_dir.replace(results_root + "/", "")
        if _training_finished(run_dir):
            return None, row, None, short, "-"
        env = dict(os.environ)
        cmd = [python, "-m", "scsf.train"] + row["args"].split()
        resume = _fit_resume_ckpt(run_dir)
        if resume:
            cmd += ["+resume_from=" + resume]
        logfile = os.path.join(log_root, short.replace("/", "__") + ".train.log")
        pid = subprocess.Popen(cmd, env=env, cwd=REPO_ROOT,
                               stdout=open(logfile, "ab"), stderr=subprocess.STDOUT)
        return run_dir, row, pid, short, resume

    queue = [r for r in rows if not _registry_test_complete(
        os.path.join(results_root, "registry.csv"), r["run_dir"])]
    queue_idx, active, done, t0 = 0, {}, 0, time.time()

    while queue_idx < len(queue) or active:
        while len(active) < max_jobs and queue_idx < len(queue):
            row = queue[queue_idx]
            if _registry_test_complete(os.path.join(results_root, "registry.csv"), row["run_dir"]):
                queue_idx += 1
                continue
            if _gpu_used_mib() >= max_gpu_mib and len(active) >= 1:
                break
            run_dir, r, pid, short, resume = _worker(row)
            if pid is None:
                queue_idx += 1
                continue
            active[run_dir] = (pid, r, short, resume)
            print(f"  start {short}  resume={resume or '-'}  gpu_mib={_gpu_used_mib()}", flush=True)
            queue_idx += 1
        finished = [run_dir for run_dir, (pid, *_rest) in active.items() if pid.poll() is not None]
        for run_dir in finished:
            pid, r, short, resume = active.pop(run_dir)
            ok = pid.returncode == 0
            done += 1
            print(f"  [train {done}/{len(queue)}] {short if ok else short + ' FAILED rc=%d' % pid.returncode} "
                  f"({(time.time() - t0) / 60:.0f} min wall)", flush=True)
        time.sleep(5 if active else 2)


def _phase_b(rows, results_root, python):
    log_root = os.path.join(results_root, "logs")
    registry_path = os.path.join(results_root, "registry.csv")
    for i, row in enumerate(rows):
        run_dir = row["run_dir"]
        short = run_dir.replace(results_root + "/", "")
        if _registry_test_complete(registry_path, run_dir):
            print(f"  eval-skip {short} (test row already complete)", flush=True)
            continue
        if not _training_finished(run_dir):
            print(f"  eval-skip {short} (training not finished)", flush=True)
            continue
        logfile = os.path.join(log_root, short.replace("/", "__") + ".eval.log")
        for split in ("val", "test"):
            cmd = [python, "-m", "scsf.evaluate", f"run_dir={run_dir}", f"split={split}"]
            rc = _run(cmd, logfile)
            if rc != 0:
                print(f"  [eval {i + 1}/{len(rows)}] {short} split={split} FAILED rc={rc}", flush=True)
                return
        print(f"  [eval {i + 1}/{len(rows)}] {short} val+test OK", flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--max-jobs", type=int, default=4)
    ap.add_argument("--max-gpu-mib", type=int, default=22000)
    ap.add_argument("--train-only", action="store_true")
    a = ap.parse_args(argv)

    rows = list(csv.DictReader(open(a.manifest, newline=""), delimiter="\t"))
    registry_path = os.path.join(a.results_root, "registry.csv")
    pending = [r for r in rows if not _registry_test_complete(registry_path, r["run_dir"])]
    print(f"manifest: {len(rows)} rows, {len(rows) - len(pending)} already complete, "
          f"{len(pending)} pending", flush=True)

    t0 = time.time()
    if not a.train_only:
        _phase_a(pending, a.results_root, a.python, a.max_jobs, a.max_gpu_mib)
        print(f"phase A (parallel train) finished in {(time.time() - t0) / 60:.1f} min", flush=True)
        _phase_b(pending, a.results_root, a.python)
        print(f"phase B (serial eval) finished; total {(time.time() - t0) / 60:.1f} min", flush=True)
    else:
        _phase_a(pending, a.results_root, a.python, a.max_jobs, a.max_gpu_mib)
        print(f"train-only finished in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()