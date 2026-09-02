"""Idempotent manifest executor for the CIFAR gate matrix.

Processes a stage TSV (see gen_manifest.py) top-to-bottom. For every run:

* if ``results/registry.csv`` already has a complete ``split=test`` row for the
  run_dir it is skipped (never re-evaluated);
* if the run directory has no ``manifest.json`` (training did not finish), it
  resumes from the highest ``epoch_NNN`` checkpoint (falling back to ``last``
  before epoch 5), or starts fresh when no checkpoint exists; the identical
  CLI args keep the config hash unchanged on resume;
* after training it evaluates ``val`` (cheap, unscores nothing) and the
  official ``test`` split exactly once.

One process only; no two jobs ever write the same output directory. Progress
is streamed to stdout (tee to a log) and appended to ``<results>/progress.tsv``
so monitoring can inspect completed runs and wall-time without touching big
artifacts.

Usage (execution host)::

    python scripts/scheduler.py --manifest results/manifests/stage_A.tsv \
        --results-root results --python /root/scsf_venv/bin/python
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


def _run(cmd, logfile):
    print(f"    cmd: {cmd}", flush=True)
    with open(logfile, "ab") as f:
        t0 = time.time()
        p = subprocess.run(cmd, check=False, stdout=f, stderr=subprocess.STDOUT)
        dt = time.time() - t0
    if p.returncode != 0:
        raise RuntimeError(f"command failed rc={p.returncode}: {cmd}")
    return dt


def scheduler(manifest: str, results_root: str, python: str, dry_run: bool = False,
              max_runs: int | None = None) -> list:
    registry_path = os.path.join(results_root, "registry.csv")
    progress_path = os.path.join(results_root, "progress.tsv")
    os.makedirs(results_root, exist_ok=True)
    rows = list(csv.DictReader(open(manifest, newline=""), delimiter="\t"))
    if max_runs:
        rows = rows[:max_runs]
    log_root = os.path.join(results_root, "logs")
    os.makedirs(log_root, exist_ok=True)
    done = 0
    for i, row in enumerate(rows):
        run_dir, args = row["run_dir"], row["args"]
        short = run_dir.replace(results_root + "/", "")
        tag = f"[{i+1}/{len(rows)}] {row['stage']} {short}"
        print(f"\n{tag}  (run_dir={run_dir})", flush=True)
        if _registry_test_complete(registry_path, run_dir):
            print("    already complete; skipping", flush=True)
            done += 1
            continue
        resume_from = None
        if not _training_finished(run_dir):
            ckpt = _highest_epoch_ckpt(run_dir)
            if ckpt:
                resume_from = ckpt
                print(f"    resume {run_dir} from {ckpt}", flush=True)
            elif os.path.exists(os.path.join(run_dir, "last.pt")):
                resume_from = "last"
                print(f"    resume {run_dir} from last", flush=True)
        t0 = time.time()
        logfile = os.path.join(log_root, short.replace("/", "__") + ".log")
        if not dry_run:
            train_cmd = [python, "-m", "scsf.train"] + args.split()
            if resume_from:
                train_cmd += ["+resume_from=" + resume_from]
            dt_tr = _run(train_cmd, logfile)
            ev_cmd = [python, "-m", "scsf.evaluate",
                      f"run_dir={run_dir}", "split=val"]
            dt_ev = _run(ev_cmd, logfile)
            ev_cmd[-1] = "split=test"
            dt_test = _run(ev_cmd, logfile)
        else:
            dt_tr = dt_ev = dt_test = 0.0
            print("    (dry-run) train+val+test", flush=True)
        dt = time.time() - t0
        if resume_from:
            resumed = resume_from
        else:
            resumed = "-"
        with open(progress_path, "a", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            if i == 0 or os.path.getsize(progress_path) == 0:
                w.writerow(["priority", "stage", "dataset", "backbone",
                            "method_name", "mode", "seed", "run_dir",
                            "resume_from", "train_s", "eval_s", "test_s",
                            "total_s", "rc_train_ok", "time"])
            w.writerow([row["priority"], row["stage"], row["dataset"],
                        row["backbone"], row["method_name"], row["mode"],
                        row["seed"], run_dir, resumed, f"{dt_tr:.1f}",
                        f"{dt_ev:.1f}", f"{dt_test:.1f}", f"{dt:.1f}",
                        "1" if dt_tr else "0",
                        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())])
        done += 1
        print(f"    done in {dt/60:.1f} min", flush=True)
    print(f"\nscheduler finished: {done}/{len(rows)} rows processed", flush=True)
    return rows


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-runs", type=int, default=None)
    a = ap.parse_args(argv)
    scheduler(a.manifest, a.results_root, a.python, dry_run=a.dry_run,
              max_runs=a.max_runs)


if __name__ == "__main__":
    main()