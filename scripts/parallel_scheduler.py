"""Parallel GPU scheduler for the CIFAR gate matrix.

Runs up to ``--max-jobs`` training+eval jobs concurrently on one GPU. Each job
is a plain background subprocess wrapping a small driver (this module's
``_job_main``) that runs train + val-eval + test-eval and prints one JSON line
on exit. Completed runs (complete=1 in registry.csv) are never restarted. A
singleton lock prevents duplicate scheduler instances.

Usage::

    python scripts/parallel_scheduler.py \\
        --manifest results/manifests/gate.tsv \\
        --results-root results \\
        --python /root/scsf_venv/bin/python \\
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


def _fit_resume_ckpt(run_dir: str) -> str | None:
    """Find the resume checkpoint, dropping any corrupt/truncated files.

    A corrupt checkpoint is deleted (it can never be validly resumed) and we
    fall back to an earlier one, or to a fresh start.
    """
    import torch  # noqa: PLC0415  (only needed inside job subprocess)
    ckpt = _highest_epoch_ckpt(run_dir)
    if ckpt is None:
        return None
    path = os.path.join(run_dir, ckpt + ".pt")
    try:
        torch.load(path, map_location="cpu", weights_only=False)
        return ckpt
    except Exception:
        os.remove(path)
        print(f"  [scheduler] dropped corrupt checkpoint {path}", flush=True)
        return _fit_resume_ckpt(run_dir)


def _registry_test_complete(registry_path: str, run_dir: str) -> bool:
    for r in load_registry(registry_path):
        if (r.get("run_dir") == run_dir and r.get("split") == "test"
                and r.get("complete") == "1"):
            return True
    return False


def _gpu_used_mib() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=5, text=True,
        ).strip()
        return int(out.split("\n")[0])
    except Exception:
        return 0


def _job_main() -> int:
    """Entry point for one job subprocess (reads args from stdin as JSON)."""
    spec = json.load(sys.stdin)
    row = spec["row"]
    results_root = spec["results_root"]
    python = spec["python"]
    logfile = spec["logfile"]
    run_dir = row["run_dir"]

    result = {"run_dir": run_dir, "status": "OK", "train_s": 0.0,
              "eval_s": 0.0, "test_s": 0.0, "total_s": 0.0}
    resumed = "-"
    exit_code = 0
    t0 = time.time()
    log = open(logfile, "ab")
    try:
        resume_from = None
        if not _training_finished(run_dir):
            ckpt = _fit_resume_ckpt(run_dir)
            if ckpt:
                resume_from = ckpt
        resumed = resume_from or "-"

        train_cmd = [python, "-m", "scsf.train"] + row["args"].split()
        if resume_from:
            train_cmd += ["+resume_from=" + resume_from]
        p = subprocess.run(train_cmd, check=False, stdout=log, stderr=subprocess.STDOUT)
        if p.returncode != 0:
            result["status"] = "TRAIN_FAIL"
            result["total_s"] = time.time() - t0
            exit_code = 1
            sys.stdout.write(json.dumps({"result": result, "ok": False,
                                         "resumed": resumed}))
            return exit_code
        result["train_s"] = time.time() - t0

        for split in ("val", "test"):
            t_ev = time.time()
            ev_cmd = [python, "-m", "scsf.evaluate",
                      f"run_dir={run_dir}", f"split={split}"]
            p = subprocess.run(ev_cmd, check=False, stdout=log, stderr=subprocess.STDOUT)
            if p.returncode != 0:
                result["status"] = f"EVAL_FAIL_{split}"
                result["total_s"] = time.time() - t0
                exit_code = 1
                sys.stdout.write(json.dumps({"result": result, "ok": False,
                                             "resumed": resumed}))
                return exit_code
            if split == "val":
                result["eval_s"] = time.time() - t_ev
            else:
                result["test_s"] = time.time() - t_ev

        result["total_s"] = time.time() - t0
        sys.stdout.write(json.dumps({"result": result, "ok": True, "resumed": resumed}))
        return exit_code
    except Exception as exc:
        result["status"] = "CRASHED"
        result["total_s"] = time.time() - t0
        result["error"] = repr(exc)
        sys.stdout.write(json.dumps({"result": result, "ok": False, "resumed": resumed}))
        return 1
    finally:
        log.close()


def parallel_scheduler(manifest: str, results_root: str, python: str,
                       max_jobs: int = 4, max_gpu_mib: int = 22000,
                       dry_run: bool = False) -> list:
    registry_path = os.path.join(results_root, "registry.csv")
    progress_path = os.path.join(results_root, "progress.tsv")
    os.makedirs(results_root, exist_ok=True)
    log_root = os.path.join(results_root, "logs")
    os.makedirs(log_root, exist_ok=True)

    lock_path = os.path.join(results_root, ".scheduler.lock")
    if os.path.exists(lock_path):
        try:
            old_pid = int(open(lock_path).read().strip())
            os.kill(old_pid, 0)
            print(f"ERROR: another scheduler (pid {old_pid}) is running. "
                  f"Remove {lock_path} if stale.", flush=True)
            sys.exit(1)
        except (OSError, ValueError):
            pass
    with open(lock_path, "w") as f:
        f.write(str(os.getpid()))

    rows = list(csv.DictReader(open(manifest, newline=""), delimiter="\t"))

    if not os.path.exists(progress_path) or os.path.getsize(progress_path) == 0:
        with open(progress_path, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(["priority", "stage", "dataset", "backbone",
                        "method_name", "mode", "seed", "run_dir",
                        "resume_from", "train_s", "eval_s", "test_s",
                        "total_s", "rc_train_ok", "time", "status"])

    pending = [r for r in rows if not _registry_test_complete(registry_path, r["run_dir"])]
    skipped = len(rows) - len(pending)
    print(f"manifest: {len(rows)} rows, {skipped} already complete, "
          f"{len(pending)} pending", flush=True)

    if dry_run:
        for r in pending:
            print(f"  (dry-run) {r['run_dir']}", flush=True)
        os.unlink(lock_path)
        return rows

    this_file = os.path.abspath(__file__)
    repo_root = os.path.dirname(os.path.dirname(this_file))

    queue_idx = 0
    active = {}
    done = 0
    failed = 0
    t_start = time.time()

    def _launch(row):
        run_dir = row["run_dir"]
        short = run_dir.replace(results_root + "/", "")
        logfile = os.path.join(log_root, short.replace("/", "__") + ".log")
        spec = {"row": row, "results_root": results_root, "python": python,
                "logfile": logfile}
        p = subprocess.Popen(
            [python, this_file, "--job-main"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True,
            cwd=repo_root,
        )
        p.stdin.write(json.dumps(spec))
        p.stdin.close()
        active[run_dir] = (p, row, logfile)

    try:
        while queue_idx < len(pending) or active:
            while len(active) < max_jobs and queue_idx < len(pending):
                row = pending[queue_idx]
                if _registry_test_complete(registry_path, row["run_dir"]):
                    queue_idx += 1
                    continue
                # GPU memory ceiling
                if _gpu_used_mib() >= max_gpu_mib and len(active) >= 1:
                    break
                _launch(row)
                queue_idx += 1

            finished = []
            for run_dir, (p, row, logfile) in list(active.items()):
                rc = p.poll()
                if rc is not None:
                    stdout = p.stdout.read() if p.stdout else ""
                    finished.append((run_dir, row, rc, stdout))
                    del active[run_dir]
            for run_dir, row, rc, stdout in finished:
                result = {"run_dir": run_dir, "status": "CRASHED", "train_s": 0.0,
                          "eval_s": 0.0, "test_s": 0.0, "total_s": 0.0}
                resumed = "-"
                try:
                    parsed = json.loads(stdout.strip().splitlines()[-1] if stdout.strip() else "{}")
                    if parsed:
                        result = parsed.get("result", result) or result
                        resumed = parsed.get("resumed", "-")
                except Exception as dexc:
                    print(f"  parse-fail {run_dir}: {dexc} stdout={stdout[-200:]}", flush=True)
                if result.get("status") == "OK":
                    done += 1
                else:
                    failed += 1
                short = run_dir.replace(results_root + "/", "")
                print(f"  [{done+failed}] {short} → {result.get('status')} "
                      f"({result.get('total_s', 0)/60:.1f} min)", flush=True)
                with open(progress_path, "a", newline="") as f:
                    w = csv.writer(f, delimiter="\t", lineterminator="\n")
                    w.writerow([row.get("priority", ""), row.get("stage", ""),
                                row.get("dataset", ""), row.get("backbone", ""),
                                row.get("method_name", ""), row.get("mode", ""),
                                row.get("seed", ""), run_dir, resumed,
                                f"{result['train_s']:.1f}", f"{result['eval_s']:.1f}",
                                f"{result['test_s']:.1f}", f"{result['total_s']:.1f}",
                                "1" if result["train_s"] else "0",
                                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                result.get("status", "?")])
            time.sleep(8 if active else 2)
    finally:
        try:
            os.unlink(lock_path)
        except OSError:
            pass

    wall = time.time() - t_start
    print(f"\nfinished: {done} OK, {failed} failed in {wall/60:.1f} min "
          f"({skipped} skipped)", flush=True)
    return rows


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--job-main" in argv:
        sys.exit(_job_main())
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--max-jobs", type=int, default=4)
    ap.add_argument("--max-gpu-mib", type=int, default=22000)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--job-main", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    parallel_scheduler(a.manifest, a.results_root, a.python,
                       max_jobs=a.max_jobs, max_gpu_mib=a.max_gpu_mib,
                       dry_run=a.dry_run)


if __name__ == "__main__":
    main()
