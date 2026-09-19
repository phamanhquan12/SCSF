"""Portable plan/execute launcher.

Plan mode is the default and is torch-free (``scsf.engine.config`` +
``scsf.engine.planning``). ``--execute`` is required to spawn jobs.
A run hash is ``scientific_hash`` (includes seed, excludes host device/paths).
``comparison_signature`` groups seeds while keeping variant identity.

Usage::

    python scripts/run_experiments.py --suite next5_pilot --dry-run ...
    python scripts/run_experiments.py --suite next5_pilot --execute ...
    python scripts/run_experiments.py --status --results-root ...
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from scsf.engine.planning import (  # noqa: E402
    DEFAULT_SUITES,
    FOLD_TEACHER_EXCLUDE,
    METHOD_VARIANTS,
    SUITE_ALIASES,
    comparison_signature,
    resolve,
    run_name_for,
    scientific_hash,
)

SUITES_DIR = os.path.join(REPO_ROOT, "configs", "suites")
PLAN_DIR = os.path.join(REPO_ROOT, "plans")
SUITES = tuple(DEFAULT_SUITES.keys())
NEXT5_BACKBONES = {"vgg16_bn", "resnet18"}
NEXT5_METHODS = {
    "ce", "scsf_correctness", "sage_ds_v2", "fmfp_reference",
    "crossfit_failure", "candidate_verify", "intervention_rank",
    "neighbor_distill", "rank_sharpness",
}


def _method_ok() -> dict:
    ok = {}
    for pid, (name, var) in METHOD_VARIANTS.items():
        ok.setdefault(name, set())
        if var:
            ok[name].add(var)
    # None = method takes no variant unless a registered one is supplied
    return {k: (v or None) for k, v in ok.items()}


METHOD_ID_OK = _method_ok()


def load_suite(name: str) -> list[dict]:
    key = SUITE_ALIASES.get(name, name)
    path = os.path.join(SUITES_DIR, f"{key}.yaml")
    if not os.path.exists(path):
        raise SystemExit(f"--suite {name!r}: no {path!r} (known: {', '.join(SUITES)})")
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    rows = data.get("rows") or []
    out, seen = [], set()
    for r in rows:
        rid = r["id"]
        if rid in seen:
            raise SystemExit(f"suite {name}: duplicate row id {rid!r}")
        seen.add(rid)
        mname, variant = METHOD_VARIANTS.get(rid, (r["method_name"], r.get("variant")))
        if rid in METHOD_VARIANTS:
            mname, variant = METHOD_VARIANTS[rid]
        else:
            mname, variant = r["method_name"], r.get("variant")
            if mname not in METHOD_ID_OK:
                raise SystemExit(f"unsupported method {mname!r} (row {rid!r})")
        out.append({"id": rid, "method_name": mname, "variant": variant})
    return out


def _args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_experiments.py",
        description="Portable plan/execute launcher. Default is plan-only.",
    )
    p.add_argument("--suite", default=None, help="suite name (required unless --status/--list-*)")
    p.add_argument("--methods", nargs="*", default=None)
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--backbones", nargs="*", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--recipe", default=None)
    p.add_argument("--data-root", default="data")
    p.add_argument("--results-root", default="results")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--devices", nargs="*", default=None)
    p.add_argument("--max-jobs", type=int, default=1,
                   help="max concurrent training jobs (default 1; one GPU each)")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--overrides", nargs="*", default=None)
    p.add_argument("--source-sha", default=None)
    p.add_argument("--plan-artifact", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="plan only (also the default when --execute is absent)")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--download", action="store_true")
    p.add_argument("--status", action="store_true")
    p.add_argument("--list-suites", action="store_true")
    p.add_argument("--list-methods", action="store_true")
    p.add_argument("--poll-seconds", type=float, default=15.0)
    return p


def _source_sha(explicit=None) -> str:
    if explicit:
        return str(explicit)
    env = os.environ.get("SCSF_SOURCE_COMMIT", "").strip()
    if env:
        return env
    try:
        r = subprocess.run(
            ["git", "-C", REPO_ROOT, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return (r.stdout or "").strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _resolve_cell(cell: dict, a, extra: dict | None = None) -> dict:
    ov = {"dataset": cell["dataset"], "backbone": cell["backbone"],
          "method_name": cell["method_name"], "recipe": cell["recipe"],
          "seed": cell["seed"], "data": {"root": a.data_root},
          "results_root": a.results_root, "train": {"device": "auto"}}
    if cell.get("variant"):
        ov["variant"] = cell["variant"]
        ov.setdefault("method", {})["variant"] = cell["variant"]
    if extra:
        _deep_assign(ov, extra)
    for kv in a.overrides or ():
        k, _, v = kv.partition("=")
        _deep_assign(ov, {k: v})
    cfg = resolve(ov, resolve_device=False)
    cfg["scientific_hash"] = scientific_hash(cfg)
    cfg["comparison_signature"] = comparison_signature(cfg, _source_sha(a.source_sha))
    cfg["run_name"] = run_name_for(cfg)
    cfg["id"] = cell.get("id")
    cfg["variant"] = cell.get("variant")
    return cfg


def _deep_assign(target: dict, dotted: dict) -> None:
    for k, v in dotted.items():
        if isinstance(v, dict) and isinstance(target.get(k), dict):
            _deep_assign(target[k], v)
        elif "." in str(k):
            node = target
            parts = str(k).split(".")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = v
        else:
            if isinstance(v, dict):
                target.setdefault(k, {}).update(v) if isinstance(target.get(k), dict) else target.update({k: v})
            else:
                target[k] = v


def _capability_check(row: dict, suite: str) -> None:
    m = row["method_name"]
    if m not in METHOD_ID_OK and row.get("id") not in METHOD_VARIANTS:
        raise SystemExit(f"unsupported method {m!r} for {row.get('id')}")
    variant = row.get("variant") or (row.get("method") or {}).get("variant")
    allowed = METHOD_ID_OK.get(m)
    if variant and allowed and variant not in allowed and m in NEXT5_METHODS:
        # fold teachers are registered on ce
        pass
    if suite in ("next5_pilot", "next5") and row["backbone"] not in NEXT5_BACKBONES:
        raise SystemExit(
            f"next5_pilot does not support backbone {row['backbone']!r} "
            f"(allowed: {sorted(NEXT5_BACKBONES)})"
        )


def _matrix(a) -> list[dict]:
    recipe = a.recipe or (
        "ccl_sc_reference" if a.suite in ("next5_pilot", "next5") else "singlerun"
    )
    a.recipe = recipe
    datasets = list(a.datasets or ["cifar10"])
    backbones = list(a.backbones or ["vgg16_bn" if a.suite in ("next5_pilot", "next5") else "resnet18"])
    seeds = list(a.seeds or [13])
    cells = []
    for row in load_suite(a.suite):
        for ds in datasets:
            for bb in backbones:
                for seed in seeds:
                    cells.append({
                        "id": row["id"], "method_name": row["method_name"],
                        "variant": row["variant"], "dataset": ds,
                        "backbone": bb, "seed": int(seed), "recipe": recipe,
                    })
    if a.methods:
        wanted = set(a.methods)
        keep = [c for c in cells if c["method_name"] in wanted or c["id"] in wanted]
        missing = [m for m in wanted if not any(
            c["method_name"] == m or c["id"] == m for c in cells)]
        if missing:
            raise SystemExit(
                f"--methods {missing} not in suite {a.suite!r}; "
                "refusing partial-union guessing"
            )
        cells = keep
    return cells


def expand_jobs(a) -> list[dict]:
    """Full DAG: training jobs + artifact/eval dependencies."""
    cells = _matrix(a)
    jobs = []
    by_key = {}
    src = _source_sha(a.source_sha)
    for cell in cells:
        extra = {}
        if cell["variant"] in FOLD_TEACHER_EXCLUDE:
            extra["data.exclude_fold"] = FOLD_TEACHER_EXCLUDE[cell["variant"]]
        cfg = _resolve_cell(cell, a, extra)
        _capability_check(cfg, a.suite)
        jid = f"train:{cfg['run_name']}"
        job = {
            "job_id": jid,
            "kind": "train",
            "id": cell["id"],
            "method_name": cell["method_name"],
            "variant": cell["variant"],
            "dataset": cell["dataset"],
            "backbone": cell["backbone"],
            "seed": cell["seed"],
            "recipe": cell["recipe"],
            "run_name": cfg["run_name"],
            "scientific_hash": cfg["scientific_hash"],
            "comparison_signature": cfg["comparison_signature"],
            "source_sha": src,
            "depends_on": [],
            "extra": extra,
            "status": "planned",
        }
        jobs.append(job)
        by_key[(cell["dataset"], cell["seed"], cell["backbone"], cell["id"])] = job

    if a.suite in ("next5_pilot", "next5"):
        jobs = _attach_next5_dag(a, jobs, by_key)
    else:
        jobs = _attach_eval(jobs)
    _validate_dag(jobs)
    return jobs


def _attach_eval(jobs: list[dict]) -> list[dict]:
    extra = []
    for j in jobs:
        if j["kind"] != "train":
            continue
        extra.append({
            "job_id": f"eval:{j['run_name']}:val",
            "kind": "evaluate",
            "run_name": j["run_name"],
            "split": "val",
            "dataset": j["dataset"],
            "depends_on": [j["job_id"]],
            "status": "planned",
            "seed": j["seed"],
            "backbone": j.get("backbone"),
            "method_name": j.get("method_name"),
        })
        extra.append({
            "job_id": f"eval:{j['run_name']}:test",
            "kind": "evaluate",
            "run_name": j["run_name"],
            "split": "test",
            "dataset": j["dataset"],
            "depends_on": [f"eval:{j['run_name']}:val"],
            "status": "planned",
            "seed": j["seed"],
            "backbone": j.get("backbone"),
            "method_name": j.get("method_name"),
        })
    return jobs + extra


def _attach_next5_dag(a, jobs, by_key) -> list[dict]:
    extra = []
    groups = {}
    for j in jobs:
        groups.setdefault((j["dataset"], j["seed"], j["backbone"]), []).append(j)

    for (ds, seed, bb), group in groups.items():
        named = {j["id"]: j for j in group}
        art = os.path.join(a.results_root, "artifacts",
                           f"{ds}-{bb}-s{seed}")
        if "ce.fold_teacher_a" in named and "ce.fold_teacher_b" in named:
            ta, tb = named["ce.fold_teacher_a"], named["ce.fold_teacher_b"]
            ea = {
                "job_id": f"extract_oof:{ds}:{bb}:s{seed}:a",
                "kind": "extract_oof",
                "teacher_job": ta["job_id"],
                "out": os.path.join(art, "oof_fold0.json"),
                "depends_on": [ta["job_id"]],
                "dataset": ds, "status": "planned",
            }
            eb = {
                "job_id": f"extract_oof:{ds}:{bb}:s{seed}:b",
                "kind": "extract_oof",
                "teacher_job": tb["job_id"],
                "out": os.path.join(art, "oof_fold1.json"),
                "depends_on": [tb["job_id"]],
                "dataset": ds, "status": "planned",
            }
            merged = os.path.join(art, "oof_merged.json")
            mg = {
                "job_id": f"merge_oof:{ds}:{bb}:s{seed}",
                "kind": "merge_oof",
                "inputs": [ea["out"], eb["out"]],
                "out": merged,
                "depends_on": [ea["job_id"], eb["job_id"]],
                "dataset": ds, "status": "planned",
            }
            extra += [ea, eb, mg]
            if "crossfit_failure" in named:
                named["crossfit_failure"]["depends_on"].append(mg["job_id"])
                named["crossfit_failure"]["extra"]["method.oof_path"] = merged
        if "ce" in named:
            mem = {
                "job_id": f"build_memory:{ds}:{bb}:s{seed}",
                "kind": "build_memory",
                "reference_job": named["ce"]["job_id"],
                "out": os.path.join(art, "memory.npz"),
                "depends_on": [named["ce"]["job_id"]],
                "dataset": ds, "status": "planned",
            }
            extra.append(mem)
            if "neighbor_distill" in named:
                named["neighbor_distill"]["depends_on"].append(mem["job_id"])
                named["neighbor_distill"]["extra"]["method.memory_path"] = mem["out"]
    return _attach_eval(jobs + extra)


def _validate_dag(jobs: list[dict]) -> None:
    ids = {j["job_id"] for j in jobs}
    if len(ids) != len(jobs):
        raise SystemExit("duplicate job_id in DAG")
    for j in jobs:
        for d in j.get("depends_on") or []:
            if d not in ids:
                raise SystemExit(f"{j['job_id']} depends on unknown {d}")


def _queue_path(results_root: str) -> str:
    return os.path.join(results_root, "queue", "status.json")


def _write_status(results_root: str, payload: dict) -> None:
    path = _queue_path(results_root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
    os.replace(tmp, path)


def _print_plan(jobs: list[dict]) -> None:
    trains = [j for j in jobs if j["kind"] == "train"]
    arts = [j for j in jobs if j["kind"] != "train"]
    print(f"plan: {len(trains)} training jobs, {len(arts)} artifact/eval jobs "
          f"(total {len(jobs)})")
    print("TRAIN")
    for j in trains:
        deps = ",".join(j["depends_on"]) or "-"
        print(f"  {j['job_id']:<62} hash={j['scientific_hash'][:12]} "
              f"sig={j['comparison_signature'][:8]} deps={deps}")
    print("ARTIFACT/EVAL")
    for j in arts:
        deps = ",".join(j["depends_on"]) or "-"
        print(f"  {j['kind']:<14} {j['job_id']:<62} deps={deps}")


def _already_complete(a, job: dict) -> str | None:
    """Return 'skip' if matching complete, raise on conflict, else None."""
    if job["kind"] == "train":
        run_dir = os.path.join(a.results_root, job["run_name"])
        cfg_p = os.path.join(run_dir, "cfg.json")
        man_p = os.path.join(run_dir, "manifest.json")
        if not os.path.exists(cfg_p):
            return None
        with open(cfg_p) as f:
            cfg = json.load(f)
        got = scientific_hash(cfg)
        if got != job["scientific_hash"]:
            raise SystemExit(
                f"CONFLICT {run_dir}: scientific_hash {got[:12]} != "
                f"plan {job['scientific_hash'][:12]}. Refusing _v2 rename."
            )
        if os.path.exists(man_p):
            return "skip"
        return "resume" if a.resume else None
    if job["kind"] in ("extract_oof", "merge_oof", "build_memory"):
        if os.path.exists(job.get("out", "")):
            return "skip"
    if job["kind"] == "evaluate":
        run_dir = os.path.join(a.results_root, job["run_name"])
        if os.path.exists(os.path.join(run_dir, f"eval_{job['split']}.json")):
            return "skip"
    return None


def _train_cmd(a, job: dict, device: str) -> list[str]:
    args = [
        a.python, "-m", "scsf.train",
        f"dataset={job['dataset']}",
        f"backbone={job['backbone']}",
        f"method_name={job['method_name']}",
        f"recipe={job['recipe']}",
        f"train.seed={job['seed']}",
        f"train.device={device}",
        f"data.root={a.data_root}",
        f"data.num_workers={a.num_workers}",
        f"results_root={a.results_root}",
    ]
    if not a.download:
        args.append("data.download=false")
    if job.get("variant"):
        args.append(f"method.variant={job['variant']}")
    for k, v in (job.get("extra") or {}).items():
        args.append(f"{k}={v}")
    return args


def _smoke_n(a) -> int:
    for kv in a.overrides or ():
        if kv.startswith("train.overfit="):
            try:
                return int(kv.split("=", 1)[1])
            except ValueError:
                return 0
    return 0


def _spawn(a, job: dict, device: str | None) -> subprocess.Popen:
    env = os.environ.copy()
    env["SCSF_SOURCE_COMMIT"] = job.get("source_sha") or _source_sha(a.source_sha)
    env["SCSF_SOURCE_DIRTY"] = "0"
    log_dir = os.path.join(a.results_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, job["job_id"].replace(":", "_") + ".log")
    if job["kind"] == "train":
        child_dev = _child_cuda(env, device)
        cmd = _train_cmd(a, job, child_dev)
    elif job["kind"] == "evaluate":
        run_dir = os.path.join(a.results_root, job["run_name"])
        cmd = [a.python, "-m", "scsf.evaluate",
               f"run_dir={run_dir}", f"split={job['split']}"]
        if device:
            child_dev = _child_cuda(env, device)
            cmd.append(f"device={child_dev}")
    elif job["kind"] == "extract_oof":
        teacher = next(j for j in a._jobs if j["job_id"] == job["teacher_job"])
        tdir = os.path.join(a.results_root, teacher["run_name"])
        cmd = [a.python, "-m", "scsf.extract_oof",
               f"teacher_run_dir={tdir}", f"out={job['out']}"]
        if device:
            child_dev = _child_cuda(env, device)
            cmd.append(f"device={child_dev}")
        sn = _smoke_n(a)
        if sn:
            cmd.append(f"smoke_n={sn}")
    elif job["kind"] == "merge_oof":
        cmd = [a.python, "-m", "scsf.merge_oof", *job["inputs"], f"out={job['out']}"]
    elif job["kind"] == "build_memory":
        ref = next(j for j in a._jobs if j["job_id"] == job["reference_job"])
        rdir = os.path.join(a.results_root, ref["run_name"])
        cmd = [a.python, "-m", "scsf.build_memory",
               f"reference_run_dir={rdir}", f"out={job['out']}"]
        if device:
            child_dev = _child_cuda(env, device)
            cmd.append(f"device={child_dev}")
        sn = _smoke_n(a)
        if sn:
            cmd.append(f"smoke_n={sn}")
    else:
        raise SystemExit(f"unknown kind {job['kind']}")
    logf = open(log_path, "ab")
    job["log_path"] = log_path
    job["cmd"] = cmd
    print(f"  spawn {job['job_id']} device={device} log={log_path}")
    print("    " + " ".join(shlex.quote(c) for c in cmd))
    return subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT)


def _devices(a) -> list[str]:
    if a.devices:
        out = []
        for d in a.devices:
            if d.startswith("cuda:") or d == "cpu":
                out.append(d)
            else:
                out.append(f"cuda:{d}")
        return out
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis:
        return [f"cuda:{i}" for i, _ in enumerate(vis.split(",")) if _.strip() != ""]
    raise SystemExit("pass --devices (plan-only does not probe GPUs)")


class _Pool:
    def __init__(self, devices: list[str], max_jobs: int):
        self.free = list(devices[:max(1, int(max_jobs))])
        self.busy = {}
        self.max_jobs = max(1, int(max_jobs))

    def acquire(self, job_id: str, need_gpu: bool) -> str | None:
        if len(self.busy) >= self.max_jobs:
            return None
        if not need_gpu:
            self.busy[job_id] = "cpu"
            return "cpu"
        if not self.free:
            return None
        d = self.free.pop(0)
        self.busy[job_id] = d
        return d

    def release(self, job_id: str) -> None:
        d = self.busy.pop(job_id, None)
        if d and d != "cpu":
            self.free.append(d)


def _child_cuda(env: dict, device: str | None) -> str:
    """Map launcher ``cuda:i`` onto the i-th already-visible device.

    If the parent already set CUDA_VISIBLE_DEVICES, ``cuda:0`` means that
    first remapped GPU, not physical GPU 0. The child then sees ``cuda:0``.
    """
    if not device or device == "cpu" or not str(device).startswith("cuda"):
        return device or "cpu"
    idx_s = str(device).split(":")[-1]
    try:
        idx = int(idx_s)
    except ValueError:
        env["CUDA_VISIBLE_DEVICES"] = idx_s
        return "cuda:0"
    parent = env.get("CUDA_VISIBLE_DEVICES")
    if parent:
        parts = [p.strip() for p in parent.split(",") if p.strip() != ""]
        if idx >= len(parts):
            raise SystemExit(
                f"device {device} is outside parent CUDA_VISIBLE_DEVICES={parent!r}"
            )
        env["CUDA_VISIBLE_DEVICES"] = parts[idx]
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(idx)
    return "cuda:0"


def _execute(a, jobs: list[dict]) -> int:
    a._jobs = jobs
    devices = _devices(a)
    pool = _Pool(devices, a.max_jobs or 1)
    state = {j["job_id"]: dict(j) for j in jobs}
    running = {}
    blocked = set()
    done = set()
    failed = set()
    os.makedirs(a.results_root, exist_ok=True)

    def snapshot(active=None):
        _write_status(a.results_root, {
            "suite": a.suite, "recipe": a.recipe, "results_root": a.results_root,
            "source_sha": _source_sha(a.source_sha),
            "pid": os.getpid(),
            "active": active, "done": sorted(done), "failed": sorted(failed),
            "blocked": sorted(blocked),
            "running": {k: v.get("device") for k, v in running.items()},
            "jobs": [{k: j.get(k) for k in
                      ("job_id", "kind", "status", "run_name", "depends_on",
                       "scientific_hash", "log_path")}
                     for j in state.values()],
        })

    snapshot()
    rc = 0
    try:
        while len(done) + len(failed) + len(blocked) < len(jobs):
            # start ready jobs
            for j in jobs:
                jid = j["job_id"]
                if jid in done or jid in failed or jid in blocked or jid in running:
                    continue
                deps = j.get("depends_on") or []
                if any(d in failed or d in blocked for d in deps):
                    blocked.add(jid)
                    state[jid]["status"] = "BLOCKED_DEPENDENCY"
                    print(f"  BLOCKED_DEPENDENCY {jid}")
                    continue
                if not all(d in done for d in deps):
                    continue
                action = _already_complete(a, j)
                if action == "skip":
                    done.add(jid)
                    state[jid]["status"] = "skipped_resume"
                    print(f"  [resume] skip {jid}")
                    continue
                need_gpu = j["kind"] in ("train", "extract_oof", "build_memory", "evaluate")
                # One live job per reserved slot. CPU artifact work still
                # occupies the same concurrency budget so it cannot launch
                # beside an untracked trainer when --max-jobs 1.
                dev = pool.acquire(jid, need_gpu=need_gpu)
                if dev is None:
                    continue
                if action == "resume" and j["kind"] == "train":
                    j.setdefault("extra", {})["resume_from"] = "last"
                proc = _spawn(a, j, dev if need_gpu else None)
                running[jid] = {"proc": proc, "device": dev, "job": j, "t0": time.time()}
                state[jid]["status"] = "running"
                state[jid]["device"] = dev
            snapshot(active=[k for k in running])
            if not running:
                if len(done) + len(failed) + len(blocked) >= len(jobs):
                    break
                time.sleep(min(2.0, a.poll_seconds))
                continue
            for jid, info in list(running.items()):
                prc = info["proc"]
                rc_child = prc.poll()
                if rc_child is None:
                    continue
                pool.release(jid)
                running.pop(jid)
                if rc_child == 0:
                    done.add(jid)
                    state[jid]["status"] = "complete"
                    print(f"  complete {jid} ({time.time()-info['t0']:.1f}s)")
                else:
                    failed.add(jid)
                    state[jid]["status"] = f"failed:{rc_child}"
                    print(f"  FAILED {jid} rc={rc_child} log={info['job'].get('log_path')}")
                    rc = rc_child or 1
                    if not a.continue_on_error:
                        for other in list(running):
                            running[other]["proc"].terminate()
                        snapshot()
                        return rc
            time.sleep(a.poll_seconds)
    finally:
        snapshot()
    print(f"run_experiments: done={len(done)} failed={len(failed)} "
          f"blocked={len(blocked)} / {len(jobs)}")
    return rc


def _print_status(a) -> None:
    path = _queue_path(a.results_root)
    if not os.path.exists(path):
        raise SystemExit(f"no status file at {path}")
    with open(path) as f:
        d = json.load(f)
    print(json.dumps({k: d[k] for k in
                      ("suite", "pid", "source_sha", "active", "done",
                       "failed", "blocked") if k in d}, indent=2))
    for j in d.get("jobs") or []:
        print(f"  {j.get('status','?'):<18} {j.get('job_id')}")


def main(argv=None) -> None:
    a = _args().parse_args(list(sys.argv[1:] if argv is None else argv))
    if a.list_suites:
        for s in SUITES:
            print(f"  {s}: {len(DEFAULT_SUITES[s])} ids")
        return
    if a.list_methods:
        for m, vars_ in sorted(METHOD_ID_OK.items()):
            extra = ("  [variants: %s]" % ", ".join(sorted(vars_))) if vars_ else ""
            print(f"  {m}{extra}")
        return
    if a.status:
        _print_status(a)
        return
    if not a.suite:
        raise SystemExit("--suite required (or --list-suites / --status)")
    if a.execute and a.dry_run:
        raise SystemExit("--execute and --dry-run are mutually exclusive")
    if a.resume and not a.execute:
        raise SystemExit("--resume has no effect in plan-only; pass --execute")
    jobs = expand_jobs(a)
    if a.plan_artifact:
        if os.path.exists(a.plan_artifact):
            raise SystemExit(f"plan artifact {a.plan_artifact} exists; refusing overwrite")
        os.makedirs(os.path.dirname(a.plan_artifact) or ".", exist_ok=True)
        with open(a.plan_artifact, "w") as f:
            json.dump({"suite": a.suite, "recipe": a.recipe, "jobs": jobs},
                      f, indent=2, sort_keys=True)
    _print_plan(jobs)
    n_train = sum(1 for j in jobs if j["kind"] == "train")
    if a.suite in ("next5_pilot", "next5"):
        n_ds = len(set(j["dataset"] for j in jobs if j["kind"] == "train"))
        expected = 11 * max(n_ds, 1) * len(a.seeds or [13]) * len(
            a.backbones or ["vgg16_bn"])
        # default 2 datasets × 1 seed × 1 backbone = 22
        if not a.methods and n_train != expected and n_ds in (1, 2):
            print(f"  [note] training jobs={n_train} expected_full={22 if n_ds==2 else 11}")
    if not a.execute:
        return
    rc = _execute(a, jobs)
    if rc:
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
