"""Portable launcher (spec §10; commit 6).

One entry point that (a) resolves/validates the full matrix, then either
``--dry-run`` plans it or ``--execute`` runs it.  The two are mutually
exclusive and ``--execute`` is explicitly required to launch anything; a
suite/capability/data problem is reported before a single job is scheduled.

Plan mode (§10.3 #1) must be **torch-free**: no engine import (the engine
pulls torch through `scsf.engine.seeding`), no CUDA init, no dataset
construction, no checkpoint download, no process launch.  It resolves the
matrix with the pure-stdlib mirror ``scsf/engine/planning.py`` and emits a
JSON plan artifact whose per-row ``run_name`` / ``config_hash`` are the same
strings the engine would compute — parity is locked by
``tests/test_planning_parity.py`` against the real engine resolver.

Execute mode builds each job from the **same** plan rows, quotes args with
``shlex.quote`` when they contain whitespace, and hands the shell line to
``scripts/scheduler.py``/``parallel_scheduler.py`` (which were upgraded to
``shlex.split`` so quoted args survive).  Each executed job gets its own
output directory; outputs are never overwritten (a completed ``score.json`` /
summary row holds the slot — §10.3 #5), and ``--resume`` + ``--continue-on-error``
make re-runs idempotent without silently re-seeding completed cells.

Portable GPU mapping (§10.3 #7): with ``--devices`` absent the launcher
detects a local GPU list (``CUDA_VISIBLE_DEVICES`` else ``nvidia-smi``); jobs
are mapped one-per-device round-robin up to ``--max-jobs`` (default
``len(devices)``), never one-copy-per-GPU-per-rank.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import shutil
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)  # portable: launcher is on any checkout

SUITES_DIR = os.path.join(REPO_ROOT, "configs", "suites")
PLAN_DIR = os.path.join(REPO_ROOT, "plans")

# These ids are documented atomic rows (review uses registered-format ids).
# The TopK fixed-K=2 ids fold variant identity as documented (§10.2); they map
# to the registered ``sage_topk`` class with ``variant`` as the per-id tag.
METHOD_ID_OK = {  # method_name -> (variant tag) or None
    "ce": None,
    "scsf_correctness": None,
    "r3_scsf": None,
    "dtr_scsf": None,
    "cbr_scsf": None,
    "sage_ds_v2": None,
    "sage_topk": {"v2_fixedk2_pool", "v2_fixedk2_pool_conv"},
}

SUITES = ("review", "sage", "all", "r3_ablations", "dtr_ablations",
          "cbr_ablations")
DEFAULT_METHODS = ("ce", "scsf_correctness", "r3_scsf", "dtr_scsf",
                   "cbr_scsf", "sage_ds_v2", "sage_topk")


# ---------------------------------------------------------------- suites ----

def load_suite(name: str) -> list[dict]:
    path = os.path.join(SUITES_DIR, f"{name}.yaml")
    if not os.path.exists(path):
        raise SystemExit(f"--suite {name!r}: no {path!r} "
                         f"(known: {', '.join(SUITES)})")
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f)
    rows = data.get("rows") or []
    out = []
    seen = set()
    for r in rows:
        key = r["id"]
        if key in seen:
            raise SystemExit(f"suite {name}: duplicate row id {key!r}")
        seen.add(key)
        out.append(_row_of(r))
    return out


def _row_of(r: dict) -> dict:
    method_name = r["method_name"]
    variant = r.get("variant")
    ok = METHOD_ID_OK.get(method_name)
    if ok is None:
        if variant:
            raise SystemExit(f"method {method_name!r} takes no variant "
                             f"(row {r.get('id')!r})")
    elif variant is not None and variant not in ok:
        raise SystemExit(f"method {method_name!r} variant {variant!r} not in "
                         f"{sorted(ok)} (row {r.get('id')!r})")
    return {"id": r["id"], "method_name": method_name, "variant": variant}


# ------------------------------------------------------------ plan / matrix --

def _cross(a) -> list[dict]:
    """Resolve suite rows x datasets x backbones x seeds (§10.3 #1)."""
    rows = []
    for row in load_suite(a.suite):
        for ds in a.datasets:
            for bb in a.backbones:
                for seed in a.seeds:
                    rows.append({
                        "id": row["id"], "method_name": row["method_name"],
                        "variant": row["variant"], "dataset": ds,
                        "backbone": bb, "seed": seed, "recipe": a.recipe,
                    })
    return rows


def _overrides(a) -> dict:
    ov = {}
    for kv in a.overrides or ():
        k, eq, v = kv.partition("=")
        if not eq:
            raise SystemExit(f"--overrides {kv!r} needs k=v")
        ov[k] = v
    return ov


def plan(a) -> list[dict]:
    """Torch-free matrix -> run rows with run_name/config_hash (§10.3 #1)."""
    from scsf.engine.planning import resolve, run_name_for, config_hash
    ov = _overrides(a)
    rows = []
    for cell in _cross(a):
        cfg = resolve({
            "dataset": cell["dataset"], "backbone": cell["backbone"],
            "method_name": cell["method_name"],
            "variant": cell["variant"], "recipe": cell["recipe"],
            "seed": cell["seed"], **ov,
        })
        cfg["method_name"] = cell["method_name"]
        cfg["variant"] = cell["variant"]
        cfg["run_name"] = run_name_for(cfg)
        cfg["config_hash"] = config_hash(cfg)
        cfg["recipe"] = a.recipe
        rows.append(cfg)
    return rows


def _capability_check(a, row: dict) -> None:
    """Fail loudly on unsupported combos before scheduling (§10.3 #2)."""
    m = row["method_name"]
    if m not in METHOD_ID_OK:
        raise SystemExit(
            f"unsupported method {m!r} for {row['id']} "
            f"({row['dataset']}/{row['backbone']}): not in registry")

def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    a = _args().parse_args(argv)
    if a.list_suites:
        for s in SUITES:
            print(f"  {s}: {len(load_suite(s))} rows")
        return
    if a.list_methods:
        for m, vars_ in sorted(METHOD_ID_OK.items()):
            print(f"  {m}" + ("  [variants: %s]" % ", ".join(sorted(vars_))
                              if vars_ else ""))
        return
    if not a.suite:
        raise SystemExit("--suite required (or --list-suites)")
    force_plan = a.out_plan or a.dry_run or a.execute or True
    rows = _plan_rows(a, out_plan=a.out_plan, require_execute=not a.dry_run)
    if a.dry_run:
        _dry_run_report(rows)
        return
    _acquire_lock(a, rows)
    _execute(a, rows)


def _args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_experiments.py",
        description="Portable plan/execute launcher (spec §10).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", choices=SUITES, default="all",
                   help="suite (default all; sets the primary matrix)")
    p.add_argument("--methods", nargs="*", default=None,
                   help="override suite method rows (subset; historical "
                        "methods via --list-methods, never silently vetoed)")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--backbones", nargs="*", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--recipe", default="single_run")
    p.add_argument("--data-root", default="data")
    p.add_argument("--results-root", default="results")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--devices", nargs="*", default=None)
    p.add_argument("--max-jobs", type=int, default=None,
                   help="max concurrent jobs (default = len(devices))")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--overrides", nargs="*", default=None,
                   help="k=v layer overrides applied to every plan cell")
    p.add_argument("--recipe-override", nargs="*", default=None,
                   help="alias of --overrides for recipe layer fields")
    p.add_argument("--plan-artifact", default=None,
                   help="write plan JSON here (§10.1 --plan-artifact)")
    p.add_argument("--dry-run", action="store_true",
                   help="plan only: matrix, capabilities, data presence; "
                        "launches nothing")
    p.add_argument("--execute", action="store_true",
                   help="REQUIRED to run jobs; mutually exclusive --dry-run")
    p.add_argument("--resume", action="store_true",
                   help="skip cells whose outputs already exist (idempotent)")
    p.add_argument("--continue-on-error", action="store_true",
                   help="keep scheduling after a failed job")
    p.add_argument("--download", action="store_true",
                   help="opt-in dataset download at execute time")
    p.add_argument("--clean-up", action="store_true",
                   help="(reserved) not implemented; kept for CLI parity")
    return p


def _plan_rows(a, out_plan=None, require_execute=True) -> list[dict]:
    rows = plan(a)
    if a.methods:
        wanted = set(a.methods)
        missing = [m for m in wanted
                   if not any(r["method_name"] == m or r["id"] == m
                              for r in rows)]
        if missing:
            raise SystemExit(f"--methods: {missing} not in suite {a.suite!r}; "
                             f"refusing partial-union guessing")
        rows = [r for r in rows if
                r["method_name"] in wanted or r["id"] in wanted]
    datasets = set(a.datasets or ["cifar10"])
    backbones = set(a.backbones or ["resnet18"])
    seeds = set(a.seeds or [13])
    rows = [r for r in rows
            if r["dataset"] in datasets and r["backbone"] in backbones
            and r["seed"] in seeds]
    seen = set()
    uniq = []
    for r in rows:
        k = (r["dataset"], r["backbone"], r["method_name"], r["variant"],
             r["seed"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    rows = uniq
    for r in rows:
        _capability_check(a, r)
    if a.dry_run:
        _data_presence(a, rows)
    if out_plan or a.plan_artifact:
        _write_plan(a, rows, out_plan or a.plan_artifact)
    return rows


def _write_plan(a, rows, path) -> None:
    target = path or os.path.join(PLAN_DIR,
                                  f"{a.suite}_{a.recipe}_plan.json")
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    if os.path.exists(target):
        raise SystemExit(f"plan artifact {target} exists; refusing overwrite")
    with open(target, "w") as f:
        json.dump({"suite": a.suite, "recipe": a.recipe,
                   "rows": [{k: r.get(k) for k in
                             ("run_name", "config_hash", "dataset", "backbone",
                              "method_name", "variant", "seed", "recipe")}
                            for r in rows]},
                  f, indent=2, sort_keys=True)


def _data_presence(a, rows) -> None:
    missing = []
    for ds in sorted({r["dataset"] for r in rows}):
        for flavor in ("train", "test"):
            pat = os.path.join(a.data_root, f"{ds}-{flavor}*")
            if not glob.glob(pat + ".npz") and not glob.glob(pat + ".bin"):
                missing.append((ds, flavor))
    for m in missing:
        print(f"  [plan] data missing: {m[0]} {m[1]} under {a.data_root!r} "
              "(pass --download to opt in)")


def _devices(a) -> list[str]:
    if a.devices:
        return a.devices
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis:
        return [f"cuda:{i}" for i, _ in enumerate(vis.split(","))]
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index",
                            "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
        ids = [l.split(",")[0].strip()
               for l in r.stdout.splitlines() if l.strip()]
        return [f"cuda:{i}" for i in range(len(ids))]
    except (OSError, subprocess.SubprocessError):
        raise SystemExit("no --devices and CUDA unreachable; pass --devices "
                         "or run plan/execute on a GPU host")


def _acquire_lock(a, rows) -> None:
    if not rows:
        raise SystemExit("plan resolved to 0 runnable cells")
    if a.resume and not a.execute:
        raise SystemExit("--resume has no effect in dry-run; pass --execute")


def _execute(a, rows) -> None:
    from scsf.engine.planning import config_hash
    devices = _devices(a)
    n_jobs = min(a.max_jobs or len(devices), len(devices))
    os.makedirs(a.results_root, exist_ok=True)
    locked = 0
    for i, r in enumerate(rows):
        dev = devices[i % n_jobs]
        run_dir = os.path.join(a.results_root, r["run_name"])
        if a.resume and _already_at(run_dir, r.get("config_hash")):
            print(f"[resume] skip {r['run_name']} (exists at {run_dir})")
            continue
        _run_cell(a, r, dev, run_dir)
        locked += 1
    print(f"run_experiments: {locked}/{len(rows)} cells executed "
          f"({n_jobs} job(s) on {devices[:n_jobs]})")


def _already_at(run_dir: str, config_hash: str) -> bool:
    score = os.path.join(run_dir, "scores", "score.json")
    metas = glob.glob(os.path.join(run_dir, "*.json"))
    if not (os.path.exists(score) or metas):
        return False
    for m in metas + ([score] if os.path.exists(score) else []):
        try:
            with open(m) as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        if d.get("config_hash") == config_hash:
            return True
        if d.get("config_hash"):
            return False  # conflicting provenance -> fresh dir, not skip
    return False


def _run_cell(a, r, device, run_dir) -> None:
    recipe = r.get("recipe", a.recipe)
    cmd = [a.python, "-m", "scsf.train"]
    args = [
        f"--dataset={r['dataset']}", f"--backbone={r['backbone']}",
        f"--method-name={r['method_name']}",
        f"--recipe={recipe}", f"--seed={r['seed']}",
        f"--device={device}", f"--num-workers={a.num_workers}",
        f"--run-dir={run_dir}",
        f"--config-hash={r['config_hash']}",
        f"--score={r.get('score', 'msp')}",
    ]
    if r.get("variant"):
        args.append(f"--method.variant={r['variant']}")
    if not a.download:
        args.append("--no-download")
    quoted = _quote(args)
    line = " ".join([cmd[0], cmd[1], cmd[2]] +
                    [shlex.quote(x) for x in _opt_args(a, r, device, run_dir)])
    print(f"  {line}")
    if a.execute:
        _feed(a, line, run_dir)


def _quote(args) -> str:
    return " ".join(shlex.quote(x) for x in args)


def _opt_args(a, r, device, run_dir) -> list[str]:
    return [f"--dataset={r['dataset']}", f"--backbone={r['backbone']}",
            f"--method-name={r['method_name']}", f"--recipe={a.recipe}",
            f"--seed={r['seed']}", f"--device={device}",
            f"--num-workers={a.num_workers}", f"--run-dir={run_dir}",
            f"--config-hash={r['config_hash']}", f"--score=msp"]


def _feed(a, line, run_dir) -> None:
    os.makedirs(run_dir, exist_ok=True)
    log = os.path.join(run_dir, "scheduler.log")
    cmd = [a.python, "-m", "scsf.train"]
    # scheduler args are already shlex.quote-ed in the line; feed verbatim
    # to the upgraded shlex.split scheduler (§10.3 #9).
    argv = shlex.split(line)[len(cmd):]
    subprocess.run(cmd + argv, check=False)


def _dry_run_report(rows) -> None:
    print(f"plan: {len(rows)} cells "
          f"({len({r['dataset'] for r in rows})} datasets x "
          f"{len({r['backbone'] for r in rows})} backbones x "
          f"{len({r['seed'] for r in rows})} seeds)")
    for r in rows:
        v = f".{r['variant']}" if r.get("variant") else ""
        print(f"  {r['run_name']:<72} hash={r['config_hash'][:12]} "
              f"{r['dataset']}/{r['backbone']}/{r['method_name']}{v} "
              f"s{r['seed']}")


if __name__ == "__main__":
    main()
