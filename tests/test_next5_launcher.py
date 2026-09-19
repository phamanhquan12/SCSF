"""Launcher DAG / plan-only / torch-free import contracts."""

from __future__ import annotations

import os
import py_compile
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAUNCHER = os.path.join(REPO, "scripts", "run_experiments.py")


def test_launcher_compiles():
    py_compile.compile(LAUNCHER, doraise=True)


def test_engine_config_import_is_torch_free():
    code = (
        "import sys; "
        "sys.path.insert(0, %r); "
        "import scsf.engine.config as c; "
        "assert 'torch' not in sys.modules, sorted(sys.modules)[:20]; "
        "assert 'numpy' not in sys.modules; "
        "cfg = c.resolve({'method_name':'ce','train':{'seed':13}}, resolve_device=False); "
        "assert cfg['run_name'].endswith('-s13')"
    ) % REPO
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + "\n" + r.stderr


def test_plan_only_default_does_not_execute(tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO
    r = subprocess.run(
        [sys.executable, LAUNCHER, "--suite", "next5_pilot",
         "--datasets", "cifar10", "cifar100",
         "--backbones", "vgg16_bn", "--seeds", "13",
         "--recipe", "ccl_sc_reference",
         "--data-root", str(tmp_path / "data"),
         "--results-root", str(tmp_path / "results")],
        cwd=REPO, capture_output=True, text=True, env=env, timeout=60,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "22 training jobs" in r.stdout or "training jobs=22" in r.stdout or "plan: 22 training" in r.stdout
    assert "extract_oof" in r.stdout
    assert "build_memory" in r.stdout
    assert "crossfit_failure" in r.stdout
    # plan-only must not create result run dirs
    results = tmp_path / "results"
    if results.exists():
        assert not any(results.iterdir()) or list(results.iterdir()) == []


def test_dry_run_and_execute_are_exclusive(tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO
    r = subprocess.run(
        [sys.executable, LAUNCHER, "--suite", "review", "--dry-run", "--execute"],
        cwd=REPO, capture_output=True, text=True, env=env, timeout=30,
    )
    assert r.returncode != 0


def test_unknown_method_fails_before_launch():
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO
    r = subprocess.run(
        [sys.executable, LAUNCHER, "--suite", "review", "--methods", "not_a_method"],
        cwd=REPO, capture_output=True, text=True, env=env, timeout=30,
    )
    assert r.returncode != 0
    assert "not_a_method" in (r.stderr + r.stdout)


def test_review_suite_still_plans():
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO
    r = subprocess.run(
        [sys.executable, LAUNCHER, "--suite", "review", "--datasets", "cifar10",
         "--backbones", "resnet18", "--seeds", "13"],
        cwd=REPO, capture_output=True, text=True, env=env, timeout=60,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "training jobs" in r.stdout


def test_pool_serializes_cpu_and_gpu_under_max_jobs_one():
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_experiments", LAUNCHER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pool = mod._Pool(["cuda:0"], max_jobs=1)
    d1 = pool.acquire("train", need_gpu=True)
    assert d1 == "cuda:0"
    assert pool.acquire("merge", need_gpu=False) is None
    pool.release("train")
    assert pool.acquire("merge", need_gpu=False) == "cpu"
    assert pool.acquire("train2", need_gpu=True) is None
    pool.release("merge")
    assert pool.acquire("train2", need_gpu=True) == "cuda:0"
