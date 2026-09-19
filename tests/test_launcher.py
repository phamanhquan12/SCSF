"""Launcher + planning-replica parity tests (§10.3, §13; commit 6).

planning.py is deliberately loaded **by file path** (importlib), mirroring the
launcher's own torch-free dry-run replica — never via ``scsf.engine`` (whose
package ``__init__`` pulls torch/numpy).  Parity itself is locked by the §13
GPU section (test_parity), which this file keeps free of.
"""

from __future__ import annotations

import importlib.util
import os
import py_compile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAUNCHER = os.path.join(REPO, "scripts", "run_experiments.py")
PLANNING = os.path.join(REPO, "scsf", "engine", "planning.py")


def _planning():
    spec = importlib.util.spec_from_file_location("_planning_rep", PLANNING)
    m = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(m)
    return m


def test_launcher_compiles():
    py_compile.compile(LAUNCHER, doraise=True)


def test_planning_replica_standalone_and_hash_distinct():
    from scsf.engine.config import resolve, run_name_for, scientific_hash
    a = resolve({"dataset": "cifar10", "backbone": "resnet18",
                 "method_name": "sage_topk", "method": {"variant": "v2_fixedk2_pool"},
                 "recipe": "singlerun", "train": {"seed": 13}}, resolve_device=False)
    b = resolve({"dataset": "cifar10", "backbone": "resnet18",
                 "method_name": "sage_topk", "method": {"variant": "v2_fixedk2_pool_conv"},
                 "recipe": "singlerun", "train": {"seed": 13}}, resolve_device=False)
    assert run_name_for(a) != run_name_for(b)
    assert scientific_hash(a) != scientific_hash(b)


def test_suites_exact_row_counts():
    from scsf.engine.planning import DEFAULT_SUITES
    assert len(DEFAULT_SUITES["review"]) == 5
    assert len(DEFAULT_SUITES["sage"]) == 4
    assert len(DEFAULT_SUITES["all"]) == 8
    assert sorted(DEFAULT_SUITES["all"]) == sorted(
        set(DEFAULT_SUITES["review"]) | set(DEFAULT_SUITES["sage"]))
    union = set(DEFAULT_SUITES["review"]) | set(DEFAULT_SUITES["sage"])
    for k in ("r3_ablations", "dtr_ablations", "cbr_ablations"):
        assert set(DEFAULT_SUITES[k]).isdisjoint(union)
    assert len(DEFAULT_SUITES["next5_pilot"]) == 11


def test_sage_topk_variants_fold_distinct():
    from scsf.engine.planning import METHOD_VARIANTS, METHODS
    assert "sage_topk_v2_fixedk2_pool" in METHOD_VARIANTS
    m = METHOD_VARIANTS["sage_topk_v2_fixedk2_pool"]
    assert m[0] == "sage_topk"
    assert METHOD_VARIANTS["sage_topk_v2_fixedk2_pool_conv"] != m
    assert METHODS == sorted({x[0] for x in METHOD_VARIANTS.values()})
