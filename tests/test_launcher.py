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
    p = _planning()
    base = {"dataset": "cifar10", "backbone": "resnet18",
            "method_name": "sage_topk", "variant": "v2_fixedk2_pool",
            "score": "msp", "recipe": "single_run", "seed": 13}
    other = dict(base, variant="v2_fixedk2_pool_conv")
    # pool vs pool_conv must never fold to the same run_name/config hash.
    assert p.run_name_for(base) != p.run_name_for(other)
    assert p.config_hash(base) != p.config_hash(other)


def test_suites_exact_row_counts():
    p = _planning()
    # §10.2: review=5, sage=4, all=dedup union=8.
    assert len(p.DEFAULT_SUITES["review"]) == 5
    assert len(p.DEFAULT_SUITES["sage"]) == 4
    assert len(p.DEFAULT_SUITES["all"]) == 8
    assert sorted(p.DEFAULT_SUITES["all"]) == sorted(
        set(p.DEFAULT_SUITES["review"]) | set(p.DEFAULT_SUITES["sage"]))
    # ablations never silently part of `all`.
    union = set(p.DEFAULT_SUITES["review"]) | set(p.DEFAULT_SUITES["sage"])
    for k in ("r3_ablations", "dtr_ablations", "cbr_ablations"):
        assert set(p.DEFAULT_SUITES[k]).isdisjoint(union)


def test_sage_topk_variants_fold_distinct():
    # §10.2: the two TopK fixed-K=2 ids share one class but never one cell.
    p = _planning()
    assert "sage_topk_v2_fixedk2_pool" in p.METHOD_VARIANTS
    m = p.METHOD_VARIANTS["sage_topk_v2_fixedk2_pool"]
    assert m[0] == "sage_topk"
    assert p.METHOD_VARIANTS["sage_topk_v2_fixedk2_pool_conv"] != m
    assert p.METHODS == sorted({x[0] for x in p.METHOD_VARIANTS.values()})
