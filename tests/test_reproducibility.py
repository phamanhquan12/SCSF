"""Reproducibility regression: package imports and pytest collection.

Guards against regressions where required source is accidentally git-ignored
(e.g. the historical ``data/`` rule shadowing ``scsf/data/``) or where the
repository-root package breaks the toolchain (e.g. an orphaned root
``__init__.py`` that imports a module which does not exist). A fresh ``git
clone`` must be runnable without copying any untracked source files.
"""

import importlib
import os


def test_import_scsf_package():
    import scsf
    from scsf.methods import build_method
    from scsf.engine.config import resolve
    # package-level imports resolve without error
    assert scsf is not None
    assert build_method is not None
    assert resolve is not None


def test_import_scsf_data_package():
    import scsf.data
    from scsf.data import (
        SPLIT_SEED,
        SplitSpec,
        build_dataloader,
        build_dataset,
        get_split,
        load_split,
        make_stratified_split,
        split_hashes,
    )
    assert SPLIT_SEED == 20260902
    assert build_dataloader is not None
    assert SplitSpec is not None


def test_data_module_imports():
    """Every required data source module imports on its own."""
    for mod in ("scsf.data.cifar", "scsf.data.splits"):
        importlib.import_module(mod)


def test_split_spec_contract_smoke():
    """The core split object constructs and covers the training fold."""
    from scsf.data.splits import DATASET_LAYOUT, make_stratified_split

    spec = make_stratified_split("cifar10")
    assert spec.train_size == DATASET_LAYOUT["cifar10"]["train_per_class"] * 10
    assert spec.val_size == DATASET_LAYOUT["cifar10"]["val_per_class"] * 10
    hashes = spec.train_label_counts()
    assert all(h == 4500 for h in hashes)  # balanced 10-way


def test_scsf_data_source_files_tracked():
    """The scsf.data source must be tracked in git, not shadowed by gitignore."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("__init__.py", "cifar.py", "splits.py"):
        path = os.path.join(repo_root, "scsf", "data", name)
        assert os.path.isfile(path), f"missing tracked source: {path}"
