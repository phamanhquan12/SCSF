"""Deterministic stratified splits: reproducibility, class balance, no leakage."""

import os

import pytest

from scsf.data.splits import (
    SPLIT_SEED,
    SplitSpec,
    assert_no_official_test_leakage,
    load_split,
    make_stratified_split,
    split_hash_of,
    split_hashes,
)


def test_split_is_deterministic_and_covers_the_training_fold():
    a = make_stratified_split("cifar10")
    b = make_stratified_split("cifar10")
    assert a.train_indices == b.train_indices
    assert a.val_indices == b.val_indices
    assert a.train_size == 45000 and a.val_size == 5000
    # union covers [0, 50000) exactly and partitions disjointly
    assert len(set(a.train_indices) | set(a.val_indices)) == 50000
    assert not set(a.train_indices) & set(a.val_indices)


def test_class_balance_exact_per_class():
    a = make_stratified_split("cifar10")
    assert a.train_label_counts() == [4500] * 10
    assert a.val_label_counts() == [500] * 10


def test_no_official_test_leakage_structural_guarantee():
    a = make_stratified_split("cifar10")
    assert_no_official_test_leakage(a)  # all indices inside [0, 50000)


def test_serialize_load_roundtrip_keeps_hashes():
    a = make_stratified_split("cifar10")
    d = a.serialize("/tmp/opencode/test_splits")
    b = load_split("/tmp/opencode/test_splits", "cifar10")
    assert b.train_indices == a.train_indices
    assert b.val_indices == a.val_indices
    assert split_hashes(a) == split_hashes(b)
    assert split_hashes(a)["split_seed"] == SPLIT_SEED


def test_split_hash_stable_checksum():
    h = split_hash_of([0, 1, 2, 3])
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)
    assert split_hash_of([0, 1, 2, 3]) == h


def test_invalid_split_rejected():
    with pytest.raises(ValueError):
        SplitSpec("cifar10", list(range(45000)), list(range(45000, 50000)))
    with pytest.raises(ValueError):
        SplitSpec("cifar10", list(range(45000)), [99999])
    with pytest.raises(ValueError):
        SplitSpec("cifar10", list(range(45000)) + [10], [99999])


def test_split_files_exist_in_results_tree():
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "results", "splits")
    if not os.path.isdir(root):
        pytest.skip("results/splits not present")
    for name in ("cifar10_train_seed20260902.txt", "cifar10_val_seed20260902.txt"):
        assert os.path.exists(os.path.join(root, name))