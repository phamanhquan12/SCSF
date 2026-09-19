"""Train-fold construction and scientific-hash identity."""

from scsf.data.splits import (
    SPLIT_SEED,
    make_stratified_split,
    make_stratified_train_folds,
)
from scsf.engine.config import comparison_signature, resolve, scientific_hash


def test_two_folds_partition_train_only():
    split = make_stratified_split("cifar10", seed=SPLIT_SEED)
    folds = make_stratified_train_folds("cifar10", split.train_indices, n_folds=2, seed=SPLIT_SEED)
    assert len(folds) == 2
    a, b = set(folds[0]), set(folds[1])
    assert not (a & b)
    assert a | b == set(split.train_indices)
    assert not (a & set(split.val_indices))
    assert not (b & set(split.val_indices))
    assert abs(len(a) - len(b)) <= 10  # stratified remainder
    assert len(a) == 22500
    assert len(b) == 22500


def test_scientific_hash_ignores_device_and_data_root():
    a = resolve({"dataset": "cifar10", "backbone": "vgg16_bn",
                 "method_name": "ce", "recipe": "ccl_sc_reference",
                 "train": {"seed": 13, "device": "cpu"},
                 "data": {"root": "/tmp/a"}}, resolve_device=False)
    b = resolve({"dataset": "cifar10", "backbone": "vgg16_bn",
                 "method_name": "ce", "recipe": "ccl_sc_reference",
                 "train": {"seed": 13, "device": "cuda:0"},
                 "data": {"root": "/other/data"}}, resolve_device=False)
    assert scientific_hash(a) == scientific_hash(b)
    c = resolve({"dataset": "cifar10", "backbone": "vgg16_bn",
                 "method_name": "ce", "recipe": "ccl_sc_reference",
                 "train": {"seed": 17}}, resolve_device=False)
    assert scientific_hash(a) != scientific_hash(c)


def test_comparison_signature_groups_seeds_keeps_variants():
    a = resolve({"method_name": "ce", "variant": "fold_teacher_a",
                 "train": {"seed": 13}, "backbone": "vgg16_bn",
                 "recipe": "ccl_sc_reference"}, resolve_device=False)
    b = resolve({"method_name": "ce", "variant": "fold_teacher_a",
                 "train": {"seed": 17}, "backbone": "vgg16_bn",
                 "recipe": "ccl_sc_reference"}, resolve_device=False)
    c = resolve({"method_name": "ce", "variant": "fold_teacher_b",
                 "train": {"seed": 13}, "backbone": "vgg16_bn",
                 "recipe": "ccl_sc_reference"}, resolve_device=False)
    assert comparison_signature(a) == comparison_signature(b)
    assert comparison_signature(a) != comparison_signature(c)
    d = resolve({"method_name": "ce", "train": {"seed": 13},
                 "backbone": "vgg16_bn", "recipe": "ccl_sc_reference"},
                resolve_device=False)
    assert comparison_signature(a) != comparison_signature(d)


def test_scientific_hash_ignores_artifact_paths():
    a = resolve({"method_name": "crossfit_failure",
                 "method": {"oof_path": "/tmp/a/oof.json"},
                 "recipe": "ccl_sc_reference", "train": {"seed": 13},
                 "backbone": "vgg16_bn"}, resolve_device=False)
    b = resolve({"method_name": "crossfit_failure",
                 "method": {"oof_path": "/other/oof.json"},
                 "recipe": "ccl_sc_reference", "train": {"seed": 13},
                 "backbone": "vgg16_bn"}, resolve_device=False)
    assert scientific_hash(a) == scientific_hash(b)
    c = resolve({"method_name": "neighbor_distill",
                 "method": {"memory_path": "/tmp/mem.npz"},
                 "recipe": "ccl_sc_reference", "train": {"seed": 13},
                 "backbone": "vgg16_bn"}, resolve_device=False)
    d = resolve({"method_name": "neighbor_distill",
                 "method": {"memory_path": "/other/mem.npz"},
                 "recipe": "ccl_sc_reference", "train": {"seed": 13},
                 "backbone": "vgg16_bn"}, resolve_device=False)
    assert scientific_hash(c) == scientific_hash(d)


def test_overfit_preserves_official_ids():
    from scsf.data.cifar import _IndexDataset, _IndexSubset, _cap_overfit

    class _Fake:
        def __len__(self):
            return 100

        def __getitem__(self, i):
            return i, 0

    official = [10, 20, 30, 40, 50]
    ds = _IndexDataset(_IndexSubset(_Fake(), official))
    capped = _cap_overfit(ds, 3, return_indices=True)
    got = [capped[i][2] for i in range(len(capped))]
    assert got == [10, 20, 30]
    assert len(capped) == 3

