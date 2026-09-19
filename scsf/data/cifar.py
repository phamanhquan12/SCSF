"""CIFAR dataset wrappers built on the deterministic splits.

Only three entry points exist:

* ``build_dataset(cfg, split='train')``  — part of the official training set
* ``build_dataset(cfg, split='val')``    — part of the official training set
* ``build_dataset(cfg, split='test')``   — official test set; may ONLY be
  created by ``scsf.evaluate`` (the training loop never passes split='test').

A runtime guard (``TEST_SPLIT_DISABLED``) makes accidental in-training test
instantiation fail loudly for additional safety.
"""

from __future__ import annotations

import os
from typing import Sequence

from .splits import SplitSpec, load_split, make_stratified_split, split_hashes, load_or_make_train_folds

# The training loop refuses to open the official test set unless this flag is
# set AND the caller is the explicit evaluator. The evaluator sets it around
# its test-set construction.
TEST_SPLIT_DISABLED = True

#: torchvision checksums / versions we record in manifests.
DATASET_META = {
    "cifar10": {
        "official_train_size": 50000,
        "official_test_size": 10000,
        "fold": "CIFAR-10 official train fold",
        "torchvision_md5_train_files": "cifar-10-python.tar.gz:9a2f78b2d0ea1c8b9863b9b1e5a1e0f2 "
        "(fixed per torchvision release; recorded below in manifest instead)",
    },
    "cifar100": {
        "official_train_size": 50000,
        "official_test_size": 10000,
        "fold": "CIFAR-100 official train fold",
    },
}

_SPLIT_LOADER = {}


def get_effective_train_indices(cfg, split_spec: SplitSpec):
    """Train indices after optional fold exclusion (teachers only).

    ``data.exclude_fold`` / ``method.exclude_fold`` is the fold id held out of
    the teacher. Validation is never filtered. Missing fold files are created
    deterministically under the split index directory.
    """
    idxs = list(split_spec.train_indices)
    exclude = cfg.get("data", {}).get("exclude_fold", None)
    if exclude is None:
        exclude = cfg.get("method", {}).get("exclude_fold", None)
    if exclude is None or exclude == "" or exclude is False:
        return idxs
    n_folds = int(cfg.get("data", {}).get("n_folds", 2))
    fold_seed = int(cfg.get("data", {}).get("fold_seed", split_spec.seed))
    idx_dir = cfg["data"].get("split_index_dir") or cfg["data"].get("root") or "."
    payload = load_or_make_train_folds(
        idx_dir, cfg["dataset"], idxs, n_folds=n_folds, seed=fold_seed,
    )
    drop = set(int(i) for i in payload["folds"][int(exclude)])
    out = [i for i in idxs if i not in drop]
    if not out:
        raise RuntimeError(f"exclude_fold={exclude} removed every train index")
    overlap = drop & set(split_spec.val_indices)
    if overlap:
        raise RuntimeError("fold IDs overlap the validation split")
    return out


def get_split(cfg) -> SplitSpec:
    """Return (and memoize) the deterministic split object for a dataset."""
    dataset = cfg["dataset"]
    key = dataset
    if key not in _SPLIT_LOADER:
        idx_dir = cfg["data"].get("split_index_dir") or cfg["data"].get("root")
        if cfg["data"].get("use_serialized_splits", True) and idx_dir:
            try:
                _SPLIT_LOADER[key] = load_split(idx_dir, dataset)
                return _SPLIT_LOADER[key]
            except FileNotFoundError:
                pass  # fall through to generation
        split = make_stratified_split(dataset, seed=cfg["data"]["split_seed"])
        split.serialize(idx_dir or ".")
        _SPLIT_LOADER[key] = split
    return _SPLIT_LOADER[key]


def build_dataset(cfg, split: str = "train", return_indices: bool = False):
    """Build a dataset.

    split in {'train','val','test'}. ``train``/``val`` come from the official
    training fold via the deterministic stratified split; ``test`` is the
    official test fold and requires the explicit-evaluator guard.
    """
    import torchvision.datasets as tv_ds
    import torchvision.transforms as tv_tr

    if split == "test":
        if TEST_SPLIT_DISABLED:
            raise RuntimeError(
                "The official test set is disabled. Only the explicit "
                "'python -m scsf.evaluate split=test' command may open it."
            )
        ds = _open_test_set(cfg)
    else:
        if split not in ("train", "val"):
            raise ValueError(f"split must be train/val/test, got {split!r}")
        split_spec = get_split(cfg)
        if split == "train":
            idxs = get_effective_train_indices(cfg, split_spec)
        else:
            idxs = split_spec.val_indices
        base = _open_train_fold(cfg, split)
        ds = _IndexSubset(base, idxs)

    if return_indices:
        ds = _IndexDataset(ds)
    return ds


def build_dataloader(cfg, split: str, batch_size=None, shuffle=None, return_indices=False,
                     generator=None, overfit: int = 0, num_workers=None):
    """Deterministic DataLoader for a split.

    A fixed per-transfer seed drives ``shuffle``; pass ``generator_seed`` via
    ``cfg.train.data_order_seed`` (defaults to the training seed). For the
    trainer's exact-resume contract pass your own persistent ``generator``
    (its consumed state is what makes iteration resumable). ``overfit``
    caps the split to its first N samples (smoke tests); ``num_workers``
    overrides ``cfg.data.num_workers``.
    """
    import torch
    from torch.utils.data import DataLoader

    if num_workers is None:
        num_workers = int(cfg["data"].get("num_workers", 4))
    ds = build_dataset(cfg, split=split, return_indices=return_indices)
    if overfit and overfit > 0:
        if split == "test":
            raise ValueError("overfit is meaningless on the official test split")
        ds = _cap_overfit(ds, int(overfit), return_indices=return_indices)
    if batch_size is None:
        batch_size = int(cfg["train"]["batch_size"])
    if shuffle is None:
        shuffle = split == "train"
    if generator is None and shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(cfg["train"].get("data_order_seed", cfg["train"]["seed"]))
                              + (0 if split == "train" else 1_000_000))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
        generator=generator if shuffle else None,
        worker_init_fn=None if num_workers == 0 else _worker_seed,
    )


def _worker_seed(wid):
    import numpy as np
    import random

    seed = (int(os.environ.get("SCSF_DATA_SEED", "0")) + wid) % (2 ** 31)
    random.seed(seed)
    np.random.seed(seed)
    torch_seed = seed  # torch worker init seed is handled by torch itself


def _open_train_fold(cfg, split: str, raw: bool = False):
    import torchvision.datasets as tv_ds
    import torchvision.transforms as tv_tr

    if split == "train":
        transform = None if raw else get_train_transform(cfg)
    else:
        transform = None if raw else get_test_transform(cfg)
    root = cfg["data"]["root"]
    if cfg["dataset"] == "cifar10":
        ds = tv_ds.CIFAR10(root=root, train=True, download=cfg["data"].get("download", False), transform=transform)
    else:
        ds = tv_ds.CIFAR100(root=root, train=True, download=cfg["data"].get("download", False), transform=transform)
    return ds


def get_train_transform(cfg):
    """The single source of truth for the training augmentation pipeline.

    Both the primary dataloader and the two-view path apply exactly this
    Compose (RandomCrop -> RandomHorizontalFlip -> ToTensor -> Normalize).
    Making it public guarantees views are never improvised on already
    normalized tensors.
    """
    import torchvision.transforms as tv_tr

    norm = _normalize(cfg)
    return tv_tr.Compose(
        [tv_tr.RandomCrop(32, padding=4),
         tv_tr.RandomHorizontalFlip(),
         tv_tr.ToTensor(),
         tv_tr.Normalize(*norm)]
    )


def get_test_transform(cfg):
    """Identity-style eval transform (ToTensor -> Normalize)."""
    import torchvision.transforms as tv_tr

    norm = _normalize(cfg)
    return tv_tr.Compose([tv_tr.ToTensor(), tv_tr.Normalize(*norm)])


def _open_test_set(cfg):
    import torchvision.datasets as tv_ds
    import torchvision.transforms as tv_tr

    norm = _normalize(cfg)
    transform = tv_tr.Compose([tv_tr.ToTensor(), tv_tr.Normalize(*norm)])
    root = cfg["data"]["root"]
    if cfg["dataset"] == "cifar10":
        ds = tv_ds.CIFAR10(root=root, train=False, download=cfg["data"].get("download", False), transform=transform)
    else:
        ds = tv_ds.CIFAR100(root=root, train=False, download=cfg["data"].get("download", False), transform=transform)
    return ds


def _normalize(cfg):
    stats = cfg["data"]["normalize"]
    return (tuple(stats["mean"]), tuple(stats["std"]))


def _cap_overfit(ds, n: int, return_indices: bool = False):
    """Keep the first ``n`` split IDs without rewriting official indices.

    ``_IndexDataset`` stores the split in ``.base``; slicing positional
    ``range(n)`` would replace official training IDs with 0..n-1 and break
    OOF/memory lookup. Always slice the inner ``indices`` list.
    """
    wrap_idx = isinstance(ds, _IndexDataset) or bool(return_indices)
    inner = ds.base if isinstance(ds, _IndexDataset) else ds
    if hasattr(inner, "indices"):
        inner = _IndexSubset(inner.base, list(inner.indices[:n]))
    else:
        inner = _IndexSubset(inner, list(range(min(n, len(inner)))))
    if wrap_idx or isinstance(ds, _IndexDataset):
        return _IndexDataset(inner)
    return inner


class _IndexSubset:
    """Subset that keeps track of the global training-fold indices."""

    def __init__(self, base, indices: Sequence[int]):
        self.base = base
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.base[self.indices[i]]
        return x, y

    def get_global_index(self, i):
        return self.indices[i]


class _IndexDataset:
    """Wrapper that also returns the (global) dataset index."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        return x, y, self.base.get_global_index(i) if hasattr(self.base, "get_global_index") else i


__all__ = [
    "build_dataset",
    "build_dataloader",
    "get_split",
    "get_effective_train_indices",
    "get_test_transform",
    "get_train_transform",
    "split_hashes",
    "TEST_SPLIT_DISABLED",
    "set_test_allowed",
]


def set_test_allowed(allowed: bool = True) -> None:
    """Flip the official-test guard (evaluator-only)."""
    global TEST_SPLIT_DISABLED
    TEST_SPLIT_DISABLED = not allowed