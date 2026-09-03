"""Deterministic, stratified, leakage-free data splits.

CIFAR-10 and CIFAR-100 validation splits are carved out of the **official
training set** (never the official test set) with fixed seed ``20260902``:

* CIFAR-10 :  45 000 train /  5 000 valid, 10 classes  (4500/450 per class)
* CIFAR-100:  45 000 train /  5 000 valid, 100 classes (450/50 per class)

The official test set is never instantiated by training code paths; only the
explicit ``scsf.evaluate split=test`` command may open it.

The index lists are produced with Python's ``random.Random`` (Mersenne
Twister), which is version-stable, so the split is reproducible without torch.
For each class we shuffle that class's contiguous index block and take the
first ``train_per_class`` indices for training and the next ``val_per_class``
for validation. The result is serialized to plain-text index files so that
runs can store a stable SHA-256 ``split_hash`` in their manifest.
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import List, Sequence, Tuple

SPLIT_SEED = 20260902

DATASET_LAYOUT = {
    "cifar10": {
        "num_classes": 10,
        "per_class": 5000,
        "train_per_class": 4500,
        "val_per_class": 500,
    },
    "cifar100": {
        "num_classes": 100,
        "per_class": 500,
        "train_per_class": 450,
        "val_per_class": 50,
    },
}

#: canonical split-file stem so run manifests can reference a hash by name.
INDEX_FILE_NAMES = {
    "cifar10": (
        "cifar10_train_seed20260902.txt",
        "cifar10_val_seed20260902.txt",
    ),
    "cifar100": (
        "cifar100_train_seed20260902.txt",
        "cifar100_val_seed20260902.txt",
    ),
}


class SplitSpec:
    """Represents one deterministic train/val split of a CIFAR training set."""

    def __init__(
        self,
        dataset: str,
        train_indices: Sequence[int],
        val_indices: Sequence[int],
        seed: int = SPLIT_SEED,
    ):
        if dataset not in DATASET_LAYOUT:
            raise ValueError(f"unknown dataset {dataset!r}")
        self.dataset = dataset
        self.seed = seed
        self.train_indices = [int(i) for i in train_indices]
        self.val_indices = [int(i) for i in val_indices]
        self._validate()

    def _validate(self) -> None:
        layout = DATASET_LAYOUT[self.dataset]
        n = layout["per_class"] * layout["num_classes"]
        for name, idxs in (("train", self.train_indices), ("val", self.val_indices)):
            for i in idxs:
                if not (0 <= i < n):
                    raise ValueError(f"{name} index {i} outside official training set [0,{n})")
            if len(set(idxs)) != len(idxs):
                raise ValueError(f"duplicate index inside {name} split")
        if set(self.train_indices) & set(self.val_indices):
            raise ValueError("train/val index overlap")
        if len(set(self.train_indices) | set(self.val_indices)) != n:
            raise ValueError(
                f"split does not cover the full official training set "
                f"({len(set(self.train_indices) | set(self.val_indices))} of {n})"
            )
        # class balance check
        labels = [i // layout["per_class"] for i in self.train_indices]
        for c in range(layout["num_classes"]):
            if labels.count(c) != layout["train_per_class"]:
                raise ValueError(
                    f"class {c} has {labels.count(c)} train examples, "
                    f"expected {layout['train_per_class']}"
                )

    @property
    def train_size(self) -> int:
        return len(self.train_indices)

    @property
    def val_size(self) -> int:
        return len(self.val_indices)

    def train_label_counts(self):
        layout = DATASET_LAYOUT[self.dataset]
        counts = [0] * layout["num_classes"]
        for i in self.train_indices:
            counts[i // layout["per_class"]] += 1
        return counts

    def val_label_counts(self):
        layout = DATASET_LAYOUT[self.dataset]
        counts = [0] * layout["num_classes"]
        for i in self.val_indices:
            counts[i // layout["per_class"]] += 1
        return counts

    def serialize(self, directory: str) -> Tuple[str, str]:
        """Write index files and return (train_path, val_path)."""
        os.makedirs(directory, exist_ok=True)
        tname, vname = INDEX_FILE_NAMES[self.dataset]
        tpath = os.path.join(directory, tname)
        vpath = os.path.join(directory, vname)
        with open(tpath, "w") as f:
            f.write("# SCSF deterministic stratified split\n")
            f.write("# dataset: %s\n" % self.dataset)
            f.write("# seed: %d\n" % self.seed)
            f.write("# source: official training set (never the official test set)\n")
            for i in self.train_indices:
                f.write("%d\n" % i)
        with open(vpath, "w") as f:
            f.write("# SCSF deterministic stratified split (validation)\n")
            f.write("# dataset: %s\n" % self.dataset)
            f.write("# seed: %d\n" % self.seed)
            f.write("# source: official training set (never the official test set)\n")
            for i in self.val_indices:
                f.write("%d\n" % i)
        return tpath, vpath


def load_split(directory: str, dataset: str) -> SplitSpec:
    """Load a previously serialized split back from text files."""
    tname, vname = INDEX_FILE_NAMES[dataset]
    tpath = os.path.join(directory, tname)
    vpath = os.path.join(directory, vname)
    return SplitSpec(
        dataset,
        _read_indices(tpath),
        _read_indices(vpath),
        seed=SPLIT_SEED,
    )


def _read_indices(path: str) -> List[int]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(int(line))
    return out


def make_stratified_split(dataset: str, seed: int = SPLIT_SEED) -> SplitSpec:
    """Build the deterministic stratified 45k/5k split from the train set."""
    if dataset not in DATASET_LAYOUT:
        raise ValueError(f"unknown dataset {dataset!r}")
    layout = DATASET_LAYOUT[dataset]
    num_classes = layout["num_classes"]
    per_class = layout["per_class"]
    train_idxs: List[int] = []
    val_idxs: List[int] = []
    for c in range(num_classes):
        rng = random.Random(seed * 1_000_003 + c)
        block = list(range(c * per_class, (c + 1) * per_class))
        rng.shuffle(block)
        train_idxs.extend(block[: layout["train_per_class"]])
        val_idxs.extend(block[layout["train_per_class"]: layout["train_per_class"] + layout["val_per_class"]])
    return SplitSpec(dataset, train_idxs, val_idxs, seed=seed)


def split_hash_of(indices: Sequence[int]) -> str:
    """SHA-256 over the canonical serialization of an index list (id list)."""
    payload = "\n".join(str(int(i)) for i in indices) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def split_hashes(split: SplitSpec) -> dict:
    """Return {train_hash, val_hash, split_version} for a run manifest."""
    return {
        "split_seed": split.seed,
        "split_version": "1",
        "train_hash": split_hash_of(split.train_indices),
        "val_hash": split_hash_of(split.val_indices),
        "train_size": split.train_size,
        "val_size": split.val_size,
    }


def assert_no_official_test_leakage(split: SplitSpec) -> None:
    """Assert every split index belongs to the official training set.

    Official-test indices live in a separate torchvision dataset with its own
    0..9999 numbering, so the strongest structural guarantee is that every
    index is inside ``[0, 50000)`` (the full official training fold) and the
    train/val partition covers it exactly.
    """
    split._validate()
    n = DATASET_LAYOUT[split.dataset]["per_class"] * DATASET_LAYOUT[split.dataset]["num_classes"]
    for name, idxs in (("train", split.train_indices), ("val", split.val_indices)):
        for i in idxs:
            assert 0 <= i < n, f"{name} index {i} outside [0,{n})"
    assert not set(split.train_indices) & set(split.val_indices), "train/val overlap"
    assert len(set(split.train_indices) | set(split.val_indices)) == n