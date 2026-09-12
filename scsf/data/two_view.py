"""Two-view stable-ID training path used by R3/DTR (§3.5 of the spec).

``build_two_view_dataloader`` yields one batch of 4-tuples per step:

    (x, v, y, idx)

where ``x`` is the augmented primary image, ``v`` a second label-preserving
augmented view of the **same original image**, ``y`` the label and ``idx`` the
stable global training-fold index. Both views go through the documented
training pipeline (``cifar.get_train_transform``) applied to the raw image,
each under a deterministic per-sample/per-view seed derived from
``(data_order_seed, sample_id, view_index)``.

Invariants (locked by tests):

* view-0 is **identical** whether or not a second view is requested
  (per-view RNG is seeded and the global RNG restored around each sample),
* ``seed(view == 0) != seed(view == 1)`` for the same sample,
* ``return_indices``-style primary methods are untouched: this path is only
  used when ``Method.needs_two_views`` is ``True``.
"""

from __future__ import annotations

from typing import Optional

from .cifar import _IndexSubset, _open_train_fold, _worker_seed
from ..methods.rc_training.views import apply_transform_gated, view_seed


__all__ = ["PairedViewDataset", "build_two_view_dataloader"]


class PairedViewDataset:
    """View-pair generator over the raw official train fold.

    ``base`` must yield ``(raw_image, label)`` and expose ``get_global_index``
    (an ``_IndexSubset`` over the deterministic split). ``transform`` is the
    shared training pipeline applied to the raw image for every view.
    """

    def __init__(self, base, transform, data_order_seed: int, n_views: int = 2):
        self.base = base
        self.transform = transform
        self.data_order_seed = int(data_order_seed)
        self.n_views = int(n_views)
        assert self.n_views >= 1

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        raw, y = self.base[i]
        idx = self.base.get_global_index(i)
        x, _ = apply_transform_gated(
            self.transform, raw, view_seed(self.data_order_seed, idx, 0))
        if self.n_views >= 2:
            v, _ = apply_transform_gated(
                self.transform, raw, view_seed(self.data_order_seed, idx, 1))
            return x, v, y, idx
        return x, y, idx


def build_two_view_dataloader(
    cfg,
    batch_size=None,
    generator=None,
    overfit: int = 0,
    num_workers=None,
    n_views: int = 2,
):
    """Deterministic two-view train DataLoader (``(x, v, y, idx)`` batches).

    Mirrors ``cifar.build_dataloader``: the reshape in ``__getitem__`` is
    fully seeded per sample, so the only cross-run state is the shuffle
    generator (the trainer passes its persistent generator for exact resume).
    """
    import torch
    from torch.utils.data import DataLoader

    if num_workers is None:
        num_workers = int(cfg["data"].get("num_workers", 4))

    from .cifar import get_split

    split_spec = get_split(cfg)
    ds = _IndexSubset(_open_train_fold(cfg, "train", raw=True),
                      split_spec.train_indices)
    if overfit and overfit > 0:
        ds = _IndexSubset(ds.base, ds.indices[: int(overfit)])
    ds = PairedViewDataset(
        ds,
        transform=_train_transform(cfg),
        data_order_seed=int(cfg["train"].get("data_order_seed", cfg["train"]["seed"])),
        n_views=n_views,
    )
    if batch_size is None:
        batch_size = int(cfg["train"]["batch_size"])
    if generator is None:
        generator = torch.Generator()
        generator.manual_seed(int(cfg["train"].get("data_order_seed", cfg["train"]["seed"])))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
        generator=generator,
        worker_init_fn=None if num_workers == 0 else _worker_seed,
    )


def _train_transform(cfg):
    from .cifar import get_train_transform

    return get_train_transform(cfg)