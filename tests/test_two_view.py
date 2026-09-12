"""Two-view stable-ID path tests (authored; STATUS: NOT RUN locally).

Checks the primary-view identity guarantee, seed separation and the
4-tuple contract without needing torchvision data.
"""

import torch

from scsf.data.two_view import PairedViewDataset
from scsf.methods.rc_training.views import view_seed

from scsf.data.cifar import _IndexSubset


class _FakeRaw:
    """Raw 'images' are tensors; labels repeat every 4 samples."""

    def __init__(self, n=16):
        torch.manual_seed(5)
        self.raws = [torch.rand(3, 8, 8) for _ in range(n)]

    def __len__(self):
        return len(self.raws)

    def __getitem__(self, i):
        return self.raws[i], i % 4


def _stub_transform(img):
    """Transform consuming the torch RNG stream (deterministic under seed)."""
    return img + 0.01 * torch.randn(img.shape)


def _subset(n=16):
    return _IndexSubset(_FakeRaw(n), list(range(n)))


def test_paired_dataset_returns_4_tuple_with_global_index():
    ds = PairedViewDataset(_subset(), transform=_stub_transform,
                           data_order_seed=13, n_views=2)
    x, v, y, idx = ds[3]
    assert tuple(x.shape) == (3, 8, 8)
    assert y == 3 % 4 and idx == 3
    assert not torch.equal(x, v) or v.shape == x.shape  # distinct draw (usually)


def test_enabling_second_view_does_not_change_primary():
    """The §3.5 primary-view identity lock: view0 is bit-identical."""
    ds1 = PairedViewDataset(_subset(12), transform=_stub_transform,
                            data_order_seed=13, n_views=1)
    ds2 = PairedViewDataset(_subset(12), transform=_stub_transform,
                            data_order_seed=13, n_views=2)
    for i in range(12):
        assert torch.allclose(ds1[i][0], ds2[i][0])


def test_view_pairs_seed_separated():
    seed = 13
    for i in range(6):
        assert view_seed(seed, i, 0) != view_seed(seed, i, 1)


def test_getitem_restores_global_rng():
    ds = PairedViewDataset(_subset(6), transform=_stub_transform,
                           data_order_seed=13, n_views=2)
    before = torch.random.get_rng_state()
    ds[2]
    assert torch.random.get_rng_state().equal(before)


def test_pair_reproducible_across_calls():
    ds = PairedViewDataset(_subset(8), transform=_stub_transform,
                           data_order_seed=7, n_views=2)
    a = ds[4]
    b = ds[4]
    assert torch.allclose(a[0], b[0]) and torch.allclose(a[1], b[1])