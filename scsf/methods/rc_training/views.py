"""Deterministic, stream-safe input-view generation for two-view training.

A paucity-safe view pair is produced at construction time only, from a
per-sample deterministic seed derived from the *data order* seed and the
sample's global index. View transforms are **the same composition** as the
primary train transform (with a weaker ``RandomCrop`` only); the view matching
rule is ``seed(view == 0) != seed(view == 1)`` enforced at unit level.

``apply_transform_gated`` captures and restores the torch / numpy / python
global RNG state around a stochastic transform so the *primary* stream
(newsamples via ``RandomCrop``/flip) is never perturbed by the view stream and
vice versa. Determinism of the view is locked by tests.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np
import torch


def view_seed(data_order_seed: int, sample_id: int, view_index: int) -> int:
    """Separate deterministic 31-bit seed per (sample, view)."""
    digest = _stable_digest(data_order_seed, sample_id, view_index)
    return int(digest, 16) % (2 ** 31)


def _stable_digest(data_order_seed, sample_id, view_index):
    payload = f"{int(data_order_seed)}:{int(sample_id)}:{int(view_index)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def apply_transform_gated(transform, img, seed: int):
    """Run ``transform`` with isolated RNG; return the transformed image.

    The global torch / numpy / python RNG states are captured, the transform
    (seeded with ``seed`` for torch and numpy) is applied, then every RNG state
    is restored so no other stream is perturbed. Returns ``(transformed,
    transformed_seed)``.
    """
    torch_state = torch.random.get_rng_state()
    np_state = np.random.get_state()
    py_state = random.getstate()
    try:
        torch.manual_seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        random.seed(seed)
        out = transform(img)
    finally:
        torch.random.set_rng_state(torch_state)
        np.random.set_state(np_state)
        random.setstate(py_state)
    return out, seed


__all__ = ["apply_transform_gated", "view_seed"]