"""New-family schedules with explicit, test-locked endpoints.

The legacy ``scsf.meta_weight_cosine`` schedule starts at the *minimum* and
climbs to the maximum, which is inverted relative to its documented decay
semantics (audit: ``docs/audits/meta_weight_cosine_audit.md``). It is left
untouched. These helpers are the correctly named and correctly oriented
schedules for the review-aligned family: decreasing cosine meta-weight and a
positive linear temperature anneal, both with explicit one-epoch definitions.
"""

from __future__ import annotations

import math


def meta_weight_cosine_decay(
    epoch: int,
    pretrain: int,
    total_epochs: int,
    start_weight: float = 1.0,
    min_weight: float = 1e-4,
) -> float:
    """Cosine decay from ``start_weight`` down to ``min_weight``.

    Returns ``0.0`` before ``pretrain`` (CE-only phase, inside the epoch
    budget). ``progress`` is clamped to ``[0, 1]``; the joint phase is
    empty when ``total_epochs <= pretrain``. Endpoints and monotonicity are
    locked by tests.
    """
    if total_epochs <= pretrain or epoch < pretrain:
        return 0.0
    progress = (epoch - pretrain) / (total_epochs - pretrain)
    progress = min(max(progress, 0.0), 1.0)
    return min_weight + 0.5 * (start_weight - min_weight) * (
        1.0 + math.cos(math.pi * progress)
    )


def temperature_schedule(
    epoch: int,
    total_epochs: int,
    start_temperature: float,
    end_temperature: float,
) -> float:
    """Linear positive temperature anneal from start to end over the run.

    The configured default for CBR's ``Ts`` (and any KD temperature) is an
    explicit, unvalidated default schedule; endpoints are locked by tests.
    """
    denom = max(int(total_epochs) - 1, 1)
    progress = min(max(int(epoch) / denom, 0.0), 1.0)
    return float(start_temperature) + progress * (
        float(end_temperature) - float(start_temperature)
    )


def phase_fn(epoch: int, start_epoch: int, warmup_epochs: int) -> str:
    """Explicit phase bookkeeping: ``pre`` / ``warmup`` / ``joint``.

    ``start_epoch`` is the first epoch of the run (usually ``0``); the warmup
    phase spans ``[start_epoch, start_epoch + warmup_epochs)`` and the joint
    phase starts at ``start_epoch + warmup_epochs``. The phase is a pure
    function of ``epoch`` (already checkpointed exactly by the trainer), so
    resume reproduces the phase sequence.
    """
    if epoch < int(start_epoch):
        return "pre"
    if epoch < int(start_epoch) + int(max(warmup_epochs, 0)):
        return "warmup"
    return "joint"


__all__ = ["meta_weight_cosine_decay", "phase_fn", "temperature_schedule"]