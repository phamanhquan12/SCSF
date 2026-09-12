"""Fixed-shape, buffer-backed training state shared by the new methods.

These are *training-time* books that never enter ``MethodPrediction.aux``
(read-only view into tracker tensors) and never serialize into checkpoints
(the engine saves only module state + config). All caches live on the same
device as their method and are rebuilt from scratch when a run restarts (their
content is derivable: the running suffices are only a training-glance, not a
scientific external artifact).

Every container is *capacity-bounded* with a documented replacement rule so a
training method can never grow unbounded memory.
"""

from __future__ import annotations

import torch


class RhoCache:
    """Direct-mapped ``(sample_id -> rho)`` cache, one epoch of capacity.

    Collision *raises* instead of silently reusing noise: a DTR/R3 run that
    actually hit a collision (50000 samples, 65536 slots) is better failed
    loudly than silently corrupted. Entries written with the current epoch
    token are valid for a window; lookups past the window expire cleanly.
    """

    def __init__(self, capacity: int = 65536, device=None, dtype=torch.float32):
        self.capacity = int(capacity)
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.reset()

    def reset(self):
        self._ids = torch.full((self.capacity,), -1, dtype=torch.long, device=self.device)
        self._rho = torch.zeros((self.capacity,), dtype=self.dtype, device=self.device)
        self._epoch = torch.full((self.capacity,), -1, dtype=torch.long, device=self.device)

    def update(self, sample_id, rho, epoch: int) -> None:
        slot = int(sample_id) % self.capacity
        prev = int(self._ids[slot].item())
        if prev >= 0 and prev != int(sample_id):
            raise RuntimeError(
                f"RhoCache collision at slot {slot}: {prev} vs {sample_id} "
                f"(capacity {self.capacity})"
            )
        self._ids[slot] = int(sample_id)
        self._rho[slot] = rho.detach()
        self._epoch[slot] = int(epoch)

    def lookup(self, sample_id, epoch: int, expiry: int = 1):
        """Return ``rho`` or ``None``; the entry must be within ``expiry`` epochs."""
        slot = int(sample_id) % self.capacity
        if int(self._ids[slot].item()) != int(sample_id):
            return None
        if epoch - int(self._epoch[slot].item()) > int(expiry):
            return None
        return self._rho[slot]


class DualState:
    """Dual accumulators for CBR (one per supported confusion edge).

    ``nu`` is dual variables (softmax-free, clamped to ``dual_max``), ascent is
    blocked only at the documented boundary (optimizer commitment: dual *moves*
    after a *successful* step, never inside the loss); ``residual`` is a
    light EMA of the constraint residual for reporting/probing only.
    ``sample`` returns a detached copy for the loss mix.
    """

    def __init__(self, num_edges, dtype=torch.float32, dual_max: float = 100.0,
                 ema: float = 0.0):
        self.dual_max = float(dual_max)
        self.ema = float(ema)
        self._nu = torch.zeros(num_edges, dtype=dtype)
        self._residual = torch.zeros(num_edges, dtype=dtype)

    def clamp_nu(self):
        torch.clamp_(self._nu, 0.0, self.dual_max)

    def add_dual_grad(self, grad_nu, lr: float):
        self._nu = torch.clamp(self._nu + lr * grad_nu.detach(), 0.0, self.dual_max)

    def set_residual(self, residual):
        r = residual.detach()
        if self.ema > 0:
            self._residual = self.ema * self._residual + (1.0 - self.ema) * r
        else:
            self._residual = r

    def sample(self):
        return self._nu.detach().clone()

    def nu(self):
        return self._nu


class EdgeSupport:
    """Confusion-edge support counts during training (dims: C x C).

    ``update_from_batch`` accumulates ``(y_predicted, y_true)`` hard counts;
    ``mask(min_support)`` returns the supported-edge boolean mask over the
    **balanced-interaction** interpretation (edges with observed mass ≥
    min_support). Support is what blocks silent edges that only ever saw 0
    samples.
    """

    def __init__(self, num_classes, device, dtype=torch.long):
        self.num_classes = int(num_classes)
        self.device = device
        self._counts = torch.zeros(
            (self.num_classes, self.num_classes), dtype=dtype, device=device
        )

    def update_from_batch(self, y_pred, y_true):
        yp = y_pred.detach().flatten().to(torch.long)
        yt = y_true.detach().flatten().to(torch.long)
        flat = yp * self.num_classes + yt
        idx, cnt = torch.unique(flat, return_counts=True)
        self._counts.flatten().index_add_(
            0, idx, cnt.to(self._counts.dtype)
        )

    def counts(self):
        return self._counts

    def mask(self, min_support: int = 1):
        return self._counts >= int(min_support)


class PhaseState:
    """Small immutable-typed phase/step scalars carried per ``train_loss`` call.

    The engine's public ``state`` object owns epoch/step; methods that need
    phase-local read-only values (pretrain, warmup, ramp) get them from here.
    This keeps ``state`` API surface minimal and the phase transition logic
    testable without a fake engine.
    """

    def __init__(self, epoch: int = 0, step: int = 0):
        self.epoch = int(epoch)
        self.step = int(step)


__all__ = [
    "DualState",
    "EdgeSupport",
    "PhaseState",
    "RhoCache",
]