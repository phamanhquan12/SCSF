"""Architecture-neutral backbone shell: BackboneOutput, taps, hooks.

`BackboneOutput` is the only object methods are allowed to consume::

    BackboneOutput(
        logits: Tensor[B, C],
        features: OrderedDict[str, Tensor],
        final_embedding: Tensor[B, D],
    )

`features` keys are registry-native tap names (e.g. ``layer1..layer4``,
``pool1..pool5``, ``group1..group3``, ``stage1..stage4``, ``block0..block11``).
Methods therefore never hard-code module names; they reference either a tap
name from the registry or a semantic *role* from the backbone's ``roles`` map
(for example ``top_l1`` = the tap immediately before the classification head).

``ForwardHook``/``MultiHook`` attach forward hooks that are removed on exit,
which satisfies the "hooks must be removed cleanly after use" requirement.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn


class BackboneOutput:
    __slots__ = ("logits", "features", "final_embedding")

    def __init__(self, logits, features: Optional[OrderedDict] = None, final_embedding=None):
        object.__setattr__(self, "logits", logits)
        object.__setattr__(self, "features", features if features is not None else OrderedDict())
        object.__setattr__(self, "final_embedding", final_embedding)

    def tap(self, name: str):
        return self.features[name]

    def role(self, backbone, role: str):
        """Resolve a semantic role to a captured tap tensor (hard-fail)."""
        name = backbone.roles[role]
        return self.features[name]

    def __repr__(self):
        return (
            f"BackboneOutput(logits={tuple(self.logits.shape)}, "
            f"features={list((k, tuple(v.shape)) for k, v in self.features.items())}, "
            f"final_embedding={tuple(self.final_embedding.shape)})"
        )


class ForwardHook:
    """Records a module's output into ``storage``; removable."""

    def __init__(self, module: nn.Module, storage: Dict[str, torch.Tensor], name: str):
        self.module = module
        self.storage = storage
        self.name = name
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self.storage[self.name] = out

    def remove(self):
        self.handle.remove()


class MultiHook:
    """Attach a set of named hooks; ``remove()`` detaches them all."""

    def __init__(self, modules: Dict[str, nn.Module], store: Optional[Dict[str, torch.Tensor]]=None):
        self.store = store if store is not None else {}
        self._hooks: List[ForwardHook] = []
        for name, mod in modules.items():
            self._hooks.append(ForwardHook(mod, self.store, name))

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def __enter__(self):
        return self.store

    def __exit__(self, *exc):
        self.remove()
        return False


class Backbone(nn.Module):
    """Base class for registry backbones.

    Subclasses implement ``forward_backbone(x, capture)`` and expose:
    ``taps``  — OrderedDict[name, module] of tap modules,
    ``roles`` — dict of semantic role -> tap name,
    ``input_size``, ``channels``, ``final_dim``.
    """

    registry_name: str = "backbone"

    def __init__(self, num_classes: int, input_size: int = 32, channels: int = 3):
        super().__init__()
        self.num_classes = int(num_classes)
        self.input_size = int(input_size)
        self.channels = int(channels)
        self.taps: "OrderedDict[str, nn.Module]" = OrderedDict()
        self.roles: Dict[str, str] = {}

    # -- shape metadata ------------------------------------------------------
    def probe_tap_shapes(self, batch: int = 1):
        """Run a deterministic dummy forward; return {tap_name: torch.Size}."""
        self.eval()
        self._hook_store: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            hooks = MultiHook(self.taps, self._hook_store)
            dummy = torch.zeros(batch, self.channels, self.input_size, self.input_size)
            out = self(dummy)
            shapes = {name: tuple(t.shape) for name, t in self._hook_store.items()}
            hooks.remove()
        return out, shapes

    def forward(self, x):
        return self.forward_backbone(x)

    def forward_embedding(self, x):
        """Return (features, final_embedding) without computing logits.

        Used by head-replacing methods (SelectiveNet builds its own three
        heads on top of the representation; the native head is not in the
        inference graph and must not be counted in deployment parameters).
        """
        raise NotImplementedError

    # -- required by subclasses ---------------------------------------------
    def forward_backbone(self, x) -> BackboneOutput:  # pragma: no cover - abstract
        raise NotImplementedError


def flatten_spatial(t: torch.Tensor) -> torch.Tensor:
    """Flatten a (B, C, 1, 1) / (B, C) / (B, C, H, W) tensor to (B, D)."""
    b = t.shape[0]
    return t.reshape(b, -1)


def adaptive_flatten(t: torch.Tensor, out: int = 1) -> torch.Tensor:
    """GAP to (out, out) if more than one spatial pixel, then flatten.

    Returns (B, C * out * out). For a (B, C, 1, 1) map this is (B, C).
    """
    if t.dim() == 2:
        return t
    if t.shape[2] > out or t.shape[3] > out:
        t = torch.nn.functional.adaptive_avg_pool2d(t, (out, out))
    b, c = t.shape[0], t.shape[1]
    return t.reshape(b, -1)


__all__ = [
    "BackboneOutput",
    "Backbone",
    "ForwardHook",
    "MultiHook",
    "flatten_spatial",
    "adaptive_flatten",
]