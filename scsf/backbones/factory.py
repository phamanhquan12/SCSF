"""Backbone registry + factory.

No method may reference module names outside this registry; a method consumes
either a tap name declared by a backbone or a semantic role from its ``roles``
map. ``build_backbone`` validates shape metadata via a dummy forward.
"""

from __future__ import annotations

from typing import Callable, Dict

from .base import Backbone, BackboneOutput  # noqa: F401
from .vgg import vgg16_bn
from .resnet import resnet18
from .wrn import wideresnet28_10
from .convnext import convnext_tiny
from .deit import deit_s

_REGISTRY: Dict[str, Callable[..., "Backbone"]] = {}


def register_backbone(name: str, builder: Callable[..., "Backbone"]) -> None:
    _REGISTRY[name] = builder


def backbone_names() -> list:
    return sorted(_REGISTRY)


def build_backbone(name: str, num_classes: int, cfg=None) -> Backbone:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown backbone {name!r}; available: {sorted(_REGISTRY)}"
        )
    bcfg = (cfg or {}).get("backbones", {}).get(name) or (cfg or {}).get("backbone_cfg") or {}
    input_size = int(bcfg.get("input_size", 32) if bcfg else 32)
    patch_size = int(bcfg.get("patch_size", 4)) if bcfg and "patch_size" in bcfg else 4
    builder = _REGISTRY[name]
    if name == "deit_s":
        backbone = builder(num_classes, input_size=input_size, patch_size=patch_size)
    else:
        backbone = builder(num_classes, input_size=input_size)
    return backbone


for _n, _b in [
    ("resnet18", resnet18),
    ("vgg16_bn", vgg16_bn),
    ("wideresnet28_10", wideresnet28_10),
    ("convnext_tiny", convnext_tiny),
    ("deit_s", deit_s),
]:
    register_backbone(_n, _b)

__all__ = [
    "Backbone",
    "BackboneOutput",
    "build_backbone",
    "register_backbone",
    "backbone_names",
]