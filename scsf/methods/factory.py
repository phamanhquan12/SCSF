"""Method registry + factory."""

from __future__ import annotations

from typing import Callable, Dict

from .base import Method, MethodPrediction  # noqa: F401
from .ce import CEMethod
from .ccl_sc import CCLSCMethod
from .dg import DeepGamblersMethod
from .sat import SATMethod
from .scsf import SCSFMethod
from .selectivenet import SelectiveNetMethod

_REGISTRY: Dict[str, Callable[..., Method]] = {}


def register_method(name: str, builder: Callable[..., Method]) -> None:
    _REGISTRY[name] = builder


def method_names() -> list:
    return sorted(_REGISTRY)


def build_method(name: str, train_cfg: dict) -> Method:
    if name not in _REGISTRY:
        raise KeyError(f"unknown method {name!r}; available: {method_names()}")
    return _REGISTRY[name](train_cfg)


for _n, _b in [
    ("ce", CEMethod),
    ("dg", DeepGamblersMethod),
    ("selectivenet", SelectiveNetMethod),
    ("sat", SATMethod),
    ("scsf", SCSFMethod),
    ("ccl_sc", CCLSCMethod),
]:
    register_method(_n, _b)

__all__ = [
    "Method",
    "MethodPrediction",
    "build_method",
    "register_method",
    "method_names",
]