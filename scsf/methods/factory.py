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
from .sage_ds import SageDSMethod
from .sage_ds_v2 import SageDSV2Method
from .depthfrag import DepthFragMethod
from .riskflow import RiskFlowMethod

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
    ("sage_ds", SageDSMethod),
    # SAGE-DS topology/ablation aliases: all resolve to the same class driven
    # by the method config (topology, safety, utility, supervision_scale).
    ("sage_ds_fixed_late", SageDSMethod),
    ("sage_ds_all_equal", SageDSMethod),
    ("sage_ds_learned_dense", SageDSMethod),
    ("sage_ds_sparse", SageDSMethod),
    ("sage_ds_ss0_1", SageDSMethod),
    ("sage_ds_ss0_3", SageDSMethod),
    ("sage_ds_ss1_0", SageDSMethod),
    ("sage_ds_ss3_0", SageDSMethod),
    # SAGE-V2: distinct class, bilevel-utility + per-site CE-safe projection.
    # Separate alias; sage_ds (v1) is preserved unchanged (protocol doc).
    ("sage_ds_v2", SageDSV2Method),
    # DepthFrag: all aliases resolve to the same class driven by the method
    # config (ablation ladder + frozen control + sensitivity control).
    ("depthfrag", DepthFragMethod),
    ("depthfrag_terminal_margin", DepthFragMethod),
    ("depthfrag_terminal", DepthFragMethod),
    ("depthfrag_intermediate", DepthFragMethod),
    ("depthfrag_raw", DepthFragMethod),
    ("depthfrag_frozen", DepthFragMethod),
    ("depthfrag_clip", DepthFragMethod),
    ("depthfrag_warm25", DepthFragMethod),
    # RiskFlow: a single class driven by the mode config (ablation ladder +
    # frozen control + hard-channel-only control).
    ("riskflow", RiskFlowMethod),
    ("riskflow_concat", RiskFlowMethod),
    ("riskflow_heads", RiskFlowMethod),
    ("riskflow_cum", RiskFlowMethod),
    ("riskflow_resid", RiskFlowMethod),
    ("riskflow_frozen", RiskFlowMethod),
    ("riskflow_hard", RiskFlowMethod),
]:
    register_method(_n, _b)

__all__ = [
    "Method",
    "MethodPrediction",
    "build_method",
    "register_method",
    "method_names",
]