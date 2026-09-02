"""DepthFrag: depth-wise decision-fragility geometry + distilled confidence.

Geometry (``scsf.depthfrag.geometry``) computes signed per-site margins,
radii, and detached regression targets. ``scsf.depthfrag.extract`` (imported
explicitly by the CLI and tests; it depends on the engine) provides the
frozen-checkpoint extraction path with the full score-ladder evaluation and
the analytic-vs-iterative audit. ``scsf.depthfrag.iterative`` implements the
DeepFool-style walk and ``scsf.depthfrag.oracle`` the validation-fitted
diagnostic oracles. The end-to-end distillation method lives in
``scsf.methods.depthfrag``.
"""

from .geometry import (  # noqa: F401
    AGG_FUNCS,
    TARGET_KINDS,
    SiteRadiiComputer,
    aggregate_profile,
    per_example_norm,
    pool_tap,
    radii_from_site,
    site_gradients,
    target_transform,
    true_class_margin,
)
from .iterative import (  # noqa: F401
    compare_analytic_iterative,
    iterative_boundary_audit,
    relative_scale_error,
    spearman,
)

__all__ = [
    "SiteRadiiComputer",
    "true_class_margin",
    "per_example_norm",
    "radii_from_site",
    "site_gradients",
    "target_transform",
    "aggregate_profile",
    "pool_tap",
    "iterative_boundary_audit",
    "compare_analytic_iterative",
    "relative_scale_error",
    "spearman",
    "TARGET_KINDS",
    "AGG_FUNCS",
]