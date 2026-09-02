"""RiskFlow diagnostics and CKA re-exports.

Submodules: :mod:`scsf.riskflow.cka` (linear CKA + correlation) and
:mod:`scsf.riskflow.diagnostics` (per-depth trace export, redundancy report,
trajectory plots and the fixed category-assignment rule).
"""

from .cka import (  # noqa: F401
    cross_depth_correlation,
    off_diagonal_mean,
    pairwise_linear_cka,
)
from .diagnostics import (  # noqa: F401
    CATEGORIES,
    assign_category,
    export_trace,
    redundancy_report,
    save_trajectory_plots,
)
from .overhead import (  # noqa: F401
    added_macs,
    deployment_params,
    measure_latency_memory,
    report_overhead,
)

__all__ = [
    "cross_depth_correlation",
    "off_diagonal_mean",
    "pairwise_linear_cka",
    "CATEGORIES",
    "assign_category",
    "export_trace",
    "redundancy_report",
    "save_trajectory_plots",
    "added_macs",
    "deployment_params",
    "measure_latency_memory",
    "report_overhead",
]
