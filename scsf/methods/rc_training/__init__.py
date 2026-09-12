"""Shared review-aligned RC-training primitives for the new family.

Baseline's primary confidence, gradient rules and weight schedules are shared
by ``scsf_correctness``, ``r3_scsf``, ``dtr_scsf`` and ``cbr_scsf`` through this
package; DTR changes only its calibrator output width to four for the main
comparison. Everything here is additive and leaves the legacy classes / state
dicts untouched.
"""

from __future__ import annotations

from . import calibration  # noqa: F401
from . import losses  # noqa: F401
from . import rc_weights  # noqa: F401
from . import schedules  # noqa: F401
from . import softquantile  # noqa: F401
from . import state  # noqa: F401
from . import views  # noqa: F401