"""Training/evaluation engine (deterministic, resumable, registry-driven)."""

from .config import resolve, overrides_from_cli, run_name_for  # noqa: F401
from .seeding import seed_all, capture_global_state, restore_global_state  # noqa: F401
from .checkpoint import CheckpointManager  # noqa: F401
from .trainer import Trainer  # noqa: F401
from .evaluator import evaluate_run  # noqa: F401
from .registry import BASE_COLUMNS, append_rows, load_registry  # noqa: F401

__all__ = [
    "resolve",
    "overrides_from_cli",
    "run_name_for",
    "seed_all",
    "capture_global_state",
    "restore_global_state",
    "CheckpointManager",
    "Trainer",
    "evaluate_run",
    "BASE_COLUMNS",
    "append_rows",
    "load_registry",
]