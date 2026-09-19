"""Training/evaluation engine (deterministic, resumable, registry-driven).

Importing this package is torch-free. Submodules that need torch (trainer,
evaluator, checkpoint, seeding) are loaded lazily so
``import scsf.engine.config`` / ``scsf.engine.planning`` can run in plan-only
launchers without initializing CUDA.
"""

from __future__ import annotations

__all__ = [
    "resolve",
    "overrides_from_cli",
    "run_name_for",
    "config_hash",
    "scientific_hash",
    "comparison_signature",
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

_LAZY = {
    "resolve": (".config", "resolve"),
    "overrides_from_cli": (".config", "overrides_from_cli"),
    "run_name_for": (".config", "run_name_for"),
    "config_hash": (".config", "config_hash"),
    "scientific_hash": (".config", "scientific_hash"),
    "comparison_signature": (".config", "comparison_signature"),
    "seed_all": (".seeding", "seed_all"),
    "capture_global_state": (".seeding", "capture_global_state"),
    "restore_global_state": (".seeding", "restore_global_state"),
    "CheckpointManager": (".checkpoint", "CheckpointManager"),
    "Trainer": (".trainer", "Trainer"),
    "evaluate_run": (".evaluator", "evaluate_run"),
    "BASE_COLUMNS": (".registry", "BASE_COLUMNS"),
    "append_rows": (".registry", "append_rows"),
    "load_registry": (".registry", "load_registry"),
}


def __getattr__(name):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr = target
    from importlib import import_module
    mod = import_module(mod_name, __name__)
    value = getattr(mod, attr)
    globals()[name] = value
    return value
