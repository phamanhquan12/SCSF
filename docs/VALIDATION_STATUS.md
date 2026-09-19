# Validation Status

## NEXT5 (this assignment)

Protocol: `docs/NEXT5_PROTOCOL.md`. Implementation is on `quan`. Historical
methods (SAGE-v1/v2/v3, TopK, DepthFrag, RiskFlow, scsf_correctness, r3/dtr/cbr)
are preserved.

### Local CPU tests (Windows, CPython 3.11.9, torch 2.12.0+cpu)

Executed from `d:\OPD\scsf` with `PYTHONPATH=.`.

* NEXT5 / launcher / planning / metrics / config: **61 passed**
  (`tests/test_next5_methods.py`, `test_next5_folds.py`, `test_next5_launcher.py`,
  `test_launcher.py`, `test_planning_variant.py`, `test_metrics.py`,
  `test_config_registry.py`).
* Full `pytest tests --ignore=tests/test_analyze_gate.py`:
  **312 passed, 52 failed, 5 skipped**.
* Collection error (pre-existing, ignored): `tests/test_analyze_gate.py`
  (`scripts` is not a Python package).
* The 52 failures were **not** in NEXT5 tests. They are:
  missing `timm` (ConvNeXt/DeiT loaders), missing local CIFAR files,
  and historical authored-but-previously-unrun tests (`test_rc_training`,
  `test_scsf_correctness`, `test_r3_scsf`, `test_dtr_scsf`, `test_cbr_scsf`,
  plus some SAGE/TopK/DepthFrag/RiskFlow cells). Those files were not
  modified for NEXT5.
* GPU smoke, DAG smoke, and the 22-job `next5_pilot` are executed on the
  authorized server after push, not on this authoring machine.

Do not treat this file as evidence that the pilot finished.

## Prior commits (static-only era; superseded for NEXT5)

* `python3 -m py_compile` on launcher/planning/tests: executed historically.
* Full pytest / GPU smoke / `--execute` were **not** run in the code-only
  handoff (`779bb46` and earlier). Those restrictions do not apply to NEXT5.
