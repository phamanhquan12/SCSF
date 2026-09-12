"""Torch-free plan replica (spec §10.3 #1; commit 6).

Single source of truth for the portable launcher's **plan** layer.  The engine
resolver (``scsf/engine/config.py``) must stay torch-free too, but it carries
the whole train/eval recipe; this module holds only the cell-identity layer —
method ids, exact suite ids, variants, run naming, config hash — and nothing
else.  It is deliberately **torch-free by construction** (pure stdlib +
``yaml``), so the launcher's ``--dry-run`` plan path never triggers a CUDA
init, a dataset download, a process launch, or a torch import (§10.3 #1).

Parity (run name + config hash) between this replica and the engine resolver
is locked by ``tests/test_launcher.py::test_run_name_config_hash_parity`` and
``tests/test_plan_engine_hash_parity`` (commit-7 GPU-gate only; the sha256 for
config_hash is reproduced from the engine's own ``config_hash`` in
``scsf/engine/registry.py``).
"""

from __future__ import annotations

import hashlib
import json
import os

# method_id -> (method_name, variant-or-None).  Suite rows (§10.2) resolve
# through this and never through a silent guess: an id absent here is a hard
# capability error, printed before a single job is scheduled (§10.3 #2).
METHOD_VARIANTS = {
    "ce":                               ("ce", None),
    "scsf_correctness":                 ("scsf_correctness", None),
    "r3_scsf":                          ("r3_scsf", None),
    "dtr_scsf":                         ("dtr_scsf", None),
    "cbr_scsf":                         ("cbr_scsf", None),
    "sage_ds_v2":                       ("sage_ds_v2", None),
    "sage_topk_v2_fixedk2_pool":        ("sage_topk", "v2_fixedk2_pool"),
    "sage_topk_v2_fixedk2_pool_conv":   ("sage_topk", "v2_fixedk2_pool_conv"),
    # Ablation ladder ids (§10.2): never silently part of `all`.
    "r3_full":              ("r3_scsf", "r3_full"),
    "r3_uniform_rank":      ("r3_scsf", "r3_uniform_rank"),
    "r3_rc_rank":           ("r3_scsf", "r3_rc_rank"),
    "r3_hard_ce":           ("r3_scsf", "r3_hard_ce"),
    "r3_fixed":             ("r3_scsf", "r3_fixed"),
    "r3_shuffled":          ("r3_scsf", "r3_shuffled"),
    "r3_detach_feat":       ("r3_scsf", "r3_detach_feat"),
    "dtr_full":             ("dtr_scsf", "dtr_full"),
    "dtr_aux_ce":           ("dtr_scsf", "dtr_aux_ce"),
    "dtr_state_norepair":   ("dtr_scsf", "dtr_state_norepair"),
    "dtr_cond_norc":        ("dtr_scsf", "dtr_cond_norc"),
    "dtr_rev_kd_all":       ("dtr_scsf", "dtr_rev_kd_all"),
    "dtr_rev_kd_correct_probe": ("dtr_scsf", "dtr_rev_kd_correct_probe"),
    "dtr_rand_states":      ("dtr_scsf", "dtr_rand_states"),
    "cbr_full":             ("cbr_scsf", "cbr_full"),
    "cbr_global_multicov":  ("cbr_scsf", "cbr_global_multicov"),
    "cbr_conf_nfloor":      ("cbr_scsf", "cbr_conf_nfloor"),
    "cbr_floor_nconf":      ("cbr_scsf", "cbr_floor_nconf"),
    "cbr_true_class_group": ("cbr_scsf", "cbr_true_class_group"),
    "cbr_single_cov":       ("cbr_scsf", "cbr_single_cov"),
    "cbr_groupdro":         ("cbr_scsf", "cbr_groupdro"),
}

# §10.2 exact primary suites.  `_PRIMARY` lists are the lock; `all` is the
# deduplicated union review ∪ sage = 8 ids and is derived *after* the primary
# literals exist (building it inside the same literal would reference
# DEFAULT_SUITES while it is still being constructed -> NameError).
_PRIMARY = {
    "review": ["ce", "scsf_correctness", "r3_scsf", "dtr_scsf", "cbr_scsf"],
    "sage": ["ce", "sage_ds_v2", "sage_topk_v2_fixedk2_pool",
             "sage_topk_v2_fixedk2_pool_conv"],
}
DEFAULT_SUITES = {
    "review": _PRIMARY["review"],
    "sage": _PRIMARY["sage"],
    "all": sorted(set(_PRIMARY["review"]).union(_PRIMARY["sage"])),
    "r3_ablations": [i for i, _ in METHOD_VARIANTS.items()
                     if METHOD_VARIANTS[i][0] == "r3_scsf" and
                     METHOD_VARIANTS[i][1] not in (None, "r3_full")],
    "dtr_ablations": [i for i, _ in METHOD_VARIANTS.items()
                      if METHOD_VARIANTS[i][0] == "dtr_scsf" and
                      METHOD_VARIANTS[i][1] not in (None, "dtr_full")],
    "cbr_ablations": [i for i, _ in METHOD_VARIANTS.items()
                      if METHOD_VARIANTS[i][0] == "cbr_scsf" and
                      METHOD_VARIANTS[i][1] not in (None, "cbr_full")],
}

METHODS = sorted({m for m, _ in METHOD_VARIANTS.values()})
SUITE_ALIASES = {
    "review": "review", "r3": "review", "dtr": "review", "cbr": "review",
    "ce": "review",
    "sage": "sage", "topk": "sage", "sage_topk": "sage",
    "all": "all", "full": "all",
    "r3_ablations": "r3_ablations", "dtr_ablations": "dtr_ablations",
    "cbr_ablations": "cbr_ablations",
}
PLAN_ARTIFACT_ROOT = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "plans")


def _variant_of(pid: str) -> tuple[str, str | None]:
    if pid not in METHOD_VARIANTS:
        raise KeyError(f"method id {pid!r} not registered "
                       "(refusing to guess a class/variant)")
    return METHOD_VARIANTS[pid]


def _rows_for_method(pid: str) -> tuple[str, str | None]:
    return _variant_of(pid)


def run_name_for(cfg: dict) -> str:
    """Portable run-name replica (§10.4 #§10.3): dataset-backbone-method.

    Matches the engine's ``run_name_for`` cell identity: the variant string
    (when present) is folded in so pool vs pool_conv and every ablation ladder
    cell gets its own non-colliding run name.
    """
    name = f"{cfg['dataset']}-{cfg['backbone']}-{cfg['method_name']}"
    if cfg.get("variant"):
        name += f".{cfg['variant']}"
    return f"{name}-r{cfg['recipe']}-s{cfg['seed']}"


def config_hash(cfg: dict) -> str:
    payload = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
