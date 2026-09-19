"""Torch-free plan identities (suites, method ids, variants).

Cell identity (run_name, scientific_hash, comparison_signature) is computed
by ``scsf.engine.config`` — the same resolver the trainer uses. This module
does not keep a drifting hash replica. Importing it must not load torch or
numpy (engine/__init__.py is lazy; config.py is stdlib + yaml).
"""

from __future__ import annotations

import os

from scsf.engine.config import (  # noqa: F401
    comparison_signature,
    config_hash,
    resolve,
    run_name_for,
    scientific_hash,
)

# method_id -> (method_name, variant-or-None). Suite rows resolve through this
# and never through a silent guess.
METHOD_VARIANTS = {
    "ce":                               ("ce", None),
    "scsf_correctness":                 ("scsf_correctness", None),
    "r3_scsf":                          ("r3_scsf", None),
    "dtr_scsf":                         ("dtr_scsf", None),
    "cbr_scsf":                         ("cbr_scsf", None),
    "sage_ds_v2":                       ("sage_ds_v2", None),
    "sage_topk_v2_fixedk2_pool":        ("sage_topk", "v2_fixedk2_pool"),
    "sage_topk_v2_fixedk2_pool_conv":   ("sage_topk", "v2_fixedk2_pool_conv"),
    "fmfp_reference":                   ("fmfp_reference", None),
    "crossfit_failure":                 ("crossfit_failure", None),
    "candidate_verify":                 ("candidate_verify", None),
    "intervention_rank":                ("intervention_rank", None),
    "neighbor_distill":                 ("neighbor_distill", None),
    "rank_sharpness":                   ("rank_sharpness", None),
    "ce.fold_teacher_a":                ("ce", "fold_teacher_a"),
    "ce.fold_teacher_b":                ("ce", "fold_teacher_b"),
    # Ablation ladder ids: never silently part of `all`.
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

_PRIMARY = {
    "review": ["ce", "scsf_correctness", "r3_scsf", "dtr_scsf", "cbr_scsf"],
    "sage": ["ce", "sage_ds_v2", "sage_topk_v2_fixedk2_pool",
             "sage_topk_v2_fixedk2_pool_conv"],
    "next5_pilot": [
        "ce",
        "ce.fold_teacher_a",
        "ce.fold_teacher_b",
        "scsf_correctness",
        "sage_ds_v2",
        "fmfp_reference",
        "candidate_verify",
        "intervention_rank",
        "rank_sharpness",
        "crossfit_failure",
        "neighbor_distill",
    ],
}
DEFAULT_SUITES = {
    "review": _PRIMARY["review"],
    "sage": _PRIMARY["sage"],
    "all": sorted(set(_PRIMARY["review"]).union(_PRIMARY["sage"])),
    "next5_pilot": _PRIMARY["next5_pilot"],
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
    "next5_pilot": "next5_pilot", "next5": "next5_pilot",
    "r3_ablations": "r3_ablations", "dtr_ablations": "dtr_ablations",
    "cbr_ablations": "cbr_ablations",
}
PLAN_ARTIFACT_ROOT = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "plans")

# Fold teachers exclude this fold id (train on the complement).
FOLD_TEACHER_EXCLUDE = {
    "fold_teacher_a": 0,
    "fold_teacher_b": 1,
}


def _variant_of(pid: str) -> tuple[str, str | None]:
    if pid not in METHOD_VARIANTS:
        raise KeyError(f"method id {pid!r} not registered "
                       "(refusing to guess a class/variant)")
    return METHOD_VARIANTS[pid]


def _rows_for_method(pid: str) -> tuple[str, str | None]:
    return _variant_of(pid)
