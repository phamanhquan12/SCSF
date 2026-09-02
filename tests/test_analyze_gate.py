"""Regression tests for the exact CIFAR passing-gate decision."""

from scripts.analyze_gate import (
    BASELINE_SCORES,
    CANDIDATE_SCORES,
    LOCKED_SEEDS,
    aggregate_by_cell,
    canonical_method,
    gate,
)
from scripts.gen_manifest import (
    _ccl_vgg_supplement_rows,
    _gate_rows,
    _reference_anchor_rows,
    _reference_ours_rows,
)


def _row(dataset, method, score, seed, aurc, acc=0.9, mode=None, commit="abc123"):
    registry_method = "scsf" if method.startswith("scsf_") else method
    if method == "scsf_e2e":
        run_method = "scsf.e2e"
    elif method == "scsf_posthoc":
        run_method = "scsf"
    else:
        run_method = method
    return {
        "run_dir": f"results/{dataset}-vgg16_bn-{run_method}-rbackbone_transfer-s{seed}",
        "dataset": dataset,
        "backbone": "vgg16_bn",
        "method_name": registry_method,
        "score": score,
        "seed": str(seed),
        "recipe": "backbone_transfer",
        "split": "test",
        "style": mode or "",
        "commit": commit,
        "complete": "1",
        "acc": str(acc),
        "err": str(1.0 - acc),
        "aurc": str(aurc),
        "auroc_error": "0.9",
        "excess_aurc": str(aurc / 2),
        "mean_class_aurc": str(aurc),
        "worst_class_aurc": str(aurc * 2),
    }


def _complete_registry(candidate_acc=0.899):
    rows = []
    for dataset in ("cifar10", "cifar100"):
        for method, score in BASELINE_SCORES.items():
            baseline_aurc = 0.10 if method == "ce" else 0.11
            for seed in LOCKED_SEEDS:
                rows.append(_row(dataset, method, score, seed, baseline_aurc))
        for index, (method, score) in enumerate(CANDIDATE_SCORES.items()):
            for seed in LOCKED_SEEDS:
                rows.append(_row(dataset, method, score, seed,
                                 0.09 - index * 0.01, acc=candidate_acc))
    return rows


def _candidate(result, dataset, method):
    return next(item for item in result["decisions"][dataset]["candidates"]
                if item["method"] == method)


def test_method_specific_scores_allow_all_candidates_to_pass():
    result = gate(aggregate_by_cell(_complete_registry()))
    assert result["decisions"]["cifar10"]["result"] == "COMPLETE"
    assert _candidate(result, "cifar10", "sage_ds")["score"] == "msp"
    assert _candidate(result, "cifar10", "depthfrag")["score"] == "depthfrag"
    assert _candidate(result, "cifar10", "riskflow")["score"] == "riskflow"
    assert set(result["pass"]) == set(CANDIDATE_SCORES)
    assert all(status["passed"] for status in result["pass"].values())


def test_gate_blocks_until_every_required_baseline_has_five_seeds():
    rows = [row for row in _complete_registry()
            if not (row["dataset"] == "cifar100"
                    and row["method_name"] == "ccl_sc"
                    and row["seed"] == "31")]
    result = gate(aggregate_by_cell(rows))
    decision = result["decisions"]["cifar100"]
    assert decision["result"] == "INCOMPLETE_BASELINES"
    assert "ccl_sc" in decision["incomplete_baselines"]
    assert result["pass"] == {}


def test_gate_blocks_an_incomplete_candidate_cell():
    rows = [row for row in _complete_registry()
            if not (row["dataset"] == "cifar10"
                    and row["method_name"] == "riskflow"
                    and row["seed"] == "31")]
    result = gate(aggregate_by_cell(rows))
    assert result["decisions"]["cifar10"]["result"] == "INCOMPLETE_CANDIDATES"
    assert _candidate(result, "cifar10", "riskflow")["result"] == "INCOMPLETE"
    assert "riskflow" not in result["pass"]


def test_accuracy_safety_uses_fractions_and_mean_drop_limit():
    per_cell_fail = gate(aggregate_by_cell(_complete_registry(candidate_acc=0.894)))
    assert not per_cell_fail["pass"]["sage_ds"]["passed"]
    assert not _candidate(per_cell_fail, "cifar10", "sage_ds")["acc_ok"]

    mean_fail = gate(aggregate_by_cell(_complete_registry(candidate_acc=0.897)))
    assert _candidate(mean_fail, "cifar10", "sage_ds")["acc_ok"]
    assert not mean_fail["pass"]["sage_ds"]["mean_acc_drop_ok"]
    assert not mean_fail["pass"]["sage_ds"]["passed"]


def test_scsf_modes_are_separate_even_for_historical_blank_style_rows():
    post = _row("cifar10", "scsf_posthoc", "scsf_conf", 13, 0.1)
    e2e = _row("cifar10", "scsf_e2e", "scsf_conf", 13, 0.09)
    assert canonical_method(post) == "scsf_posthoc"
    assert canonical_method(e2e) == "scsf_e2e"
    agg = aggregate_by_cell([post, e2e])
    assert len(agg) == 2


def test_missing_commit_requires_explicit_frozen_source_commit():
    rows = [{**row, "commit": ""} for row in _complete_registry()]
    blocked = gate(aggregate_by_cell(rows))
    assert blocked["decisions"]["cifar10"]["result"] == "INCOMPLETE_BASELINES"
    accepted = gate(aggregate_by_cell(rows), source_commit="f4fa743")
    assert accepted["decisions"]["cifar10"]["result"] == "COMPLETE"


def test_gate_manifest_contains_matched_vgg_ccl_sc_runs():
    rows = list(_gate_rows())
    ccl_vgg = [row for row in rows if row[3] == "vgg16_bn" and row[4] == "ccl_sc"]
    assert len(ccl_vgg) == 2 * len(LOCKED_SEEDS)
    assert {(row[2], row[6]) for row in ccl_vgg} == {
        (dataset, seed)
        for dataset in ("cifar10", "cifar100")
        for seed in LOCKED_SEEDS
    }

    supplement = list(_ccl_vgg_supplement_rows())
    assert len(supplement) == len(ccl_vgg)
    assert {(row[2], row[3], row[4], row[6]) for row in supplement} == {
        (dataset, "vgg16_bn", "ccl_sc", seed)
        for dataset in ("cifar10", "cifar100")
        for seed in LOCKED_SEEDS
    }


def test_reference_manifest_prioritizes_ours_and_uses_reference_recipe():
    rows = list(_reference_ours_rows())
    assert len(rows) == 30
    assert {row[4] for row in rows} == {"sage_ds", "depthfrag", "riskflow"}
    assert {row[2] for row in rows} == {"cifar10", "cifar100"}
    assert {row[3] for row in rows} == {"vgg16_bn"}
    assert all("recipe=ccl_sc_reference" in row[8] for row in rows)
    assert len([row for row in rows if row[1] == "R0"]) == 6
    assert {row[6] for row in rows if row[1] == "R0"} == {13}

    anchors = list(_reference_anchor_rows())
    assert len(anchors) == 20
    assert {row[4] for row in anchors} == {"ce", "ccl_sc"}
    assert all("recipe=ccl_sc_reference" in row[8] for row in anchors)
