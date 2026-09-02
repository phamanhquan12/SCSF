"""Selective scalar metrics on hand-computed mini-cases."""

import math

import numpy as np
import pytest

from scsf.metrics.selective import (
    COVERAGE_GRID_PERCENT,
    auroc_error,
    aurc,
    excess_aurc,
    optimal_aurc,
    per_class_aurc,
    risk_coverage_curve,
    selective_risk_at_coverages,
    stable_confidence_order,
    worst_class_aurc,
)

# labels/predictions so errors == [1, 0, 1, 0]; confidence descending order
# sorts indices so the error cases land near the *end* (low-confidence keep
# decision would drop them).
LAB = np.array([0, 0, 1, 1])
PRED = np.array([3, 0, 2, 1])          # errors at positions 0 and 2
CONF = np.array([0.1, 0.9, 0.4, 0.8])  # ascending? see test
IDS = np.array([10, 20, 30, 40])


def test_stable_confidence_order_desc_and_id_tiebreak():
    conf = np.array([0.5, 0.5, 0.5, 0.5])
    ids = np.array([4, 1, 3, 2])
    order = stable_confidence_order(conf, ids=ids)
    assert list(order) == [1, 3, 2, 0]  # ascending ids among equal confidence


def test_risk_coverage_curve_perfectly_sorted():
    # errors at positions 0 (1 != 0) and 1 (0 != 1); the two errors are the
    # least-confident samples so they sort last (keep-first = descending conf)
    labels = np.array([0, 1, 0, 1])
    pred = np.array([1, 0, 0, 1])
    conf = np.array([0.0, 0.1, 0.9, 1.0])
    cov, risk = risk_coverage_curve(labels, pred, conf)
    assert np.allclose(risk, [0.0, 0.0, 1 / 3, 0.5])
    assert np.allclose(cov, [0.25, 0.5, 0.75, 1.0])


def test_aurc_equals_mean_of_prefix_risks():
    cov, risk = risk_coverage_curve(LAB, PRED, CONF)
    assert math.isclose(aurc(LAB, PRED, CONF), float(np.mean(risk)))
    # fully predictive selector: errors first => mean of [1, 0.5*2... ] rebuilt
    conf_rev = np.array([1.0, 0.8, 0.4, 0.1])  # good first
    cov_r, risk_r = risk_coverage_curve(LAB, PRED, conf_rev)
    assert math.isclose(aurc(LAB, PRED, conf_rev), float(np.mean(risk_r)))


def test_auroc_error_perfect_and_anti_perfect():
    labels = np.array([0, 1, 0, 1])
    pred = np.array([1, 0, 0, 1])
    assert math.isclose(auroc_error(labels, pred, np.array([0.2, 0.4, 0.6, 0.8])), 1.0)
    assert math.isclose(auroc_error(labels, pred, np.array([0.8, 0.6, 0.4, 0.2])), 0.0)


def test_selective_risk_at_coverages_grid_formula():
    n = 100
    labels = np.zeros(n, dtype=int)
    pred = np.ones(n, dtype=int)  # all wrong
    conf = np.linspace(0, 1, n)
    rows = selective_risk_at_coverages(labels, pred, conf)
    assert [r["coverage"] for r in rows] == COVERAGE_GRID_PERCENT
    for r in rows:
        assert r["k"] == max(1, int(np.floor(r["coverage"] * n / 100.0)))
    # every risk here is 1.0 because every sample is wrong
    assert all(r["risk"] == pytest.approx(1.0) for r in rows)


def test_optimal_and_excess_aurc():
    n = 100
    assert math.isclose(optimal_aurc(0, n), 0.0)
    # one error out of 100: optimal predictor risks e/k for k<=e
    opt = optimal_aurc(1, 100)
    k = np.arange(1, 101, dtype=float)
    expected = float(np.mean(np.maximum(0.0, k - 99) / k))
    assert math.isclose(opt, expected)
    # perfect selector (one error least-confident) achieves optimal => excess 0
    labels = np.array([0] * 99 + [1])
    pred = np.array([0] * 100)
    conf = np.linspace(1.0, 0.0, 100)  # lone error (idx 99) sorts last
    assert excess_aurc(labels, pred, conf) == pytest.approx(0.0, abs=1e-9)
    # same construction with the error most-confident is maximally degraded:
    # excess must equal aurc - optimal_aurc(1, 100) = aurc - 1e-4
    conf_rev = np.linspace(0.0, 1.0, 100)
    assert excess_aurc(labels, pred, conf_rev) == pytest.approx(
        aurc(labels, pred, conf_rev) - optimal_aurc(1, 100), abs=1e-9
    )


def test_per_class_and_worst_class_aurc():
    labels = np.array([0, 0, 1, 1])
    pred = np.array([0, 0, 2, 1])  # one error per class
    conf = np.array([0.9, 0.1, 0.9, 0.1])  # good examples confident first
    pc = per_class_aurc(labels, pred, conf, ids=None, num_classes=2)
    assert worst_class_aurc(labels, pred, conf, num_classes=2) == pytest.approx(
        max(pc[0], pc[1])
    )


def test_empty_and_single_error_edge_cases():
    labels = np.array([0, 0, 0])
    pred = np.array([0, 0, 0])
    conf = np.array([0.5, 0.7, 0.3])
    assert aurc(labels, pred, conf) == pytest.approx(0.0)
    # all-correct: AUROC is undefined by contract -> NaN, never a fake number
    assert math.isnan(auroc_error(labels, pred, conf))
    # single error among correct examples: unique failure detection ordering
    labels2 = np.array([0, 0, 0, 1])
    pred2 = np.array([0, 0, 0, 0])
    assert math.isclose(auroc_error(labels2, pred2, np.array([0.2, 0.4, 0.6, 0.8])), 0.0)
    assert math.isclose(auroc_error(labels2, pred2, np.array([0.8, 0.6, 0.4, 0.2])), 1.0)