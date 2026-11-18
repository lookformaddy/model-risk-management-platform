"""Tests for metrics and statistical screens."""

import math

from mrmp.metrics import (
    accuracy,
    bootstrap_accuracy_ci,
    brier_score,
    expected_calibration_error,
    kfold_variance,
    ks_statistic,
    mcnemar_pvalue,
    psi,
    roc_auc,
)


def test_accuracy_and_auc_perfect_and_random():
    assert accuracy([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert roc_auc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert roc_auc([1, 0, 1, 0], [0.1, 0.9, 0.2, 0.8]) == 0.0
    assert roc_auc([1, 1, 1], [0.5, 0.5, 0.5]) == 0.5  # single class


def test_brier_and_ece_well_calibrated():
    # Perfectly calibrated probabilities: 0.5 for a 50/50 outcome.
    y = [1, 0] * 50
    probs = [0.5] * 100
    assert brier_score(y, probs) == 0.25
    assert expected_calibration_error(y, probs) < 0.01


def test_ece_detects_overconfidence():
    y = [1, 0] * 50
    probs = [0.99 if t == 1 else 0.01 for t in y]
    # Confident and always wrong is maximally miscalibrated.
    wrong = [1, 0] * 50
    probs_wrong = [0.01 if t == 1 else 0.99 for t in wrong]
    assert expected_calibration_error(wrong, probs_wrong) > 0.9


def test_bootstrap_ci_contains_accuracy():
    y = [1, 0, 1, 1, 0, 0, 1, 0] * 25
    preds = [1, 0, 1, 0, 0, 1, 1, 0] * 25
    lo, hi = bootstrap_accuracy_ci(y, preds, n_boot=200, seed=1)
    acc = accuracy(y, preds)
    assert lo <= acc <= hi
    assert hi - lo < 0.3


def test_kfold_variance_stable_model():
    y = [1, 0] * 100
    preds = [1, 0] * 100
    mean, std = kfold_variance(y, preds, k=5, seed=1)
    assert mean == 1.0 and std == 0.0


def test_mcnemar_identical_models():
    y = [1, 0, 1, 0] * 25
    res = mcnemar_pvalue(y, y, y)
    assert res["b"] == 0 and res["c"] == 0 and res["p_value"] == 1.0


def test_mcnemar_clear_winner():
    # B is right exactly where A is wrong, 40 times; A never beats B.
    y = [1] * 40 + [0] * 40
    a = [0] * 40 + [1] * 40  # always wrong
    b = [1] * 40 + [0] * 40  # always right
    res = mcnemar_pvalue(y, a, b)
    assert res["p_value"] < 0.001


def test_psi_zero_for_identical_samples():
    sample = [float(i % 10) for i in range(1000)]
    assert psi(sample, sample) < 1e-6


def test_psi_large_for_shifted_samples():
    ref = [float(i % 10) for i in range(1000)]
    shifted = [v + 5.0 for v in ref]
    assert psi(ref, shifted) > 0.5


def test_ks_zero_identical_large_shifted():
    ref = [1.0, 2.0, 3.0] * 100
    assert ks_statistic(ref, ref) == 0.0
    assert ks_statistic(ref, [10.0, 11.0, 12.0] * 100) == 1.0
