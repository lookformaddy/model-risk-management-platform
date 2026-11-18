"""Tests for the tier-driven validation suites."""

from mrmp.data import generate_dataset
from mrmp.tiering import RiskTier
from mrmp.trainer import LogisticRegression
from mrmp.validation import run_validation


def _trained():
    train = generate_dataset(n=1500, seed=42)
    model = LogisticRegression(epochs=300).fit(train.X, train.y)
    valid = generate_dataset(n=1500, seed=142)
    return model, valid


def test_champion_passes_medium_validation():
    model, valid = _trained()
    report = run_validation(model, valid, "m", "v1", RiskTier.MEDIUM, "VAL-1")
    assert report.passed, [c.name for c in report.failures]
    names = {c.name for c in report.checks}
    assert {"accuracy_floor", "auc_floor", "must_beat_baseline",
            "bootstrap_ci_width", "cross_fold_std", "demographic_parity_gap",
            "equal_opportunity_gap", "ece", "brier_score"} <= names


def test_low_tier_runs_light_suite():
    model, valid = _trained()
    report = run_validation(model, valid, "m", "v1", RiskTier.LOW, "VAL-2")
    names = {c.name for c in report.checks}
    assert "demographic_parity_gap" not in names
    assert "ece" not in names
    assert "accuracy_floor" in names


def test_critical_tier_is_strictest():
    model, valid = _trained()
    medium = run_validation(model, valid, "m", "v1", RiskTier.MEDIUM, "VAL-3")
    critical = run_validation(model, valid, "m", "v1", RiskTier.CRITICAL, "VAL-4")
    med_acc = next(c for c in medium.checks if c.name == "accuracy_floor")
    crit_acc = next(c for c in critical.checks if c.name == "accuracy_floor")
    assert crit_acc.threshold > med_acc.threshold


def test_bad_model_fails_performance_gates():
    # A model trained on shuffled labels cannot beat the baseline.
    import random

    train = generate_dataset(n=1000, seed=42)
    rng = random.Random(0)
    shuffled = list(train.y)
    rng.shuffle(shuffled)
    model = LogisticRegression(epochs=200).fit(train.X, shuffled)
    valid = generate_dataset(n=1000, seed=142)
    report = run_validation(model, valid, "m", "v1", RiskTier.MEDIUM, "VAL-5")
    assert not report.passed
    assert any(c.name in ("accuracy_floor", "must_beat_baseline") for c in report.failures)


def test_report_hash_is_stable_and_bound():
    model, valid = _trained()
    a = run_validation(model, valid, "m", "v1", RiskTier.MEDIUM, "VAL-6")
    b = run_validation(model, valid, "m", "v1", RiskTier.MEDIUM, "VAL-6")
    assert a.report_hash() == b.report_hash()
    c = run_validation(model, valid, "m", "v1", RiskTier.MEDIUM, "VAL-7")
    assert a.report_hash() != c.report_hash()
