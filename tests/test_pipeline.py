"""Tests for the end-to-end governed pipeline."""

from mrmp.config import Config
from mrmp.pipeline import run_pipeline


def test_full_pipeline_happy_path():
    result = run_pipeline(Config(seed=42, n_train=2000, n_validate=2000))
    assert result.validation.passed
    assert result.tier.tier.value == "CRITICAL"
    assert result.stage == "PRODUCTION"
    assert len(result.approvals) == 2
    assert result.audit_ok
    assert result.audit_entries >= 8
    assert result.champion_accuracy > 0.70
    assert result.champion_auc > 0.70


def test_pipeline_challenger_rejected_and_drift_triggers_revalidation():
    result = run_pipeline(Config(seed=42, n_train=2000, n_validate=2000))
    assert result.challenger.decision == "REJECT"
    assert result.challenger.overall_delta > 0
    assert result.drift.breached
    # Drift breach alone triggers revalidation; accuracy itself barely moves.
    assert result.decay_level == "OK"
    assert result.revalidation_due == "2027-02-16"  # 90 days after 2026-11-18


def test_pipeline_is_deterministic():
    a = run_pipeline(Config(seed=42, n_train=2000, n_validate=2000))
    b = run_pipeline(Config(seed=42, n_train=2000, n_validate=2000))
    assert a.champion_accuracy == b.champion_accuracy
    assert a.challenger.to_dict() == b.challenger.to_dict()
    assert a.validation.report_hash() == b.validation.report_hash()
