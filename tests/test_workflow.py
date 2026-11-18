"""Tests for the staged approval workflow (fail-closed gates)."""

from datetime import date

import pytest

from mrmp.data import generate_dataset
from mrmp.registry import ModelVersion, ReproducibilityManifest, Stage
from mrmp.tiering import ModelAttributes, RiskTier, assign_tier
from mrmp.trainer import LogisticRegression
from mrmp.validation import run_validation
from mrmp.workflow import Approval, Workflow, WorkflowError


def _setup(tier_attrs=(3, 2, 2, 3), passing=True):
    train = generate_dataset(n=1500, seed=42)
    model = LogisticRegression(epochs=300).fit(train.X, train.y)
    valid = generate_dataset(n=1500, seed=142)
    tier = assign_tier(ModelAttributes(*tier_attrs, affects_protected_class=True))
    version = ModelVersion(
        model_id="m", version="v1", owner="o", purpose="p",
        data_lineage=["synthetic"], training_config={},
        manifest=ReproducibilityManifest("d", "c", "w"),
    )
    if not passing:
        # Train on shuffled labels so validation fails.
        import random

        rng = random.Random(0)
        shuffled = list(train.y)
        rng.shuffle(shuffled)
        model = LogisticRegression(epochs=200).fit(train.X, shuffled)
    report = run_validation(model, valid, "m", "v1", tier.tier, "VAL-1")
    workflow = Workflow(version=version, tier=tier.tier, secret="test-secret")
    return workflow, report, tier


def test_happy_path_to_production():
    workflow, report, tier = _setup()
    assert report.passed
    workflow.submit_for_validation(report)
    assert workflow.version.stage == Stage.VALIDATION
    today = date(2026, 11, 18)
    workflow.approve("ana@example.com", "model_validator", report, today)
    assert workflow.version.stage == Stage.VALIDATION  # still needs risk officer
    workflow.approve("omar@example.com", "risk_officer", report, today)
    assert workflow.version.stage == Stage.APPROVED
    workflow.promote_to_production(on=today)
    assert workflow.version.stage == Stage.PRODUCTION


def test_approve_without_passing_report_is_blocked():
    workflow, report, _ = _setup(passing=False)
    assert not report.passed
    workflow.submit_for_validation(report)
    with pytest.raises(WorkflowError, match="FAILED"):
        workflow.approve("ana@example.com", "model_validator", report, date(2026, 11, 18))
    assert workflow.version.stage == Stage.VALIDATION


def test_approve_without_validation_stage_is_blocked():
    workflow, report, _ = _setup()
    with pytest.raises(WorkflowError):
        workflow.approve("ana@example.com", "model_validator", report, date(2026, 11, 18))


def test_promote_without_approval_is_blocked():
    workflow, report, _ = _setup()
    workflow.submit_for_validation(report)
    with pytest.raises(WorkflowError):
        workflow.promote_to_production(on=date(2026, 11, 18))


def test_wrong_role_and_duplicate_approver_blocked():
    workflow, report, _ = _setup()
    workflow.submit_for_validation(report)
    with pytest.raises(WorkflowError):
        workflow.approve("mallory@example.com", "model_owner", report, date(2026, 11, 18))
    workflow.approve("ana@example.com", "model_validator", report, date(2026, 11, 18))
    with pytest.raises(WorkflowError):
        workflow.approve("ana@example.com", "risk_officer", report, date(2026, 11, 18))


def test_expired_approval_blocks_promotion():
    workflow, report, _ = _setup()
    workflow.submit_for_validation(report)
    # Approvals signed 2026-01-01 with 30-day validity expire before Nov.
    workflow.approve("ana@example.com", "model_validator", report, date(2026, 1, 1), validity_days=30)
    workflow.approve("omar@example.com", "risk_officer", report, date(2026, 1, 1), validity_days=30)
    assert workflow.version.stage == Stage.APPROVED
    with pytest.raises(WorkflowError, match="expired"):
        workflow.promote_to_production(on=date(2026, 11, 18))


def test_illegal_transition_blocked():
    workflow, _, _ = _setup()
    with pytest.raises(WorkflowError):
        workflow.transition(Stage.PRODUCTION)


def test_approval_signature_binds_to_report():
    approval = Approval("ana", "model_validator", "VAL-1", "abc", "m", "v1",
                        "2026-11-18", "2027-05-17")
    approval.sign("secret")
    assert approval.verify("secret")
    assert not approval.verify("wrong-secret")
    tampered = Approval("ana", "model_validator", "VAL-2", "abc", "m", "v1",
                        "2026-11-18", "2027-05-17", signature=approval.signature)
    assert not tampered.verify("secret")
