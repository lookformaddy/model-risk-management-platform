"""Tests for rule-based risk tiering."""

import pytest

from mrmp.tiering import ModelAttributes, RiskTier, assign_tier


def test_critical_for_max_impact_and_autonomy():
    tier = assign_tier(ModelAttributes(3, 2, 3, 3))
    assert tier.tier == RiskTier.CRITICAL
    assert tier.revalidation_days == 90
    assert tier.required_approvers == ["model_validator", "risk_officer"]


def test_high_for_strong_score():
    tier = assign_tier(ModelAttributes(2, 2, 2, 2))
    assert tier.tier == RiskTier.HIGH
    assert tier.revalidation_days == 180


def test_medium_and_low():
    assert assign_tier(ModelAttributes(1, 1, 1, 1)).tier == RiskTier.MEDIUM
    low = assign_tier(ModelAttributes(0, 0, 0, 0))
    assert low.tier == RiskTier.LOW
    assert low.required_approvers == ["model_owner"]
    assert low.revalidation_days == 730


def test_auto_escalation_rule():
    # Score alone would be HIGH (0.35*3 + 0.20*0 + 0.25*3 + 0.20*0 = 1.8),
    # but max impact + full autonomy escalates to CRITICAL.
    tier = assign_tier(ModelAttributes(3, 0, 3, 0))
    assert tier.tier == RiskTier.CRITICAL
    assert any("auto-escalated" in r for r in tier.reasons)


def test_reasons_name_the_drivers():
    tier = assign_tier(ModelAttributes(3, 3, 1, 1, affects_protected_class=True))
    assert any("decision impact" in r for r in tier.reasons)
    assert any("protected class" in r for r in tier.reasons)


def test_invalid_rating_rejected():
    with pytest.raises(ValueError):
        assign_tier(ModelAttributes(4, 2, 2, 2))
    with pytest.raises(ValueError):
        assign_tier(ModelAttributes(2, -1, 2, 2))
