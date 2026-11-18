"""Rule-based model risk tiering (SR 11-7 style).

A model is tiered from four scored attributes — decision impact, data
sensitivity, autonomy, and exposure — each rated 0–3. The weighted score
maps to CRITICAL / HIGH / MEDIUM / LOW, and the tier drives how deep the
validation suite goes, who must approve, and how often the model is
revalidated. The rules are explicit and auditable: every assignment
carries its reasons, never just a number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

WEIGHTS = {"decision_impact": 0.35, "data_sensitivity": 0.20, "autonomy": 0.25, "exposure": 0.20}


class RiskTier(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class ModelAttributes:
    """Risk attributes of a model, each rated 0 (none) to 3 (severe)."""

    decision_impact: int  # 0 informational … 3 decides money / access for a person
    data_sensitivity: int  # 0 public/synthetic … 3 regulated personal data
    autonomy: int  # 0 human decides … 3 fully automated decision
    exposure: int  # 0 internal pilot … 3 all customers / all decisions
    affects_protected_class: bool = False
    uses_external_data: bool = False

    def validate(self) -> None:
        for name in ("decision_impact", "data_sensitivity", "autonomy", "exposure"):
            v = getattr(self, name)
            if not isinstance(v, int) or v < 0 or v > 3:
                raise ValueError(f"{name} must be an int in 0..3, got {v!r}")


@dataclass
class TierAssignment:
    tier: RiskTier
    score: float  # weighted score in 0..3
    reasons: list[str] = field(default_factory=list)
    validation_depth: str = ""
    required_approvers: list[str] = field(default_factory=list)
    revalidation_days: int = 365

    def to_dict(self) -> dict:
        return {
            "tier": self.tier.value,
            "score": round(self.score, 3),
            "reasons": self.reasons,
            "validation_depth": self.validation_depth,
            "required_approvers": self.required_approvers,
            "revalidation_days": self.revalidation_days,
        }


DEPTH = {
    RiskTier.CRITICAL: (
        "full",
        ["model_validator", "risk_officer"],
        90,
        "Full suite: performance, stability, data quality, fairness, outcome "
        "analysis, challenger comparison, independent review, quarterly revalidation.",
    ),
    RiskTier.HIGH: (
        "standard",
        ["model_validator", "risk_officer"],
        180,
        "Standard suite: performance, stability, data quality, fairness, "
        "outcome analysis; semi-annual revalidation.",
    ),
    RiskTier.MEDIUM: (
        "standard",
        ["model_validator"],
        365,
        "Standard suite with relaxed floors; annual revalidation.",
    ),
    RiskTier.LOW: (
        "light",
        ["model_owner"],
        730,
        "Light suite: performance + data quality only; biennial revalidation.",
    ),
}


def assign_tier(attrs: ModelAttributes) -> TierAssignment:
    """Assign a risk tier from model attributes, with reasons."""
    attrs.validate()
    score = sum(WEIGHTS[k] * getattr(attrs, k) for k in WEIGHTS)
    reasons: list[str] = []
    if attrs.decision_impact == 3:
        reasons.append("decision impact 3: model directly decides an outcome for a person")
    if attrs.data_sensitivity == 3:
        reasons.append("data sensitivity 3: regulated personal data in scope")
    if attrs.autonomy == 3:
        reasons.append("autonomy 3: fully automated decision, no human in the loop")
    if attrs.exposure == 3:
        reasons.append("exposure 3: serves all customers / all decisions")
    if attrs.affects_protected_class:
        reasons.append("affects a protected class: fairness slices are mandatory")
    if attrs.uses_external_data:
        reasons.append("uses external data: lineage and data-quality checks are mandatory")

    if score >= 2.5 or (attrs.decision_impact == 3 and attrs.autonomy == 3):
        tier = RiskTier.CRITICAL
        if score < 2.5:
            reasons.append("auto-escalated to CRITICAL: max decision impact + full autonomy")
    elif score >= 1.8:
        tier = RiskTier.HIGH
    elif score >= 1.0:
        tier = RiskTier.MEDIUM
    else:
        tier = RiskTier.LOW

    depth, approvers, days, depth_text = DEPTH[tier]
    return TierAssignment(
        tier=tier,
        score=round(score, 3),
        reasons=reasons,
        validation_depth=depth_text,
        required_approvers=approvers,
        revalidation_days=days,
    )
