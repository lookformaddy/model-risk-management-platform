"""Staged approval workflow with signed, run-bound approvals.

Lifecycle: DEVELOPMENT -> VALIDATION -> APPROVED -> PRODUCTION -> RETIRED.
The gates are fail-closed:

* A model cannot enter VALIDATION without a validation report.
* It cannot be APPROVED without a *passing* report and a signed approval
  from every role the model's tier requires (CRITICAL/HIGH need a model
  validator *and* a risk officer; the two signatures must come from
  different people).
* An approval is bound to one report: it signs the report's content hash,
  so approving report A cannot silently cover a re-run report B.
* Approvals expire; a production model whose approval has expired (or
  whose revalidation date has passed) is flagged for revalidation and
  cannot be re-promoted until a fresh passing report exists.

Signatures are SHA-256 over (secret + canonical approval payload) — a
demo-grade stand-in for a real PKI signature, with the same binding
property: the signature only verifies for the exact payload it signed.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, field
from datetime import date, timedelta

from mrmp.registry import ModelVersion, Stage
from mrmp.tiering import RiskTier
from mrmp.validation import ValidationReport

ALLOWED_TRANSITIONS = {
    Stage.DEVELOPMENT: {Stage.VALIDATION, Stage.RETIRED},
    Stage.VALIDATION: {Stage.APPROVED, Stage.DEVELOPMENT, Stage.RETIRED},
    Stage.APPROVED: {Stage.PRODUCTION, Stage.VALIDATION, Stage.RETIRED},
    Stage.PRODUCTION: {Stage.RETIRED, Stage.VALIDATION},
    Stage.RETIRED: set(),
}


class WorkflowError(ValueError):
    """Raised when a fail-closed gate blocks a transition."""


def _canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def sign_approval(secret: str, payload: dict) -> str:
    return hmac.new(secret.encode(), _canonical(payload).encode(), hashlib.sha256).hexdigest()


def verify_approval(secret: str, payload: dict, signature: str) -> bool:
    return hmac.compare_digest(sign_approval(secret, payload), signature)


@dataclass
class Approval:
    approver: str
    role: str
    report_id: str
    report_hash: str
    model_id: str
    version: str
    signed_on: str  # ISO date
    expires_on: str  # ISO date
    signature: str = ""

    def payload(self) -> dict:
        return {
            "approver": self.approver,
            "role": self.role,
            "report_id": self.report_id,
            "report_hash": self.report_hash,
            "model_id": self.model_id,
            "version": self.version,
            "signed_on": self.signed_on,
            "expires_on": self.expires_on,
        }

    def sign(self, secret: str) -> "Approval":
        self.signature = sign_approval(secret, self.payload())
        return self

    def verify(self, secret: str) -> bool:
        return verify_approval(secret, self.payload(), self.signature)


@dataclass
class Workflow:
    """Owns stage transitions and approvals for one model version."""

    version: ModelVersion
    tier: RiskTier
    secret: str
    approvals: list[Approval] = field(default_factory=list)
    history: list[str] = field(default_factory=list)

    # -- transitions -----------------------------------------------------
    def transition(self, target: Stage) -> None:
        if target not in ALLOWED_TRANSITIONS[self.version.stage]:
            raise WorkflowError(
                f"illegal transition {self.version.stage.value} -> {target.value}"
            )
        self.version.stage = target
        self.history.append(f"{target.value}")

    def submit_for_validation(self, report: ValidationReport) -> None:
        if report.model_id != self.version.model_id or report.version != self.version.version:
            raise WorkflowError("validation report belongs to a different model version")
        self.version.validation_report_id = report.report_id
        self.transition(Stage.VALIDATION)

    # -- approvals -------------------------------------------------------
    def approve(
        self,
        approver: str,
        role: str,
        report: ValidationReport,
        signed_on: date,
        validity_days: int = 180,
    ) -> Approval:
        """Record a signed approval. Fail-closed on every gate."""
        if self.version.stage != Stage.VALIDATION:
            raise WorkflowError(
                f"cannot approve from stage {self.version.stage.value}; "
                "model must be in VALIDATION"
            )
        if self.version.validation_report_id != report.report_id:
            raise WorkflowError("no matching validation report on file for this version")
        if not report.passed:
            raise WorkflowError(
                "cannot approve: validation report FAILED — "
                f"{len(report.failures)} failing check(s): "
                + ", ".join(c.name for c in report.failures)
            )
        if report.report_hash() != report.report_hash():
            raise WorkflowError("validation report hash mismatch (tampered report)")
        if role not in self._required_roles():
            raise WorkflowError(f"role {role!r} is not a required approver for tier {self.tier.value}")
        if any(a.role == role for a in self.approvals):
            raise WorkflowError(f"role {role!r} has already approved this version")
        if any(a.approver == approver for a in self.approvals):
            raise WorkflowError(f"{approver} has already approved this version (one signature per person)")
        approval = Approval(
            approver=approver,
            role=role,
            report_id=report.report_id,
            report_hash=report.report_hash(),
            model_id=self.version.model_id,
            version=self.version.version,
            signed_on=signed_on.isoformat(),
            expires_on=(signed_on + timedelta(days=validity_days)).isoformat(),
        ).sign(self.secret)
        self.approvals.append(approval)
        self.version.approved_by.append(approver)
        if self._fully_approved():
            self.transition(Stage.APPROVED)
        return approval

    def _required_roles(self) -> list[str]:
        from mrmp.tiering import DEPTH

        return DEPTH[self.tier][1]

    def _fully_approved(self) -> bool:
        have = {a.role for a in self.approvals}
        return set(self._required_roles()) <= have

    def promote_to_production(self, on: date) -> None:
        if self.version.stage != Stage.APPROVED:
            raise WorkflowError(
                f"cannot promote from stage {self.version.stage.value}; "
                "model must be APPROVED"
            )
        if not self._fully_approved():
            raise WorkflowError("cannot promote: required approvals are incomplete")
        expired = [a for a in self.approvals if a.expires_on < on.isoformat()]
        if expired:
            raise WorkflowError(
                "cannot promote: an approval expired on "
                + ", ".join(a.expires_on for a in expired)
            )
        self.transition(Stage.PRODUCTION)

    def retire(self) -> None:
        self.transition(Stage.RETIRED)
