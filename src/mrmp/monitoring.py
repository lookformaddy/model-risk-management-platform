"""Ongoing monitoring: drift screens, performance decay, revalidation calendar.

* **Drift screens** — PSI and two-sample KS per input feature against the
  reference (training) distribution; a breach on any feature raises an
  alert and schedules a revalidation.
* **Performance decay** — accuracy on fresh labelled data vs the
  validation baseline; a drop beyond the alert threshold escalates.
* **Revalidation calendar** — every production model has a due date set
  by its tier; the calendar reports due-soon, overdue, and escalated
  (overdue by more than 30 days) models. Nothing auto-extends: an
  overdue model stays overdue until a fresh passing validation lands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from mrmp.metrics import accuracy, ks_statistic, psi


@dataclass
class DriftAlert:
    feature: str
    psi_value: float
    ks_value: float
    breached: bool
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "psi": round(self.psi_value, 4),
            "ks": round(self.ks_value, 4),
            "breached": self.breached,
            "reason": self.reason,
        }


@dataclass
class DriftScreen:
    alerts: list[DriftAlert] = field(default_factory=list)

    @property
    def breached(self) -> bool:
        return any(a.breached for a in self.alerts)

    def to_dict(self) -> dict:
        return {"breached": self.breached, "alerts": [a.to_dict() for a in self.alerts]}


def screen_drift(
    reference: list[list[float]],
    current: list[list[float]],
    feature_names: list[str],
    psi_threshold: float = 0.20,
    ks_threshold: float = 0.15,
) -> DriftScreen:
    """PSI/KS per feature; breach if either statistic exceeds its threshold."""
    alerts: list[DriftAlert] = []
    for j, name in enumerate(feature_names):
        ref_col = [row[j] for row in reference]
        cur_col = [row[j] for row in current]
        p = psi(ref_col, cur_col)
        k = ks_statistic(ref_col, cur_col)
        breached = p > psi_threshold or k > ks_threshold
        reason = ""
        if p > psi_threshold:
            reason += f"PSI {p:.3f} > {psi_threshold}; "
        if k > ks_threshold:
            reason += f"KS {k:.3f} > {ks_threshold}; "
        alerts.append(DriftAlert(name, p, k, breached, reason.strip()))
    return DriftScreen(alerts)


@dataclass
class DecayAlert:
    baseline_accuracy: float
    current_accuracy: float
    drop: float
    level: str  # "OK" | "WATCH" | "ALERT"

    def to_dict(self) -> dict:
        return {
            "baseline_accuracy": round(self.baseline_accuracy, 4),
            "current_accuracy": round(self.current_accuracy, 4),
            "drop": round(self.drop, 4),
            "level": self.level,
        }


def check_performance_decay(
    baseline_accuracy: float,
    y_true: list[int],
    y_pred: list[int],
    watch_drop: float = 0.03,
    alert_drop: float = 0.06,
) -> DecayAlert:
    current = accuracy(y_true, y_pred)
    drop = baseline_accuracy - current
    level = "OK" if drop < watch_drop else ("WATCH" if drop < alert_drop else "ALERT")
    return DecayAlert(baseline_accuracy, current, drop, level)


@dataclass
class RevalidationEntry:
    model_id: str
    version: str
    tier: str
    due_on: date
    last_validated_on: date

    def status(self, on: date) -> str:
        if on <= self.due_on:
            return "DUE_SOON" if (self.due_on - on).days <= 30 else "CURRENT"
        overdue_days = (on - self.due_on).days
        return "ESCALATED" if overdue_days > 30 else "OVERDUE"

    def to_dict(self, on: date) -> dict:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "tier": self.tier,
            "due_on": self.due_on.isoformat(),
            "last_validated_on": self.last_validated_on.isoformat(),
            "status": self.status(on),
        }


@dataclass
class RevalidationCalendar:
    entries: list[RevalidationEntry] = field(default_factory=list)

    def schedule(
        self, model_id: str, version: str, tier: str, validated_on: date, revalidation_days: int
    ) -> RevalidationEntry:
        entry = RevalidationEntry(
            model_id=model_id,
            version=version,
            tier=tier,
            due_on=validated_on + timedelta(days=revalidation_days),
            last_validated_on=validated_on,
        )
        self.entries.append(entry)
        return entry

    def status_report(self, on: date) -> dict:
        report = [e.to_dict(on) for e in self.entries]
        return {
            "as_of": on.isoformat(),
            "entries": report,
            "overdue": [e for e in report if e["status"] in ("OVERDUE", "ESCALATED")],
            "escalated": [e for e in report if e["status"] == "ESCALATED"],
        }
