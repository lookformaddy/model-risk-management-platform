"""Tier-driven validation suites.

A validation run executes five check families against held-out data:

1. **Performance gates** — accuracy / AUC floors and a must-beat-baseline
   delta (the model must beat the majority-class rate by a margin).
2. **Stability** — bootstrap 95% CI for accuracy (width cap) and
   cross-fold accuracy variance (cap).
3. **Data quality** — missing / out-of-range rates on the validation data,
   duplicate rows, and minimum group sizes for slice checks.
4. **Fairness slices** — demographic parity and equal-opportunity gaps per
   group, with small-sample guards: a group below ``min_group_n`` is
   reported as INSUFFICIENT DATA, never failed on noise.
5. **Outcome analysis** — error concentration (share of errors in the
   worst decile of scores) and calibration (ECE / Brier).

Thresholds tighten with the risk tier (CRITICAL is strictest; LOW runs
the light suite — performance + data quality only). The report carries
its own content hash so an approval can bind to exactly this report.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from mrmp.data import Dataset
from mrmp.metrics import (
    accuracy,
    bootstrap_accuracy_ci,
    brier_score,
    expected_calibration_error,
    kfold_variance,
    roc_auc,
)
from mrmp.tiering import RiskTier
from mrmp.trainer import LogisticRegression


@dataclass
class ValidationCheck:
    name: str
    passed: bool
    value: float | str
    threshold: float | str
    details: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "value": self.value,
            "threshold": self.threshold,
            "details": self.details,
        }


@dataclass
class ValidationReport:
    report_id: str
    model_id: str
    version: str
    tier: RiskTier
    checks: list[ValidationCheck] = field(default_factory=list)
    n_validation_rows: int = 0

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[ValidationCheck]:
        return [c for c in self.checks if not c.passed]

    def report_hash(self) -> str:
        payload = {
            "report_id": self.report_id,
            "model_id": self.model_id,
            "version": self.version,
            "tier": self.tier.value,
            "checks": [c.to_dict() for c in self.checks],
            "n_validation_rows": self.n_validation_rows,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def to_dict(self) -> dict:
        return {
            "report_id": self.report_id,
            "model_id": self.model_id,
            "version": self.version,
            "tier": self.tier.value,
            "passed": self.passed,
            "n_validation_rows": self.n_validation_rows,
            "report_hash": self.report_hash(),
            "checks": [c.to_dict() for c in self.checks],
        }


# Tier -> (accuracy floor, AUC floor, baseline delta, CI width cap,
#          fold-std cap, parity gap cap, equal-opp gap cap, ECE cap)
TIER_THRESHOLDS = {
    RiskTier.CRITICAL: dict(acc=0.72, auc=0.75, delta=0.10, ci_width=0.06, fold_std=0.03,
                            parity=0.12, eq_opp=0.12, ece=0.08),
    RiskTier.HIGH: dict(acc=0.70, auc=0.72, delta=0.08, ci_width=0.08, fold_std=0.04,
                        parity=0.15, eq_opp=0.15, ece=0.10),
    RiskTier.MEDIUM: dict(acc=0.68, auc=0.70, delta=0.05, ci_width=0.10, fold_std=0.05,
                          parity=0.20, eq_opp=0.20, ece=0.12),
    RiskTier.LOW: dict(acc=0.65, auc=0.65, delta=0.03, ci_width=0.12, fold_std=0.06,
                       parity=0.25, eq_opp=0.25, ece=0.15),
}

MIN_GROUP_N = 50  # below this, a fairness slice is INSUFFICIENT DATA, not a failure


def _slice_metrics(
    y_true: list[int], y_pred: list[int], probs: list[float], groups: list[str]
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for g in sorted(set(groups)):
        idx = [i for i, x in enumerate(groups) if x == g]
        out[g] = {
            "n": len(idx),
            "accuracy": accuracy([y_true[i] for i in idx], [y_pred[i] for i in idx]),
            "positive_rate": sum(y_pred[i] for i in idx) / len(idx),
            "true_positive_rate": (
                sum(1 for i in idx if y_true[i] == 1 and y_pred[i] == 1)
                / max(1, sum(1 for i in idx if y_true[i] == 1))
            ),
        }
    return out


def run_validation(
    model: LogisticRegression,
    data: Dataset,
    model_id: str,
    version: str,
    tier: RiskTier,
    report_id: str,
) -> ValidationReport:
    """Run the tier-driven validation suite on held-out data."""
    th = TIER_THRESHOLDS[tier]
    probs = model.predict_proba(data.X)
    preds = [1 if p >= 0.5 else 0 for p in probs]
    y = data.y
    checks: list[ValidationCheck] = []

    # 1. Performance gates
    acc = accuracy(y, preds)
    checks.append(ValidationCheck("accuracy_floor", acc >= th["acc"], round(acc, 4), th["acc"]))
    auc = roc_auc(y, probs)
    checks.append(ValidationCheck("auc_floor", auc >= th["auc"], round(auc, 4), th["auc"]))
    majority_rate = max(sum(y) / len(y), 1 - sum(y) / len(y))
    delta = acc - majority_rate
    checks.append(ValidationCheck(
        "must_beat_baseline", delta >= th["delta"], round(delta, 4), th["delta"],
        f"majority-class rate {majority_rate:.4f}"))

    # 2. Stability
    lo, hi = bootstrap_accuracy_ci(y, preds, n_boot=500, seed=7)
    width = hi - lo
    checks.append(ValidationCheck(
        "bootstrap_ci_width", width <= th["ci_width"], round(width, 4), th["ci_width"],
        f"95% CI [{lo:.4f}, {hi:.4f}]"))
    _, fold_std = kfold_variance(y, preds, k=5, seed=7)
    checks.append(ValidationCheck(
        "cross_fold_std", fold_std <= th["fold_std"], round(fold_std, 4), th["fold_std"]))

    # 3. Data quality on the validation data
    n_missing = sum(1 for row in data.X for v in row if v != v)  # NaN check
    missing_rate = n_missing / max(1, len(data.X) * len(data.X[0]))
    checks.append(ValidationCheck("missing_rate", missing_rate <= 0.01, round(missing_rate, 4), 0.01))
    out_of_range = sum(
        1 for row in data.X for v in row if abs(v) > 6.0
    ) / max(1, len(data.X) * len(data.X[0]))
    checks.append(ValidationCheck("out_of_range_rate", out_of_range <= 0.02, round(out_of_range, 4), 0.02))
    seen: set[tuple] = set()
    dups = 0
    for row in data.X:
        key = tuple(round(v, 6) for v in row)
        if key in seen:
            dups += 1
        seen.add(key)
    dup_rate = dups / max(1, len(data.X))
    checks.append(ValidationCheck("duplicate_rate", dup_rate <= 0.01, round(dup_rate, 4), 0.01))

    slices = _slice_metrics(y, preds, probs, data.groups)
    min_group = min(s["n"] for s in slices.values()) if slices else 0
    checks.append(ValidationCheck(
        "min_group_size", min_group >= MIN_GROUP_N, min_group, MIN_GROUP_N,
        "smallest fairness slice"))

    # 4. Fairness slices (skipped for LOW — light suite)
    if tier != RiskTier.LOW:
        rates = [s["positive_rate"] for s in slices.values() if s["n"] >= MIN_GROUP_N]
        tprs = [s["true_positive_rate"] for s in slices.values() if s["n"] >= MIN_GROUP_N]
        if len(rates) >= 2:
            parity_gap = max(rates) - min(rates)
            checks.append(ValidationCheck(
                "demographic_parity_gap", parity_gap <= th["parity"],
                round(parity_gap, 4), th["parity"],
                f"positive rates { {g: round(s['positive_rate'], 3) for g, s in slices.items()} }"))
            eq_gap = max(tprs) - min(tprs)
            checks.append(ValidationCheck(
                "equal_opportunity_gap", eq_gap <= th["eq_opp"],
                round(eq_gap, 4), th["eq_opp"],
                f"TPRs { {g: round(s['true_positive_rate'], 3) for g, s in slices.items()} }"))
        else:
            checks.append(ValidationCheck(
                "demographic_parity_gap", True, "insufficient-data", th["parity"],
                "fewer than 2 slices meet the minimum group size — not failed"))
            checks.append(ValidationCheck(
                "equal_opportunity_gap", True, "insufficient-data", th["eq_opp"],
                "fewer than 2 slices meet the minimum group size — not failed"))

    # 5. Outcome analysis (skipped for LOW)
    if tier != RiskTier.LOW:
        ece = expected_calibration_error(y, probs)
        checks.append(ValidationCheck("ece", ece <= th["ece"], round(ece, 4), th["ece"]))
        brier = brier_score(y, probs)
        checks.append(ValidationCheck("brier_score", brier <= 0.25, round(brier, 4), 0.25))
        # Error concentration: share of all errors sitting in the worst
        # decile of predicted probability (near the 0.5 boundary = hardest).
        scored = sorted(range(len(y)), key=lambda i: abs(probs[i] - 0.5))
        worst = scored[: max(1, len(y) // 10)]
        total_errors = sum(1 for a, b in zip(y, preds) if a != b)
        worst_errors = sum(1 for i in worst if y[i] != preds[i])
        concentration = worst_errors / max(1, total_errors)
        checks.append(ValidationCheck(
            "error_concentration", concentration >= 0.15, round(concentration, 4), 0.15,
            "errors should concentrate near the decision boundary, not spread evenly"))

    return ValidationReport(
        report_id=report_id,
        model_id=model_id,
        version=version,
        tier=tier,
        checks=checks,
        n_validation_rows=len(data),
    )
