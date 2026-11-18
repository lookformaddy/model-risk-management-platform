"""Demo: the full governed model lifecycle, end to end.

1. Train a champion (from-scratch logistic regression, base features).
2. Register it with a reproducibility manifest, tier it, validate it.
3. Collect signed approvals and promote it to PRODUCTION.
4. Evaluate a challenger that adds a channel score: it wins overall but
   regresses on group C — the slice rule rejects it.
5. Screen fresh shifted production data for drift: PSI/KS breach +
   performance decay trigger a revalidation.
6. Verify the hash-chained audit log.

Run:  PYTHONPATH=src python scripts/demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mrmp.config import Config  # noqa: E402
from mrmp.pipeline import run_pipeline  # noqa: E402


def main() -> int:
    result = run_pipeline(Config.from_env())
    print("=== Model Risk Management Platform — demo ===\n")

    print("1. CHAMPION  credit-decision-logreg:v1 (from-scratch logistic regression)")
    print(f"   validation accuracy {result.champion_accuracy:.4f}   AUC {result.champion_auc:.4f}")
    print(f"   manifest hash {result.details['manifest_hash']}...")

    print(f"\n2. RISK TIER  {result.tier.tier.value} (score {result.tier.score})")
    for reason in result.tier.reasons:
        print(f"   - {reason}")
    print(f"   {result.tier.validation_depth}")

    print(f"\n3. VALIDATION  {result.validation.report_id}: "
          f"{'PASSED' if result.validation.passed else 'FAILED'} "
          f"({len(result.validation.checks)} checks, "
          f"{len(result.validation.failures)} failing)")
    for check in result.validation.checks:
        mark = "PASS" if check.passed else "FAIL"
        print(f"   [{mark}] {check.name:26s} {check.value} (limit {check.threshold})")

    print(f"\n4. APPROVALS  stage {result.stage}")
    for name in result.approvals:
        print(f"   signed: {name}")
    print(f"   revalidation due: {result.revalidation_due}")

    ch = result.challenger
    print(f"\n5. CHALLENGER  v2-challenger (adds channel_score): {ch.decision}")
    print(f"   champion {ch.champion_accuracy:.4f} -> challenger {ch.challenger_accuracy:.4f} "
          f"(delta {ch.overall_delta:+.4f}, McNemar p={ch.mcnemar['p_value']:.4f})")
    for g, d in ch.slice_deltas.items():
        print(f"   slice {g}: accuracy delta {d:+.4f} "
              f"(champion {result.details['champion_slice_accuracies'][g]:.4f} -> "
              f"challenger {result.details['challenger_slice_accuracies'][g]:.4f})")
    for reason in ch.reasons:
        print(f"   - {reason}")

    print("\n6. MONITORING  fresh production data (mean shift +0.6)")
    for alert in result.drift.alerts:
        mark = "BREACH" if alert.breached else "ok"
        print(f"   [{mark}] {alert.feature:18s} PSI {alert.psi_value:.3f}  KS {alert.ks_value:.3f}  {alert.reason}")
    print(f"   drift breached: {result.drift.breached}")
    print(f"   accuracy on drifted data: {result.details['drifted_accuracy']:.4f} "
          f"(decay level {result.decay_level})")
    if result.drift.breached or result.decay_level == "ALERT":
        print("   -> REVALIDATION TRIGGERED for credit-decision-logreg:v1")

    print(f"\n7. AUDIT  hash-chained log: {result.audit_entries} entries, "
          f"chain {'OK' if result.audit_ok else 'BROKEN'}, head {result.details['audit_head']}...")
    print("\nDemo complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
