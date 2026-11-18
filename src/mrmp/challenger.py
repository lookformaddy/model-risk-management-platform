"""Champion vs challenger comparison with a per-slice regression rule.

A challenger promotes only if it wins overall **and** regresses on no
protected slice. The overall win is tested with the McNemar paired test
(the two models are evaluated on the same rows, so an unpaired test
would be wrong); the slice rule is a hard gate, not a tiebreaker — a
challenger that is better on average but worse for one group is
rejected, and the rejection names the regressing slice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mrmp.metrics import accuracy, mcnemar_pvalue

OVERALL_MIN_DELTA = 0.005  # challenger must improve overall accuracy by this much
SLICE_MAX_REGRESSION = 0.02  # no slice may lose more than this much accuracy
MCNEMAR_ALPHA = 0.05


@dataclass
class ChallengerReport:
    champion_accuracy: float
    challenger_accuracy: float
    overall_delta: float
    mcnemar: dict
    slice_deltas: dict[str, float]
    decision: str  # "PROMOTE" or "REJECT"
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "champion_accuracy": round(self.champion_accuracy, 4),
            "challenger_accuracy": round(self.challenger_accuracy, 4),
            "overall_delta": round(self.overall_delta, 4),
            "mcnemar": self.mcnemar,
            "slice_deltas": {g: round(d, 4) for g, d in self.slice_deltas.items()},
            "decision": self.decision,
            "reasons": self.reasons,
        }


def compare_challenger(
    y_true: list[int],
    champion_pred: list[int],
    challenger_pred: list[int],
    groups: list[str],
) -> ChallengerReport:
    """Compare champion vs challenger; apply the promotion rule."""
    champ_acc = accuracy(y_true, champion_pred)
    chall_acc = accuracy(y_true, challenger_pred)
    delta = chall_acc - champ_acc
    mc = mcnemar_pvalue(y_true, champion_pred, challenger_pred)

    slice_deltas: dict[str, float] = {}
    for g in sorted(set(groups)):
        idx = [i for i, x in enumerate(groups) if x == g]
        a = accuracy([y_true[i] for i in idx], [champion_pred[i] for i in idx])
        b = accuracy([y_true[i] for i in idx], [challenger_pred[i] for i in idx])
        slice_deltas[g] = b - a

    reasons: list[str] = []
    decision = "PROMOTE"
    if delta < OVERALL_MIN_DELTA:
        decision = "REJECT"
        reasons.append(
            f"overall accuracy delta {delta:+.4f} is below the required "
            f"+{OVERALL_MIN_DELTA:.3f} improvement"
        )
    regressing = {g: d for g, d in slice_deltas.items() if d < -SLICE_MAX_REGRESSION}
    if regressing:
        decision = "REJECT"
        for g, d in sorted(regressing.items()):
            reasons.append(
                f"slice regression on group {g}: accuracy {d:+.4f} "
                f"(limit -{SLICE_MAX_REGRESSION:.2f})"
            )
    if decision == "PROMOTE" and mc["p_value"] >= MCNEMAR_ALPHA and delta < 0.02:
        # A tiny, statistically insignificant win is not a promotion.
        decision = "REJECT"
        reasons.append(
            f"overall win is not statistically significant "
            f"(McNemar p={mc['p_value']:.3f}) and delta {delta:+.4f} < 0.02"
        )
    if decision == "PROMOTE":
        reasons.append(
            f"overall delta {delta:+.4f} (McNemar p={mc['p_value']:.4f}) with "
            f"no slice regressing more than {SLICE_MAX_REGRESSION:.2f}"
        )
    return ChallengerReport(
        champion_accuracy=champ_acc,
        challenger_accuracy=chall_acc,
        overall_delta=delta,
        mcnemar={k: (round(v, 4) if isinstance(v, float) else v) for k, v in mc.items()},
        slice_deltas=slice_deltas,
        decision=decision,
        reasons=reasons,
    )
