"""Pure-stdlib evaluation metrics and statistical screens.

Covers the numbers a validation suite needs: accuracy, ROC-AUC
(rank-based), Brier score, log-loss, expected calibration error (ECE),
confusion counts, paired bootstrap confidence intervals, k-fold
cross-validated variance, the McNemar paired test for champion vs
challenger, and PSI / two-sample KS drift statistics.
"""

from __future__ import annotations

import math
import random


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true:
        return 0.0
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)


def confusion(y_true: list[int], y_pred: list[int]) -> dict[str, int]:
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def roc_auc(y_true: list[int], scores: list[float]) -> float:
    """Mann-Whitney rank AUC; 0.5 when only one class is present."""
    pos = [s for t, s in zip(y_true, scores) if t == 1]
    neg = [s for t, s in zip(y_true, scores) if t == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def brier_score(y_true: list[int], probs: list[float]) -> float:
    if not y_true:
        return 0.0
    return sum((p - t) ** 2 for t, p in zip(y_true, probs)) / len(y_true)


def log_loss(y_true: list[int], probs: list[float]) -> float:
    if not y_true:
        return 0.0
    total = 0.0
    for t, p in zip(y_true, probs):
        p = min(max(p, 1e-12), 1.0 - 1e-12)
        total += -(t * math.log(p) + (1 - t) * math.log(1 - p))
    return total / len(y_true)


def expected_calibration_error(
    y_true: list[int], probs: list[float], n_bins: int = 10
) -> float:
    """ECE: weighted mean |mean prob - observed rate| per confidence bin."""
    if not y_true:
        return 0.0
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, p in enumerate(probs) if lo <= p < hi or (b == n_bins - 1 and p == 1.0)]
        if not idx:
            continue
        mean_p = sum(probs[i] for i in idx) / len(idx)
        rate = sum(y_true[i] for i in idx) / len(idx)
        ece += (len(idx) / n) * abs(mean_p - rate)
    return ece


def bootstrap_ci(
    values: list[float],
    statistic,
    n_boot: int = 500,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for ``statistic(values)`` (deterministic)."""
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    estimates = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        estimates.append(statistic(sample))
    estimates.sort()
    lo_idx = max(0, int((alpha / 2) * n_boot))
    hi_idx = min(n_boot - 1, int((1 - alpha / 2) * n_boot))
    return (estimates[lo_idx], estimates[hi_idx])


def bootstrap_accuracy_ci(
    y_true: list[int], y_pred: list[int], n_boot: int = 500, seed: int = 42
) -> tuple[float, float]:
    """Bootstrap 95% CI for accuracy, resampling (truth, pred) pairs."""
    pairs = list(zip(y_true, y_pred))
    return bootstrap_ci(
        pairs,
        lambda s: sum(1 for a, b in s if a == b) / len(s) if s else 0.0,
        n_boot=n_boot,
        seed=seed,
    )


def kfold_variance(
    y_true: list[int], y_pred: list[int], k: int = 5, seed: int = 42
) -> tuple[float, float]:
    """Mean and standard deviation of per-fold accuracy (seeded shuffle)."""
    rng = random.Random(seed)
    idx = list(range(len(y_true)))
    rng.shuffle(idx)
    folds: list[list[int]] = [[] for _ in range(k)]
    for i, j in enumerate(idx):
        folds[i % k].append(j)
    accs = []
    for fold in folds:
        if fold:
            accs.append(
                sum(1 for j in fold if y_true[j] == y_pred[j]) / len(fold)
            )
    if not accs:
        return (0.0, 0.0)
    mean = sum(accs) / len(accs)
    var = sum((a - mean) ** 2 for a in accs) / len(accs)
    return (mean, math.sqrt(var))


def mcnemar_pvalue(y_true: list[int], pred_a: list[int], pred_b: list[int]) -> dict:
    """McNemar test (with continuity correction) for two paired classifiers.

    Returns the discordant counts and a chi-square(1 df) p-value computed
    from the normal survival function: p = erfc(sqrt(chi2 / 2)).
    """
    b = sum(1 for t, a, c in zip(y_true, pred_a, pred_b) if a == t and c != t)
    c = sum(1 for t, a, d in zip(y_true, pred_a, pred_b) if a != t and d == t)
    n_disc = b + c
    if n_disc == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p_value": 1.0}
    chi2 = (abs(b - c) - 1.0) ** 2 / n_disc
    p_value = math.erfc(math.sqrt(chi2 / 2.0))
    return {"b": b, "c": c, "chi2": chi2, "p_value": p_value}


def psi(expected: list[float], actual: list[float], n_bins: int = 10) -> float:
    """Population Stability Index between two samples (decile bins)."""
    if not expected or not actual:
        return 0.0
    ref = sorted(expected)
    edges = [ref[int(i * len(ref) / n_bins)] for i in range(n_bins)] + [ref[-1] + 1e-9]
    total = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        e = sum(1 for v in expected if lo <= v < hi or (i == n_bins - 1 and v == hi)) / len(expected)
        a = sum(1 for v in actual if lo <= v < hi or (i == n_bins - 1 and v == hi)) / len(actual)
        e = max(e, 1e-4)
        a = max(a, 1e-4)
        total += (a - e) * math.log(a / e)
    return total


def ks_statistic(sample_a: list[float], sample_b: list[float]) -> float:
    """Two-sample Kolmogorov-Smirnov statistic."""
    if not sample_a or not sample_b:
        return 0.0
    combined = sorted(set(sample_a) | set(sample_b))
    max_diff = 0.0
    for v in combined:
        cdf_a = sum(1 for x in sample_a if x <= v) / len(sample_a)
        cdf_b = sum(1 for x in sample_b if x <= v) / len(sample_b)
        max_diff = max(max_diff, abs(cdf_a - cdf_b))
    return max_diff
