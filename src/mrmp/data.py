"""Deterministic synthetic data for model risk management demos.

Domain (generic, synthetic): a consumer-credit-style binary decision —
approve / decline — driven by four generic financial-behaviour features.
No real customer data, no employer data. Three synthetic population
groups (``A`` / ``B`` / ``C``) stand in for protected slices in the
fairness checks.

The data is deliberately constructed so the champion/challenger story in
the demo is real, not staged by hand-editing predictions:

* The label depends on the four base features through a logistic link.
* An extra feature, ``channel_score``, is strongly predictive of the
  label for groups A and B but *anti*-predictive for group C (it was
  built from a channel that behaves differently for that segment).
  A challenger that adds ``channel_score`` therefore wins overall while
  regressing sharply on group C — exactly the failure the slice
  regression rule exists to catch.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field

BASE_FEATURE_NAMES = ["income_stability", "tenure_years", "utilization", "payment_history"]
ALL_FEATURE_NAMES = BASE_FEATURE_NAMES + ["channel_score"]
GROUPS = ("A", "B", "C")
GROUP_SHARES = {"A": 0.50, "B": 0.30, "C": 0.20}


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass
class Dataset:
    """One synthetic dataset.

    ``X`` holds the four base features; ``channel`` holds the extra
    channel score (kept separate so the champion can train without it).
    """

    X: list[list[float]]
    channel: list[float]
    y: list[int]
    groups: list[str]
    feature_names: list[str] = field(default_factory=lambda: list(BASE_FEATURE_NAMES))
    seed: int = 42
    shift: float = 0.0

    def __len__(self) -> int:
        return len(self.y)

    def with_channel(self) -> list[list[float]]:
        """Feature matrix including the channel score."""
        return [row + [c] for row, c in zip(self.X, self.channel)]

    def subset(self, indices: list[int]) -> "Dataset":
        return Dataset(
            X=[self.X[i] for i in indices],
            channel=[self.channel[i] for i in indices],
            y=[self.y[i] for i in indices],
            groups=[self.groups[i] for i in indices],
            feature_names=list(self.feature_names),
            seed=self.seed,
            shift=self.shift,
        )

    def content_hash(self) -> str:
        """Deterministic SHA-256 over the rounded rows (reproducibility)."""
        h = hashlib.sha256()
        for row, c, label, g in zip(self.X, self.channel, self.y, self.groups):
            h.update(
                ",".join(f"{v:.6f}" for v in row).encode()
                + f"|{c:.6f}|{label}|{g}\n".encode()
            )
        return h.hexdigest()


def generate_dataset(
    n: int = 2000,
    seed: int = 42,
    shift: float = 0.0,
    label_noise: float = 0.0,
) -> Dataset:
    """Generate ``n`` seeded synthetic rows.

    ``shift`` moves every base feature mean (simulates distribution drift
    in production data). ``label_noise`` flips that fraction of labels.
    """
    rng = random.Random(seed)
    X: list[list[float]] = []
    channel: list[float] = []
    y: list[int] = []
    groups: list[str] = []
    for _ in range(n):
        r = rng.random()
        group = "A" if r < 0.50 else ("B" if r < 0.80 else "C")
        x1 = rng.gauss(0.0, 1.0) + shift
        x2 = rng.gauss(0.0, 1.0) + shift
        x3 = rng.gauss(0.0, 1.0) + shift
        x4 = rng.gauss(0.0, 1.0) + shift
        logit = -0.30 + 1.10 * x1 + 0.70 * x2 - 0.55 * x3 + 0.45 * x4
        label = 1 if rng.random() < _sigmoid(logit) else 0
        if label_noise > 0.0 and rng.random() < label_noise:
            label = 1 - label
        # Channel score: helpful for A/B, misleading for C.
        if group == "C":
            chan = (1 - label) + rng.gauss(0.0, 0.5)
        else:
            chan = float(label) + rng.gauss(0.0, 0.5)
        X.append([x1, x2, x3, x4])
        channel.append(chan)
        y.append(label)
        groups.append(group)
    return Dataset(X=X, channel=channel, y=y, groups=groups, seed=seed, shift=shift)


def split_dataset(data: Dataset, seed: int, train_frac: float = 0.5) -> tuple[Dataset, Dataset]:
    """Deterministic shuffled split."""
    rng = random.Random(seed)
    idx = list(range(len(data)))
    rng.shuffle(idx)
    cut = int(len(idx) * train_frac)
    return data.subset(idx[:cut]), data.subset(idx[cut:])
