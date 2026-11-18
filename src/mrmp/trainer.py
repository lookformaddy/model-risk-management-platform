"""From-scratch logistic regression (pure stdlib).

Full-batch gradient descent on standardized features with L2
regularisation. Deterministic: fixed initialisation (zeros) and a fixed
epoch count, so the same data and config always produce the same weights
— a prerequisite for the reproducibility manifest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass
class StandardScaler:
    mean: list[float] = field(default_factory=list)
    std: list[float] = field(default_factory=list)

    def fit(self, X: list[list[float]]) -> "StandardScaler":
        n = len(X)
        d = len(X[0])
        self.mean = [sum(row[j] for row in X) / n for j in range(d)]
        self.std = []
        for j in range(d):
            var = sum((row[j] - self.mean[j]) ** 2 for row in X) / n
            self.std.append(math.sqrt(var) if var > 1e-12 else 1.0)
        return self

    def transform(self, X: list[list[float]]) -> list[list[float]]:
        return [[(v - self.mean[j]) / self.std[j] for j, v in enumerate(row)] for row in X]


@dataclass
class LogisticRegression:
    """Binary logistic regression trained from scratch."""

    learning_rate: float = 0.5
    epochs: int = 400
    l2: float = 0.01
    weights: list[float] = field(default_factory=list)
    bias: float = 0.0
    scaler: StandardScaler = field(default_factory=StandardScaler)
    n_features: int = 0
    train_loss: float = 0.0

    def fit(self, X: list[list[float]], y: list[int]) -> "LogisticRegression":
        if not X:
            raise ValueError("cannot fit on empty data")
        self.n_features = len(X[0])
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.weights = [0.0] * self.n_features
        self.bias = 0.0
        n = len(Xs)
        for _ in range(self.epochs):
            grad_w = [0.0] * self.n_features
            grad_b = 0.0
            loss = 0.0
            for row, label in zip(Xs, y):
                z = self.bias + sum(w * v for w, v in zip(self.weights, row))
                p = _sigmoid(z)
                p = min(max(p, 1e-12), 1.0 - 1e-12)
                loss += -(label * math.log(p) + (1 - label) * math.log(1 - p))
                err = p - label
                for j in range(self.n_features):
                    grad_w[j] += err * row[j]
                grad_b += err
            for j in range(self.n_features):
                grad_w[j] = grad_w[j] / n + self.l2 * self.weights[j]
                self.weights[j] -= self.learning_rate * grad_w[j]
            self.bias -= self.learning_rate * (grad_b / n)
            self.train_loss = loss / n
        return self

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        if not self.weights:
            raise ValueError("model is not fitted")
        Xs = self.scaler.transform(X)
        return [
            _sigmoid(self.bias + sum(w * v for w, v in zip(self.weights, row)))
            for row in Xs
        ]

    def predict(self, X: list[list[float]], threshold: float = 0.5) -> list[int]:
        return [1 if p >= threshold else 0 for p in self.predict_proba(X)]

    def to_dict(self) -> dict:
        return {
            "weights": [round(w, 6) for w in self.weights],
            "bias": round(self.bias, 6),
            "mean": [round(m, 6) for m in self.scaler.mean],
            "std": [round(s, 6) for s in self.scaler.std],
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "l2": self.l2,
            "train_loss": round(self.train_loss, 6),
        }
