"""Tests for the from-scratch logistic regression trainer."""

from mrmp.data import generate_dataset
from mrmp.trainer import LogisticRegression


def test_trainer_beats_majority():
    data = generate_dataset(n=1000, seed=42)
    model = LogisticRegression(epochs=400).fit(data.X, data.y)
    preds = model.predict(data.X)
    acc = sum(1 for a, b in zip(data.y, preds) if a == b) / len(data.y)
    majority = max(sum(data.y), len(data.y) - sum(data.y)) / len(data.y)
    assert acc > majority + 0.05
    assert 0.0 < model.train_loss < 0.7


def test_trainer_deterministic():
    data = generate_dataset(n=500, seed=42)
    a = LogisticRegression(epochs=200).fit(data.X, data.y)
    b = LogisticRegression(epochs=200).fit(data.X, data.y)
    assert a.to_dict() == b.to_dict()


def test_predict_proba_in_unit_interval():
    data = generate_dataset(n=200, seed=1)
    model = LogisticRegression(epochs=100).fit(data.X, data.y)
    probs = model.predict_proba(data.X)
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_unfitted_model_raises():
    import pytest

    model = LogisticRegression()
    with pytest.raises(ValueError):
        model.predict_proba([[0.0, 0.0, 0.0, 0.0]])


def test_empty_fit_raises():
    import pytest

    with pytest.raises(ValueError):
        LogisticRegression().fit([], [])
