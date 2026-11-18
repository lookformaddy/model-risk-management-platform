"""Tests for champion vs challenger comparison."""

import pytest

from mrmp.challenger import compare_challenger
from mrmp.data import generate_dataset
from mrmp.trainer import LogisticRegression


def _models():
    train = generate_dataset(n=2000, seed=42)
    valid = generate_dataset(n=2000, seed=142)
    champ = LogisticRegression(epochs=300).fit(train.X, train.y)
    chall = LogisticRegression(epochs=300).fit(train.with_channel(), train.y)
    return valid, champ.predict(valid.X), chall.predict(valid.with_channel())


def test_challenger_rejected_on_slice_regression():
    valid, champ_preds, chall_preds = _models()
    report = compare_challenger(valid.y, champ_preds, chall_preds, valid.groups)
    assert report.decision == "REJECT"
    assert report.overall_delta > 0  # it really does win overall
    assert report.slice_deltas["C"] < -0.02  # and regresses on group C
    assert any("group C" in r for r in report.reasons)


def test_identical_challenger_rejected_no_improvement():
    valid, champ_preds, _ = _models()
    report = compare_challenger(valid.y, champ_preds, list(champ_preds), valid.groups)
    assert report.decision == "REJECT"
    assert report.overall_delta == 0.0


def test_clearly_better_challenger_promotes():
    # Synthetic: challenger fixes 60 champion errors and breaks none.
    y = [1, 0] * 500
    champ = [1, 0] * 470 + [0, 1] * 30  # 94% accurate
    chall = [1, 0] * 500  # 100% accurate
    groups = (["A", "B", "C"] * 334)[:1000]
    report = compare_challenger(y, champ, chall, groups)
    assert report.decision == "PROMOTE"
    assert report.overall_delta == pytest.approx(0.06)


def test_small_insignificant_win_rejected():
    # One flipped prediction: not a real improvement.
    y = [1, 0] * 500
    champ = [1, 0] * 500
    chall = list(champ)
    chall[0] = 0  # one extra error -> worse overall anyway
    groups = (["A", "B", "C"] * 334)[:1000]
    report = compare_challenger(y, champ, chall, groups)
    assert report.decision == "REJECT"
