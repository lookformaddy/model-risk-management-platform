"""Tests for monitoring: drift screens, decay alerts, revalidation calendar."""

from datetime import date

from mrmp.data import generate_dataset
from mrmp.monitoring import (
    RevalidationCalendar,
    check_performance_decay,
    screen_drift,
)
from mrmp.trainer import LogisticRegression


def test_drift_screen_quiet_on_same_distribution():
    ref = generate_dataset(n=1000, seed=42)
    cur = generate_dataset(n=1000, seed=43)
    screen = screen_drift(ref.X, cur.X, ["f1", "f2", "f3", "f4"])
    assert not screen.breached


def test_drift_screen_breaches_on_shift():
    ref = generate_dataset(n=1000, seed=42)
    cur = generate_dataset(n=1000, seed=43, shift=0.6)
    screen = screen_drift(ref.X, cur.X, ["f1", "f2", "f3", "f4"])
    assert screen.breached
    assert any(a.ks_value > 0.15 for a in screen.alerts)


def test_performance_decay_levels():
    train = generate_dataset(n=1000, seed=42)
    model = LogisticRegression(epochs=200).fit(train.X, train.y)
    valid = generate_dataset(n=1000, seed=142)
    preds = model.predict(valid.X)
    acc = sum(1 for a, b in zip(valid.y, preds) if a == b) / len(valid.y)
    ok = check_performance_decay(acc, valid.y, preds)
    assert ok.level == "OK"
    # Deliberately wrong predictions: a big drop.
    wrong = [1 - p for p in preds]
    alert = check_performance_decay(acc, valid.y, wrong)
    assert alert.level == "ALERT"


def test_revalidation_calendar_statuses():
    cal = RevalidationCalendar()
    cal.schedule("m", "v1", "HIGH", date(2026, 11, 18), 180)
    report = cal.status_report(date(2026, 11, 18))
    assert report["entries"][0]["status"] == "CURRENT"
    assert report["entries"][0]["due_on"] == "2027-05-17"
    overdue = cal.status_report(date(2027, 6, 1))
    assert overdue["entries"][0]["status"] == "OVERDUE"
    assert len(overdue["overdue"]) == 1
    escalated = cal.status_report(date(2027, 7, 1))
    assert escalated["entries"][0]["status"] == "ESCALATED"
    assert len(escalated["escalated"]) == 1
