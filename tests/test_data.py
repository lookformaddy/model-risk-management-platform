"""Tests for synthetic data generation."""

from mrmp.data import generate_dataset, split_dataset


def test_generate_dataset_shape_and_groups():
    data = generate_dataset(n=1000, seed=42)
    assert len(data) == 1000
    assert all(len(row) == 4 for row in data.X)
    assert set(data.groups) == {"A", "B", "C"}
    assert set(data.y) == {0, 1}
    # Group shares are approximately the designed 50/30/20.
    share_c = sum(1 for g in data.groups if g == "C") / 1000
    assert 0.15 < share_c < 0.25


def test_generate_dataset_deterministic():
    a = generate_dataset(n=500, seed=7)
    b = generate_dataset(n=500, seed=7)
    assert a.content_hash() == b.content_hash()
    c = generate_dataset(n=500, seed=8)
    assert a.content_hash() != c.content_hash()


def test_shift_changes_distribution():
    base = generate_dataset(n=1000, seed=42, shift=0.0)
    shifted = generate_dataset(n=1000, seed=42, shift=1.0)
    mean_base = sum(r[0] for r in base.X) / 1000
    mean_shifted = sum(r[0] for r in shifted.X) / 1000
    assert mean_shifted - mean_base > 0.8


def test_channel_is_helpful_for_a_b_and_harmful_for_c():
    """The planted structure: channel tracks the label for A/B, flips for C."""
    data = generate_dataset(n=4000, seed=42)
    for group, expected_sign in (("A", 1), ("B", 1), ("C", -1)):
        idx = [i for i, g in enumerate(data.groups) if g == group]
        # Mean channel for positives minus mean channel for negatives.
        pos = [data.channel[i] for i in idx if data.y[i] == 1]
        neg = [data.channel[i] for i in idx if data.y[i] == 0]
        diff = sum(pos) / len(pos) - sum(neg) / len(neg)
        assert diff * expected_sign > 0.5


def test_split_is_disjoint_and_covers():
    data = generate_dataset(n=1000, seed=42)
    train, holdout = split_dataset(data, seed=3)
    assert len(train) + len(holdout) == 1000
    assert len(train) == 500
    assert train.content_hash() != holdout.content_hash()


def test_with_channel_shape():
    data = generate_dataset(n=10, seed=1)
    assert all(len(row) == 5 for row in data.with_channel())
