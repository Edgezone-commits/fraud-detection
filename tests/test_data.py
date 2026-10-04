"""Data loading and splitting: sizes, stratification and no leakage between parts."""

import pytest

from fraud_detection.data import load_data, split_data
from helpers import make_synthetic, requires_data


def test_split_sizes_are_about_60_20_20():
    X, y = make_synthetic(n=5000, fraud_rate=0.02)
    _, _, _, y_tr, y_va, y_te = split_data(X, y)
    n = len(y)
    assert len(y_tr) / n == pytest.approx(0.6, abs=0.01)
    assert len(y_va) / n == pytest.approx(0.2, abs=0.01)
    assert len(y_te) / n == pytest.approx(0.2, abs=0.01)


def test_split_keeps_fraud_rate_in_every_part():
    X, y = make_synthetic(n=5000, fraud_rate=0.02)
    _, _, _, y_tr, y_va, y_te = split_data(X, y)
    overall = y.mean()
    for part in (y_tr, y_va, y_te):
        assert part.mean() == pytest.approx(overall, abs=0.002)


def test_no_row_appears_in_two_parts():
    X, y = make_synthetic(n=5000)
    X_tr, X_va, X_te, *_ = split_data(X, y)
    train, val, test = set(X_tr.index), set(X_va.index), set(X_te.index)
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)


def test_parts_cover_every_row():
    X, y = make_synthetic(n=5000)
    X_tr, X_va, X_te, *_ = split_data(X, y)
    assert set(X_tr.index) | set(X_va.index) | set(X_te.index) == set(X.index)


def test_split_is_reproducible():
    X, y = make_synthetic(n=3000)
    first = split_data(X, y)
    second = split_data(X, y)
    assert list(first[2].index) == list(second[2].index)


@requires_data
def test_full_dataset_split_matches_known_counts():
    # Counts from the Phase 1 run. The split is deterministic (seed 42), so they must not change.
    X, y = load_data()
    assert int(y.sum()) == 492
    _, _, _, y_tr, y_va, y_te = split_data(X, y)
    assert (len(y_tr), len(y_va), len(y_te)) == (170883, 56962, 56962)
    assert (int(y_tr.sum()), int(y_va.sum()), int(y_te.sum())) == (295, 99, 98)
