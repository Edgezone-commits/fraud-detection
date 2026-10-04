"""Cost function and metrics: hand-checked examples, so the arithmetic is verified."""

import json

import numpy as np
import pytest

from fraud_detection.evaluate import (
    THRESHOLD_GRID,
    best_threshold,
    bootstrap_ci,
    confusion_at,
    summarize,
    total_cost,
)


def test_cost_counts_missed_frauds_ten_times_and_false_alarms_once():
    # Scores >= 0.5 are flagged. Row 4 is a missed fraud (FN). Row 2 is a false alarm (FP).
    y = [1, 0, 0, 1]
    p = [0.9, 0.8, 0.1, 0.2]
    assert confusion_at(y, p, 0.5) == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert total_cost(y, p, 0.5) == 10 * 1 + 1 * 1  # 11


def test_cost_uses_given_cost_ratio():
    y = [1, 0, 0, 1]
    p = [0.9, 0.8, 0.1, 0.2]
    assert total_cost(y, p, 0.5, cost_fn=5, cost_fp=2) == 5 * 1 + 2 * 1  # 7


def test_best_threshold_reaches_zero_cost_when_classes_separate():
    y = [0, 0, 0, 1, 1]
    p = [0.05, 0.10, 0.20, 0.80, 0.90]
    threshold, curve = best_threshold(y, p)
    assert total_cost(y, p, threshold) == 0
    assert curve["cost"].min() == 0


def test_best_threshold_is_on_the_grid():
    y = [0, 1, 0, 1]
    p = [0.2, 0.7, 0.4, 0.3]
    threshold, _ = best_threshold(y, p)
    assert any(np.isclose(threshold, THRESHOLD_GRID))


def test_threshold_grid_is_log_spaced_from_1e6_to_1():
    assert THRESHOLD_GRID[0] == pytest.approx(1e-6)
    assert THRESHOLD_GRID[-1] == pytest.approx(1.0)
    assert len(THRESHOLD_GRID) == 1000


def test_summarize_returns_json_safe_values():
    y = [0, 1, 0, 1, 0]
    p = [0.1, 0.9, 0.3, 0.4, 0.2]
    result = summarize(y, p, 0.5)
    json.dumps(result)  # raises if a numpy type leaked in
    assert result["tp"] == 1 and result["fn"] == 1 and result["fp"] == 0
    assert result["recall"] == pytest.approx(0.5)
    assert result["precision"] == pytest.approx(1.0)


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    y = np.concatenate([np.zeros(950), np.ones(50)]).astype(int)
    p = np.clip(0.3 * y + rng.random(len(y)) * 0.5, 0, 1)
    point = summarize(y, p, 0.5)["auprc"]
    ci = bootstrap_ci(y, p, 0.5, n_boot=200)
    assert ci["auprc"]["low"] <= point <= ci["auprc"]["high"]
    assert set(ci) == {"auprc", "roc_auc", "recall", "cost"}
