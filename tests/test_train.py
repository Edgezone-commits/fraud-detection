"""Training helpers: class weights and the rules for choosing probabilities."""

import numpy as np
import pandas as pd
import pytest

from fraud_detection.train import choose_probabilities, make_xgboost, predict_fraud_proba


class FakeModel:
    """Stands in for a fitted model. It returns fixed fraud probabilities."""

    def __init__(self, proba):
        self.proba = np.asarray(proba, dtype=float)

    def predict_proba(self, X):
        return np.column_stack([1 - self.proba, self.proba])


def test_scale_pos_weight_is_legit_over_fraud():
    y = pd.Series([0] * 980 + [1] * 20)
    assert make_xgboost(y).get_params()["scale_pos_weight"] == pytest.approx(49.0)


def test_unweighted_model_has_no_scale_pos_weight():
    y = pd.Series([0] * 980 + [1] * 20)
    assert make_xgboost(y, class_weighted=False).get_params()["scale_pos_weight"] is None


def test_weight_comes_from_the_labels_it_is_given():
    # Changing the labels changes the weight, so the weight depends on training labels only.
    a = make_xgboost(pd.Series([0] * 90 + [1] * 10)).get_params()["scale_pos_weight"]
    b = make_xgboost(pd.Series([0] * 50 + [1] * 50)).get_params()["scale_pos_weight"]
    assert a == pytest.approx(9.0) and b == pytest.approx(1.0)


def test_choose_probabilities_prefers_the_lower_log_loss():
    y = np.array([0, 0, 1, 1])
    raw = FakeModel([0.3, 0.4, 0.6, 0.7])
    calibrated = FakeModel([0.01, 0.02, 0.98, 0.99])
    assert choose_probabilities(raw, calibrated, None, y)["chosen"] == "calibrated"
    assert choose_probabilities(calibrated, raw, None, y)["chosen"] == "raw"


def test_predict_fraud_proba_uses_the_calibrator_when_given():
    model = FakeModel([0.2, 0.2])
    calibrator = FakeModel([0.9, 0.1])
    assert list(predict_fraud_proba(model, calibrator, None)) == [0.9, 0.1]


def test_predict_fraud_proba_uses_the_raw_model_without_a_calibrator():
    model = FakeModel([0.2, 0.7])
    assert list(predict_fraud_proba(model, None, None)) == [0.2, 0.7]
