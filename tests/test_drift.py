"""Drift check: PSI and KS flag real shifts and stay quiet on the same distribution."""

import numpy as np
import pytest

from fraud_detection.drift import build_reference, check_drift, psi
from helpers import make_synthetic


def test_psi_is_zero_for_identical_distributions():
    shares = np.array([0.25, 0.25, 0.25, 0.25])
    assert psi(shares, shares) == pytest.approx(0.0, abs=1e-9)


def test_psi_is_large_when_mass_moves_to_other_bins():
    expected = np.array([0.5, 0.5, 0.0, 0.0])
    actual = np.array([0.0, 0.0, 0.5, 0.5])
    assert psi(expected, actual) > 0.25


def test_check_drift_flags_a_shifted_feature():
    X, _ = make_synthetic(n=4000, seed=0)
    reference = build_reference(X)
    shifted = X.copy()
    shifted["Amount"] = shifted["Amount"] * 3
    status = check_drift(reference, shifted).set_index("feature")["status"]
    assert status["Amount"] == "significant"


def test_check_drift_is_quiet_on_the_same_distribution():
    X_train, _ = make_synthetic(n=4000, seed=0)
    X_new, _ = make_synthetic(n=4000, seed=1)
    result = check_drift(build_reference(X_train), X_new)
    assert (result["status"] == "significant").sum() == 0
    assert result["psi"].max() < 0.25


def test_reference_keeps_a_training_sample_for_ks():
    X, _ = make_synthetic(n=4000)
    reference = build_reference(X)
    assert len(reference["sample"]) == 4000
    assert set(reference["sample"].columns) == set(X.columns)
