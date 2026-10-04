"""Shared helpers for the tests.

Tests marked requires_data or requires_model are skipped when those gitignored files
are missing. That lets CI (which has neither file) run everything else.
"""

import numpy as np
import pandas as pd
import pytest

from fraud_detection.config import DATA_PATH, MODELS_DIR

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

MODEL_FILE = MODELS_DIR / "fraud_model.joblib"

requires_data = pytest.mark.skipif(
    not DATA_PATH.exists(), reason="data/creditcard.csv not present (gitignored)"
)
requires_model = pytest.mark.skipif(
    not MODEL_FILE.exists(), reason="models/fraud_model.joblib not present; run scripts/train.py"
)


def make_synthetic(n=4000, fraud_rate=0.02, seed=0):
    """A small fake dataset with the same 30 columns.

    Frauds are shifted in four features so that a model has something to learn.
    """
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(int)
    X = pd.DataFrame(rng.normal(size=(n, 30)), columns=FEATURES)
    X["Amount"] = np.abs(X["Amount"]) * 100
    X["Time"] = np.abs(X["Time"]) * 1000
    for col in ["V1", "V2", "V3", "V4"]:
        X.loc[y == 1, col] += 3.0
    return X, pd.Series(y, name="Class")
