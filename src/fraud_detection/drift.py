"""Drift check: has new data moved away from the training data, feature by feature?

Two tests per feature:
- PSI (Population Stability Index). Bins are set from TRAINING quantiles, then we
  compare the share of rows per bin. Rule of thumb: < 0.10 stable, 0.10-0.25 watch,
  > 0.25 significant shift.
- KS test (Kolmogorov-Smirnov). Compares the two distributions directly. A p-value
  below 0.01 is flagged. It needs raw training values, so we save a sample of them.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from fraud_detection.config import RANDOM_STATE

N_BINS = 10
REFERENCE_SAMPLE_SIZE = 5000
PSI_WATCH = 0.10
PSI_SIGNIFICANT = 0.25
KS_ALERT_P = 0.01
EPS = 1e-6  # avoids log(0) when a bin is empty


def _bin_shares(values, inner_edges):
    """Share of values in each bin. The outer bins catch values outside the training range."""
    idx = np.digitize(values, inner_edges)  # bin index 0 .. len(inner_edges)
    counts = np.bincount(idx, minlength=len(inner_edges) + 1)
    return counts / counts.sum()


def build_reference(X_train, seed=RANDOM_STATE):
    """Save what the drift check needs: bin cut points, training bin shares and a sample."""
    inner_edges, train_shares = {}, {}
    for col in X_train.columns:
        quantiles = np.unique(np.quantile(X_train[col], np.linspace(0, 1, N_BINS + 1)))
        cuts = quantiles[1:-1]  # interior cut points only
        inner_edges[col] = cuts
        train_shares[col] = _bin_shares(X_train[col].to_numpy(), cuts)

    sample = X_train.sample(n=min(REFERENCE_SAMPLE_SIZE, len(X_train)), random_state=seed)
    return {"inner_edges": inner_edges, "train_shares": train_shares, "sample": sample}


def psi(expected_shares, actual_shares):
    """Population Stability Index between two sets of bin shares."""
    e = np.clip(expected_shares, EPS, None)
    a = np.clip(actual_shares, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def check_drift(reference, X_new):
    """One row per feature: PSI, KS p-value and a status label."""
    rows = []
    for col, cuts in reference["inner_edges"].items():
        actual = _bin_shares(X_new[col].to_numpy(), cuts)
        score = psi(reference["train_shares"][col], actual)
        ks_p = float(ks_2samp(reference["sample"][col], X_new[col]).pvalue)

        if score > PSI_SIGNIFICANT:
            status = "significant"
        elif score > PSI_WATCH or ks_p < KS_ALERT_P:
            status = "watch"
        else:
            status = "stable"
        rows.append({"feature": col, "psi": score, "ks_p_value": ks_p, "status": status})

    return pd.DataFrame(rows)
