"""Metrics for the fraud model.

- AUPRC (area under the precision-recall curve): the main metric. It focuses on the
  rare fraud class. A random model scores about 0.0017 (the fraud rate).
- ROC-AUC: shown for comparison. It looks high even when there are many false alarms.
- Cost: cost_fn * missed frauds + cost_fp * false alarms. The ratio is an assumption.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)

from fraud_detection.config import COST_FN, COST_FP, N_BOOTSTRAP, RANDOM_STATE

# Candidate thresholds, spaced evenly on a log scale from 1e-6 to 1.
# Calibrated fraud probabilities are often tiny, so a linear grid misses the best cut.
THRESHOLD_GRID = np.logspace(-6, 0, 1000)


def confusion_at(y_true, y_proba, threshold):
    """Counts of tn, fp, fn, tp. A row is predicted fraud if its score >= threshold."""
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def total_cost(y_true, y_proba, threshold, cost_fn=COST_FN, cost_fp=COST_FP):
    """Business cost at one threshold."""
    c = confusion_at(y_true, y_proba, threshold)
    return cost_fn * c["fn"] + cost_fp * c["fp"]


def cost_curve(y_true, y_proba, cost_fn=COST_FN, cost_fp=COST_FP):
    """Cost, missed frauds and false alarms for every threshold in THRESHOLD_GRID."""
    rows = []
    for t in THRESHOLD_GRID:
        c = confusion_at(y_true, y_proba, t)
        rows.append({
            "threshold": t,
            "cost": cost_fn * c["fn"] + cost_fp * c["fp"],
            "fn": c["fn"],
            "fp": c["fp"],
        })
    return pd.DataFrame(rows)


def best_threshold(y_true, y_proba, cost_fn=COST_FN, cost_fp=COST_FP):
    """Return (threshold with the lowest cost, full cost curve).

    Call this with VALIDATION data only. If several thresholds tie, the lowest one wins.
    """
    curve = cost_curve(y_true, y_proba, cost_fn, cost_fp)
    best_row = curve.loc[curve["cost"].idxmin()]
    return float(best_row["threshold"]), curve


def bootstrap_best_threshold(y_true, y_proba, n_boot=300, seed=RANDOM_STATE):
    """Best threshold for many bootstrap resamples. A wide spread means an unstable choice."""
    y = np.asarray(y_true)
    p = np.asarray(y_proba)
    rng = np.random.default_rng(seed)
    thresholds = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        t, _ = best_threshold(y[idx], p[idx])
        thresholds.append(t)
    return np.array(thresholds)


def summarize(y_true, y_proba, threshold, cost_fn=COST_FN, cost_fp=COST_FP):
    """All headline metrics at one threshold, as plain Python numbers (JSON-safe)."""
    c = confusion_at(y_true, y_proba, threshold)
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return {
        "threshold": float(threshold),
        "auprc": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, y_proba)),
        "precision": float(precision),
        "recall": float(recall),
        "cost": cost_fn * fn + cost_fp * fp,
        **c,
    }


def bootstrap_ci(y_true, y_proba, threshold, n_boot=N_BOOTSTRAP, seed=RANDOM_STATE, alpha=0.05):
    """95% percentile bootstrap confidence intervals for AUPRC, ROC-AUC, recall and cost.

    Each resample draws len(y) rows WITH replacement from the test set, recomputes the
    metrics, and we take the 2.5th and 97.5th percentiles of those results. The interval
    shows how much the numbers could move from sampling noise alone.
    """
    y = np.asarray(y_true)
    p = np.asarray(y_proba)
    rng = np.random.default_rng(seed)
    draws = {"auprc": [], "roc_auc": [], "recall": [], "cost": []}

    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yb, pb = y[idx], p[idx]
        c = confusion_at(yb, pb, threshold)
        draws["auprc"].append(average_precision_score(yb, pb))
        draws["roc_auc"].append(roc_auc_score(yb, pb))
        draws["recall"].append(c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] > 0 else 0.0)
        draws["cost"].append(COST_FN * c["fn"] + COST_FP * c["fp"])

    low_pct, high_pct = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        name: {
            "low": float(np.percentile(values, low_pct)),
            "high": float(np.percentile(values, high_pct)),
        }
        for name, values in draws.items()
    }
