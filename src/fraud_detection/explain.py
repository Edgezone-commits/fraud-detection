"""Explain one prediction with SHAP: which features pushed the fraud score up or down."""

import numpy as np
import shap


def make_explainer(model):
    """Build a SHAP TreeExplainer. Create it once and reuse it, since building it is slow."""
    return shap.TreeExplainer(model)


def top_reasons(explainer, feature_names, row, k=5):
    """The k features with the largest effect on one transaction's score.

    row: a pandas DataFrame with exactly one row and the model's feature columns.
    Values are SHAP contributions in log-odds of the raw XGBoost model:
    positive pushes toward fraud, negative pushes toward legitimate.
    The raw model is used on purpose, because a calibrator has no SHAP support.
    """
    contributions = explainer.shap_values(row)[0]
    top_idx = np.argsort(-np.abs(contributions))[:k]
    return [
        {
            "feature": feature_names[i],
            "value": float(row.iloc[0, i]),
            "contribution": float(contributions[i]),
        }
        for i in top_idx
    ]
