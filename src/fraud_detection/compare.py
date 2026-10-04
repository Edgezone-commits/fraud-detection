"""Models and imbalance strategies compared in Phase 2.

Each builder takes the TRAINING labels and returns an UNFITTED estimator, so any
class weight or resampling is computed from training data only.
"""

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_detection.config import CV_FOLDS, RANDOM_STATE
from fraud_detection.train import make_xgboost


def logistic_regression(y_train):
    """Scaled logistic regression with balanced class weights."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)),
    ])


def random_forest(y_train):
    """200 trees with balanced class weights."""
    return RandomForestClassifier(
        n_estimators=200, class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE
    )


def xgb_no_weights(y_train):
    """XGBoost with no imbalance handling at all (the baseline for the strategy test)."""
    return make_xgboost(y_train, class_weighted=False)


def xgb_smote(y_train):
    """SMOTE creates synthetic fraud rows, but only inside each training fold."""
    return ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("model", xgb_no_weights(y_train)),
    ])


def xgb_undersample(y_train):
    """Random undersampling throws away legitimate rows, inside each training fold."""
    return ImbPipeline([
        ("under", RandomUnderSampler(random_state=RANDOM_STATE)),
        ("model", xgb_no_weights(y_train)),
    ])


MODEL_BUILDERS = {
    "logistic_regression": logistic_regression,
    "random_forest": random_forest,
    "xgboost": make_xgboost,
}

STRATEGY_BUILDERS = {
    "none": xgb_no_weights,
    "class_weight": make_xgboost,
    "smote": xgb_smote,
    "undersample": xgb_undersample,
}


def cv_auprc(builder, X, y):
    """Stratified 5-fold CV. Each fold gets a fresh model trained on its own training rows."""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = []
    for train_idx, test_idx in skf.split(X, y):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        model = builder(y_tr)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X.iloc[test_idx])[:, 1]
        fold_scores.append(float(average_precision_score(y.iloc[test_idx], proba)))
    return {
        "mean": float(np.mean(fold_scores)),
        "std": float(np.std(fold_scores, ddof=1)),
        "folds": fold_scores,
    }


def fit_on_train(builder, X_train, y_train):
    """Fit one model on the full training split and return it."""
    model = builder(y_train)
    model.fit(X_train, y_train)
    return model
