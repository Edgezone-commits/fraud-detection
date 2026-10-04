"""Train the fraud model. Calibration and threshold are chosen on VALIDATION data only.

- make_xgboost: an untrained XGBoost model with class weighting.
- train_model: fits the raw model on the training data.
- fit_calibrator: sigmoid (Platt) calibration, cross-validated on the training data.
- choose_probabilities: picks raw or calibrated scores using validation log loss.
- predict_fraud_proba: the fraud probability used by evaluation and by the API.
"""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

from fraud_detection.config import CV_FOLDS, RANDOM_STATE


def make_xgboost(y_train, class_weighted=True):
    """XGBoost, by default with class weighting.

    scale_pos_weight = (legitimate rows) / (fraud rows), about 577 here. Each fraud
    then counts in the training loss like ~577 legitimate rows, so the model cannot
    score well by ignoring fraud. The weight comes from training labels only.
    Pass class_weighted=False for the no-weighting baseline.
    """
    if not class_weighted:
        return XGBClassifier(n_estimators=100, eval_metric="aucpr", random_state=RANDOM_STATE, n_jobs=-1)
    n_legit = int((y_train == 0).sum())
    n_fraud = int((y_train == 1).sum())
    return XGBClassifier(
        n_estimators=100,
        scale_pos_weight=n_legit / n_fraud,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def train_model(X_train, y_train):
    """Fit the raw (uncalibrated) XGBoost model on training data."""
    model = make_xgboost(y_train)
    model.fit(X_train, y_train)
    return model


def fit_calibrator(X_train, y_train, method="isotonic"):
    """Calibration trained on training data only. method: "sigmoid" or "isotonic".

    CalibratedClassifierCV trains CV_FOLDS copies of the model on parts of the
    training data, then learns a map from raw scores to probabilities.
    Sigmoid (Platt) has 2 parameters. Isotonic is a flexible step function.
    The method is chosen in scripts/run_experiments.py by validation log loss.
    """
    calibrator = CalibratedClassifierCV(
        estimator=make_xgboost(y_train), method=method, cv=CV_FOLDS
    )
    calibrator.fit(X_train, y_train)
    return calibrator


def choose_probabilities(raw_model, calibrator, X_val, y_val):
    """Choose raw or calibrated probabilities using validation log loss.

    Log loss punishes confident wrong predictions, so it measures probability quality
    and not just ranking. The lower value wins. Only validation data is used here.
    """
    raw_val = raw_model.predict_proba(X_val)[:, 1]
    cal_val = calibrator.predict_proba(X_val)[:, 1]
    raw_ll = float(log_loss(y_val, raw_val))
    cal_ll = float(log_loss(y_val, cal_val))
    return {
        "chosen": "calibrated" if cal_ll < raw_ll else "raw",
        "raw_log_loss": raw_ll,
        "calibrated_log_loss": cal_ll,
    }


def predict_fraud_proba(model, calibrator, X):
    """Fraud probability. Uses the calibrator when one was chosen (calibrator is not None)."""
    scorer = calibrator if calibrator is not None else model
    return scorer.predict_proba(X)[:, 1]
