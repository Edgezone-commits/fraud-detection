"""Run the full training pipeline and save the model. Run from the project root:

    python scripts/train.py

The order below is deliberate, so no choice is made using the test set:
  1. Split the data 60/20/20 (train / validation / test).
  2. Train XGBoost on TRAIN.
  3. Fit the calibrator on TRAIN, then choose raw or calibrated scores on VALIDATION.
  4. Choose the decision threshold on VALIDATION (lowest cost).
  5. Score TEST once, with bootstrap confidence intervals, for reporting.
  6. Save models/fraud_model.joblib and reports/metrics.json.
"""

import json
import sys
from pathlib import Path

import joblib

# Make the src/ package importable when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud_detection.config import COST_FN, COST_FP, MODELS_DIR, RANDOM_STATE  # noqa: E402
from fraud_detection.data import load_data, split_data  # noqa: E402
from fraud_detection.drift import build_reference  # noqa: E402
from fraud_detection.evaluate import best_threshold, bootstrap_ci, summarize  # noqa: E402
from fraud_detection.reporting import read_report, update_report  # noqa: E402
from fraud_detection.train import (  # noqa: E402
    choose_probabilities,
    fit_calibrator,
    predict_fraud_proba,
    train_model,
)


# Chosen in Phase 2 by validation log loss (see "decisions" in reports/metrics.json).
CALIBRATION_METHOD = "isotonic"


def main():
    print("1. Loading and splitting data (60/20/20, stratified)...")
    X, y = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    splits = {
        "train": {"rows": len(y_train), "frauds": int(y_train.sum())},
        "validation": {"rows": len(y_val), "frauds": int(y_val.sum())},
        "test": {"rows": len(y_test), "frauds": int(y_test.sum())},
    }
    for name, info in splits.items():
        print(f"   {name:<10} rows={info['rows']:>7,}  frauds={info['frauds']}")

    print("2. Training XGBoost on TRAIN (class-weighted)...")
    raw_model = train_model(X_train, y_train)

    print(f"3. Fitting {CALIBRATION_METHOD} calibrator on TRAIN; choosing raw vs calibrated on VALIDATION...")
    calibrator = fit_calibrator(X_train, y_train, method=CALIBRATION_METHOD)
    choice = choose_probabilities(raw_model, calibrator, X_val, y_val)
    print(f"   validation log loss: raw={choice['raw_log_loss']:.5f}  "
          f"calibrated={choice['calibrated_log_loss']:.5f}  -> chosen: {choice['chosen']}")
    chosen_calibrator = calibrator if choice["chosen"] == "calibrated" else None

    print("4. Choosing the decision threshold on VALIDATION (lowest cost)...")
    val_proba = predict_fraud_proba(raw_model, chosen_calibrator, X_val)
    threshold, _ = best_threshold(y_val, val_proba)
    val_summary = summarize(y_val, val_proba, threshold)
    print(f"   threshold={threshold:.3f}  cost={val_summary['cost']}  "
          f"missed={val_summary['fn']}/{val_summary['fn'] + val_summary['tp']}  "
          f"false alarms={val_summary['fp']}")

    print("5. Scoring TEST once (this is the reported result; no more tuning after this)...")
    test_proba = predict_fraud_proba(raw_model, chosen_calibrator, X_test)
    test_summary = summarize(y_test, test_proba, threshold)
    test_ci = bootstrap_ci(y_test, test_proba, threshold)
    print(f"   AUPRC={test_summary['auprc']:.4f}  "
          f"CI [{test_ci['auprc']['low']:.4f}, {test_ci['auprc']['high']:.4f}]")
    print(f"   cost={test_summary['cost']}  missed={test_summary['fn']}  "
          f"false alarms={test_summary['fp']}")

    print("6. Saving model bundle and metrics...")
    MODELS_DIR.mkdir(exist_ok=True)
    bundle = {
        "model": raw_model,
        "calibrator": chosen_calibrator,      # None when raw scores were chosen
        "threshold": threshold,
        "feature_names": list(X.columns),     # the order the model expects
        "drift_reference": build_reference(X_train),
        "cost_fn": COST_FN,
        "cost_fp": COST_FP,
        "random_state": RANDOM_STATE,
    }
    joblib.dump(bundle, MODELS_DIR / "fraud_model.joblib")

    # Keep earlier test results instead of silently overwriting them. They are
    # reported as history, not as the final result.
    previous = read_report()
    test_history = previous.get("test_history", [])
    if "test" in previous:
        test_history.append({
            "label": "phase1_sigmoid_calibration (superseded)",
            "probability_choice": previous.get("probability_choice"),
            "threshold": previous.get("threshold"),
            "test": previous["test"],
            "test_bootstrap_95ci": previous.get("test_bootstrap_95ci"),
        })

    report_path = update_report({
        "final_model": {
            "model": "xgboost",
            "strategy": "class_weight",
            "calibration": CALIBRATION_METHOD,
            "chosen_probabilities": choice["chosen"],
        },
        "cost_fn": COST_FN,
        "cost_fp": COST_FP,
        "random_state": RANDOM_STATE,
        "splits": splits,
        "probability_choice": choice,
        "threshold": threshold,
        "validation": val_summary,
        "test": test_summary,
        "test_bootstrap_95ci": test_ci,
        "test_history": test_history,
    })
    print(f"   saved {MODELS_DIR / 'fraud_model.joblib'}")
    print(f"   saved {report_path}")


if __name__ == "__main__":
    main()
