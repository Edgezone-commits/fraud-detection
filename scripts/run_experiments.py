"""Phase 2: honest comparison of models, imbalance strategies, calibration and thresholds.

Run from the project root:  python scripts/run_experiments.py

Protocol, fixed before the run:
- Dev set = train + validation (80% of rows). 5-fold stratified CV on it gives
  mean +/- std AUPRC for model and strategy comparisons.
- Validation set = the 20% holdout used for calibration and threshold choices.
- The TEST set is NOT loaded here. Final test scoring happens only in scripts/train.py.

Decision rules, applied in code:
1. Model family: highest mean CV AUPRC.
2. Imbalance strategy (XGBoost only): highest mean CV AUPRC. Class weighting is kept
   unless another strategy beats it by more than class weighting's CV std, because
   the simpler method wins ties.
3. Calibration: lowest validation log loss (raw, sigmoid, isotonic).
4. Threshold: lowest validation cost at the 10:1 cost ratio.

Outputs: sections in reports/metrics.json and plots in outputs/.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save plots to files without opening windows
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV, calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud_detection.compare import (  # noqa: E402
    MODEL_BUILDERS,
    STRATEGY_BUILDERS,
    cv_auprc,
    fit_on_train,
)
from fraud_detection.config import CV_FOLDS, OUTPUTS_DIR, RANDOM_STATE  # noqa: E402
from fraud_detection.data import load_data, split_data  # noqa: E402
from fraud_detection.evaluate import best_threshold, bootstrap_best_threshold, summarize  # noqa: E402
from fraud_detection.reporting import update_report  # noqa: E402


def compare_models(X_dev, y_dev, X_train, y_train, X_val, y_val):
    """Step 1: which model family? Returns the results and the fitted validation models."""
    results, fitted = {}, {}
    for name, builder in MODEL_BUILDERS.items():
        print(f"   {name}: 5-fold CV on dev...")
        cv = cv_auprc(builder, X_dev, y_dev)
        model = fit_on_train(builder, X_train, y_train)
        proba = model.predict_proba(X_val)[:, 1]
        results[name] = {
            "cv_auprc_mean": cv["mean"],
            "cv_auprc_std": cv["std"],
            "cv_auprc_folds": cv["folds"],
            "validation_auprc": float(average_precision_score(y_val, proba)),
        }
        fitted[name] = proba
        print(f"      CV AUPRC {cv['mean']:.4f} +/- {cv['std']:.4f}   validation {results[name]['validation_auprc']:.4f}")
    return results, fitted


def compare_strategies(X_dev, y_dev, X_train, y_train, X_val, y_val):
    """Step 2: which way to handle the 0.17% fraud rate? (XGBoost only)."""
    results, fitted = {}, {}
    for name, builder in STRATEGY_BUILDERS.items():
        print(f"   {name}: 5-fold CV on dev...")
        cv = cv_auprc(builder, X_dev, y_dev)
        model = fit_on_train(builder, X_train, y_train)
        proba = model.predict_proba(X_val)[:, 1]
        results[name] = {
            "cv_auprc_mean": cv["mean"],
            "cv_auprc_std": cv["std"],
            "cv_auprc_folds": cv["folds"],
            "validation_auprc": float(average_precision_score(y_val, proba)),
        }
        fitted[name] = proba
        print(f"      CV AUPRC {cv['mean']:.4f} +/- {cv['std']:.4f}   validation {results[name]['validation_auprc']:.4f}")
    return results, fitted


def choose_strategy(results):
    """Rule 2: class weighting stays unless another strategy beats it by more than its std."""
    cw = results["class_weight"]
    best_name = max(results, key=lambda k: results[k]["cv_auprc_mean"])
    gain = results[best_name]["cv_auprc_mean"] - cw["cv_auprc_mean"]
    if best_name != "class_weight" and gain > cw["cv_auprc_std"]:
        return best_name
    return "class_weight"


def compare_calibration(builder, X_train, y_train, X_val, y_val):
    """Step 3: raw vs sigmoid vs isotonic, judged by validation log loss."""
    raw = fit_on_train(builder, X_train, y_train)
    candidates = {"raw": raw}
    for method in ["sigmoid", "isotonic"]:
        cal = CalibratedClassifierCV(estimator=builder(y_train), method=method, cv=CV_FOLDS)
        cal.fit(X_train, y_train)
        candidates[method] = cal

    results, probas = {}, {}
    for name, model in candidates.items():
        p = model.predict_proba(X_val)[:, 1]
        probas[name] = p
        results[name] = {
            "validation_log_loss": float(log_loss(y_val, p)),
            "validation_brier": float(brier_score_loss(y_val, p)),
        }
    chosen = min(results, key=lambda k: results[k]["validation_log_loss"])
    return results, probas, chosen


def plot_pr_curves(fitted, y_val, title, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, proba in fitted.items():
        precision, recall, _ = precision_recall_curve(y_val, proba)
        ax.plot(recall, precision, label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_reliability(probas, y_val, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    for name, p in probas.items():
        frac_pos, mean_pred = calibration_curve(y_val, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Predicted probability (log scale)")
    ax.set_ylabel("Observed fraud rate (log scale)")
    ax.set_title("Validation calibration")
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cost_curve(curve, threshold, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(curve["threshold"], curve["cost"], color="tab:blue")
    ax.axvline(threshold, color="green", linestyle="--", label=f"Validation optimum ({threshold:.5f})")
    ax.axvline(0.5, color="gray", linestyle=":", label="Default 0.5")
    ax.set_xscale("log")
    ax.set_xlabel("Threshold (log scale)")
    ax.set_ylabel("Cost = 10*FN + FP")
    ax.set_title("Validation cost curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUTS_DIR.mkdir(exist_ok=True)

    print("Loading data. The test split is discarded here and not used.")
    X, y = load_data()
    X_train, X_val, _, y_train, y_val, _ = split_data(X, y)
    X_dev = pd.concat([X_train, X_val])
    y_dev = pd.concat([y_train, y_val])
    print(f"   dev rows={len(y_dev):,} (frauds={int(y_dev.sum())})   validation rows={len(y_val):,} (frauds={int(y_val.sum())})")

    print("\nStep 1. Model families")
    model_results, model_probas = compare_models(X_dev, y_dev, X_train, y_train, X_val, y_val)
    best_model = max(model_results, key=lambda k: model_results[k]["cv_auprc_mean"])
    print(f"   -> winner by CV AUPRC: {best_model}")

    print("\nStep 2. Imbalance strategies (XGBoost)")
    strategy_results, strategy_probas = compare_strategies(X_dev, y_dev, X_train, y_train, X_val, y_val)
    chosen_strategy = choose_strategy(strategy_results)
    print(f"   -> chosen strategy (rule 2): {chosen_strategy}")

    # Calibration and threshold steps need a builder. They use the chosen XGBoost strategy.
    # If the model rule chose a non-XGBoost model, the pipeline stops and reports it.
    if best_model != "xgboost":
        print("\nSTOP: a non-XGBoost model won Step 1. Calibration and threshold steps")
        print("are built for XGBoost, so they were not run. Review before continuing.")
        update_report({"model_comparison": model_results, "imbalance_strategies": strategy_results,
                       "decisions": {"model": best_model, "strategy": chosen_strategy,
                                     "status": "stopped: non-XGBoost model won"}})
        return

    builder = STRATEGY_BUILDERS[chosen_strategy]

    print("\nStep 3. Calibration (validation log loss decides)")
    cal_results, cal_probas, chosen_cal = compare_calibration(builder, X_train, y_train, X_val, y_val)
    for name, r in cal_results.items():
        print(f"   {name:<9} log loss={r['validation_log_loss']:.5f}  brier={r['validation_brier']:.6f}")
    print(f"   -> chosen calibration (rule 3): {chosen_cal}")
    p_val = cal_probas[chosen_cal]

    print("\nStep 4. Threshold (validation cost at 10:1)")
    threshold, curve = best_threshold(y_val, p_val)
    val_at_best = summarize(y_val, p_val, threshold)
    val_at_half = summarize(y_val, p_val, 0.5)
    print(f"   optimum threshold={threshold:.6f}  cost={val_at_best['cost']}  "
          f"missed={val_at_best['fn']}  false alarms={val_at_best['fp']}")
    print(f"   at default 0.5: cost={val_at_half['cost']}  missed={val_at_half['fn']}  false alarms={val_at_half['fp']}")

    print("   bootstrap stability of the threshold (300 resamples of validation)...")
    boot = bootstrap_best_threshold(y_val, p_val, n_boot=300)
    stability = {
        "median": float(np.median(boot)),
        "p05": float(np.percentile(boot, 5)),
        "p95": float(np.percentile(boot, 95)),
    }
    print(f"      median={stability['median']:.6f}  5th-95th pct=[{stability['p05']:.6f}, {stability['p95']:.6f}]")

    print("   cost-ratio sensitivity (validation):")
    ratio_table = {}
    for ratio in [5, 10, 20]:
        t_r, _ = best_threshold(y_val, p_val, cost_fn=ratio, cost_fp=1)
        ratio_table[f"{ratio}:1"] = {"threshold": t_r}
        print(f"      {ratio}:1 -> threshold {t_r:.6f}")

    print("\nSaving plots and metrics...")
    plot_pr_curves(model_probas, y_val, "Validation PR curves: model families", OUTPUTS_DIR / "phase2_model_pr.png")
    plot_pr_curves(strategy_probas, y_val, "Validation PR curves: imbalance strategies (XGBoost)",
                   OUTPUTS_DIR / "phase2_strategy_pr.png")
    plot_reliability(cal_probas, y_val, OUTPUTS_DIR / "phase2_calibration_reliability.png")
    plot_cost_curve(curve, threshold, OUTPUTS_DIR / "phase2_threshold_cost_curve.png")

    report_path = update_report({
        "phase2_protocol": {
            "dev_rows": len(y_dev), "dev_frauds": int(y_dev.sum()),
            "validation_rows": len(y_val), "validation_frauds": int(y_val.sum()),
            "cv_folds": CV_FOLDS, "random_state": RANDOM_STATE,
            "test_set_used": False,
        },
        "model_comparison": model_results,
        "imbalance_strategies": strategy_results,
        "calibration_validation": cal_results,
        "threshold_validation": {
            "optimum": threshold,
            "at_optimum": val_at_best,
            "at_default_0_5": val_at_half,
            "bootstrap_stability": stability,
            "cost_ratio_sensitivity": ratio_table,
        },
        "decisions": {
            "model": best_model,
            "strategy": chosen_strategy,
            "calibration": chosen_cal,
            "threshold": threshold,
        },
    })
    print(f"   saved {report_path}")
    print("   saved outputs/phase2_*.png (4 plots)")


if __name__ == "__main__":
    main()
