# Decisions and why

Each section gives the decision, the alternatives, the reason, and what to say about it in an interview. Numbers come from `reports/metrics.json`.

## 1. Measure with AUPRC, not accuracy

**Decision:** AUPRC is the main metric. Recall, precision and a business cost are reported alongside it.

**Why:** Only 0.17% of transactions are fraud. A model that never flags fraud gets 99.83% accuracy and catches none. AUPRC focuses on the rare class: a random classifier scores about 0.0017, and the score rises only when the model ranks frauds above legitimate transactions.

**Why not ROC-AUC alone:** ROC-AUC is also reported (0.973 on test), but it can look strong while there are many false alarms, because the legitimate class is so large.

**Interview line:** "Accuracy is meaningless at this imbalance. I used AUPRC for ranking and a cost function for the decision."

## 2. A business cost with a 10:1 ratio

**Decision:** Cost = 10 × missed frauds + 1 × false alarm.

**Why:** The two errors are not equally expensive. A missed fraud loses the money and the chargeback. A false alarm costs a customer's time and some goodwill. The ratio is an assumption, and it is stated as one everywhere it appears. A ratio of 5:1 gives the same threshold on validation; 20:1 moves it to 0.031.

**Interview line:** "The 10:1 ratio is my assumption, not a measured number. I show how the threshold moves when it changes."

## 3. A stratified 60/20/20 split with a fixed seed

**Decision:** Train 60%, validation 20%, test 20%. Stratified on the label, seed 42.

**Why:** Stratification keeps the fraud rate (about 0.17%) the same in every part, and each part still has frauds to measure. The validation set is where every choice is made. The test set is for the final report. Without a validation set, the choices would be tuned on the test set, which inflates the results.

**Evidence:** The earlier notebooks chose the threshold and the calibration on the same test set they reported. Their cost reduction of 18% (182 → 149) cannot be reproduced as an honest test result. On validation, the chosen threshold reduces cost from 274 to 210 (23%).

**Interview line:** "I found my earlier notebooks were tuning on the test set. I moved every choice to validation and kept the test set for one final score."

## 4. Test set touched once, plus one documented exception

**Decision:** Experiments use validation or 5-fold CV on train + validation. Only `scripts/train.py` scores the test set.

**Exception:** The test set was scored twice. Phase 1 used sigmoid calibration, and Phase 2 then chose isotonic on validation, so the final model was scored again. Both results are kept in `reports/metrics.json` (the sigmoid run is `test_history`). The reported result is the final model.

**Why this matters:** Scoring the test set more than once creates a risk of picking whichever run looks best. The README states the exception, and the chosen model was fixed by a rule on validation before the second run.

**Interview line:** "I broke my own rule once. I say so in the README and show both runs, and the final choice was made on validation."

## 5. Class weights over SMOTE and undersampling

**Decision:** XGBoost with `scale_pos_weight` = legitimate / fraud. No resampling.

**Evidence (5-fold CV AUPRC):** no weighting 0.745 ± 0.056, class weights 0.849 ± 0.029, SMOTE 0.857 ± 0.022, undersampling 0.722 ± 0.063.

**Why:** SMOTE scores 0.008 higher than class weights, which is less than class weighting's standard deviation, so the difference is not meaningful. Class weights are simpler: no synthetic rows, and no risk of creating fraud examples that look nothing like real fraud. Undersampling throws away most legitimate data and is the weakest option.

**Rule used:** a resampling method must beat class weights by more than class weights' CV standard deviation to be chosen. This rule was fixed before the comparison ran.

**Interview line:** "SMOTE was slightly ahead, but within noise, so I kept the simpler method. I decided that rule before I ran the comparison."

## 6. XGBoost as the model

**Decision:** Gradient-boosted trees (XGBoost), 100 trees.

**Evidence (5-fold CV AUPRC):** XGBoost 0.849 ± 0.029, random forest 0.848 ± 0.026, logistic regression 0.750 ± 0.044.

**Why:** XGBoost ties random forest within noise, and it handles the rare class well through weights. Its SHAP explanations are fast and exact for tree models, which the API needs. Logistic regression is clearly weaker, so the extra complexity of trees is justified.

**Caveat:** Unweighted XGBoost behaved unstably on one fit (validation AUPRC 0.0017) while its CV folds scored 0.69–0.81. Weighting the classes avoids this, but the instability was not fully explained.

**Interview line:** "XGBoost and random forest are tied within noise. I chose XGBoost because it works well with weights and gives fast exact SHAP values."

## 7. Calibration: why, and which method

**Decision:** Isotonic regression, cross-validated on the training split.

**Why calibrate at all:** A class-weighted model's scores are not probabilities. A score of 0.9 does not mean a 90% chance of fraud. The calibrated probability is what the API reports, and it is needed to reason about thresholds in cost terms.

**Choice:** Raw, sigmoid and isotonic were compared on validation log loss: raw 0.00431, sigmoid 0.00358, isotonic 0.00293. Isotonic won.

**Caveat:** I had recommended sigmoid earlier, because isotonic has more parameters and can overfit with few positives. The rule picked isotonic, and I followed it. On test, isotonic (AUPRC 0.876) and sigmoid (0.883) are not clearly different. Isotonic produces many scores of exactly 1.0, which limits resolution at the top of the scale.

**Interview line:** "Log loss on validation picked isotonic, even though I expected sigmoid to be more stable. I followed the rule and report the caveat."

## 8. Threshold chosen on validation by minimising cost

**Decision:** Threshold = the value on a log-spaced grid (1e-6 to 1, 1000 points) that minimises cost on validation. The chosen value is 0.0785.

**Why a log grid:** Calibrated fraud probabilities are often tiny. A linear grid from 0.001 stopped at its lowest value, which showed that the search range was too narrow.

**Why this threshold:** It gives validation cost 210, against 274 at 0.5. On validation, that is 19 frauds missed and 20 false alarms.

**Stability:** Bootstrap resampling of validation gives a 5th–95th percentile range of 0.031 – 0.237. Cost is 210–220 for any threshold between 0.05 and 0.2, and it rises to 348 at 0.005 and 274 at 0.5. So the choice matters more on the low side.

**Interview line:** "The threshold is chosen by minimising a cost function on validation. I show how much it moves under resampling, and where the cost rises."

## 9. Drift check with PSI and KS

**Decision:** For each feature, compare new data with the training data using PSI (bins from training quantiles) and the KS test. PSI above 0.25 is "significant" and above 0.10 is "watch". KS p-value below 0.01 is "watch".

**Why:** PSI is standard in credit-risk work and easy to explain. KS compares full distributions and catches changes PSI can miss. The training sample is saved so KS can run later.

**Check:** Validation data from the same distribution gives a maximum PSI of 0.0004. Scaling `Amount` by 3 is flagged as significant (PSI 0.42).

**Limitation:** This is a snapshot check, not monitoring. It has no schedule and no alerting.

**Interview line:** "PSI tells me how much a feature's distribution moved, KS tells me whether it moved significantly, and both are cheap to run."

## 10. SHAP for explanations, from the raw model

**Decision:** SHAP TreeExplainer on the raw XGBoost model. Return the top 5 contributions for one transaction.

**Why raw:** Calibration wraps the model in a way SHAP does not support. The explanation therefore reflects the raw model's log-odds, not the calibrated probability. The API says so in an `explanation_note`.

**Why top-5:** It is short enough to read, and the direction (toward fraud or legitimate) is shown for each feature.

**Limitation:** Features are PCA components (V1–V28), so the reasons are not business concepts.

**Interview line:** "SHAP explains the raw model because the calibrator is a wrapper around it. I say that in the response rather than hiding it."

## 11. The autoencoder as a side experiment

**Decision:** A simple reconstruction autoencoder trained on legitimate transactions only. Its reconstruction error is the anomaly score. It is evaluated on validation only, in a notebook. It is not used for predictions.

**Result:** Validation AUPRC 0.232, against 0.809 for the supervised XGBoost on the same rows.

**Why keep it:** It shows what an unsupervised approach can and cannot do here. It also documents a comparison that was in the original project.

**Not the same as the cited paper:** The 2025 paper uses an adversarial autoencoder. This project's autoencoder is a plain reconstruction model.

**Interview line:** "The autoencoder works without labels, but on this data it is much weaker than a supervised model, so I did not use it."

## 12. Serving design

**Decisions:**

- **Load once at startup.** The model and SHAP explainer are expensive to create, so they are loaded at startup rather than per request.
- **Strict input schema.** Pydantic checks the 30 fields, their types and their ranges. Unknown keys are rejected. The schema is checked against the saved model's feature names at startup.
- **NaN and Infinity return 422.** The default FastAPI handler crashed with a 500, because it echoes NaN back as JSON. A custom handler fixes this.
- **Degraded mode.** If the model file is missing, `/health` reports `degraded` and `/predict` returns 503 with the fix.
- **Non-root Docker user and a health check.** The health check reads `model_loaded`, because `/health` returns 200 even when the model is missing.

**Interview line:** "I tested bad input, not just good input. The NaN case exposed a real 500 error, which I fixed."

## 13. Project layout, dependencies and CI

**Decisions:**

- **`src/` package, with scripts and an app on top.** Logic is in one tested place, and the notebooks are kept as history.
- **Pinned dependencies, split by purpose.** `requirements.txt` for runtime, `requirements-test.txt` for tests and CI, `requirements-dev.txt` for notebooks and plots. CI does not install torch.
- **Tests that need data are skipped in CI.** The dataset and model are gitignored, so CI cannot run those tests. The skips are shown in the log so they are not silent.

**Interview line:** "CI runs without the dataset, so I designed the tests to run on synthetic data and skip the rest visibly."

## 14. Git history rewritten to a standalone repo

**Decision:** The project was a subfolder of the home-directory repository. I rebuilt its history as a standalone repository, keeping the commits and messages. The commit hashes changed.

**Consequence:** The remote `main` on GitHub still has the old history, so pushing needs a force-push. That has not been done.

**Interview line:** "The project's history was in a subfolder of my home repo. I moved it into its own repository, which means the commit hashes changed."
