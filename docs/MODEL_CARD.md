# Model card: credit card fraud detector

## Model details

- **Model:** XGBoost classifier (100 trees, default depth), trained with class weights (`scale_pos_weight` = legitimate / fraud = about 577 on the training split).
- **Calibration:** isotonic regression (`CalibratedClassifierCV`, 5-fold on the training split). Chosen because it had the lowest validation log loss.
- **Decision threshold:** 0.0785, chosen on the validation split to minimise cost = 10 × missed frauds + 1 × false alarm.
- **Version:** 0.1.0. The trained artifact `models/fraud_model.joblib` is not committed; regenerate it with `python scripts/train.py`.
- **Explanations:** SHAP TreeExplainer on the raw (uncalibrated) XGBoost model.

## Intended use

- Ranking card transactions by fraud risk, and flagging the highest-risk ones for review.
- A portfolio and learning project that shows an end-to-end, honestly evaluated fraud pipeline.

**Out of scope:**

- Automatically blocking or declining transactions without human review.
- Use on card data from other countries, periods, or payment networks without re-training and re-validating.
- Any decision that affects a person's access to credit or financial services.

## Data

- **Source:** public dataset of European card transactions from September 2013 (Kaggle, Machine Learning Group, ULB). 284,807 transactions, 492 frauds (0.17%).
- **Features:** `Time` (seconds since the first transaction), `V1`–`V28` (PCA components, anonymised; their original meaning is not disclosed), and `Amount` (transaction amount).
- **Split:** stratified 60/20/20 with seed 42. Train 170,883 rows (295 frauds), validation 56,962 (99), test 56,962 (98).
- **Not in the data:** customer identity, merchant, location, device, or any demographic attribute. So the model cannot be audited for demographic bias with this data.

## Evaluation

All numbers come from `reports/metrics.json`.

**Held-out test set (scored once for the final model):**

| Metric | Value | 95% bootstrap CI |
|---|---|---|
| AUPRC | 0.876 | 0.811 – 0.932 |
| ROC-AUC | 0.973 | 0.946 – 0.994 |
| Recall | 0.867 (85 of 98) | 0.800 – 0.932 |
| Precision | 0.773 | not computed |
| Cost | 155 (13 missed, 25 false alarms) | 86 – 232 |

**Validation (used for every choice):** AUPRC 0.825, recall 0.808, precision 0.800, cost 210 (19 missed, 20 false alarms) at threshold 0.0785.

**Cross-validation on train + validation (5 folds, 394 frauds):** XGBoost AUPRC 0.849 ± 0.029.

## Limitations

- **98 test frauds.** Confidence intervals are wide. Small differences between models are not meaningful.
- **Single, old dataset.** Fraud patterns from 2013 may not reflect current fraud. Performance on other data is unknown.
- **Anonymised features.** Explanations refer to V-numbers, so a human reviewer cannot connect them to merchant or location without the original mapping.
- **Cost ratio is an assumption.** The 10:1 ratio is not measured. The chosen threshold depends on it: at 20:1 it would be 0.031.
- **Threshold stability.** Under bootstrap resampling of the validation set, the chosen threshold ranges from 0.031 to 0.237 (5th–95th percentile). The cost is 210–220 for thresholds from 0.05 to 0.2, and rises to 274 at 0.5.
- **Saturated probabilities.** Isotonic calibration produces many scores of exactly 1.0, so probabilities at the top of the scale carry little information.
- **Test set reuse.** The test split comes from data explored in the early notebooks, and it was scored twice in this project (sigmoid, then isotonic). The reported result is the final model, selected on validation.
- **Unstable baseline.** Unweighted XGBoost gave validation AUPRC 0.0017 on one fit while CV folds gave 0.69–0.81. This was not investigated further.
- **Autoencoder is weaker.** The unsupervised autoencoder scores validation AUPRC 0.232 against 0.809 for the supervised model, so it is not used for predictions.

## Ethical and operational considerations

- **Customer impact.** A false alarm blocks or delays a legitimate customer's transaction. At the chosen threshold, validation shows 20 false alarms per 56,962 transactions. A real system would need a review process and a way for customers to contest flagged transactions.
- **Missed fraud.** At the chosen threshold, validation shows 19 of 99 frauds missed. Lowering the threshold catches more fraud at the cost of more false alarms. That trade-off should be set by the business, not by this project's assumed 10:1 ratio.
- **Explanations are not reasons for a decision.** SHAP shows what pushed a score up or down in this model. It does not prove why a transaction was fraudulent.
- **Monitoring.** The drift check (PSI and KS per feature) is a one-off comparison. A deployed system would need scheduled drift checks, performance tracking once labels arrive, and a plan for retraining.
- **Privacy.** The data is already anonymised and public. The API stores nothing about requests.

## Maintenance

- Re-run `scripts/run_experiments.py` and `scripts/train.py` after any change to data, features or cost assumptions.
- Re-validate on new data before any use. Do not reuse the test set for further model selection.
