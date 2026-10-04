# Credit Card Fraud Detection

An end-to-end fraud detector for a public dataset of 284,807 card transactions, of which 492 (0.17%) are fraud. The final model is a class-weighted XGBoost classifier with isotonic calibration. Its decision threshold is chosen on a validation set using a 10:1 cost ratio. It is served by a FastAPI service that returns a fraud probability, a decision and the top SHAP reasons for each transaction. A drift check compares new data with the training data. Results are reported on a held-out test set, scored once, with bootstrap confidence intervals.

## Why accuracy fails here

A model that never flags fraud scores **99.83% accuracy** and catches **0 of 98** frauds in the test set. This project therefore uses:

- **AUPRC** (area under the precision-recall curve) as the main ranking metric. A random model scores about 0.0017.
- **Recall, precision and business cost.** Cost = 10 × missed frauds + 1 × false alarm. **The 10:1 ratio is an assumption, not a measured value.**

## Results

Held-out **test set**: 56,962 transactions, 98 frauds. Scored once, for the final model.

| Metric | Test | 95% bootstrap CI |
|---|---|---|
| AUPRC | **0.876** | 0.811 – 0.932 |
| ROC-AUC | 0.973 | 0.946 – 0.994 |
| Recall | **0.867** (85 of 98 caught) | 0.800 – 0.932 |
| Precision | 0.773 | not computed |
| Cost (10 × missed + false alarms) | **155** (13 missed, 25 false alarms) | 86 – 232 |

The intervals are wide because the test set has only 98 frauds. Differences of a few frauds are within noise.

**Model comparison** (5-fold stratified CV on train + validation: 227,845 rows, 394 frauds; mean ± std AUPRC):

| Model | CV AUPRC |
|---|---|
| XGBoost, class-weighted (chosen) | 0.849 ± 0.029 |
| Random forest, balanced | 0.848 ± 0.026 |
| Logistic regression, balanced | 0.750 ± 0.044 |

**Imbalance strategy** (XGBoost; CV AUPRC):

| Strategy | CV AUPRC |
|---|---|
| No weighting | 0.745 ± 0.056 |
| Class weights (chosen) | 0.849 ± 0.029 |
| SMOTE | 0.857 ± 0.022 |
| Random undersampling | 0.722 ± 0.063 |

SMOTE is 0.008 higher, which is less than class weighting's standard deviation. The simpler method was kept.

**Calibration** (validation log loss, lower is better): raw 0.00431, sigmoid 0.00358, **isotonic 0.00293 (chosen)**.

**Threshold** (chosen on validation only): **0.0785**. On validation, this gives a cost of 210 against 274 at the default 0.5, a 23% reduction. The threshold is unstable under resampling (bootstrap 5th–95th percentile 0.031 – 0.237). The validation cost is 210–220 for any threshold between 0.05 and 0.2, but it rises to 348 at 0.005 and to 274 at 0.5, so the choice matters more on the low side.

## Key findings

1. **The honest numbers are lower than the earlier notebooks.** The old notebooks reported AUPRC 0.879 and a cost of 149. That cost was tuned on the same test set it was reported on. The validation-based comparison above is the honest one.
2. **Class weighting and SMOTE are statistically indistinguishable** in CV. Class weighting was kept as the simpler method.
3. **Random undersampling is the weakest strategy.** The old notebook reported 0.387 on the test set. Under 5-fold CV it is 0.722 ± 0.063.
4. **Isotonic calibration gave the best validation log loss.** Its output is a step function, so many transactions get a score of exactly 1.0. Ranking is still good, but the top of the probability scale has little resolution.
5. **Unweighted XGBoost is unstable here.** One training fit scored validation AUPRC 0.0017 (the fraud rate itself), while the five CV folds of the same configuration scored between 0.69 and 0.81. Class weights avoid this.
6. **The autoencoder is clearly weaker.** On the same validation rows, it scores AUPRC 0.232 against 0.809 for the class-weighted XGBoost. This is an unsupervised side experiment, and it is a simple reconstruction autoencoder, not the adversarial method of the paper cited below.

## Limitations

- **Anonymised features.** V1–V28 are PCA components, so the explanations refer to V-numbers, not to business concepts such as merchant or location.
- **Small test set.** 98 frauds gives wide intervals. The AUPRC interval spans about ±0.06.
- **One dataset from September 2013** (European cardholders). Fraud patterns, rates and card usage change over time and between countries.
- **The cost ratio is assumed.** Changing it changes the threshold. At 20:1 the validation threshold drops to 0.031.
- **The test set is not a pristine holdout.** It comes from the same dataset explored in the early notebooks. It was also scored twice in this project: once with sigmoid calibration (AUPRC 0.883, cost 162, kept in `reports/metrics.json` as `test_history`) and once with the final isotonic model. The final model was chosen on validation, not on test results.
- **Calibration saturates.** Isotonic calibration produces many probabilities of exactly 1.0, so probabilities at the top are coarse.
- **The drift check is a snapshot tool, not monitoring.** It compares a batch with the training data. There is no scheduled monitoring or retraining.
- **Explanations and probabilities come from different models.** SHAP explains the raw XGBoost model. The reported probability is calibrated, so the two are related but not identical.
- **Not production-tested.** There are no load or latency tests. The Docker image has not been built on this machine.

## Architecture

```
data/creditcard.csv
   |
   v  split (stratified 60/20/20, seed 42)
   |-- train ------> XGBoost (class-weighted) ------> isotonic calibration (cross-validated on train)
   |                      |
   |                      +--> drift reference (bin edges, training sample)
   |-- validation ---> choose raw vs calibrated (log loss) ---> choose threshold (cost 10:1)
   '-- test ---------> final report, scored once (with bootstrap CIs)
                                   |
                                   v
                       models/fraud_model.joblib
                                   |
                                   v
            FastAPI: GET /health · POST /predict · POST /explain (top-5 SHAP reasons)
```

## Project structure

```
src/fraud_detection/   config, data (load + split), train (XGBoost, calibration), evaluate
                       (cost, AUPRC, bootstrap CI), compare (Phase 2 candidates), explain (SHAP),
                       drift (PSI + KS), reporting (writes reports/metrics.json)
scripts/train.py            final pipeline: split, train, calibrate, threshold, test report, save model
scripts/run_experiments.py  model, strategy, calibration and threshold comparisons (CV + validation)
app/                        FastAPI service: main.py (routes), schemas.py (input validation)
tests/                      pytest suite (logic tests always run; data/model tests skip if files are absent)
notebooks/                  01–07 exploration history. Their numbers predate the honest protocol; do not cite them.
outputs/                    plots (phase2_*.png are current; the other PNGs are from the old notebooks)
reports/metrics.json        every number in this README comes from here
models/                     trained artifact (gitignored; regenerate with scripts/train.py)
docs/                       MODEL_CARD.md, DECISIONS.md
```

## Quickstart

**Prerequisites:** Python 3.12. Download `creditcard.csv` from [Kaggle (mlg-ulb/creditcardfraud)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at `data/creditcard.csv`. The data is gitignored and must not be committed.

### Local (Windows PowerShell; on Linux or macOS use `source venv/bin/activate`)

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
python scripts/train.py                 # writes models/fraud_model.joblib and reports/metrics.json
uvicorn app.main:app --reload           # then open http://127.0.0.1:8000/docs
```

To rerun the comparisons (optional; needs the dev requirements):

```powershell
pip install -r requirements-dev.txt
python scripts/run_experiments.py       # CV + validation comparisons, writes outputs/phase2_*.png
python scripts/train.py                 # final model, using the calibration method set in the script
```

To run the tests:

```powershell
pip install -r requirements-test.txt
pytest
```

### Docker

```bash
python scripts/train.py                 # the image copies models/, so train first
docker build -t fraud-api .
docker run --rm -p 8000:8000 fraud-api
```

> The Dockerfile has not been built or run on the machine where this project was developed, because Docker was not installed there. Treat it as untested until you have built it.

### Example requests

The sample is the first fraud row in the dataset (a training row, not a held-out example), with the values as the API received them.

```bash
curl http://127.0.0.1:8000/health
```

```json
{"status":"ok","model_loaded":true,"threshold":0.07850456200204509,"calibration":"calibrated"}
```

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample.json
```

<details>
<summary>sample.json (30 features)</summary>

```json
{"Time": 406.0, "V1": -2.3122265423263, "V2": 1.95199201064158, "V3": -1.60985073229769, "V4": 3.9979055875468, "V5": -0.522187864667764, "V6": -1.42654531920595, "V7": -2.53738730624579, "V8": 1.39165724829804, "V9": -2.77008927719433, "V10": -2.77227214465915, "V11": 3.20203320709635, "V12": -2.89990738849473, "V13": -0.595221881324605, "V14": -4.28925378244217, "V15": 0.389724120274487, "V16": -1.14074717980657, "V17": -2.83005567450437, "V18": -0.0168224681808257, "V19": 0.416955705037907, "V20": 0.126910559061474, "V21": 0.517232370861764, "V22": -0.0350493686052974, "V23": -0.465211076182388, "V24": 0.320198198514526, "V25": 0.0445191674731724, "V26": 0.177839798284401, "V27": 0.261145002567677, "V28": -0.143275874698919, "Amount": 0.0}
```
</details>

Response:

```json
{"fraud_probability":1.0,"decision":"fraud","threshold":0.07850456200204509}
```

`POST /explain` with the same body returns the decision plus the five largest SHAP contributions:

```json
{"fraud_probability":1.0,"decision":"fraud","threshold":0.07850456200204509,
 "top_reasons":[
  {"feature":"V14","value":-4.28925378244217,"contribution":7.590172290802002,"direction":"toward_fraud"},
  {"feature":"V10","value":-2.77227214465915,"contribution":1.776408314704895,"direction":"toward_fraud"},
  {"feature":"V4","value":3.9979055875468,"contribution":1.7129422426223755,"direction":"toward_fraud"},
  {"feature":"V12","value":-2.89990738849473,"contribution":1.5861390829086304,"direction":"toward_fraud"},
  {"feature":"V23","value":-0.465211076182388,"contribution":-1.005346655845642,"direction":"toward_legitimate"}],
 "explanation_note":"SHAP contributions are in log-odds of the raw (uncalibrated) XGBoost model. fraud_probability uses the calibrated score, so the two are related but not identical."}
```

Invalid input returns HTTP 422 with the field and the rule. For example, `{"Time": -1}` gives "Input should be greater than or equal to 0", and missing fields are listed. NaN and Infinity are rejected the same way.

## Engineering notes

- **Leakage control.** Thresholds, calibration choice and model choice use validation data or CV only. The test set is scored by `scripts/train.py` and nowhere else. A test confirms the splits are disjoint and cover every row.
- **Reproducibility.** Fixed seed (42), pinned versions in `requirements*.txt`, and a Python 3.12 venv.
- **Serving.** The model and SHAP explainer load once at startup. The schema is checked against the saved model's feature names. If the model file is missing, `/health` reports `degraded` and `/predict` returns 503 with the fix.
- **CI.** GitHub Actions runs the test suite on every push and pull request. Tests needing the dataset or model are skipped in CI, and the skips are shown in the log.

## References

- Le Borgne, Y.-A. and Bontempi, G. *Reproducible Machine Learning for Credit Card Fraud Detection – Practical Handbook.* Online handbook: https://fraud-detection-handbook.github.io/fraud-detection-handbook
- Dal Pozzolo, A., Caelen, O., Johnson, R. A. and Bontempi, G. (2015). *Calibrating Probability with Undersampling for Unbalanced Classification.* IEEE SSCI 2015. The dataset is the one used in that work.
- Kaggle dataset: *Credit Card Fraud Detection*, Machine Learning Group, ULB: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- *Addressing Credit Card Fraud Detection Challenges with Adversarial Autoencoders.* Big Data and Cognitive Computing, 9(7), 168 (2025): https://www.mdpi.com/2504-2289/9/7/168. The authors are not listed here because they were not verified. This project's autoencoder is not that paper's method.

## License

MIT. See [LICENSE](LICENSE).
