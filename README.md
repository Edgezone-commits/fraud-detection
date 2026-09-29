# Credit Card Fraud Detection — Beyond 99.8% Accuracy

A model that **never predicts fraud** scores **99.83% accuracy** on this dataset. It catches exactly **zero** out of 98 fraudulent transactions in the test set.

This project builds a fraud detection system that actually works — using the right metrics, cost-aware threshold optimization, calibrated probabilities, an unsupervised second approach, and per-transaction explainability.

---

## The Problem

284,807 credit card transactions from European cardholders (September 2013), only **492 fraud (0.17%)** — a 577:1 imbalance. At this imbalance:

- **Accuracy is meaningless.** A dummy "always legitimate" classifier scores 99.83% accuracy and 0% recall.
- **AUPRC (Area Under the Precision-Recall Curve)** is the metric that actually matters — it's used as the primary metric throughout this project.
- **Costs are asymmetric.** A missed fraud costs far more than a false alarm, so the right decision threshold isn't the default 0.5 — it has to be derived, not assumed.

## Approach & Results

| Step | What we did | Key result |
|------|-------------|------------|
| EDA | Confirmed the accuracy trap; correlation analysis | Dummy model: 99.83% acc, 0% recall. Top correlated: V17, V14, V12, V10 |
| Baseline models | Logistic Regression → Random Forest → XGBoost | XGBoost best: **AUPRC 0.879** vs. RF 0.863 vs. LogReg 0.719 |
| Imbalance strategies | Class weighting vs. SMOTE vs. undersampling (head-to-head) | Class weighting (0.879) ≈ SMOTE (0.878); undersampling collapsed to **0.387** |
| Threshold optimization | Cost curve (10:1 FN:FP cost ratio), swept 0.01–0.99 | Moved threshold 0.5 → 0.31: caught 2 more frauds, **cost dropped 182 → 165 (9.3%)** |
| Calibration | Isotonic calibration (`CalibratedClassifierCV`) | AUPRC preserved (0.879 → 0.877); re-optimized threshold (0.08) **caught 4 more frauds than baseline — cost 182 → 149 (18.1% reduction)** |
| Autoencoder | Unsupervised — trained on legitimate transactions only | AUPRC 0.385 (expected, no labels) — but median reconstruction error 38x higher for fraud; complementary to XGBoost (catches different cases) |
| SHAP explainability | Global + per-transaction explanations | V14/V4/V12 dominate; V4 non-linear (weak correlation, high SHAP); confident (100%) and uncertain (44%) case examples |

## Key Findings

### 1. The Accuracy Trap
A model predicting "legitimate" for every transaction gets 99.83% accuracy while catching 0 of 98 frauds. Every result in this project is reported using AUPRC, never raw accuracy.

### 2. Tree Ensembles Beat Linear Models — and Beat Deep Learning Too
XGBoost (AUPRC 0.879) and Random Forest (0.863) clearly outperformed Logistic Regression (0.719). This matches published research on this exact dataset — deep learning rarely beats tree ensembles here, since there's too little labeled fraud (492 examples) for a neural network to out-learn a well-tuned tree ensemble on already-structured, PCA-transformed features.

### 3. SMOTE Isn't Automatically Better Than Class Weighting
A head-to-head comparison (same model, same data, only the imbalance strategy changed) showed class weighting and SMOTE performing identically (0.879 vs. 0.878), while undersampling — which throws away ~226,700 real legitimate transactions — cost more than half the AUPRC (0.387). **Simpler won.**

### 4. Threshold Tuning + Calibration Reduced Business Cost by 18%
Using the default 0.5 threshold is arbitrary. Sweeping thresholds against a cost function (missed fraud = 10x cost of a false alarm) and calibrating the model's raw scores into true probabilities first reduced total cost from 182 (naive) to 149 (optimized + calibrated) — catching 4 additional real frauds for a modest increase in false alarms, entirely through better decision-making on the same trained model.

### 5. Unsupervised and Supervised Approaches Catch Different Fraud
An autoencoder trained only on legitimate transactions (no fraud labels at all) scored much lower overall (AUPRC 0.385) — expected, since it has no labeled signal. But cross-checking showed XGBoost still confidently caught 75%+ of the fraud cases the autoencoder found hardest to reconstruct, while roughly 25% of that group scored low on *both* models — a genuinely hard segment that neither approach reliably catches with the available anonymized features alone.

### 6. Explainability Reveals Non-Linear Signal Correlation Misses
SHAP's global feature importance mostly agreed with Day 1's correlation analysis (V14, V12 topped both), but ranked **V4 as the #2 most important feature** despite it showing weak linear correlation (0.133) — evidence of a real, non-linear pattern invisible to a simple correlation check. Conversely, V17 (the strongest linear correlation) ranked lower in SHAP importance.

## Project Structure

```
fraud-detection/
├── notebooks/
│   ├── 01_eda.ipynb                    
│   ├── 02_baseline_models.ipynb        
│   ├── 03_imbalance_strategies.ipynb   
│   ├── 04_threshold_optimization.ipynb 
│   ├── 05_calibration.ipynb            
│   ├── 06_autoencoder.ipynb            
│   └── 07_shap_explainability.ipynb    
├── data/
│   ├── creditcard.csv      (not committed — see .gitignore)
│   ├── train.csv
│   └── test.csv
├── outputs/                # Saved charts from every notebook
├── requirements.txt
└── README.md
```

## Dataset

[Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions, 31 features. `V1`–`V28` are PCA-anonymized for confidentiality; `Time`, `Amount`, and `Class` (0/1) are the only interpretable columns.

## Setup

```powershell
git clone https://github.com/Edgezone-commits/fraud-detection.git
cd fraud-detection
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Download creditcard.csv from Kaggle into data/, then run notebooks in order
```

## References

- Le Borgne, Siblini, Lebichot, Bontempi — *Reproducible Machine Learning for Credit Card Fraud Detection* (open-access handbook, ULB fraud detection research group — creators of this dataset)
- MDPI (2025) — *Addressing Credit Card Fraud Detection Challenges with Adversarial Autoencoders*
- Alarfaj et al. (2022) — *Credit Card Fraud Detection Using State-of-the-Art ML and DL Algorithms*, IEEE Access

## What's Next

- Serve the calibrated XGBoost model as a FastAPI endpoint with SHAP-based per-prediction explanations, input validation, tests, and a Dockerfile (mirroring the `credit-risk-api` project structure).