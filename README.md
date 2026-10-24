# Financial Crime Detection & Risk Scoring System using Ensemble Machine Learning

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Ensemble-green)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-orange)
![Status](https://img.shields.io/badge/Status-Deployable-brightgreen)

## Overview

The **Financial Crime Detection & Risk Scoring System** is an end-to-end machine learning application that detects potentially fraudulent card transactions and supports financial crime investigations.

Rather than only labelling a transaction as fraud or not, the system provides:

- Fraud probability prediction
- Risk score generation (0–100)
- Risk categorization (Low / Medium / High / Critical)
- Investigation recommendations
- SHAP explanations for every prediction
- An interactive Streamlit dashboard (single-transaction investigation, what-if analysis, batch scoring)
- Downloadable PDF investigation reports and CSV batch results

The model is a **stacked ensemble** of LightGBM, XGBoost and CatBoost, combined by a Logistic Regression meta-learner, trained on the **IEEE-CIS Fraud Detection** dataset (590,540 labelled transactions, 3.5% fraud).

---

# Results

Measured on the **most recent 10% of transactions**, which the model never saw during training or threshold tuning (chronological split - see [Evaluation design](#evaluation-design)).

| Metric | Value |
|--------|-------|
| PR-AUC | 0.557 (random baseline: 0.037) |
| ROC-AUC | 0.911 |
| Precision | 69.3% |
| Recall | 46.5% |
| F1-score | 0.557 |
| Balanced accuracy | 0.728 |
| Specificity | 0.9920 |
| False positive rate | 0.0080 |
| Decision threshold | 0.402 (tuned for max F1 on the threshold split) |

Confusion matrix on 59,054 test transactions: TN 56,386 · FP 455 · FN 1,184 · TP 1,029.

**Risk categories work as intended** - the observed fraud rate climbs steeply across bands:

| Category | Share of transactions | Observed fraud rate | Share of all fraud caught |
|----------|----------------------:|--------------------:|--------------------------:|
| Low | 96.8% | 1.9% | 49.2% |
| Medium | 0.7% | 22.9% | 4.3% |
| High | 0.6% | 39.0% | 6.1% |
| Critical | 1.9% | 78.7% | 40.4% |

**Ensemble vs. individual models** (test split):

| Model | PR-AUC | ROC-AUC |
|-------|-------:|--------:|
| LightGBM | 0.557 | 0.911 |
| XGBoost | 0.533 | 0.895 |
| CatBoost | 0.528 | 0.896 |
| **Stacked ensemble** | 0.557 | 0.911 |

The stack matches the strongest single model (LightGBM) and outperforms XGBoost and CatBoost; its main benefit here is robustness rather than a large accuracy gain. Top-ranked IEEE-CIS solutions reach higher scores with extensive customer-ID reconstruction and target-period tuning; this project prioritises a leak-free, chronologically honest evaluation and a deployable inference path.

---

# Quick Start

```bash
git clone https://github.com/zameer-Gimba/financial-crime-risk-scoring-system
cd financial-crime-risk-scoring-system
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

streamlit run streamlit_app.py      # dashboard (uses the committed model)
pytest                              # tests
```

The trained model, evaluation reports and a 500-transaction demo sample are committed, so the dashboard runs without downloading the 1.3 GB dataset.

## Getting the data

The raw dataset (~1.3 GB) is not stored in this repository. Download it from the
[IEEE-CIS Fraud Detection competition on Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data)
(you must sign in and accept the competition rules), then extract the CSVs into `data/raw/`:

```bash
pip install kaggle                    # needs an API token in ~/.kaggle/kaggle.json
kaggle competitions download -c ieee-fraud-detection -p data/raw
unzip data/raw/ieee-fraud-detection.zip -d data/raw
```

Expected files: `train_transaction.csv`, `train_identity.csv`, `test_transaction.csv`, `test_identity.csv`.

## Retraining

```bash
python -m src.pipeline                # full run, ~20 min on 4 CPU cores
python -m src.pipeline --nrows 100000 --n-estimators 50   # quick experiment
```

`src.pipeline` writes every artifact the dashboard reads:

| Artifact | Contents |
|----------|----------|
| `models/fraud_risk_model.joblib` | Fitted preprocessing, feature engineering, selected features, ensemble, threshold, metadata |
| `reports/metrics.json` | Holdout metrics, threshold, risk-band analysis, base-model comparison |
| `reports/pr_curve.csv`, `reports/roc_curve.csv` | Evaluation curves |
| `reports/global_shap_importance.csv` | Global SHAP importance |
| `reports/feature_importance.csv` | Feature selection ranking |
| `data/sample/sample_transactions.csv` | Demo transactions from the test period |

The individual stages can still be run on their own (`python -m src.merge_data`, `src.preprocessing`, `src.feature_engineering`, `src.feature_selection`, `src.model_training`).

---

# Deploying to Streamlit Community Cloud

1. Push this repository to GitHub (already done if you are reading it there).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **Create app** → **Deploy a public app from GitHub**.
3. Repository `zameer-Gimba/financial-crime-risk-scoring-system`, branch `main`, main file `streamlit_app.py`.
4. Under **Advanced settings**, choose **Python 3.12** (the version the model was trained with).
5. Deploy. The first build installs the pinned requirements (a few minutes).

The app only needs the committed model, reports and demo sample, so deployment is fast and does not touch the raw dataset.

---

# Dashboard

| Page | What it does |
|------|--------------|
| **Overview** | Holdout KPIs, observed fraud rate per risk category, PR/ROC curves, confusion matrix, base-model comparison |
| **Investigate a transaction** | Risk gauge, probability, recommendation, SHAP drivers, what-if edits (amount, e-mail domain, device, hour), PDF case report |
| **Batch scoring** | Upload a CSV (IEEE-CIS format; missing columns allowed) or use the sample, score up to 50,000 rows, download results |
| **Explainability** | Global SHAP importance and feature-selection ranking |
| **About** | Methodology, limitations, model build information |

---

# Machine Learning Pipeline

```
Load Data (train_transaction + train_identity)
    │   normalise identity column names (test files use id-01, train id_01)
Merge Datasets on TransactionID
    │
Validate Data
    │
Preprocess  ── fit on training period ──┐
    │                                    │  the same fitted objects are
Feature Engineering ── fit ──────────────┤  reused for the holdout, the
    │                                    │  Kaggle test set, and every
Feature Selection ── fit ────────────────┘  transaction scored in the app
    │
Chronological split: train 80% │ threshold 10% │ test 10%
    │
Stacked Ensemble (5-fold stacking)
    │
Threshold Optimization (max F1 on threshold split)
    │
Evaluation (test split, touched once)
    │
Risk Scoring → Category → Recommendation
    │
SHAP Explainability
    │
Model bundle + reports → Streamlit
```

## Feature engineering

- **Time** - hour of day, day of week, night-time flag (absolute day counters are excluded because they do not generalise to future dates)
- **Amount** - log amount, decimal "cents" part, quantile bin
- **Frequency encoding** - how common each card, address, e-mail domain, device, browser and amount is
- **Behavioural** - how the amount compares with what is typical for the same card, address, e-mail domain and pseudo-customer (`card1 + addr1 + P_emaildomain`): mean, std, median, ratio to mean, z-score
- **Identity** - identity-record present, e-mail provider, purchaser/recipient domain match
- **Missingness** - count of missing fields per transaction

## Feature selection

Constant, duplicate and highly correlated (|r| > 0.95) features are removed, then the top 100 are kept by LightGBM gain. `TransactionID` and `TransactionDT` are never used as model inputs.

## Ensemble

| Role | Model |
|------|-------|
| Base learner | LightGBM |
| Base learner | XGBoost |
| Base learner | CatBoost |
| Meta-learner | Logistic Regression |

Class imbalance is handled with `scale_pos_weight` (LightGBM, XGBoost) and balanced class weights (CatBoost).

## Evaluation design

The data is temporal, and a deployed fraud model always scores transactions that happen after the ones it learned from. Transactions are therefore sorted by `TransactionDT`:

- **train** - earliest 80%
- **threshold** - next 10%, used only to choose the decision threshold
- **test** - latest 10%, used only for the reported metrics

A random split would let the model learn from the future and overstate performance.

---

# Risk Scoring

The fraud probability `p` is mapped to a 0–100 score anchored on the tuned threshold `t`, so the threshold always lands at 50:

```
score = 50 · p / t                   if p < t
score = 50 + 50 · (p − t) / (1 − t)  if p ≥ t
```

| Risk Score | Category | Recommendation |
|------------|----------|----------------|
| 0–24 | Low | Approve |
| 25–49 | Medium | Monitor |
| 50–74 | High | Manual Review |
| 75–100 | Critical | Immediate Investigation |

High and Critical are exactly the transactions the model flags.

---

# Explainability

Each base learner is explained exactly with TreeSHAP; contributions are combined using the meta-learner's normalised weights to give an ensemble-level explanation on the log-odds scale. The dashboard shows the top drivers for every transaction, and the PDF report includes them.

---

# Repository Structure

```
financial-crime-risk-scoring-system/
├── streamlit_app.py          # dashboard entry point
├── config/                   # paths, settings, constants, logging
├── src/
│   ├── data_loader.py        # load + normalise raw CSVs
│   ├── merge_data.py         # transaction ⋈ identity
│   ├── data_validation.py    # schema / duplicate / missing checks
│   ├── preprocessing.py      # DataPreprocessor (fit/transform)
│   ├── feature_engineering.py# FeatureEngineer (fit/transform)
│   ├── feature_selection.py  # FeatureSelector (fit/transform)
│   ├── model_training.py     # stacked ensemble
│   ├── evaluation.py         # threshold optimisation + metrics
│   ├── risk_scoring.py       # score, category, recommendation
│   ├── explainability.py     # ensemble SHAP
│   ├── report_generator.py   # PDF / CSV reports
│   ├── predict.py            # FraudRiskPredictor (inference)
│   └── pipeline.py           # end-to-end training
├── models/                   # fraud_risk_model.joblib
├── reports/                  # metrics and curves
├── data/raw/                 # IEEE-CIS CSVs (download from Kaggle)
├── data/sample/              # demo transactions
├── docs/ARCHITECTURE.md
└── tests/
```

---

# Technologies

Python · Pandas · NumPy · Scikit-learn · LightGBM · XGBoost · CatBoost · SHAP · Streamlit · Plotly · ReportLab · Joblib

---

# Limitations

- IEEE-CIS features are anonymised and come from a single e-commerce provider; the model demonstrates the method and is not a drop-in control for another institution's data.
- Scores are decision support; every flagged case needs analyst review.

---

# License

This project is released under the MIT License.

---

# Author

**Muhammad Ibrahim Gimba**

Computer Scientist | Data Scientist | Machine Learning Engineer | Project Manager

GitHub:

https://github.com/zameer-Gimba/financial-crime-risk-scoring-system

LinkedIn:

https://www.linkedin.com/in/muhammad-ibrahim-gimba-60b87718b
