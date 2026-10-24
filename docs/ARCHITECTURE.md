# System Architecture

## Project

Financial Crime Detection & Risk Scoring System using Ensemble Machine Learning

---

# Overview

This project is an end-to-end machine learning system for detecting potentially fraudulent financial transactions and supporting financial crime investigations.

The system transforms raw IEEE-CIS fraud detection data into actionable intelligence through data processing, feature engineering, ensemble learning, explainable AI, and an interactive investigation dashboard.

---

# High-Level Architecture

```
                IEEE-CIS Dataset
                       │
                       ▼
              Data Loading Layer
                       │
                       ▼
             Data Validation Layer
                       │
                       ▼
             Data Preprocessing Layer
                       │
                       ▼
           Feature Engineering Layer
                       │
                       ▼
           Feature Selection Layer
                       │
                       ▼
        Ensemble Machine Learning Layer
                       │
                       ▼
      Threshold Optimization & Evaluation
                       │
                       ▼
          Risk Intelligence Engine
                       │
                       ▼
          SHAP Explainability Layer
                       │
                       ▼
        Investigation Report Generator
                       │
                       ▼
          Streamlit Dashboard
```

---

# Machine Learning Architecture

## Base Models

- LightGBM
- XGBoost
- CatBoost

## Meta Learner

- Logistic Regression

The ensemble combines predictions from the three base learners into a single probability estimate.

---

# Risk Intelligence Pipeline

```
Fraud Probability
        │
        ▼
Risk Score (0–100)
        │
        ▼
Risk Category
        │
        ▼
Recommendation
```

---

# Risk Categories

| Score | Category |
|-------:|----------|
| 0–24 | Low |
| 25–49 | Medium |
| 50–74 | High |
| 75–100 | Critical |

---

# Recommendations

- Approve
- Monitor
- Manual Review
- Immediate Investigation

---

# Explainability

Predictions are interpreted using SHAP to provide both global feature importance and transaction-level explanations.

---

# Module Map

| Layer | Module | Key class / function |
|-------|--------|----------------------|
| Data loading | `src/data_loader.py` | `DataLoader`, `normalize_columns` |
| Merge | `src/merge_data.py` | `DataMerger` |
| Validation | `src/data_validation.py` | `DataValidator` |
| Preprocessing | `src/preprocessing.py` | `DataPreprocessor` (fit / transform) |
| Feature engineering | `src/feature_engineering.py` | `FeatureEngineer` (fit / transform) |
| Feature selection | `src/feature_selection.py` | `FeatureSelector` (fit / transform) |
| Training | `src/model_training.py` | `ModelTrainer` |
| Threshold & evaluation | `src/evaluation.py` | `ModelEvaluator`, `optimize_threshold` |
| Risk intelligence | `src/risk_scoring.py` | `RiskScorer`, `probability_to_score` |
| Explainability | `src/explainability.py` | `EnsembleExplainer` |
| Reports | `src/report_generator.py` | `build_transaction_report` |
| Inference | `src/predict.py` | `FraudRiskPredictor` |
| Orchestration | `src/pipeline.py` | `TrainingPipeline` |
| Dashboard | `streamlit_app.py` | Streamlit pages |

---

# Design Decisions

## Fit on the past, transform anything

Preprocessing, feature engineering and feature selection follow the
scikit-learn `fit` / `transform` convention. Every learned quantity
(schema, frequency counts, per-card / address / e-mail amount statistics,
amount bins, selected columns) comes from the training period only, and
the same fitted objects transform the holdout, the Kaggle test set, and
single transactions in the dashboard. This prevents leakage and
guarantees a transaction gets the same features in training and in
production.

## Chronological evaluation

Transactions are ordered by `TransactionDT`:

```
|<-------------- train (80%) -------------->|<- threshold (10%) ->|<- test (10%) ->|
```

The threshold split tunes the decision threshold (max F1); the test split
is touched once, for the final metrics. A random split would leak future
behaviour into training and overstate performance.

## Missing values

Numeric gaps stay as `NaN` - all three gradient-boosting libraries handle
them natively, and missingness itself (for example, no identity record)
is predictive. Categorical gaps become `"Unknown"`.

## Risk score

```
score = 50 * p / t                     if p < t
score = 50 + 50 * (p - t) / (1 - t)     if p >= t
```

where `t` is the tuned threshold. The threshold always lands on 50, the
start of the High band, so High and Critical are exactly the transactions
the model flags.

## Ensemble explanations

TreeSHAP is exact for each base learner. Per-feature contributions are
combined using the normalised meta-learner coefficients, giving an
ensemble-level explanation on the log-odds scale.

## Deployable artifact

`models/fraud_risk_model.joblib` is a single bundle: fitted preprocessor,
feature engineer, selected feature list, preprocessing + stacking
pipeline, threshold and metadata (metrics, library versions). Library
versions are pinned in `requirements.txt` because the bundle is a pickle.

---

# Deployment

Streamlit Community Cloud runs `streamlit_app.py` with the committed
model bundle, evaluation reports and a 500-row demo sample. The raw
dataset is not needed at runtime.

- Interactive dashboard
- Fraud prediction (single and batch)
- Risk analysis
- SHAP visualizations
- Investigation report generation (PDF)
