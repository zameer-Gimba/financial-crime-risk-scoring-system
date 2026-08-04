# Financial Crime Detection & Risk Scoring System using Ensemble Machine Learning

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Ensemble-green)
![License](https://img.shields.io/badge/License-MIT-orange)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

## Overview

The **Financial Crime Detection & Risk Scoring System** is an end-to-end machine learning application designed to detect potentially fraudulent financial transactions and support financial crime investigations.

Unlike traditional fraud detection projects that only classify transactions as fraudulent or legitimate, this system provides:

- Fraud probability prediction
- Risk score generation (0–100)
- Risk categorization
- SHAP-based explainability
- Investigation recommendations
- Interactive Streamlit dashboard
- Downloadable investigation reports

The project is built using the **IEEE-CIS Fraud Detection Dataset** and employs a **stacked ensemble model** consisting of:

- LightGBM
- XGBoost
- CatBoost

combined through a Logistic Regression meta-learner.

---

# Objectives

The system aims to:

- Detect fraudulent financial transactions.
- Generate calibrated fraud probabilities (if calibration improves performance).
- Produce risk scores between 0 and 100.
- Categorize transactions into Low, Medium, High, and Critical risk.
- Explain predictions using SHAP.
- Generate investigation-ready reports.
- Provide an interactive dashboard for analysts.

---

# Dataset

The project uses the IEEE-CIS Fraud Detection Dataset.

Files used:

```
train_transaction.csv
train_identity.csv
test_transaction.csv
test_identity.csv
```

The transaction and identity datasets are merged using:

```
TransactionID
```

---

# Repository Structure

```
financial-crime-risk-scoring-system/

├── app/
├── config/
├── data/
├── docs/
├── models/
├── notebooks/
├── reports/
├── src/
├── tests/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Machine Learning Pipeline

```
Load Data
    │
Merge Datasets
    │
Validate Data
    │
Data Cleaning
    │
Exploratory Data Analysis
    │
Feature Engineering
    │
Feature Selection
    │
Train/Test Split
    │
5-Fold Stratified Cross Validation
    │
Stacked Ensemble Training
    │
Threshold Optimization
    │
Evaluation
    │
Risk Scoring
    │
Explainability
    │
Deployment
```

---

# Ensemble Architecture

Base Models

- LightGBM
- XGBoost
- CatBoost

Meta Learner

- Logistic Regression

---

# Evaluation Metrics

Primary Metrics

- Precision
- Recall
- F1-score
- PR-AUC

Secondary Metrics

- ROC-AUC
- Balanced Accuracy
- Specificity
- False Positive Rate
- False Negative Rate
- Confusion Matrix

---

# Risk Categories

| Risk Score | Category |
|------------|----------|
| 0–24 | Low |
| 25–49 | Medium |
| 50–74 | High |
| 75–100 | Critical |

---

# Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- XGBoost
- CatBoost
- SHAP
- Streamlit
- Plotly
- Matplotlib
- Joblib

---

# Project Status

Current Version

```
v0.1.0
```

Status

```
Milestone 1 – Foundation
```

---

# Future Milestones

- Foundation
- Data Layer + EDA
- Feature Engineering
- Feature Selection
- Ensemble Training
- Threshold Optimization
- Risk Intelligence Engine
- Explainability
- Investigation Reports
- Streamlit Dashboard
- Testing & Deployment

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
