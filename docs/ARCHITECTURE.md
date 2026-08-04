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

# Deployment

The completed system will be deployed using Streamlit with support for:

- Interactive dashboard
- Fraud prediction
- Risk analysis
- SHAP visualizations
- Investigation report generation
