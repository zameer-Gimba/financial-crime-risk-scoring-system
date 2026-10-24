"""
Project constants.

Author: Muhammad Ibrahim Gimba
"""

# MODEL NAMES

LIGHTGBM = "LightGBM"

XGBOOST = "XGBoost"

CATBOOST = "CatBoost"

STACKING = "Stacking Ensemble"

META_MODEL = "Logistic Regression"

# RISK LABELS

LOW = "Low"

MEDIUM = "Medium"

HIGH = "High"

CRITICAL = "Critical"

# RECOMMENDATIONS

APPROVE = "Approve"

MONITOR = "Monitor"

MANUAL_REVIEW = "Manual Review"

IMMEDIATE_INVESTIGATION = "Immediate Investigation"

# DASHBOARD TITLES

APP_TITLE = "Financial Crime Detection & Risk Scoring System"

SIDEBAR_TITLE = "Navigation"

# TARGET

TARGET_COLUMN = "isFraud"

TRANSACTION_ID = "TransactionID"

# COLORS

# Status palette (good / warning / serious / critical). Always shown
# together with the category label, never as color alone.

LOW_COLOR = "#0CA30C"

MEDIUM_COLOR = "#FAB219"

HIGH_COLOR = "#EC835A"

CRITICAL_COLOR = "#D03B3B"

# REPORT

REPORT_TITLE = "Financial Crime Investigation Report"
