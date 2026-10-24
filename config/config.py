"""
Central configuration for the Financial Crime Detection &
Risk Scoring System.

Author: Muhammad Ibrahim Gimba
"""

from pathlib import Path

# PROJECT PATHS

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = PROJECT_ROOT / "models"
BASE_MODEL_DIR = MODEL_DIR / "base_models"

REPORT_DIR = PROJECT_ROOT / "reports"
DOCS_DIR = PROJECT_ROOT / "docs"
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

LOG_DIR = PROJECT_ROOT / "logs"

# DATA FILES

TRAIN_TRANSACTION = RAW_DATA_DIR / "train_transaction.csv"
TRAIN_IDENTITY = RAW_DATA_DIR / "train_identity.csv"

TEST_TRANSACTION = RAW_DATA_DIR / "test_transaction.csv"
TEST_IDENTITY = RAW_DATA_DIR / "test_identity.csv"

# OUTPUT FILES

MERGED_TRAIN = PROCESSED_DATA_DIR / "merged_train.parquet"
MERGED_TEST = PROCESSED_DATA_DIR / "merged_test.parquet"

PREPROCESSED_TRAIN = (
    PROCESSED_DATA_DIR / "preprocessed_train.parquet"
)
PREPROCESSED_TEST = (
    PROCESSED_DATA_DIR / "preprocessed_test.parquet"
)

ENGINEERED_TRAIN = (
    PROCESSED_DATA_DIR / "engineered_train.parquet"
)
ENGINEERED_TEST = (
    PROCESSED_DATA_DIR / "engineered_test.parquet"
)

SELECTED_TRAIN = (
    PROCESSED_DATA_DIR / "selected_train.parquet"
)
SELECTED_TEST = (
    PROCESSED_DATA_DIR / "selected_test.parquet"
)

STACKING_MODEL = MODEL_DIR / "ensemble_model.joblib"

# Single deployable artifact: fitted preprocessing, feature engineering,
# selected features, ensemble pipeline, decision threshold and metadata.
MODEL_BUNDLE = MODEL_DIR / "fraud_risk_model.joblib"

# DEPLOYMENT ASSETS (small files committed for the Streamlit app)

SAMPLE_DIR = DATA_DIR / "sample"
SAMPLE_TRANSACTIONS = SAMPLE_DIR / "sample_transactions.csv"

METRICS_REPORT = REPORT_DIR / "metrics.json"
PR_CURVE_REPORT = REPORT_DIR / "pr_curve.csv"
ROC_CURVE_REPORT = REPORT_DIR / "roc_curve.csv"
GLOBAL_SHAP_REPORT = REPORT_DIR / "global_shap_importance.csv"
FEATURE_IMPORTANCE_REPORT = REPORT_DIR / "feature_importance.csv"

# RANDOMNESS

RANDOM_STATE = 42

# TRAINING

# The IEEE-CIS data is temporal, so the holdout is the most recent
# TEST_SIZE share of training transactions (ordered by TransactionDT).
# The holdout is split again in time: the first THRESHOLD_SHARE is used
# to tune the decision threshold, the remainder for final reporting.
TEST_SIZE = 0.20
THRESHOLD_SHARE = 0.50
N_FOLDS = 5

# Full cross-validation of the stacked ensemble refits it N_FOLDS times
# on top of the stack's own internal CV. Off by default (slow).
RUN_CROSS_VALIDATION = False

N_ESTIMATORS = 1000
LEARNING_RATE = 0.05

TARGET = "isFraud"

TOP_FEATURES = 100
CORRELATION_THRESHOLD = 0.95

# Correlation filtering on a row sample keeps memory and time bounded.
CORRELATION_SAMPLE_SIZE = 100_000

# Columns never used as model inputs: identifiers and raw timestamps are
# monotonic in time and do not generalise to future transactions.
EXCLUDED_FEATURES = [
    "TransactionID",
    "TransactionDT",
]

# EXPLAINABILITY

SHAP_SAMPLE_SIZE = 2_000
TOP_EXPLANATION_FEATURES = 10

# SAMPLE DATA FOR THE DASHBOARD

SAMPLE_FRAUD_ROWS = 150
SAMPLE_LEGIT_ROWS = 350

# RISK SCORE

LOW_RISK_MAX = 24
MEDIUM_RISK_MAX = 49
HIGH_RISK_MAX = 74

# DIRECTORIES

DIRECTORIES = [
    RAW_DATA_DIR,
    SAMPLE_DIR,
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    BASE_MODEL_DIR,
    REPORT_DIR,
    DOCS_DIR,
    NOTEBOOK_DIR,
    LOG_DIR,
]


def create_directories() -> None:
    """
    Create all required project directories if they do not already exist.
    """

    for directory in DIRECTORIES:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


if __name__ == "__main__":
    create_directories()
