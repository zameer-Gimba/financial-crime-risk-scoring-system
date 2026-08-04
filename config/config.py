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

ENGINEERED_TRAIN = PROCESSED_DATA_DIR / "engineered_train.parquet"
ENGINEERED_TEST = PROCESSED_DATA_DIR / "engineered_test.parquet"

STACKING_MODEL = MODEL_DIR / "ensemble_model.joblib"

# RANDOMNESS

RANDOM_STATE = 42

# TRAINING

TEST_SIZE = 0.20
N_FOLDS = 5

TARGET = "isFraud"

# RISK SCORE

LOW_RISK_MAX = 24
MEDIUM_RISK_MAX = 49
HIGH_RISK_MAX = 74

# DIRECTORIES

DIRECTORIES = [
    RAW_DATA_DIR,
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
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    create_directories()
