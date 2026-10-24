"""
Data loading module for the Financial Crime Detection &
Risk Scoring System.

Responsible for loading the IEEE-CIS Fraud Detection datasets.

Author: Muhammad Ibrahim Gimba
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from config.config import (
    TEST_IDENTITY,
    TEST_TRANSACTION,
    TRAIN_IDENTITY,
    TRAIN_TRANSACTION,
)
from config.logging_config import get_logger
from src.utils import load_csv

logger = get_logger(__name__)

DATASET_URL = "https://www.kaggle.com/c/ieee-fraud-detection/data"


def load_raw_csv(file_path) -> pd.DataFrame:
    """
    Load one raw IEEE-CIS file with a helpful error if it is missing.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"{file_path} not found. Download the IEEE-CIS Fraud Detection "
            f"dataset from {DATASET_URL} and extract the CSV files into "
            f"{file_path.parent} (see README: 'Getting the data')."
        )

    return prepare_raw(load_csv(file_path))


def normalize_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize column names across datasets.

    The IEEE-CIS test identity file names its columns ``id-01`` ...
    ``id-38`` while the training file uses ``id_01`` ... ``id_38``.
    Without this step every identity feature is silently lost for the
    test set and for any new transaction scored by the application.
    """

    return dataframe.rename(
        columns=lambda column: column.replace("id-", "id_")
        if column.startswith("id-")
        else column
    )


def downcast_floats(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert float64 columns to float32 to halve memory usage.
    """

    float_columns = dataframe.select_dtypes(
        include=["float64"]
    ).columns

    dataframe[float_columns] = dataframe[float_columns].astype(
        np.float32
    )

    return dataframe


def prepare_raw(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply the standard raw-data fixes (column names, dtypes).
    """

    return downcast_floats(normalize_columns(dataframe))


class DataLoader:
    """
    Loads the raw IEEE-CIS Fraud Detection datasets.

    Methods
    -------
    load_train_transaction()
        Load the training transaction dataset.

    load_train_identity()
        Load the training identity dataset.

    load_test_transaction()
        Load the test transaction dataset.

    load_test_identity()
        Load the test identity dataset.

    load_all()
        Load all four datasets and return them as a dictionary.
    """

    def __init__(self) -> None:
        """Initialize the DataLoader."""
        logger.info("DataLoader initialized.")

    def load_train_transaction(self) -> pd.DataFrame:
        """
        Load the training transaction dataset.

        Returns
        -------
        pd.DataFrame
            Training transaction dataset.
        """
        logger.info("Loading training transaction dataset...")
        return load_raw_csv(TRAIN_TRANSACTION)

    def load_train_identity(self) -> pd.DataFrame:
        """
        Load the training identity dataset.

        Returns
        -------
        pd.DataFrame
            Training identity dataset.
        """
        logger.info("Loading training identity dataset...")
        return load_raw_csv(TRAIN_IDENTITY)

    def load_test_transaction(self) -> pd.DataFrame:
        """
        Load the test transaction dataset.

        Returns
        -------
        pd.DataFrame
            Test transaction dataset.
        """
        logger.info("Loading test transaction dataset...")
        return load_raw_csv(TEST_TRANSACTION)

    def load_test_identity(self) -> pd.DataFrame:
        """
        Load the test identity dataset.

        Returns
        -------
        pd.DataFrame
            Test identity dataset.
        """
        logger.info("Loading test identity dataset...")
        return load_raw_csv(TEST_IDENTITY)

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """
        Load all IEEE-CIS datasets.

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary containing all datasets.
        """
        logger.info("Loading all IEEE-CIS datasets...")

        datasets = {
            "train_transaction": self.load_train_transaction(),
            "train_identity": self.load_train_identity(),
            "test_transaction": self.load_test_transaction(),
            "test_identity": self.load_test_identity(),
        }

        logger.info("Successfully loaded all datasets.")

        return datasets


def main() -> None:
    """
    Entry point for standalone execution.
    """

    loader = DataLoader()

    datasets = loader.load_all()

    print("\nLoaded Datasets\n" + "-" * 40)

    for name, df in datasets.items():
        print(f"{name:<20} {df.shape}")


if __name__ == "__main__":
    main()
