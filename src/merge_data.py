"""
Data merging module for the Financial Crime Detection &
Risk Scoring System.

Responsible for merging IEEE-CIS transaction and identity datasets.

Author: Muhammad Ibrahim Gimba
"""

from __future__ import annotations

import pandas as pd

from config.config import (
    MERGED_TEST,
    MERGED_TRAIN,
)
from config.logging_config import get_logger
from src.data_loader import DataLoader
from src.utils import save_dataframe

logger = get_logger(__name__)


class DataMerger:
    """
    Merge transaction and identity datasets.

    Merge key:
        TransactionID
    """

    def __init__(self) -> None:
        """
        Initialize merger.
        """
        self.loader = DataLoader()

        logger.info("DataMerger initialized.")

    @staticmethod
    def merge_train_data(
        train_transaction: pd.DataFrame,
        train_identity: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge training datasets.

        Parameters
        ----------
        train_transaction : pd.DataFrame
        train_identity : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """

        logger.info("Merging training datasets...")

        merged = pd.merge(
            train_transaction,
            train_identity,
            how="left",
            on="TransactionID"
        )

        logger.info(
            "Training merge complete. Shape: %s",
            merged.shape
        )

        return merged

    @staticmethod
    def merge_test_data(
        test_transaction: pd.DataFrame,
        test_identity: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge test datasets.

        Parameters
        ----------
        test_transaction : pd.DataFrame
        test_identity : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """

        logger.info("Merging test datasets...")

        merged = pd.merge(
            test_transaction,
            test_identity,
            how="left",
            on="TransactionID"
        )

        logger.info(
            "Test merge complete. Shape: %s",
            merged.shape
        )

        return merged

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute full merge process.

        Returns
        -------
        tuple
            merged_train, merged_test
        """

        logger.info("Loading datasets...")

        datasets = self.loader.load_all()

        merged_train = self.merge_train_data(
            datasets["train_transaction"],
            datasets["train_identity"]
        )

        merged_test = self.merge_test_data(
            datasets["test_transaction"],
            datasets["test_identity"]
        )

        logger.info("Saving merged datasets...")

        save_dataframe(
            merged_train,
            MERGED_TRAIN
        )

        save_dataframe(
            merged_test,
            MERGED_TEST
        )

        logger.info("Merged datasets saved successfully.")

        return merged_train, merged_test


def main() -> None:
    """
    Standalone execution.
    """

    merger = DataMerger()

    train_df, test_df = merger.run()

    print("\nMerge Summary")
    print("-" * 40)
    print(f"Merged Train Shape: {train_df.shape}")
    print(f"Merged Test Shape: {test_df.shape}")


if __name__ == "__main__":
    main()
