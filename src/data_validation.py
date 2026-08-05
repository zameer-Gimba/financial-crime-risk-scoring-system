"""
Data validation module for the Financial Crime Detection &
Risk Scoring System.

Responsible for validating merged IEEE-CIS datasets before
preprocessing.

Author: Muhammad Ibrahim Gimba
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from config.logging_config import get_logger
from src.merge_data import DataMerger

logger = get_logger(__name__)


class DataValidator:
    """
    Validate merged datasets before preprocessing.
    """

    REQUIRED_COLUMNS = [
        "TransactionID",
        "TransactionDT",
        "TransactionAmt",
    ]

    TARGET_COLUMN = "isFraud"

    def __init__(self) -> None:
        logger.info("DataValidator initialized.")

    def validate_schema(
        self,
        dataframe: pd.DataFrame,
        is_training: bool = True,
    ) -> bool:
        """
        Validate that required columns exist.
        """

        required = self.REQUIRED_COLUMNS.copy()

        if is_training:
            required.append(self.TARGET_COLUMN)

        missing = [
            column
            for column in required
            if column not in dataframe.columns
        ]

        if missing:
            logger.error(
                "Missing required columns: %s",
                missing,
            )
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        logger.info("Schema validation passed.")

        return True

    def validate_duplicates(
        self,
        dataframe: pd.DataFrame,
    ) -> int:
        """
        Count duplicate TransactionIDs.
        """

        duplicates = dataframe["TransactionID"].duplicated().sum()

        logger.info(
            "Duplicate TransactionIDs: %d",
            duplicates,
        )

        return int(duplicates)

    def validate_missing_values(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.Series:
        """
        Return missing value counts.
        """

        missing = dataframe.isnull().sum()

        missing = missing[missing > 0].sort_values(
            ascending=False
        )

        logger.info(
            "Columns with missing values: %d",
            len(missing),
        )

        return missing

    def generate_validation_report(
        self,
        dataframe: pd.DataFrame,
    ) -> Dict:
        """
        Generate validation summary.
        """

        report = {
            "rows": dataframe.shape[0],
            "columns": dataframe.shape[1],
            "duplicate_transaction_ids": self.validate_duplicates(
                dataframe
            ),
            "total_missing_values": int(
                dataframe.isnull().sum().sum()
            ),
            "columns_with_missing": int(
                (dataframe.isnull().sum() > 0).sum()
            ),
        }

        return report

    def validate(
        self,
        dataframe: pd.DataFrame,
        is_training: bool = True,
    ) -> Dict:
        """
        Execute complete validation.
        """

        logger.info("Running dataset validation...")

        self.validate_schema(
            dataframe,
            is_training=is_training,
        )

        report = self.generate_validation_report(
            dataframe
        )

        logger.info("Validation complete.")

        return report


def main() -> None:
    """
    Standalone execution.
    """

    merger = DataMerger()

    train_df, test_df = merger.run()

    validator = DataValidator()

    train_report = validator.validate(
        train_df,
        is_training=True,
    )

    test_report = validator.validate(
        test_df,
        is_training=False,
    )

    print("\nTraining Dataset Validation")
    print("-" * 40)

    for key, value in train_report.items():
        print(f"{key:<30}: {value}")

    print("\nTest Dataset Validation")
    print("-" * 40)

    for key, value in test_report.items():
        print(f"{key:<30}: {value}")


if __name__ == "__main__":
    main()
