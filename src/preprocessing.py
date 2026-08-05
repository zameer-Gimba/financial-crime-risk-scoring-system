"""
Data preprocessing module for the Financial Crime Detection &
Risk Scoring System.

Responsible for cleaning and preparing the merged IEEE-CIS
datasets for feature engineering.

Author: Muhammad Ibrahim Gimba
"""

from __future__ import annotations

import pandas as pd

from config.config import (
    PREPROCESSED_TEST,
    PREPROCESSED_TRAIN,
)
from config.logging_config import get_logger
from src.data_validation import DataValidator
from src.merge_data import DataMerger
from src.utils import save_dataframe

logger = get_logger(__name__)


class DataPreprocessor:
    """
    Clean and prepare datasets before feature engineering.
    """

    def __init__(self) -> None:
        logger.info("DataPreprocessor initialized.")

    @staticmethod
    def remove_invalid_records(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Remove invalid observations.
        """

        logger.info("Removing invalid records...")

        df = dataframe.copy()

        if "TransactionAmt" in df.columns:
            df = df[df["TransactionAmt"] > 0]

        df.reset_index(drop=True, inplace=True)

        logger.info("Remaining rows: %d", len(df))

        return df

    @staticmethod
    def convert_data_types(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert selected columns to efficient data types.
        """

        logger.info("Optimizing data types...")

        df = dataframe.copy()

        if "TransactionDT" in df.columns:
            df["TransactionDT"] = pd.to_numeric(
                df["TransactionDT"],
                errors="coerce"
            )

        if "TransactionAmt" in df.columns:
            df["TransactionAmt"] = pd.to_numeric(
                df["TransactionAmt"],
                errors="coerce"
            )

        return df

    @staticmethod
    def handle_missing_values(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Handle missing values.

        Numeric columns:
            Median

        Categorical columns:
            'Unknown'
        """

        logger.info("Handling missing values...")

        df = dataframe.copy()

        numeric_columns = df.select_dtypes(
            include=["number"]
        ).columns

        categorical_columns = df.select_dtypes(
            exclude=["number"]
        ).columns

        for column in numeric_columns:

            median = df[column].median()

            df[column] = df[column].fillna(median)

        for column in categorical_columns:

            df[column] = df[column].fillna("Unknown")

        return df

    def preprocess(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Execute preprocessing pipeline.
        """

        logger.info("Starting preprocessing pipeline...")

        df = self.remove_invalid_records(dataframe)

        df = self.convert_data_types(df)

        df = self.handle_missing_values(df)

        logger.info("Preprocessing completed.")

        return df

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute complete preprocessing workflow.
        """

        logger.info("Running preprocessing workflow...")

        merger = DataMerger()

        validator = DataValidator()

        train_df, test_df = merger.run()

        validator.validate(
            train_df,
            is_training=True,
        )

        validator.validate(
            test_df,
            is_training=False,
        )

        processed_train = self.preprocess(train_df)

        processed_test = self.preprocess(test_df)

        save_dataframe(
            processed_train,
            PREPROCESSED_TRAIN,
        )

        save_dataframe(
            processed_test,
            PREPROCESSED_TEST,
        )

        logger.info(
            "Processed datasets saved successfully."
        )

        return processed_train, processed_test


def main() -> None:
    """
    Standalone execution.
    """

    preprocessor = DataPreprocessor()

    train_df, test_df = preprocessor.run()

    print("\nPreprocessing Summary")
    print("-" * 40)
    print(f"Processed Train Shape : {train_df.shape}")
    print(f"Processed Test Shape  : {test_df.shape}")


if __name__ == "__main__":
    main()
