"""
Data preprocessing module for the Financial Crime Detection &
Risk Scoring System.

Responsible for cleaning and preparing the merged IEEE-CIS
datasets for feature engineering.

The preprocessor follows the scikit-learn fit/transform convention:
everything it learns (the raw schema and which columns are categorical)
comes from the training data only, and the same fitted object is used
for the test set and for new transactions scored by the application.

Author: Muhammad Ibrahim Gimba
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.config import (
    PREPROCESSED_TEST,
    PREPROCESSED_TRAIN,
    TARGET,
)
from config.logging_config import get_logger
from src.data_validation import DataValidator
from src.merge_data import DataMerger
from src.utils import categorical_columns, save_dataframe

logger = get_logger(__name__)

MISSING_CATEGORY = "Unknown"


class DataPreprocessor:
    """
    Clean and prepare datasets before feature engineering.

    Missing numeric values are intentionally left as NaN: LightGBM,
    XGBoost and CatBoost all handle missing values natively, and in this
    dataset *whether* a value is missing (e.g. no identity record) is
    itself a strong fraud signal that median imputation would erase.
    Missing categorical values are replaced with ``"Unknown"``.
    """

    def __init__(self) -> None:
        self.raw_columns_: list[str] = []
        self.categorical_columns_: list[str] = []
        self.is_fitted = False

        logger.info("DataPreprocessor initialized.")

    @staticmethod
    def remove_invalid_records(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Remove invalid observations.

        Only applied to training data - transactions submitted for
        scoring are never dropped.
        """

        logger.info("Removing invalid records...")

        df = dataframe

        if "TransactionAmt" in df.columns:
            df = df[df["TransactionAmt"] > 0]

        df = df.reset_index(drop=True)

        logger.info("Remaining rows: %d", len(df))

        return df

    @staticmethod
    def convert_data_types(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert selected columns to numeric types.
        """

        logger.info("Converting data types...")

        df = dataframe.copy()

        for column in ["TransactionDT", "TransactionAmt"]:

            if column in df.columns:
                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce"
                )

        return df

    def fit(
        self,
        dataframe: pd.DataFrame,
    ) -> "DataPreprocessor":
        """
        Learn the raw schema from the training data.
        """

        self.raw_columns_ = [
            column
            for column in dataframe.columns
            if column != TARGET
        ]

        self.categorical_columns_ = [
            column
            for column in categorical_columns(dataframe)
            if column != TARGET
        ]

        self.is_fitted = True

        logger.info(
            "Preprocessor fitted: %d raw columns, %d categorical.",
            len(self.raw_columns_),
            len(self.categorical_columns_),
        )

        return self

    def handle_missing_values(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Fill missing categorical values with ``"Unknown"``.
        """

        logger.info("Handling missing values...")

        df = dataframe

        for column in self.categorical_columns_:
            df[column] = (
                df[column]
                .astype(object)
                .where(df[column].notna(), MISSING_CATEGORY)
                .astype(str)
            )

        return df

    def align_schema(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Make sure every training column exists.

        Uploaded files may contain only a subset of the 430+ raw
        IEEE-CIS columns; absent columns are added as missing values so
        the downstream pipeline always receives the training schema.
        """

        missing = [
            column
            for column in self.raw_columns_
            if column not in dataframe.columns
        ]

        if missing:
            logger.info(
                "Adding %d absent raw columns as missing.",
                len(missing),
            )

            filler = pd.DataFrame(
                {
                    column: (
                        pd.Series(np.nan, index=dataframe.index, dtype=object)
                        if column in self.categorical_columns_
                        else pd.Series(np.nan, index=dataframe.index, dtype=np.float32)
                    )
                    for column in missing
                }
            )

            dataframe = pd.concat([dataframe, filler], axis=1)

        for column in self.raw_columns_:
            if column not in self.categorical_columns_:
                dataframe[column] = pd.to_numeric(
                    dataframe[column],
                    errors="coerce",
                )

        keep = self.raw_columns_ + (
            [TARGET] if TARGET in dataframe.columns else []
        )

        return dataframe[keep].copy()

    def transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply the fitted preprocessing to any dataset.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "DataPreprocessor must be fitted before transform()."
            )

        df = self.convert_data_types(dataframe)

        df = self.align_schema(df)

        df = self.handle_missing_values(df)

        return df

    def preprocess(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Fit on (cleaned) training data and transform it.
        """

        logger.info("Starting preprocessing pipeline...")

        df = self.remove_invalid_records(dataframe)

        df = self.fit(df).transform(df)

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

        processed_test = self.transform(test_df)

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
