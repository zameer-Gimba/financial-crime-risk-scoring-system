"""
Feature engineering module for the Financial Crime Detection &
Risk Scoring System.

Responsible for creating predictive features from the
preprocessed IEEE-CIS datasets.

Author: Muhammad Ibrahim Gimba
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.config import (
    ENGINEERED_TEST,
    ENGINEERED_TRAIN,
    PREPROCESSED_TEST,
    PREPROCESSED_TRAIN,
)
from config.logging_config import get_logger
from src.utils import load_parquet, save_dataframe

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Create predictive features from the preprocessed datasets.
    """

    def __init__(self) -> None:
        logger.info("FeatureEngineer initialized.")

    def _create_time_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create time-related features.
        """

        logger.info("Creating time features...")

        df = dataframe.copy()

        seconds_per_day = 24 * 60 * 60

        if "TransactionDT" in df.columns:

            df["transaction_hour"] = (
                (df["TransactionDT"] // 3600) % 24
            ).astype("int16")

            df["transaction_day"] = (
                df["TransactionDT"] // seconds_per_day
            ).astype("int32")

            df["transaction_week"] = (
                df["transaction_day"] // 7
            ).astype("int16")

            df["is_weekend"] = (
                df["transaction_day"] % 7 >= 5
            ).astype("int8")

        logger.info("Time features created.")

        return df

    def _create_transaction_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create transaction-based features.
        """

        logger.info("Creating transaction features...")

        df = dataframe.copy()

        if "TransactionAmt" in df.columns:

            df["transaction_amount_log"] = np.log1p(
                df["TransactionAmt"]
            )

            df["transaction_amount_round"] = (
                df["TransactionAmt"]
                .round(0)
                .astype("float32")
            )

            df["transaction_amount_bin"] = pd.qcut(
                df["TransactionAmt"],
                q=10,
                duplicates="drop",
                labels=False,
            )

            frequency = (
                df["TransactionAmt"]
                .value_counts(dropna=False)
                .to_dict()
            )

            df["transaction_amount_frequency"] = (
                df["TransactionAmt"]
                .map(frequency)
                .astype("int32")
            )

        logger.info("Transaction features created.")

        return df

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute the feature engineering pipeline.
        """

        logger.info(
            "Loading preprocessed datasets..."
        )

        train_df = load_parquet(
            PREPROCESSED_TRAIN
        )

        test_df = load_parquet(
            PREPROCESSED_TEST
        )

        logger.info(
            "Generating features for training dataset..."
        )

        train_df = self._create_time_features(
            train_df
        )

        train_df = self._create_transaction_features(
            train_df
        )

        logger.info(
            "Generating features for test dataset..."
        )

        test_df = self._create_time_features(
            test_df
        )

        test_df = self._create_transaction_features(
            test_df
        )

        train_df = self._create_card_features(
    train_df
)

train_df = self._create_identity_features(
    train_df
)

test_df = self._create_card_features(
    test_df
)

test_df = self._create_identity_features(
    test_df
)

                train_df = self._create_behavior_features(
            train_df
        )

        train_df = self._create_aggregation_features(
            train_df
        )

        test_df = self._create_behavior_features(
            test_df
        )

        test_df = self._create_aggregation_features(
            test_df
        )

        save_dataframe(
            train_df,
            ENGINEERED_TRAIN,
        )

        save_dataframe(
            test_df,
            ENGINEERED_TEST,
        )

        logger.info(
            "Feature engineering completed successfully."
        )

        return train_df, test_df

        save_dataframe(
            train_df,
            ENGINEERED_TRAIN,
        )

        save_dataframe(
            test_df,
            ENGINEERED_TEST,
        )

        logger.info(
            "Feature engineering completed."
        )

        return train_df, test_df
          def _create_card_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create card-related features.
        """

        logger.info("Creating card features...")

        df = dataframe.copy()

        card_columns = [
            "card1",
            "card2",
            "card3",
            "card5",
        ]

        for column in card_columns:

            if column not in df.columns:
                continue

            frequency = (
                df[column]
                .value_counts(dropna=False)
                .to_dict()
            )

            df[f"{column}_frequency"] = (
                df[column]
                .map(frequency)
                .astype("int32")
            )

            if "TransactionAmt" in df.columns:

                df[f"{column}_amount_mean"] = (
                    df.groupby(column)["TransactionAmt"]
                    .transform("mean")
                )

                df[f"{column}_amount_std"] = (
                    df.groupby(column)["TransactionAmt"]
                    .transform("std")
                    .fillna(0)
                )

        logger.info("Card features created.")

        return df

    def _create_identity_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create identity-related features.
        """

        logger.info("Creating identity features...")

        df = dataframe.copy()

        if "DeviceType" in df.columns:

            df["device_type"] = (
                df["DeviceType"]
                .fillna("Unknown")
                .astype("category")
                .cat.codes
            )

        if "DeviceInfo" in df.columns:

            df["device_info_frequency"] = (
                df["DeviceInfo"]
                .fillna("Unknown")
                .map(
                    df["DeviceInfo"]
                    .fillna("Unknown")
                    .value_counts()
                )
                .astype("int32")
            )

        if "P_emaildomain" in df.columns:

            df["p_email_frequency"] = (
                df["P_emaildomain"]
                .fillna("Unknown")
                .map(
                    df["P_emaildomain"]
                    .fillna("Unknown")
                    .value_counts()
                )
                .astype("int32")
            )

            df["p_email_provider"] = (
                df["P_emaildomain"]
                .fillna("Unknown")
                .str.split(".")
                .str[0]
            )

        if "R_emaildomain" in df.columns:

            df["r_email_frequency"] = (
                df["R_emaildomain"]
                .fillna("Unknown")
                .map(
                    df["R_emaildomain"]
                    .fillna("Unknown")
                    .value_counts()
                )
                .astype("int32")
            )

            df["r_email_provider"] = (
                df["R_emaildomain"]
                .fillna("Unknown")
                .str.split(".")
                .str[0]
            )

        logger.info("Identity features created.")

        return df
          def _create_behavior_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create customer behavioral features.
        """

        logger.info("Creating behavioral features...")

        df = dataframe.copy()

        grouping_columns = [
            "card1",
            "card2",
        ]

        for column in grouping_columns:

            if column not in df.columns:
                continue

            if "TransactionAmt" in df.columns:

                grouped = df.groupby(column)["TransactionAmt"]

                df[f"{column}_transaction_count"] = (
                    grouped.transform("count")
                )

                df[f"{column}_transaction_mean"] = (
                    grouped.transform("mean")
                )

                df[f"{column}_transaction_std"] = (
                    grouped.transform("std")
                    .fillna(0)
                )

                df[f"{column}_transaction_max"] = (
                    grouped.transform("max")
                )

                df[f"{column}_transaction_min"] = (
                    grouped.transform("min")
                )

        logger.info("Behavioral features created.")

        return df

    def _create_aggregation_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create aggregation-based statistical features.
        """

        logger.info("Creating aggregation features...")

        df = dataframe.copy()

        grouping_columns = [
            "card1",
            "card2",
            "addr1",
            "P_emaildomain",
        ]

        statistics = [
            "mean",
            "median",
            "std",
            "count",
        ]

        for column in grouping_columns:

            if column not in df.columns:
                continue

            if "TransactionAmt" not in df.columns:
                continue

            grouped = df.groupby(column)["TransactionAmt"]

            for statistic in statistics:

                feature_name = (
                    f"{column}_amount_{statistic}"
                )

                df[feature_name] = (
                    grouped
                    .transform(statistic)
                    .fillna(0)
                )

        logger.info("Aggregation features created.")

        return df
def main() -> None:
    """
    Standalone execution.
    """

    engineer = FeatureEngineer()

    train_df, test_df = engineer.run()

    print("\nFeature Engineering Summary")
    print("-" * 40)
    print(f"Engineered Train Shape : {train_df.shape}")
    print(f"Engineered Test Shape  : {test_df.shape}")


if __name__ == "__main__":
    main()

      
