"""
Feature engineering module for the Financial Crime Detection &
Risk Scoring System.

Responsible for creating predictive features from the
preprocessed IEEE-CIS datasets.

All lookup tables (frequency counts, per-group amount statistics and
amount bin edges) are learned from the training data in ``fit()`` and
re-used by ``transform()``. This keeps the test set and every
transaction scored in the dashboard on exactly the same scale as the
data the model was trained on - computing them separately per dataset
would give the same raw value a different feature value at inference.

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

SECONDS_PER_HOUR = 60 * 60
SECONDS_PER_DAY = 24 * SECONDS_PER_HOUR

# Columns whose value frequency in the training data is informative
# (rare cards, devices and e-mail domains are riskier).
FREQUENCY_COLUMNS = [
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceInfo",
    "id_30",
    "id_31",
    "TransactionAmt",
    "client_key",
]

# Entities for which "is this amount unusual for them?" is computed.
GROUP_COLUMNS = [
    "card1",
    "card2",
    "addr1",
    "P_emaildomain",
    "client_key",
]

# A pseudo customer identifier: card + billing address + e-mail domain.
CLIENT_KEY_COLUMNS = [
    "card1",
    "addr1",
    "P_emaildomain",
]

AMOUNT_BINS = 10


def _as_key(series: pd.Series) -> pd.Series:
    """
    Convert a column into stable string keys.

    Numeric values are rounded and cast to float64 first so that a value
    read as float32 from parquet and the same value read as int64 or
    float64 from an uploaded CSV produce the same key.
    """

    if pd.api.types.is_numeric_dtype(series):
        return (
            series.astype("float64")
            .round(3)
            .astype(str)
        )

    return series.astype(str)


class FeatureEngineer:
    """
    Create predictive features from the preprocessed datasets.
    """

    def __init__(self) -> None:
        self.frequency_maps_: dict[str, dict[str, int]] = {}
        self.group_stats_: dict[str, pd.DataFrame] = {}
        self.amount_bin_edges_: np.ndarray | None = None
        self.is_fitted = False

        logger.info("FeatureEngineer initialized.")

    @staticmethod
    def _add_client_key(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add the pseudo customer identifier used for behaviour features.
        """

        df = dataframe

        if all(column in df.columns for column in CLIENT_KEY_COLUMNS):
            df["client_key"] = (
                _as_key(df["card1"])
                + "_"
                + _as_key(df["addr1"])
                + "_"
                + _as_key(df["P_emaildomain"])
            )

        return df

    def fit(
        self,
        dataframe: pd.DataFrame,
    ) -> "FeatureEngineer":
        """
        Learn frequency maps, group statistics and amount bins.
        """

        logger.info("Fitting feature engineer...")

        df = self._add_client_key(dataframe.copy())

        for column in FREQUENCY_COLUMNS:

            if column not in df.columns:
                continue

            self.frequency_maps_[column] = (
                _as_key(df[column])
                .value_counts()
                .to_dict()
            )

        if "TransactionAmt" in df.columns:

            amount = df["TransactionAmt"].astype("float64")

            for column in GROUP_COLUMNS:

                if column not in df.columns:
                    continue

                self.group_stats_[column] = (
                    amount
                    .groupby(_as_key(df[column]))
                    .agg(["mean", "std", "median"])
                )

            edges = np.unique(
                np.nanquantile(
                    amount,
                    np.linspace(0, 1, AMOUNT_BINS + 1)[1:-1],
                )
            )

            self.amount_bin_edges_ = edges

        self.is_fitted = True

        logger.info("Feature engineer fitted.")

        return self

    @staticmethod
    def _create_time_features(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create time-related features.

        TransactionDT is a time delta in seconds from a reference date.
        Only cyclical features (hour, weekday) are created; absolute
        day/week counters grow monotonically and do not generalise to
        future transactions.
        """

        df = dataframe

        if "TransactionDT" in df.columns:

            seconds = df["TransactionDT"].astype("float64")

            df["transaction_hour"] = (
                (seconds // SECONDS_PER_HOUR) % 24
            )

            df["transaction_weekday"] = (
                (seconds // SECONDS_PER_DAY) % 7
            )

            df["is_night"] = (
                df["transaction_hour"].between(0, 5).astype("int8")
            )

        return df

    def _create_transaction_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create transaction-amount features.
        """

        df = dataframe

        if "TransactionAmt" in df.columns:

            amount = df["TransactionAmt"].astype("float64")

            df["transaction_amount_log"] = np.log1p(amount)

            # Fraudsters often test cards with odd, non-round amounts.
            df["transaction_amount_cents"] = (
                (amount - np.floor(amount)).round(3)
            )

            if self.amount_bin_edges_ is not None:
                df["transaction_amount_bin"] = np.digitize(
                    amount.fillna(0),
                    self.amount_bin_edges_,
                ).astype("int8")

        return df

    def _create_frequency_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Map each value to how often it appeared in the training data.

        Values never seen during training get a frequency of 0.
        """

        df = dataframe

        for column, frequency in self.frequency_maps_.items():

            if column not in df.columns:
                continue

            df[f"{column}_frequency"] = (
                _as_key(df[column])
                .map(frequency)
                .fillna(0)
                .astype("int32")
            )

        return df

    def _create_behavior_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compare each transaction amount with what is typical for the
        card, address, e-mail domain and pseudo customer.
        """

        df = dataframe

        if "TransactionAmt" not in df.columns:
            return df

        amount = df["TransactionAmt"].astype("float64")

        new_columns = {}

        for column, stats in self.group_stats_.items():

            if column not in df.columns:
                continue

            keys = _as_key(df[column])

            mean = keys.map(stats["mean"])
            std = keys.map(stats["std"])
            median = keys.map(stats["median"])

            new_columns[f"{column}_amount_mean"] = mean
            new_columns[f"{column}_amount_std"] = std
            new_columns[f"{column}_amount_median"] = median
            new_columns[f"{column}_amount_to_mean"] = amount / mean
            new_columns[f"{column}_amount_zscore"] = (
                (amount - mean) / std.replace(0, np.nan)
            )

        if new_columns:
            df = pd.concat(
                [df, pd.DataFrame(new_columns, index=df.index)],
                axis=1,
            )

        return df

    @staticmethod
    def _create_identity_features(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create identity and e-mail features.
        """

        df = dataframe

        identity_numeric = [
            column
            for column in ["id_01", "id_02", "id_05", "id_06"]
            if column in df.columns
        ]

        if identity_numeric:
            df["has_identity"] = (
                df[identity_numeric].notna().any(axis=1).astype("int8")
            )

        for prefix, column in [
            ("p", "P_emaildomain"),
            ("r", "R_emaildomain"),
        ]:

            if column in df.columns:
                df[f"{prefix}_email_provider"] = (
                    df[column]
                    .astype(str)
                    .str.split(".")
                    .str[0]
                )

        if {"P_emaildomain", "R_emaildomain"}.issubset(df.columns):
            df["email_domain_match"] = (
                (df["P_emaildomain"] == df["R_emaildomain"])
                & (df["P_emaildomain"] != "Unknown")
            ).astype("int8")

        return df

    @staticmethod
    def _create_missing_features(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Count missing numeric values per transaction.
        """

        df = dataframe

        numeric = df.select_dtypes(include=["number"])

        df["missing_count"] = (
            numeric.isna().sum(axis=1).astype("int16")
        )

        return df

    def transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create all features using the fitted lookup tables.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "FeatureEngineer must be fitted before transform()."
            )

        df = dataframe.copy()

        df = self._create_missing_features(df)

        df = self._add_client_key(df)

        df = self._create_time_features(df)

        df = self._create_transaction_features(df)

        df = self._create_frequency_features(df)

        df = self._create_behavior_features(df)

        df = self._create_identity_features(df)

        df = df.drop(columns=["client_key"], errors="ignore")

        return df

    def fit_transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Fit on training data and transform it.
        """

        return self.fit(dataframe).transform(dataframe)

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

        train_df = self.fit_transform(train_df)

        logger.info(
            "Generating features for test dataset..."
        )

        test_df = self.transform(test_df)

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
