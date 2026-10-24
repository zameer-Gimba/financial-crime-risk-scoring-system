"""
Feature selection module for the Financial Crime Detection &
Risk Scoring System.

Responsible for selecting the most informative features
before model training.

Selection is fitted on the chronological training portion only (never on
the evaluation holdout), and the result is a list of column names that
is applied identically to every dataset.

Author: Muhammad Ibrahim Gimba
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from config.config import (
    CORRELATION_SAMPLE_SIZE,
    CORRELATION_THRESHOLD,
    ENGINEERED_TEST,
    ENGINEERED_TRAIN,
    EXCLUDED_FEATURES,
    FEATURE_IMPORTANCE_REPORT,
    RANDOM_STATE,
    SELECTED_TEST,
    SELECTED_TRAIN,
    TARGET,
    TEST_SIZE,
    TOP_FEATURES,
)
from config.logging_config import get_logger
from src.utils import (
    categorical_columns,
    load_parquet,
    save_dataframe,
    time_split,
)

logger = get_logger(__name__)


class FeatureSelector:
    """
    Perform feature selection on engineered datasets.
    """

    def __init__(
        self,
        top_features: int = TOP_FEATURES,
    ) -> None:
        self.top_features = top_features
        self.selected_features_: list[str] = []
        self.importance_: pd.DataFrame | None = None

        logger.info(
            "FeatureSelector initialized."
        )

    @staticmethod
    def candidate_features(
        dataframe: pd.DataFrame,
    ) -> list[str]:
        """
        All columns except the target and excluded identifiers.
        """

        return [
            column
            for column in dataframe.columns
            if column != TARGET
            and column not in EXCLUDED_FEATURES
        ]

    @staticmethod
    def find_constant_features(
        dataframe: pd.DataFrame,
    ) -> list[str]:
        """
        Features containing only one unique value.
        """

        return [
            column
            for column in dataframe.columns
            if dataframe[column].nunique(dropna=False) <= 1
        ]

    @staticmethod
    def find_duplicate_features(
        dataframe: pd.DataFrame,
    ) -> list[str]:
        """
        Columns that are exact copies of an earlier column.

        Columns are bucketed by a content hash first, so only columns
        with matching hashes are compared in full (instead of comparing
        every pair of the ~500 columns).
        """

        buckets: dict[int, list[str]] = {}

        duplicated_columns = []

        for column in dataframe.columns:

            digest = int(
                pd.util.hash_pandas_object(
                    dataframe[column],
                    index=False,
                ).sum()
            )

            candidates = buckets.setdefault(digest, [])

            if any(
                dataframe[column].equals(dataframe[other])
                for other in candidates
            ):
                duplicated_columns.append(column)
            else:
                candidates.append(column)

        return duplicated_columns

    @staticmethod
    def find_correlated_features(
        dataframe: pd.DataFrame,
        threshold: float = CORRELATION_THRESHOLD,
    ) -> list[str]:
        """
        Numeric features highly correlated with an earlier feature.

        Correlations are computed on a row sample to bound memory/time.
        """

        numeric_df = dataframe.select_dtypes(include=["number"])

        if len(numeric_df) > CORRELATION_SAMPLE_SIZE:
            numeric_df = numeric_df.sample(
                CORRELATION_SAMPLE_SIZE,
                random_state=RANDOM_STATE,
            )

        correlation_matrix = (
            numeric_df.astype("float32").corr().abs()
        )

        upper_triangle = correlation_matrix.where(
            np.triu(
                np.ones(
                    correlation_matrix.shape
                ),
                k=1,
            ).astype(bool)
        )

        return [
            column
            for column in upper_triangle.columns
            if (upper_triangle[column] > threshold).any()
        ]

    def filter_features(
        self,
        dataframe: pd.DataFrame,
    ) -> list[str]:
        """
        Remove constant, duplicated and highly correlated features.
        """

        logger.info(
            "Starting feature filtering..."
        )

        features = self.candidate_features(dataframe)

        constant = self.find_constant_features(dataframe[features])
        features = [f for f in features if f not in set(constant)]
        logger.info("Removed %d constant features.", len(constant))

        duplicated = self.find_duplicate_features(dataframe[features])
        features = [f for f in features if f not in set(duplicated)]
        logger.info("Removed %d duplicated features.", len(duplicated))

        correlated = self.find_correlated_features(dataframe[features])
        features = [f for f in features if f not in set(correlated)]
        logger.info("Removed %d correlated features.", len(correlated))

        logger.info(
            "Remaining Features: %d",
            len(features),
        )

        return features

    def rank_features(
        self,
        dataframe: pd.DataFrame,
        features: list[str],
    ) -> pd.DataFrame:
        """
        Rank features by LightGBM gain importance.

        Categorical columns are passed as pandas ``category`` so LightGBM
        handles them natively - no separate (and potentially inconsistent)
        integer encoding is needed.
        """

        logger.info(
            "Training LightGBM for feature ranking..."
        )

        X_train = dataframe[features].copy()

        y_train = dataframe[TARGET]

        for column in categorical_columns(X_train):
            X_train[column] = X_train[column].astype("category")

        model = lgb.LGBMClassifier(
            objective="binary",
            random_state=RANDOM_STATE,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            n_jobs=-1,
            verbose=-1,
        )

        model.fit(
            X_train,
            y_train,
        )

        importance = pd.DataFrame(
            {
                "feature": features,
                "importance": model.booster_.feature_importance(
                    importance_type="gain"
                ),
            }
        )

        return (
            importance
            .sort_values(
                by="importance",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

    def fit(
        self,
        dataframe: pd.DataFrame,
    ) -> "FeatureSelector":
        """
        Choose the top features using the training data.
        """

        features = self.filter_features(dataframe)

        self.importance_ = self.rank_features(dataframe, features)

        self.selected_features_ = (
            self.importance_
            .head(self.top_features)["feature"]
            .tolist()
        )

        logger.info(
            "Selected %d features.",
            len(self.selected_features_),
        )

        return self

    def transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Keep the selected features, plus the target and TransactionDT
        when present (TransactionDT is carried only so the training
        step can split chronologically; it is never a model input).
        """

        columns = self.selected_features_ + [
            column
            for column in ["TransactionDT", TARGET]
            if column in dataframe.columns
        ]

        return dataframe[columns]

    def load_data(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load engineered datasets.
        """

        logger.info(
            "Loading engineered datasets..."
        )

        train_df = load_parquet(
            ENGINEERED_TRAIN
        )

        test_df = load_parquet(
            ENGINEERED_TEST
        )

        logger.info(
            "Training Shape: %s",
            train_df.shape,
        )

        logger.info(
            "Test Shape: %s",
            test_df.shape,
        )

        return train_df, test_df

    def run(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute the complete feature
        selection pipeline.
        """

        logger.info(
            "Starting feature selection..."
        )

        train_df, test_df = self.load_data()

        fit_df, _ = time_split(train_df, TEST_SIZE)

        self.fit(fit_df)

        selected_train = self.transform(train_df)

        selected_test = self.transform(test_df)

        save_dataframe(
            selected_train,
            SELECTED_TRAIN,
        )

        save_dataframe(
            selected_test,
            SELECTED_TEST,
        )

        FEATURE_IMPORTANCE_REPORT.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.importance_.to_csv(
            FEATURE_IMPORTANCE_REPORT,
            index=False,
        )

        logger.info(
            "Selected Train Shape: %s",
            selected_train.shape,
        )

        logger.info(
            "Selected Test Shape: %s",
            selected_test.shape,
        )

        return (
            selected_train,
            selected_test,
        )


def main() -> None:
    """
    Standalone execution.
    """

    selector = FeatureSelector()

    selected_train, selected_test = (
        selector.run()
    )

    print("\nFeature Selection Summary")
    print("-" * 40)
    print(
        f"Selected Train Shape : "
        f"{selected_train.shape}"
    )
    print(
        f"Selected Test Shape  : "
        f"{selected_test.shape}"
    )


if __name__ == "__main__":
    main()
