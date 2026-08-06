"""
Feature selection module for the Financial Crime Detection &
Risk Scoring System.

Responsible for selecting the most informative features
before model training.

Author: Muhammad Ibrahim Gimba
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import lightgbm as lgb

from config.config import (
    CORRELATION_THRESHOLD,
    ENGINEERED_TEST,
    ENGINEERED_TRAIN,
    RANDOM_STATE,
    SELECTED_TEST,
    SELECTED_TRAIN,
    TARGET,
    TOP_FEATURES,
)
from config.logging_config import get_logger
from src.utils import (
    load_parquet,
    save_dataframe,
)

logger = get_logger(__name__)


class FeatureSelector:
    """
    Perform feature selection on engineered datasets.
    """

    def __init__(self) -> None:
        logger.info(
            "FeatureSelector initialized."
        )

    @staticmethod
    def remove_constant_features(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Remove features containing only one unique value.
        """

        logger.info(
            "Removing constant features..."
        )

        df = dataframe.copy()

        constant_columns = [
            column
            for column in df.columns
            if df[column].nunique(dropna=False) <= 1
        ]

        if constant_columns:

            logger.info(
                "Removed %d constant features.",
                len(constant_columns),
            )

            df = df.drop(
                columns=constant_columns
            )

        return df

    @staticmethod
    def remove_duplicate_features(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Remove duplicate columns.
        """

        logger.info(
            "Removing duplicate features..."
        )

        df = dataframe.copy()

        duplicated_columns = []

        columns = df.columns

        for index, column in enumerate(columns):

            for other_column in columns[index + 1:]:

                if df[column].equals(
                    df[other_column]
                ):

                    duplicated_columns.append(
                        other_column
                    )

        if duplicated_columns:

            logger.info(
                "Removed %d duplicated features.",
                len(duplicated_columns),
            )

            df = df.drop(
                columns=duplicated_columns
            )

        return df
      
    @staticmethod
    def remove_correlated_features(
        dataframe: pd.DataFrame,
        threshold: float = CORRELATION_THRESHOLD,
    ) -> pd.DataFrame:
        """
        Remove highly correlated numeric features.
        """

        logger.info(
            "Removing highly correlated features..."
        )

        df = dataframe.copy()
        
        numeric_df = (
          df.select_dtypes(include=["number"])
          .drop(columns=[TARGET], errors="ignore")
        )

        correlation_matrix = (
            numeric_df.corr().abs()
        )

        upper_triangle = correlation_matrix.where(
            np.triu(
                np.ones(
                    correlation_matrix.shape
                ),
                k=1,
            ).astype(bool)
        )

        correlated_columns = [
            column
            for column in upper_triangle.columns
            if any(
                upper_triangle[column]
                > threshold
            )
        ]

        if correlated_columns:

            logger.info(
                "Removed %d correlated features.",
                len(correlated_columns),
            )

            df = df.drop(
                columns=correlated_columns
            )

        return df

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
            "Datasets loaded successfully."
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

    def preprocess_features(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute feature filtering before
        feature importance ranking.
        """

        logger.info(
            "Starting feature filtering..."
        )

        train_df = self.remove_constant_features(
            train_df
        )

        train_df = self.remove_duplicate_features(
            train_df
        )

        train_df = self.remove_correlated_features(
            train_df
        )

        common_columns = [
            column
            for column in train_df.columns
            if column == TARGET
            or column in test_df.columns
        ]

        train_df = train_df[
            common_columns
        ]

        test_df = test_df[
            [
                column
                for column in common_columns
                if column != TARGET
            ]
        ]

        logger.info(
            "Feature filtering completed."
        )

        logger.info(
            "Remaining Features: %d",
            len(train_df.columns) - 1,
        )

        return train_df, test_df
      
    def select_top_features(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Select the most important features using
        LightGBM feature importance.
        """

        logger.info(
            "Selecting top features..."
        )

        X_train = train_df.drop(
            columns=[TARGET]
        )

        y_train = train_df[TARGET]

        categorical_columns = (
            X_train.select_dtypes(
                include=[
                    "object",
                    "category",
                ]
            ).columns
        )

        for column in categorical_columns:

            X_train[column] = (
                X_train[column]
                .astype("category")
                .cat.codes
            )

            if column in test_df.columns:

                test_df[column] = (
                    test_df[column]
                    .astype("category")
                    .cat.codes
                )

        model = lgb.LGBMClassifier(
            objective="binary",
            random_state=RANDOM_STATE,
            n_estimators=300,
            learning_rate=0.05,
            n_jobs=-1,
        )

        logger.info(
            "Training LightGBM for feature selection..."
        )

        model.fit(
            X_train,
            y_train,
            categorical_feature=list(categorical_columns),
        )

        importance = pd.DataFrame(
            {
                "feature": X_train.columns,
                "importance": (
                    model.feature_importances_
                ),
            }
        )

        importance = (
            importance
            .sort_values(
                by="importance",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        selected_features = (
            importance
            .head(TOP_FEATURES)[
                "feature"
            ]
            .tolist()
        )

        logger.info(
            "Selected %d features.",
            len(selected_features),
        )

        selected_train = train_df[
            selected_features + [TARGET]
        ]

        selected_test = test_df[
            selected_features
        ]

        return (
            selected_train,
            selected_test,
        )

  
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

        train_df, test_df = (
            self.preprocess_features(
                train_df,
                test_df,
            )
        )

        selected_train, selected_test = (
            self.select_top_features(
                train_df,
                test_df,
            )
        )

        save_dataframe(
            selected_train,
            SELECTED_TRAIN,
        )

        save_dataframe(
            selected_test,
            SELECTED_TEST,
        )

        logger.info(
            "Selected datasets saved successfully."
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
