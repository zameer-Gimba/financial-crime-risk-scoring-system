"""
Model training module for the Financial Crime Detection &
Risk Scoring System.

Responsible for training the ensemble fraud detection
model using LightGBM, XGBoost, CatBoost,
and Logistic Regression.

Author: Muhammad Ibrahim Gimba
"""

from __future__ import annotations

import joblib
import pandas as pd

import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
)
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from config.config import (
    BASE_MODEL_DIR,
    EXCLUDED_FEATURES,
    LEARNING_RATE,
    N_ESTIMATORS,
    N_FOLDS,
    RANDOM_STATE,
    RUN_CROSS_VALIDATION,
    SELECTED_TRAIN,
    STACKING_MODEL,
    TARGET,
    TEST_SIZE,
    THRESHOLD_SHARE,
)
from config.logging_config import get_logger
from src.utils import categorical_columns, load_parquet, time_split

logger = get_logger(__name__)


class ModelTrainer:
    """
    Train the fraud detection ensemble model.
    """

    def __init__(self) -> None:

        logger.info(
            "ModelTrainer initialized."
        )

    def load_data(
        self,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Load the selected training dataset.
        """

        logger.info(
            "Loading selected dataset..."
        )

        try:
            dataframe = load_parquet(
                SELECTED_TRAIN
            )
        except FileNotFoundError as error:
            logger.error(
                "Selected training dataset not found at %s",
                SELECTED_TRAIN,
            )
            raise FileNotFoundError(
                f"Selected training dataset not found at {SELECTED_TRAIN}. "
                "Run feature selection before model training."
            ) from error

        if dataframe.empty:
            logger.error(
                "Selected training dataset at %s is empty.",
                SELECTED_TRAIN,
            )
            raise ValueError(
                f"Selected training dataset at {SELECTED_TRAIN} is empty."
            )

        if TARGET not in dataframe.columns:
            logger.error(
                "Target column '%s' not found in selected dataset.",
                TARGET,
            )
            raise ValueError(
                f"Target column '{TARGET}' not found in selected dataset. "
                f"Available columns: {list(dataframe.columns)}"
            )

        X = dataframe.drop(
            columns=[TARGET]
        )

        y = dataframe[TARGET]

        if y.nunique(dropna=True) < 2:
            logger.error(
                "Target column '%s' contains only one class: %s",
                TARGET,
                y.unique().tolist(),
            )
            raise ValueError(
                f"Target column '{TARGET}' must contain both classes "
                f"(fraud and non-fraud); found only {y.unique().tolist()}."
            )

        unexpected_values = (
            set(y.dropna().unique()) - {0, 1}
        )

        if unexpected_values:
            logger.error(
                "Target column '%s' contains values other than 0/1: %s",
                TARGET,
                sorted(unexpected_values, key=str),
            )
            raise ValueError(
                f"Target column '{TARGET}' must be binary (0 = non-fraud, "
                f"1 = fraud); found unexpected value(s): "
                f"{sorted(unexpected_values, key=str)}."
            )

        logger.info(
            "Dataset loaded successfully."
        )

        logger.info(
            "Samples: %d",
            len(dataframe),
        )

        logger.info(
            "Features: %d",
            X.shape[1],
        )

        return X, y

    def prepare_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> dict:
        """
        Split the data chronologically for training and evaluation.

        The IEEE-CIS data is ordered in time, and a deployed fraud model
        always scores transactions that happen *after* the ones it was
        trained on. A random split would leak future behaviour into
        training and overstate performance, so:

        - train     : the earliest (1 - TEST_SIZE) share of transactions
        - threshold : the next part of the holdout, used only to tune the
                      decision threshold
        - test      : the most recent transactions, used only to report
                      final metrics

        Categorical encoding is NOT done here - it is fitted later,
        inside the training pipeline, using X_train only.
        """

        logger.info(
            "Preparing training data (time-based split)..."
        )

        data = X.copy()
        data[TARGET] = y.values

        if "TransactionDT" not in data.columns:
            logger.warning(
                "TransactionDT not available - assuming rows are "
                "already in chronological order."
            )
            data["TransactionDT"] = range(len(data))

        train_df, holdout_df = time_split(data, TEST_SIZE)

        threshold_df, test_df = time_split(
            holdout_df,
            1 - THRESHOLD_SHARE,
        )

        def split_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
            return (
                frame.drop(
                    columns=[TARGET] + EXCLUDED_FEATURES,
                    errors="ignore",
                ),
                frame[TARGET],
            )

        X_train, y_train = split_xy(train_df)
        X_threshold, y_threshold = split_xy(threshold_df)
        X_test, y_test = split_xy(test_df)

        positive_count = int((y_train == 1).sum())
        negative_count = int((y_train == 0).sum())

        if positive_count == 0 or negative_count == 0:
            logger.error(
                "Training split contains only one class after "
                "train/validation split (positive=%d, negative=%d).",
                positive_count,
                negative_count,
            )
            raise ValueError(
                "Training split must contain both classes; got "
                f"positive={positive_count}, negative={negative_count}."
            )

        scale_pos_weight = negative_count / positive_count

        logger.info(
            "Training samples: %d | Threshold samples: %d | "
            "Test samples: %d",
            len(X_train),
            len(X_threshold),
            len(X_test),
        )

        logger.info(
            "Positive samples: %d",
            positive_count,
        )

        logger.info(
            "Negative samples: %d",
            negative_count,
        )

        logger.info(
            "Scale Pos Weight: %.4f",
            scale_pos_weight,
        )

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_threshold": X_threshold,
            "y_threshold": y_threshold,
            "X_test": X_test,
            "y_test": y_test,
            "scale_pos_weight": scale_pos_weight,
        }

    def build_models(
        self,
        X_train: pd.DataFrame,
        scale_pos_weight: float,
        n_estimators: int = N_ESTIMATORS,
        n_folds: int = N_FOLDS,
    ) -> Pipeline:
        """
        Build the preprocessing step, the base learners, and the
        stacking ensemble, assembled into a single inference-capable
        pipeline: raw selected features -> preprocessing -> stacking.

        The preprocessing step is fitted using X_train only (here, and
        again independently inside every cross-validation fold), never
        on validation or held-out data.
        """

        logger.info(
            "Building ensemble models..."
        )

        categorical_features = categorical_columns(X_train)

        logger.info(
            "Categorical columns detected: %s",
            categorical_features,
        )

        preprocessing = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                        encoded_missing_value=-1,
                    ),
                    categorical_features,
                ),
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )

        # Each base learner uses all CPU cores itself; the stacking
        # ensemble therefore fits them one at a time (n_jobs=1) to avoid
        # oversubscribing the CPU with nested parallelism.

        lightgbm_model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=n_estimators,
            learning_rate=LEARNING_RATE,
            num_leaves=127,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.6,
            random_state=RANDOM_STATE,
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            verbose=-1,
        )

        xgboost_model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            n_estimators=n_estimators,
            learning_rate=LEARNING_RATE,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.6,
            random_state=RANDOM_STATE,
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            n_jobs=-1,
        )

        catboost_model = CatBoostClassifier(
            iterations=n_estimators,
            learning_rate=LEARNING_RATE * 2,
            depth=8,
            random_seed=RANDOM_STATE,
            auto_class_weights="Balanced",
            thread_count=-1,
            allow_writing_files=False,
            verbose=False,
        )

        base_models = [
            (
                "lightgbm",
                lightgbm_model,
            ),
            (
                "xgboost",
                xgboost_model,
            ),
            (
                "catboost",
                catboost_model,
            ),
        ]

        meta_model = LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000,
        )

        stacking_model = (
            StackingClassifier(
                estimators=base_models,
                final_estimator=meta_model,
                cv=StratifiedKFold(
                    n_splits=n_folds,
                    shuffle=True,
                    random_state=RANDOM_STATE,
                ),
                n_jobs=1,
                passthrough=False,
            )
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessing", preprocessing),
                ("stacking", stacking_model),
            ]
        )

        logger.info(
            "Ensemble model created successfully."
        )

        return pipeline

    def cross_validate(
        self,
        model: Pipeline,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> float:
        """
        Perform stratified cross-validation (PR-AUC).

        Optional and slow: it refits the whole stacked ensemble N_FOLDS
        times. Enable with RUN_CROSS_VALIDATION in config.

        model is the full preprocessing + stacking pipeline, so every
        fold fits its own preprocessing on that fold's training portion
        only - no leakage from a fold's held-out data.
        """

        logger.info(
            "Performing %d-fold cross-validation...",
            N_FOLDS,
        )

        scores = cross_val_score(
            estimator=model,
            X=X_train,
            y=y_train,
            cv=StratifiedKFold(
                n_splits=N_FOLDS,
                shuffle=True,
                random_state=RANDOM_STATE,
            ),
            scoring="average_precision",
            n_jobs=1,
        )

        logger.info(
            "Cross-validation PR-AUC scores: %s",
            scores,
        )

        logger.info(
            "Mean PR-AUC: %.4f",
            scores.mean(),
        )

        return scores.mean()

    def train_model(
        self,
        model: Pipeline,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Pipeline:
        """
        Train the full preprocessing + stacking pipeline on the
        training dataset only (never on the threshold or test sets).
        """

        logger.info(
            "Training stacking ensemble..."
        )

        model.fit(
            X_train,
            y_train,
        )

        logger.info(
            "Stacking ensemble training completed."
        )

        return model

    def save_base_models(
        self,
        model: Pipeline,
    ) -> None:
        """
        Save the trained (fitted) base models.

        IMPORTANT: these are the individual base learners as fitted
        inside the stacking ensemble, which means they operate on the
        *preprocessed/transformed* feature matrix produced by the
        "preprocessing" pipeline step - NOT on raw selected features.
        They are saved here for inspection/analysis purposes only.

        They are not standalone raw-input inference artifacts. The
        only artifact that accepts raw selected features directly is
        the complete fitted pipeline saved by save_stacking_model()
        (preprocessing + stacking together) at STACKING_MODEL.

        model is the full pipeline; the fitted StackingClassifier lives
        at model.named_steps["stacking"], and its .named_estimators_
        attribute maps each base learner's name to its actual fitted
        instance (StackingClassifier clones and fits its estimators
        internally, so these are the fitted models, not the unfitted
        definitions originally built in build_models()).
        """

        logger.info(
            "Saving trained base models "
            "(fitted on transformed features, "
            "for inspection only - not standalone raw-input models)..."
        )

        stacking_model = model.named_steps["stacking"]

        BASE_MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        for name, estimator in stacking_model.named_estimators_.items():

            model_path = (
                BASE_MODEL_DIR
                / f"{name}_model.joblib"
            )

            joblib.dump(
                estimator,
                model_path,
            )

            logger.info(
                "Saved %s model to %s",
                name,
                model_path,
            )

        logger.info(
            "Base models saved successfully."
        )

    def save_stacking_model(
        self,
        model: Pipeline,
    ) -> None:
        """
        Save the complete, fitted preprocessing + stacking pipeline.

        Saving the full pipeline (not just the StackingClassifier) means
        a downstream module can call joblib.load(STACKING_MODEL) and
        immediately use .predict() / .predict_proba() on raw selected
        features, without separately reconstructing any encoder.
        """

        logger.info(
            "Saving stacking ensemble..."
        )

        STACKING_MODEL.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            model,
            STACKING_MODEL,
        )

        logger.info(
            "Stacking model saved to %s",
            STACKING_MODEL,
        )

    def fit(
        self,
        splits: dict,
    ) -> Pipeline:
        """
        Build, optionally cross-validate, and train the ensemble.
        """

        pipeline = self.build_models(
            splits["X_train"],
            splits["scale_pos_weight"],
        )

        if RUN_CROSS_VALIDATION:
            self.cross_validate(
                pipeline,
                splits["X_train"],
                splits["y_train"],
            )

        return self.train_model(
            pipeline,
            splits["X_train"],
            splits["y_train"],
        )

    def run(self) -> tuple[Pipeline, dict]:
        """
        Execute the complete model training pipeline.

        Returns the fitted pipeline and the data splits (the threshold
        and test splits are used downstream for evaluation).
        """

        logger.info(
            "Starting model training pipeline..."
        )

        X, y = self.load_data()

        splits = self.prepare_data(
            X,
            y,
        )

        pipeline = self.fit(splits)

        self.save_base_models(
            pipeline
        )

        self.save_stacking_model(
            pipeline
        )

        logger.info(
            "Model training pipeline completed successfully."
        )

        return pipeline, splits


def main() -> None:
    """
    Standalone execution.
    """

    trainer = ModelTrainer()

    model, _ = trainer.run()

    print("\nModel Training Summary")
    print("-" * 40)
    print(
        "Stacking Model : "
        f"{type(model).__name__}"
    )
    print(
        f"Saved Model    : {STACKING_MODEL}"
    )


if __name__ == "__main__":
    main()
