"""
End-to-end training pipeline for the Financial Crime Detection &
Risk Scoring System.

Runs every stage in order and produces the artifacts the Streamlit
dashboard needs:

- models/fraud_risk_model.joblib       deployable model bundle
- reports/metrics.json                 holdout metrics and risk bands
- reports/pr_curve.csv, roc_curve.csv  evaluation curves
- reports/global_shap_importance.csv   global explainability
- reports/feature_importance.csv       feature selection ranking
- data/sample/sample_transactions.csv  demo transactions for the app

Missing raw data is downloaded from Kaggle automatically (see
src/download_data.py for the one-time Kaggle token setup).

Usage:
    python -m src.pipeline                      # full training run
    python -m src.pipeline --nrows 100000 --n-estimators 50   # quick run

Author: Muhammad Ibrahim Gimba
"""

from __future__ import annotations

import argparse
import platform
from datetime import datetime, timezone
from importlib.metadata import version

import pandas as pd

from config.config import (
    FEATURE_IMPORTANCE_REPORT,
    GLOBAL_SHAP_REPORT,
    MERGED_TRAIN,
    MODEL_BUNDLE,
    N_ESTIMATORS,
    RANDOM_STATE,
    SAMPLE_FRAUD_ROWS,
    SAMPLE_LEGIT_ROWS,
    SAMPLE_TRANSACTIONS,
    SHAP_SAMPLE_SIZE,
    TARGET,
    TEST_SIZE,
    create_directories,
)
from config.constants import TRANSACTION_ID
from config.logging_config import get_logger
from src.data_loader import DataLoader
from src.data_validation import DataValidator
from src.download_data import download_dataset
from src.evaluation import ModelEvaluator
from src.explainability import EnsembleExplainer
from src.feature_engineering import FeatureEngineer
from src.feature_selection import FeatureSelector
from src.merge_data import DataMerger
from src.model_training import ModelTrainer
from src.preprocessing import DataPreprocessor
from src.utils import load_parquet, save_dataframe, save_model, time_split

logger = get_logger(__name__)

TRACKED_PACKAGES = [
    "scikit-learn",
    "lightgbm",
    "xgboost",
    "catboost",
    "shap",
    "pandas",
    "numpy",
]


class TrainingPipeline:
    """
    Orchestrate data preparation, training, evaluation and packaging.
    """

    def __init__(
        self,
        nrows: int | None = None,
        n_estimators: int = N_ESTIMATORS,
        reuse_merged: bool = False,
    ) -> None:
        self.nrows = nrows
        self.n_estimators = n_estimators
        self.reuse_merged = reuse_merged

    def load_training_data(self) -> pd.DataFrame:
        """
        Load and merge the labelled IEEE-CIS training data.
        """

        if self.reuse_merged and MERGED_TRAIN.exists():
            logger.info("Reusing merged training data at %s", MERGED_TRAIN)
            merged = load_parquet(MERGED_TRAIN)
        else:
            # Fetches the CSVs from Kaggle if they are not in data/raw yet.
            download_dataset()
            loader = DataLoader()
            merged = DataMerger.merge_train_data(
                loader.load_train_transaction(),
                loader.load_train_identity(),
            )
            save_dataframe(merged, MERGED_TRAIN)

        if self.nrows:
            merged = (
                merged
                .sort_values("TransactionDT", kind="stable")
                .head(self.nrows)
                .reset_index(drop=True)
            )

        DataValidator().validate(merged, is_training=True)

        return merged

    @staticmethod
    def build_sample(
        raw: pd.DataFrame,
        transaction_ids: pd.Series,
    ) -> pd.DataFrame:
        """
        Pick demo transactions from the unseen test period.
        """

        pool = raw[raw[TRANSACTION_ID].isin(set(transaction_ids))]

        fraud = pool[pool[TARGET] == 1]
        legit = pool[pool[TARGET] == 0]

        sample = pd.concat(
            [
                fraud.sample(
                    min(SAMPLE_FRAUD_ROWS, len(fraud)),
                    random_state=RANDOM_STATE,
                ),
                legit.sample(
                    min(SAMPLE_LEGIT_ROWS, len(legit)),
                    random_state=RANDOM_STATE,
                ),
            ]
        )

        return sample.sort_values("TransactionDT").reset_index(drop=True)

    def run(self) -> dict:
        """
        Execute the full pipeline and save all artifacts.
        """

        create_directories()

        raw = self.load_training_data()

        logger.info("Stage: preprocessing")
        preprocessor = DataPreprocessor()
        clean = preprocessor.preprocess(raw)

        fit_part, holdout = time_split(clean, TEST_SIZE)

        logger.info("Stage: feature engineering (fitted on training period)")
        engineer = FeatureEngineer().fit(fit_part)
        engineered_fit = engineer.transform(fit_part)
        engineered_holdout = engineer.transform(holdout)

        logger.info("Stage: feature selection")
        selector = FeatureSelector().fit(engineered_fit)
        selector.importance_.to_csv(FEATURE_IMPORTANCE_REPORT, index=False)

        selected = pd.concat(
            [
                selector.transform(engineered_fit),
                selector.transform(engineered_holdout),
            ]
        )

        del engineered_fit, engineered_holdout

        logger.info("Stage: model training")
        trainer = ModelTrainer()

        splits = trainer.prepare_data(
            selected.drop(columns=[TARGET]),
            selected[TARGET],
        )

        model = trainer.build_models(
            splits["X_train"],
            splits["scale_pos_weight"],
            n_estimators=self.n_estimators,
        )

        model = trainer.train_model(
            model,
            splits["X_train"],
            splits["y_train"],
        )

        logger.info("Stage: evaluation")
        report = ModelEvaluator().run(model, splits)

        logger.info("Stage: explainability")
        explainer = EnsembleExplainer(model)

        shap_sample = splits["X_test"].sample(
            min(SHAP_SAMPLE_SIZE, len(splits["X_test"])),
            random_state=RANDOM_STATE,
        )

        explainer.global_importance(shap_sample).to_csv(
            GLOBAL_SHAP_REPORT,
            index=False,
        )

        logger.info("Stage: demo sample")
        # Row labels are preserved from `clean` through every stage.
        test_ids = clean.loc[splits["X_test"].index, TRANSACTION_ID]
        sample = self.build_sample(raw, test_ids)
        SAMPLE_TRANSACTIONS.parent.mkdir(parents=True, exist_ok=True)
        sample.to_csv(SAMPLE_TRANSACTIONS, index=False)

        logger.info("Stage: packaging model bundle")
        bundle = {
            "preprocessor": preprocessor,
            "feature_engineer": engineer,
            "selected_features": selector.selected_features_,
            "model": model,
            "threshold": report["threshold"],
            "metadata": {
                "trained_at": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "training_rows": int(len(splits["X_train"])),
                "training_fraud_rate": float(splits["y_train"].mean()),
                "n_estimators": self.n_estimators,
                "n_features": len(selector.selected_features_),
                "test_metrics": report["test_metrics"],
                "python": platform.python_version(),
                "packages": {
                    package: version(package) for package in TRACKED_PACKAGES
                },
            },
        }

        save_model(bundle, MODEL_BUNDLE)

        logger.info("Pipeline completed. Bundle saved to %s", MODEL_BUNDLE)

        return report


def main() -> None:
    """
    Command-line entry point.
    """

    parser = argparse.ArgumentParser(
        description="Train the fraud risk scoring model end to end."
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Use only the earliest N transactions (quick experiments).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=N_ESTIMATORS,
        help="Trees / iterations per base model.",
    )
    parser.add_argument(
        "--reuse-merged",
        action="store_true",
        help="Reuse data/processed/merged_train.parquet if it exists.",
    )
    args = parser.parse_args()

    report = TrainingPipeline(
        nrows=args.nrows,
        n_estimators=args.n_estimators,
        reuse_merged=args.reuse_merged,
    ).run()

    metrics = report["test_metrics"]

    print("\nTest-period performance")
    print("-" * 40)
    for key in [
        "threshold",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "roc_auc",
        "balanced_accuracy",
    ]:
        print(f"{key:<20}: {metrics[key]:.4f}")


if __name__ == "__main__":
    main()
