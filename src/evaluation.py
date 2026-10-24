"""
Model evaluation module for the Financial Crime Detection &
Risk Scoring System.

Responsible for decision-threshold optimization and for measuring the
trained ensemble on the chronological holdout.

Author: Muhammad Ibrahim Gimba
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from config.config import (
    METRICS_REPORT,
    PR_CURVE_REPORT,
    ROC_CURVE_REPORT,
)
from config.logging_config import get_logger
from src.risk_scoring import RISK_CATEGORIES, RiskScorer

logger = get_logger(__name__)

MAX_CURVE_POINTS = 400


def optimize_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    beta: float = 1.0,
) -> float:
    """
    Find the probability threshold that maximizes the F-beta score.

    beta = 1 balances precision and recall; beta > 1 favours recall
    (catching more fraud at the cost of more false alarms).
    """

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities,
    )

    precision = precision[:-1]
    recall = recall[:-1]

    beta_sq = beta ** 2

    with np.errstate(divide="ignore", invalid="ignore"):
        f_beta = (
            (1 + beta_sq) * precision * recall
            / (beta_sq * precision + recall)
        )

    f_beta = np.nan_to_num(f_beta)

    best = int(np.argmax(f_beta))

    threshold = float(thresholds[best])

    logger.info(
        "Optimal threshold %.4f (F%.1f = %.4f)",
        threshold,
        beta,
        f_beta[best],
    )

    return threshold


def compute_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    """
    Compute classification metrics at a given threshold.
    """

    y_true = np.asarray(y_true).astype(int)

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "samples": int(len(y_true)),
        "fraud_rate": float(y_true.mean()),
    }


def _downsample(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Keep at most MAX_CURVE_POINTS evenly spaced rows of a curve.
    """

    if len(frame) <= MAX_CURVE_POINTS:
        return frame

    index = np.linspace(0, len(frame) - 1, MAX_CURVE_POINTS).astype(int)

    return frame.iloc[index].reset_index(drop=True)


def risk_band_summary(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> list[dict]:
    """
    Transactions and observed fraud rate per risk category.

    Shows whether the categories are meaningful: the fraud rate should
    rise sharply from Low to Critical.
    """

    table = RiskScorer(threshold).score(probabilities)

    table["is_fraud"] = np.asarray(y_true).astype(int)

    summary = []

    for category in RISK_CATEGORIES:

        band = table[table["risk_category"] == category]

        summary.append(
            {
                "category": category,
                "transactions": int(len(band)),
                "share": float(len(band) / len(table)),
                "fraud_cases": int(band["is_fraud"].sum()),
                "fraud_rate": (
                    float(band["is_fraud"].mean()) if len(band) else 0.0
                ),
                "fraud_captured": float(
                    band["is_fraud"].sum() / max(table["is_fraud"].sum(), 1)
                ),
            }
        )

    return summary


class ModelEvaluator:
    """
    Tune the decision threshold and evaluate the ensemble.
    """

    def __init__(
        self,
        beta: float = 1.0,
    ) -> None:
        self.beta = beta

        logger.info("ModelEvaluator initialized.")

    @staticmethod
    def base_model_scores(
        model: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> dict:
        """
        PR-AUC and ROC-AUC of each base learner vs. the ensemble.
        """

        transformed = model.named_steps["preprocessing"].transform(X)

        stacking = model.named_steps["stacking"]

        scores = {}

        for name, estimator in stacking.named_estimators_.items():

            probabilities = estimator.predict_proba(transformed)[:, 1]

            scores[name] = {
                "pr_auc": float(average_precision_score(y, probabilities)),
                "roc_auc": float(roc_auc_score(y, probabilities)),
            }

        probabilities = model.predict_proba(X)[:, 1]

        scores["stacking_ensemble"] = {
            "pr_auc": float(average_precision_score(y, probabilities)),
            "roc_auc": float(roc_auc_score(y, probabilities)),
        }

        return scores

    def run(
        self,
        model: Pipeline,
        splits: dict,
    ) -> dict:
        """
        Tune the threshold on the threshold split and report final
        metrics on the (later, unseen) test split.
        """

        logger.info("Optimizing decision threshold...")

        threshold_probabilities = model.predict_proba(
            splits["X_threshold"]
        )[:, 1]

        threshold = optimize_threshold(
            splits["y_threshold"],
            threshold_probabilities,
            beta=self.beta,
        )

        logger.info("Evaluating on the test split...")

        y_test = splits["y_test"].to_numpy()

        test_probabilities = model.predict_proba(
            splits["X_test"]
        )[:, 1]

        metrics = compute_metrics(
            y_test,
            test_probabilities,
            threshold,
        )

        precision, recall, _ = precision_recall_curve(
            y_test,
            test_probabilities,
        )

        fpr, tpr, _ = roc_curve(
            y_test,
            test_probabilities,
        )

        report = {
            "threshold": threshold,
            "threshold_objective": f"F{self.beta:g}",
            "test_metrics": metrics,
            "threshold_split_metrics": compute_metrics(
                splits["y_threshold"].to_numpy(),
                threshold_probabilities,
                threshold,
            ),
            "model_comparison": self.base_model_scores(
                model,
                splits["X_test"],
                splits["y_test"],
            ),
            "risk_bands": risk_band_summary(
                y_test,
                test_probabilities,
                threshold,
            ),
            "split_sizes": {
                "train": int(len(splits["X_train"])),
                "threshold": int(len(splits["X_threshold"])),
                "test": int(len(splits["X_test"])),
            },
        }

        self.save(
            report,
            pd.DataFrame({"precision": precision, "recall": recall}),
            pd.DataFrame({"fpr": fpr, "tpr": tpr}),
        )

        logger.info(
            "Test metrics: precision=%.4f recall=%.4f f1=%.4f "
            "pr_auc=%.4f roc_auc=%.4f",
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["pr_auc"],
            metrics["roc_auc"],
        )

        return report

    @staticmethod
    def save(
        report: dict,
        pr_curve: pd.DataFrame,
        roc: pd.DataFrame,
    ) -> None:
        """
        Write the metrics report and curves to the reports directory.
        """

        METRICS_REPORT.parent.mkdir(parents=True, exist_ok=True)

        METRICS_REPORT.write_text(json.dumps(report, indent=2))

        _downsample(pr_curve).to_csv(PR_CURVE_REPORT, index=False)

        _downsample(roc).to_csv(ROC_CURVE_REPORT, index=False)

        logger.info("Evaluation reports saved to %s", METRICS_REPORT.parent)
