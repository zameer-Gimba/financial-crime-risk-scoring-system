"""
Explainability module for the Financial Crime Detection &
Risk Scoring System.

Explains ensemble predictions with SHAP.

The stacked ensemble has no exact tree-SHAP form, so explanations are
built from its parts: exact TreeSHAP values are computed for each base
learner (LightGBM, XGBoost, CatBoost) and combined using the weights
the Logistic Regression meta-learner assigns to each of them. The result
is a per-feature contribution on the log-odds scale: positive values push
a transaction towards fraud, negative values towards legitimate.

Author: Muhammad Ibrahim Gimba
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from config.config import TOP_EXPLANATION_FEATURES
from config.logging_config import get_logger

logger = get_logger(__name__)

# SHAP announces a change in LightGBM output shape on every call; the
# shape is normalized in _positive_class() below.
warnings.filterwarnings(
    "ignore",
    message="LightGBM binary classifier with TreeExplainer",
)


def _positive_class(values) -> np.ndarray:
    """
    Normalize the different shapes SHAP returns for binary models.
    """

    if isinstance(values, list):
        values = values[1]

    values = np.asarray(values)

    if values.ndim == 3:
        values = values[:, :, 1]

    return values


class EnsembleExplainer:
    """
    SHAP explanations for the preprocessing + stacking pipeline.
    """

    def __init__(
        self,
        model: Pipeline,
    ) -> None:
        self.preprocessing = model.named_steps["preprocessing"]

        stacking = model.named_steps["stacking"]

        self.feature_names = list(
            self.preprocessing.get_feature_names_out()
        )

        coefficients = stacking.final_estimator_.coef_.ravel()

        self.weights = dict(
            zip(
                stacking.named_estimators_.keys(),
                coefficients / np.abs(coefficients).sum(),
            )
        )

        self.explainers = {}

        for name, estimator in stacking.named_estimators_.items():
            try:
                self.explainers[name] = shap.TreeExplainer(estimator)
            except Exception as error:  # pragma: no cover - library specific
                logger.warning(
                    "SHAP unavailable for %s: %s",
                    name,
                    error,
                )

        if not self.explainers:
            raise RuntimeError("No base model could be explained with SHAP.")

        logger.info(
            "EnsembleExplainer ready (weights: %s)",
            {k: round(float(v), 3) for k, v in self.weights.items()},
        )

    def shap_values(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Ensemble-weighted SHAP values, one column per model feature.
        """

        transformed = np.asarray(
            self.preprocessing.transform(X),
            dtype=float,
        )

        total_weight = sum(
            abs(self.weights[name]) for name in self.explainers
        )

        combined = np.zeros_like(transformed, dtype=float)

        for name, explainer in self.explainers.items():

            values = _positive_class(
                explainer.shap_values(transformed)
            )

            combined += values * self.weights[name] / total_weight

        return pd.DataFrame(
            combined,
            columns=self.feature_names,
            index=X.index,
        )

    def explain_transaction(
        self,
        X_row: pd.DataFrame,
        top_n: int = TOP_EXPLANATION_FEATURES,
    ) -> pd.DataFrame:
        """
        Top features driving the score of a single transaction.

        Returns
        -------
        pd.DataFrame
            feature, value, contribution, direction
        """

        contributions = self.shap_values(X_row.iloc[[0]]).iloc[0]

        values = X_row.iloc[0]

        table = pd.DataFrame(
            {
                "feature": contributions.index,
                "value": [values.get(name) for name in contributions.index],
                "contribution": contributions.to_numpy(),
            }
        )

        table["direction"] = np.where(
            table["contribution"] >= 0,
            "Increases risk",
            "Decreases risk",
        )

        return (
            table
            .reindex(table["contribution"].abs().sort_values(ascending=False).index)
            .head(top_n)
            .reset_index(drop=True)
        )

    def global_importance(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Mean absolute SHAP value per feature over a sample.
        """

        values = self.shap_values(X)

        return (
            pd.DataFrame(
                {
                    "feature": values.columns,
                    "mean_abs_shap": values.abs().mean().to_numpy(),
                    "mean_shap": values.mean().to_numpy(),
                }
            )
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
