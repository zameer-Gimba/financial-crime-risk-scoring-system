"""
Inference module for the Financial Crime Detection &
Risk Scoring System.

Loads the deployable model bundle and scores raw transactions (the
merged IEEE-CIS transaction + identity format) end to end:

raw columns -> preprocessing -> feature engineering -> selected features
-> stacked ensemble -> probability -> risk score / category /
recommendation -> SHAP explanation.

Author: Muhammad Ibrahim Gimba
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.config import MODEL_BUNDLE
from config.constants import TRANSACTION_ID
from config.logging_config import get_logger
from src.data_loader import normalize_columns
from src.explainability import EnsembleExplainer
from src.risk_scoring import RiskScorer
from src.utils import load_model

logger = get_logger(__name__)


class FraudRiskPredictor:
    """
    Score raw transactions with the trained model bundle.
    """

    def __init__(
        self,
        bundle: dict,
    ) -> None:
        self.preprocessor = bundle["preprocessor"]
        self.feature_engineer = bundle["feature_engineer"]
        self.selected_features = bundle["selected_features"]
        self.model = bundle["model"]
        self.threshold = bundle["threshold"]
        self.metadata = bundle.get("metadata", {})

        self.scorer = RiskScorer(self.threshold)
        self._explainer: EnsembleExplainer | None = None

    @classmethod
    def load(
        cls,
        path: Path = MODEL_BUNDLE,
    ) -> "FraudRiskPredictor":
        """
        Load a predictor from a saved bundle.
        """

        return cls(load_model(path))

    @property
    def explainer(self) -> EnsembleExplainer:
        """
        SHAP explainer, created on first use.
        """

        if self._explainer is None:
            self._explainer = EnsembleExplainer(self.model)

        return self._explainer

    def prepare_features(
        self,
        raw: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Turn raw transactions into the model's input features.
        """

        df = normalize_columns(raw)

        df = self.preprocessor.transform(df)

        df = self.feature_engineer.transform(df)

        return df[self.selected_features]

    def predict_proba(
        self,
        raw: pd.DataFrame,
    ):
        """
        Fraud probability for each transaction.
        """

        features = self.prepare_features(raw)

        return self.model.predict_proba(features)[:, 1]

    def score(
        self,
        raw: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Full risk table for a batch of raw transactions.
        """

        results = self.scorer.score(self.predict_proba(raw))

        if TRANSACTION_ID in raw.columns:
            results.insert(
                0,
                TRANSACTION_ID,
                raw[TRANSACTION_ID].to_numpy(),
            )

        return results

    def explain(
        self,
        raw_row: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Top SHAP drivers for a single raw transaction.
        """

        features = self.prepare_features(raw_row.iloc[[0]])

        return self.explainer.explain_transaction(features)
