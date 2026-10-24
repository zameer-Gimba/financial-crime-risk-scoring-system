"""
Risk scoring module for the Financial Crime Detection &
Risk Scoring System.

Turns fraud probabilities into a 0-100 risk score, a risk category and
an investigation recommendation.

Author: Muhammad Ibrahim Gimba
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.config import (
    HIGH_RISK_MAX,
    LOW_RISK_MAX,
    MEDIUM_RISK_MAX,
)
from config.constants import (
    APPROVE,
    CRITICAL,
    CRITICAL_COLOR,
    HIGH,
    HIGH_COLOR,
    IMMEDIATE_INVESTIGATION,
    LOW,
    LOW_COLOR,
    MANUAL_REVIEW,
    MEDIUM,
    MEDIUM_COLOR,
    MONITOR,
)

RISK_CATEGORIES = [LOW, MEDIUM, HIGH, CRITICAL]

RECOMMENDATIONS = {
    LOW: APPROVE,
    MEDIUM: MONITOR,
    HIGH: MANUAL_REVIEW,
    CRITICAL: IMMEDIATE_INVESTIGATION,
}

CATEGORY_COLORS = {
    LOW: LOW_COLOR,
    MEDIUM: MEDIUM_COLOR,
    HIGH: HIGH_COLOR,
    CRITICAL: CRITICAL_COLOR,
}

RECOMMENDATION_DETAILS = {
    APPROVE: (
        "Transaction is consistent with legitimate behaviour. "
        "Process normally."
    ),
    MONITOR: (
        "Some unusual characteristics. Approve, but keep the account "
        "under enhanced monitoring for related activity."
    ),
    MANUAL_REVIEW: (
        "Above the fraud decision threshold. Hold for analyst review "
        "and verify the customer before releasing funds."
    ),
    IMMEDIATE_INVESTIGATION: (
        "Strong fraud indicators. Block or hold the transaction, "
        "contact the cardholder and open an investigation case."
    ),
}


def probability_to_score(
    probability: np.ndarray | float,
    threshold: float,
) -> np.ndarray:
    """
    Map fraud probabilities to a 0-100 risk score.

    The mapping is piecewise linear and anchored on the model's tuned
    decision threshold, which always maps to a score of 50 (the start
    of the High band):

    - probability 0          -> 0
    - probability = threshold -> 50
    - probability 1          -> 100

    Anchoring on the threshold (rather than using probability * 100)
    keeps the score meaningful for a heavily imbalanced problem where
    the optimal threshold is far from 0.5: every High/Critical
    transaction is one the model would flag.
    """

    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1.")

    p = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)

    below = 50 * p / threshold
    above = 50 + 50 * (p - threshold) / (1 - threshold)

    score = np.where(p < threshold, below, above)

    # Never round a below-threshold probability up into the High band.
    score = np.where(p < threshold, np.minimum(score, 49.0), score)

    return np.clip(np.floor(score + 0.5), 0, 100).astype(int)


def score_to_category(
    score: int | float,
) -> str:
    """
    Map a 0-100 risk score to a risk category.
    """

    if score <= LOW_RISK_MAX:
        return LOW
    if score <= MEDIUM_RISK_MAX:
        return MEDIUM
    if score <= HIGH_RISK_MAX:
        return HIGH
    return CRITICAL


class RiskScorer:
    """
    Convert model probabilities into risk intelligence.
    """

    def __init__(
        self,
        threshold: float,
    ) -> None:
        self.threshold = threshold

    def score(
        self,
        probabilities: np.ndarray,
    ) -> pd.DataFrame:
        """
        Build the risk table for a batch of probabilities.

        Returns
        -------
        pd.DataFrame
            fraud_probability, risk_score, risk_category,
            recommendation, flagged
        """

        probabilities = np.asarray(probabilities, dtype=float)

        scores = probability_to_score(probabilities, self.threshold)

        categories = [score_to_category(score) for score in scores]

        return pd.DataFrame(
            {
                "fraud_probability": probabilities,
                "risk_score": scores,
                "risk_category": categories,
                "recommendation": [
                    RECOMMENDATIONS[category]
                    for category in categories
                ],
                "flagged": probabilities >= self.threshold,
            }
        )
