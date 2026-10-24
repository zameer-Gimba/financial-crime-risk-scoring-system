"""
Tests for the risk scoring module.
"""

import numpy as np
import pytest

from src.risk_scoring import (
    RECOMMENDATIONS,
    RiskScorer,
    probability_to_score,
    score_to_category,
)


def test_threshold_maps_to_fifty():
    assert probability_to_score(0.3, threshold=0.3)[()] == 50
    assert probability_to_score(0.0, threshold=0.3)[()] == 0
    assert probability_to_score(1.0, threshold=0.3)[()] == 100


def test_score_is_monotonic():
    probabilities = np.linspace(0, 1, 501)
    scores = probability_to_score(probabilities, threshold=0.2)
    assert np.all(np.diff(scores) >= 0)


def test_below_threshold_never_high():
    scores = probability_to_score(np.array([0.2999, 0.29]), threshold=0.3)
    assert np.all(scores <= 49)


def test_invalid_threshold():
    with pytest.raises(ValueError):
        probability_to_score(0.5, threshold=1.0)


@pytest.mark.parametrize(
    "score, category",
    [(0, "Low"), (24, "Low"), (25, "Medium"), (49, "Medium"),
     (50, "High"), (74, "High"), (75, "Critical"), (100, "Critical")],
)
def test_categories(score, category):
    assert score_to_category(score) == category


def test_scorer_table():
    table = RiskScorer(threshold=0.4).score(np.array([0.01, 0.4, 0.99]))
    assert list(table["risk_category"]) == ["Low", "High", "Critical"]
    assert list(table["flagged"]) == [False, True, True]
    assert table["recommendation"].tolist() == [
        RECOMMENDATIONS[c] for c in table["risk_category"]
    ]
