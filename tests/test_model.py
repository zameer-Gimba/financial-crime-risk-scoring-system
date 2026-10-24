"""
End-to-end test: train a tiny ensemble, evaluate, score and explain.
"""

import numpy as np
import pandas as pd

from src.evaluation import compute_metrics, optimize_threshold
from src.feature_engineering import FeatureEngineer
from src.feature_selection import FeatureSelector
from src.model_training import ModelTrainer
from src.predict import FraudRiskPredictor
from src.preprocessing import DataPreprocessor
from src.report_generator import build_transaction_report


def test_optimize_threshold_and_metrics():
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.35, 0.8, 0.9])
    threshold = optimize_threshold(y, p)
    assert 0.35 < threshold <= 0.8

    metrics = compute_metrics(y, p, threshold)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["confusion_matrix"] == {"tn": 4, "fp": 0, "fn": 0, "tp": 2}


def test_end_to_end(raw_transactions):
    preprocessor = DataPreprocessor()
    clean = preprocessor.preprocess(raw_transactions)

    engineer = FeatureEngineer().fit(clean)
    engineered = engineer.transform(clean)

    selector = FeatureSelector(top_features=15).fit(engineered)
    assert "TransactionID" not in selector.selected_features_
    assert "TransactionDT" not in selector.selected_features_

    selected = selector.transform(engineered)

    trainer = ModelTrainer()
    splits = trainer.prepare_data(
        selected.drop(columns=["isFraud"]), selected["isFraud"]
    )
    assert "TransactionDT" not in splits["X_train"].columns

    model = trainer.build_models(
        splits["X_train"], splits["scale_pos_weight"],
        n_estimators=15, n_folds=2,
    )
    model = trainer.train_model(model, splits["X_train"], splits["y_train"])

    predictor = FraudRiskPredictor(
        {
            "preprocessor": preprocessor,
            "feature_engineer": engineer,
            "selected_features": selector.selected_features_,
            "model": model,
            "threshold": 0.5,
        }
    )

    results = predictor.score(raw_transactions.head(20))
    assert len(results) == 20
    assert results["fraud_probability"].between(0, 1).all()
    assert results["risk_score"].between(0, 100).all()
    assert "TransactionID" in results.columns

    explanation = predictor.explain(raw_transactions.head(1))
    assert len(explanation) > 0
    assert set(explanation.columns) == {
        "feature", "value", "contribution", "direction"
    }

    pdf = build_transaction_report(
        raw_transactions.iloc[0],
        results.iloc[0].to_dict(),
        explanation,
        {"threshold": 0.5, "test_metrics": {}},
    )
    assert pdf.startswith(b"%PDF")

    # Uploads with only a few columns must still score.
    minimal = pd.DataFrame(
        {"TransactionDT": [100_000], "TransactionAmt": [49.99]}
    )
    assert len(predictor.score(minimal)) == 1
