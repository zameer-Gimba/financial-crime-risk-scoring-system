"""
Tests for loading, preprocessing, feature engineering and selection.
"""

import numpy as np
import pandas as pd

from src.data_loader import normalize_columns
from src.feature_engineering import FeatureEngineer
from src.feature_selection import FeatureSelector
from src.preprocessing import DataPreprocessor
from src.utils import time_split


def test_normalize_identity_columns():
    df = pd.DataFrame(columns=["TransactionID", "id-01", "id-38", "DeviceType"])
    assert list(normalize_columns(df).columns) == [
        "TransactionID", "id_01", "id_38", "DeviceType"
    ]


def test_preprocessor_aligns_partial_input(raw_transactions):
    preprocessor = DataPreprocessor()
    preprocessor.preprocess(raw_transactions)

    partial = raw_transactions[["TransactionDT", "TransactionAmt"]].head(3)
    out = preprocessor.transform(partial)

    assert list(out.columns) == preprocessor.raw_columns_
    assert (out["P_emaildomain"] == "Unknown").all()
    assert out["card1"].isna().all()


def test_preprocessor_does_not_drop_rows_at_inference(raw_transactions):
    preprocessor = DataPreprocessor()
    preprocessor.preprocess(raw_transactions)

    bad = raw_transactions.head(2).assign(TransactionAmt=0)
    assert len(preprocessor.transform(bad)) == 2


def test_feature_engineering_uses_training_statistics(raw_transactions):
    preprocessor = DataPreprocessor()
    clean = preprocessor.preprocess(raw_transactions)

    engineer = FeatureEngineer().fit(clean)
    batch = engineer.transform(clean)

    # A single row must get exactly the same features as in a batch.
    single = engineer.transform(clean.iloc[[5]])
    pd.testing.assert_series_equal(
        single.iloc[0], batch.iloc[5], check_names=False
    )

    # Unseen values get frequency 0 and missing group statistics.
    unseen = clean.iloc[[0]].copy()
    unseen["card1"] = 999_999
    out = engineer.transform(unseen)
    assert out["card1_frequency"].iloc[0] == 0
    assert np.isnan(out["card1_amount_mean"].iloc[0])


def test_feature_keys_ignore_numeric_dtype(raw_transactions):
    preprocessor = DataPreprocessor()
    clean = preprocessor.preprocess(raw_transactions)
    engineer = FeatureEngineer().fit(clean)

    as_float32 = clean.iloc[[0]].copy()
    as_float32["card1"] = as_float32["card1"].astype("float32")

    a = engineer.transform(clean.iloc[[0]])["card1_frequency"].iloc[0]
    b = engineer.transform(as_float32)["card1_frequency"].iloc[0]
    assert a == b > 0


def test_duplicate_and_constant_detection():
    df = pd.DataFrame(
        {"a": [1, 2, 3], "b": [1, 2, 3], "c": [7, 7, 7], "d": [3, 1, 2]}
    )
    assert FeatureSelector.find_duplicate_features(df) == ["b"]
    assert FeatureSelector.find_constant_features(df) == ["c"]


def test_time_split_is_chronological(raw_transactions):
    shuffled = raw_transactions.sample(frac=1, random_state=1)
    early, late = time_split(shuffled, 0.25)
    assert len(late) == 300
    assert early["TransactionDT"].max() <= late["TransactionDT"].min()
