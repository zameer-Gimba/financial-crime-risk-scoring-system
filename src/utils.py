"""
Common utility functions for the Financial Crime Detection &
Risk Scoring System.

Author: Muhammad Ibrahim Gimba
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from config.logging_config import get_logger

logger = get_logger(__name__)


def load_csv(file_path: Path) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Parameters
    ----------
    file_path : Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """

    if not file_path.exists():
        logger.error("File not found: %s", file_path)
        raise FileNotFoundError(f"{file_path} does not exist.")

    logger.info("Loading CSV file: %s", file_path)

    return pd.read_csv(file_path)


def save_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save a DataFrame as a Parquet file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    output_path : Path
        Output file path.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_path, index=False)

    logger.info("Saved DataFrame to %s", output_path)


def load_parquet(file_path: Path) -> pd.DataFrame:
    """
    Load a Parquet file.

    Parameters
    ----------
    file_path : Path
        Path to the parquet file.

    Returns
    -------
    pd.DataFrame
    """

    if not file_path.exists():
        logger.error("Parquet file not found: %s", file_path)
        raise FileNotFoundError(f"{file_path} does not exist.")

    return pd.read_parquet(file_path)


def save_model(model: Any, output_path: Path) -> None:
    """
    Save a trained model using Joblib.

    Parameters
    ----------
    model : Any
        Trained model.
    output_path : Path
        Destination path.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_path)

    logger.info("Model saved to %s", output_path)


def load_model(model_path: Path) -> Any:
    """
    Load a trained Joblib model.

    Parameters
    ----------
    model_path : Path

    Returns
    -------
    Any
    """

    if not model_path.exists():
        logger.error("Model not found: %s", model_path)
        raise FileNotFoundError(f"{model_path} does not exist.")

    logger.info("Loading model from %s", model_path)

    return joblib.load(model_path)


def dataframe_summary(df: pd.DataFrame) -> dict:
    """
    Generate a basic DataFrame summary.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict
    """

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def file_exists(file_path: Path) -> bool:
    """
    Check if a file exists.

    Parameters
    ----------
    file_path : Path

    Returns
    -------
    bool
    """

    return file_path.exists()
