from pathlib import Path

import pandas as pd

from src.utils import (
    dataframe_summary,
    file_exists,
    save_dataframe,
)


def test_dataframe_summary():
    """
    Test DataFrame summary generation.
    """

    df = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [4, None, 6],
        }
    )

    summary = dataframe_summary(df)

    assert summary["rows"] == 3
    assert summary["columns"] == 2
    assert summary["missing_values"] == 1


def test_file_exists():
    """
    Test file existence utility.
    """

    assert file_exists(Path(".")) is True


def test_save_dataframe(tmp_path):
    """
    Test parquet saving.
    """

    df = pd.DataFrame({"A": [1, 2, 3]})

    output = tmp_path / "sample.parquet"

    save_dataframe(df, output)

    assert output.exists()
