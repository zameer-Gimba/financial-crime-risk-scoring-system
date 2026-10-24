"""
Dataset download module for the Financial Crime Detection &
Risk Scoring System.

Downloads the IEEE-CIS Fraud Detection CSVs from Kaggle into data/raw.
Only needed for retraining - the Streamlit dashboard runs entirely from
the committed model bundle, reports and demo sample.

Requirements (one-time):
1. A Kaggle account that has joined the competition and accepted its
   rules: https://www.kaggle.com/c/ieee-fraud-detection/rules
2. A Kaggle API token, either:
   - environment variable  KAGGLE_API_TOKEN=<token>, or
   - file ~/.kaggle/access_token, or
   - legacy ~/.kaggle/kaggle.json / KAGGLE_USERNAME + KAGGLE_KEY
   Generate one at https://www.kaggle.com/settings/api

Usage:
    python -m src.download_data            # training files only
    python -m src.download_data --all      # training + test files

Author: Muhammad Ibrahim Gimba
"""

from __future__ import annotations

import argparse
import contextlib
import io
import zipfile
from pathlib import Path

from config.config import (
    RAW_DATA_DIR,
    TEST_IDENTITY,
    TEST_TRANSACTION,
    TRAIN_IDENTITY,
    TRAIN_TRANSACTION,
)
from config.logging_config import get_logger

logger = get_logger(__name__)

COMPETITION = "ieee-fraud-detection"

TRAIN_FILES = [TRAIN_TRANSACTION, TRAIN_IDENTITY]
TEST_FILES = [TEST_TRANSACTION, TEST_IDENTITY]

SETUP_HELP = (
    "Kaggle download failed. To enable automatic downloads:\n"
    "  1. pip install kaggle\n"
    "  2. Join the competition and accept its rules:\n"
    f"     https://www.kaggle.com/c/{COMPETITION}/rules\n"
    "  3. Create an API token at https://www.kaggle.com/settings/api and\n"
    "     export KAGGLE_API_TOKEN=<token>  (or save it to ~/.kaggle/access_token)\n"
    "Alternatively, download the CSVs manually from\n"
    f"  https://www.kaggle.com/c/{COMPETITION}/data\n"
    f"and extract them into {RAW_DATA_DIR}"
)


def missing_files(include_test: bool = False) -> list[Path]:
    """
    Raw CSVs that are not present locally.
    """

    files = TRAIN_FILES + (TEST_FILES if include_test else [])

    return [path for path in files if not path.exists()]


def _kaggle_api():
    """
    Return an authenticated Kaggle API client.
    """

    try:
        # The kaggle package prints an authentication notice on import
        # when no credentials exist; keep the console clean.
        with contextlib.redirect_stdout(io.StringIO()):
            from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as error:
        raise RuntimeError(SETUP_HELP) from error

    api = KaggleApi()

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            api.authenticate()
    except (Exception, SystemExit) as error:
        raise RuntimeError(SETUP_HELP) from error

    return api


def _extract(archive: Path, destination: Path) -> None:
    """
    Extract a downloaded zip archive and delete it.
    """

    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(destination)

    archive.unlink()


def download_dataset(include_test: bool = False) -> list[Path]:
    """
    Download any missing raw CSVs from Kaggle.

    Files that already exist are skipped, so this is safe to call before
    every training run.

    Returns
    -------
    list[Path]
        Files that were downloaded.
    """

    needed = missing_files(include_test)

    if not needed:
        logger.info("Raw dataset already present in %s", RAW_DATA_DIR)
        return []

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    api = _kaggle_api()

    for path in needed:

        logger.info("Downloading %s from Kaggle...", path.name)

        try:
            api.competition_download_file(
                COMPETITION,
                path.name,
                path=str(RAW_DATA_DIR),
                quiet=False,
            )
        except Exception as error:
            raise RuntimeError(
                f"Could not download {path.name}: {error}\n\n{SETUP_HELP}"
            ) from error

        archive = RAW_DATA_DIR / f"{path.name}.zip"

        if archive.exists():
            _extract(archive, RAW_DATA_DIR)

        if not path.exists():
            raise RuntimeError(
                f"{path.name} was not found after downloading.\n\n{SETUP_HELP}"
            )

        logger.info("Saved %s", path)

    return needed


def main() -> None:
    """
    Command-line entry point.
    """

    parser = argparse.ArgumentParser(
        description="Download the IEEE-CIS Fraud Detection data from Kaggle."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Also download the (unlabelled) Kaggle test files.",
    )
    args = parser.parse_args()

    try:
        downloaded = download_dataset(include_test=args.all)
    except RuntimeError as error:
        raise SystemExit(str(error))

    print(
        f"Downloaded {len(downloaded)} file(s) to {RAW_DATA_DIR}"
        if downloaded
        else f"All files already present in {RAW_DATA_DIR}"
    )


if __name__ == "__main__":
    main()
