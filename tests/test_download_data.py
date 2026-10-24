"""
Tests for the Kaggle download module (no network access needed).
"""

import zipfile

import pytest

from src import download_data


class FakeKaggleApi:
    """Writes each requested file as a zip, like the Kaggle API does."""

    def __init__(self):
        self.requested = []

    def competition_download_file(self, competition, file_name, path, quiet):
        self.requested.append(file_name)
        with zipfile.ZipFile(f"{path}/{file_name}.zip", "w") as archive:
            archive.writestr(file_name, "TransactionID\n1\n")


@pytest.fixture
def raw_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(download_data, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        download_data,
        "TRAIN_FILES",
        [tmp_path / "train_transaction.csv", tmp_path / "train_identity.csv"],
    )
    monkeypatch.setattr(
        download_data,
        "TEST_FILES",
        [tmp_path / "test_transaction.csv", tmp_path / "test_identity.csv"],
    )
    return tmp_path


def test_downloads_and_extracts_missing_files(raw_dir, monkeypatch):
    api = FakeKaggleApi()
    monkeypatch.setattr(download_data, "_kaggle_api", lambda: api)

    downloaded = download_data.download_dataset()

    assert api.requested == ["train_transaction.csv", "train_identity.csv"]
    assert [p.name for p in downloaded] == api.requested
    assert (raw_dir / "train_transaction.csv").exists()
    assert not list(raw_dir.glob("*.zip"))


def test_skips_existing_files(raw_dir, monkeypatch):
    (raw_dir / "train_transaction.csv").write_text("x")
    (raw_dir / "train_identity.csv").write_text("x")

    def fail():
        raise AssertionError("Kaggle should not be contacted")

    monkeypatch.setattr(download_data, "_kaggle_api", fail)

    assert download_data.download_dataset() == []


def test_missing_credentials_give_setup_help(raw_dir, monkeypatch):
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(raw_dir / "no-config"))
    monkeypatch.setenv("HOME", str(raw_dir))

    with pytest.raises(RuntimeError, match="KAGGLE_API_TOKEN"):
        download_data.download_dataset()
