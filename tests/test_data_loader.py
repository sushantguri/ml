"""
Unit Tests for Data Ingestion and Schema Validation.
"""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_dataset, get_attack_taxonomy, get_dataset_summary, load_config


@pytest.fixture
def sample_csv_with_header(tmp_path):
    """Create a temporary CSV file with standard headers and sample records."""
    config = load_config()
    cols = config["schema"]["columns"]

    # 5 sample records
    data = []
    for i in range(5):
        row = [0, "tcp", "http", "SF", 200 + i * 10, 500, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               2, 2, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 150, 25, 0.17, 0.03, 0.17, 0.0, 0.0, 0.0, 0.05, 0.0,
               "normal" if i < 3 else "neptune", 20]
        data.append(row)

    df = pd.DataFrame(data, columns=cols)
    file_path = tmp_path / "test_with_header.csv"
    df.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def sample_txt_no_header(tmp_path):
    """Create a raw comma-separated file without header (standard NSL-KDD format)."""
    config = load_config()
    cols = config["schema"]["columns"]

    data = []
    for i in range(5):
        row = [0, "tcp", "ftp_data", "SF", 300, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               1, 1, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 100, 10, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0,
               "normal" if i < 2 else "satan", 18]
        data.append(row)

    df = pd.DataFrame(data, columns=cols)
    file_path = tmp_path / "test_no_header.txt"
    df.to_csv(file_path, header=False, index=False)
    return file_path


def test_load_dataset_with_header(sample_csv_with_header):
    """Test loading CSV file that includes header."""
    X, y, cat = load_dataset(sample_csv_with_header)
    assert len(X) == 5
    assert "label" not in X.columns
    assert "difficulty_level" not in X.columns
    assert y is not None
    assert len(y) == 5
    assert y.iloc[0] == 0  # normal
    assert y.iloc[3] == 1  # neptune attack
    assert cat.iloc[3] == "dos"


def test_load_dataset_without_header(sample_txt_no_header):
    """Test loading raw file without header."""
    X, y, cat = load_dataset(sample_txt_no_header)
    assert len(X) == 5
    assert "protocol_type" in X.columns
    assert "src_bytes" in X.columns
    assert y is not None
    assert cat.iloc[2] == "probe"  # satan is a probe attack


def test_load_dataset_nonexistent_file():
    """Test that loading a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_dataset("non_existent_file_path_12345.csv")


def test_missing_values_handled(tmp_path):
    """Test that missing values are imputed gracefully upon ingestion."""
    config = load_config()
    cols = config["schema"]["columns"]

    data = [
        [np.nan, "tcp", "http", "SF", np.nan, 500, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         2, 2, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 150, 25, 0.17, 0.03, 0.17, 0.0, 0.0, 0.0, 0.05, 0.0,
         "normal", 20],
        [0, "udp", "domain_u", "SF", 100, 200, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         1, 1, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 50, 50, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         "normal", 21],
    ]
    df = pd.DataFrame(data, columns=cols)
    file_path = tmp_path / "with_missing.csv"
    df.to_csv(file_path, index=False)

    X, y, _ = load_dataset(file_path)
    assert X.isnull().sum().sum() == 0


def test_attack_taxonomy_mapping():
    """Test attack categories correctly map known attack strings."""
    taxonomy = get_attack_taxonomy()
    assert taxonomy["normal"] == "normal"
    assert taxonomy["neptune"] == "dos"
    assert taxonomy["smurf"] == "dos"
    assert taxonomy["ipsweep"] == "probe"
    assert taxonomy["nmap"] == "probe"
    assert taxonomy["guess_passwd"] == "r2l"
    assert taxonomy["buffer_overflow"] == "u2r"


def test_dataset_summary(sample_csv_with_header):
    """Test dataset exploration summary generation."""
    X, y, cat = load_dataset(sample_csv_with_header)
    summary = get_dataset_summary(X, y, cat)
    assert summary["total_records"] == 5
    assert summary["has_labels"] is True
    assert summary["label_distribution"]["normal_count"] == 3
    assert summary["label_distribution"]["attack_count"] == 2
