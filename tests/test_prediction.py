"""
Unit Tests for End-to-End Inference and Prediction Pipeline.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.predict import InferencePipeline

ROOT_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture
def inference_pipeline():
    """Load the trained inference pipeline."""
    return InferencePipeline()


def test_inference_pipeline_loaded(inference_pipeline):
    """Test that pipeline successfully loads trained artifacts."""
    assert inference_pipeline.is_loaded is True
    assert inference_pipeline.preprocessor is not None
    assert inference_pipeline.detector is not None


def test_predict_sample_csv(inference_pipeline):
    """Test inference on a valid sample CSV file."""
    sample_file = ROOT_DIR / "data" / "samples" / "sample_mixed_traffic.csv"
    if not sample_file.exists():
        pytest.skip("Sample CSV not generated yet.")

    results = inference_pipeline.predict_csv(sample_file)
    assert len(results) > 0

    expected_cols = [
        "anomaly_prediction",
        "is_anomaly",
        "anomaly_score",
        "review_priority",
        "anomaly_reasons",
    ]
    for col in expected_cols:
        assert col in results.columns

    # Verify priority categories
    valid_priorities = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
    assert set(results["review_priority"].unique()).issubset(valid_priorities)


def test_predict_with_extra_unexpected_columns(inference_pipeline):
    """Test inference gracefully ignores unexpected extra columns."""
    sample_file = ROOT_DIR / "data" / "samples" / "sample_normal_traffic.csv"
    if not sample_file.exists():
        pytest.skip("Sample CSV not generated yet.")

    df = pd.read_csv(sample_file).head(5)
    # Add unexpected dummy columns
    df["unrelated_col_1"] = "testing"
    df["random_metric"] = 999.9

    results = inference_pipeline.predict_records(df)
    assert len(results) == 5
    assert "anomaly_prediction" in results.columns


def test_predict_with_missing_columns(inference_pipeline):
    """Test inference handles missing feature columns via default imputation."""
    sample_file = ROOT_DIR / "data" / "samples" / "sample_normal_traffic.csv"
    if not sample_file.exists():
        pytest.skip("Sample CSV not generated yet.")

    df = pd.read_csv(sample_file).head(5)
    # Drop several columns
    df = df.drop(columns=["duration", "src_bytes", "dst_bytes"], errors="ignore")

    results = inference_pipeline.predict_records(df)
    assert len(results) == 5
    assert not results["anomaly_score"].isnull().any()


def test_predict_empty_dataframe_raises_error(inference_pipeline):
    """Test that predicting on an empty DataFrame raises ValueError."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError):
        inference_pipeline.predict_records(empty_df)


def test_missing_model_directory_handled():
    """Test pipeline handles non-existent models directory gracefully."""
    empty_pipeline = InferencePipeline(models_dir="/tmp/non_existent_models_dir_abc123")
    assert empty_pipeline.is_loaded is False

    dummy_df = pd.DataFrame({"duration": [0]})
    with pytest.raises(RuntimeError):
        empty_pipeline.predict_records(dummy_df)
