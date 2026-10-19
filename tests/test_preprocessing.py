"""
Unit Tests for Data Preprocessing and Feature Pipelines.
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import TrafficPreprocessor


@pytest.fixture
def dummy_train_df():
    """Create a dummy training DataFrame with categorical and numerical features."""
    return pd.DataFrame({
        "protocol_type": ["tcp", "udp", "icmp", "tcp"],
        "service": ["http", "dns", "eco_i", "smtp"],
        "flag": ["SF", "SF", "SF", "S0"],
        "duration": [0, 10, 0, 5],
        "src_bytes": [200, 50, 0, 1000],
        "dst_bytes": [400, 100, 0, 0],
        "logged_in": [1, 0, 0, 1],
    })


def test_preprocessor_fit_transform(dummy_train_df):
    """Test fitting and transforming preprocessor on training data."""
    preprocessor = TrafficPreprocessor(
        categorical_features=["protocol_type", "service", "flag"],
    )
    X_proc = preprocessor.fit_transform(dummy_train_df)

    assert isinstance(X_proc, np.ndarray)
    assert X_proc.shape[0] == 4
    # Encoded dimensions must be greater than original raw column count
    assert X_proc.shape[1] > dummy_train_df.shape[1]
    assert len(preprocessor.get_feature_names()) == X_proc.shape[1]


def test_unseen_categorical_handling(dummy_train_df):
    """Test that unseen categories in inference data are ignored gracefully without crashing."""
    preprocessor = TrafficPreprocessor(
        categorical_features=["protocol_type", "service", "flag"],
    )
    preprocessor.fit(dummy_train_df)

    # Test DataFrame with novel/unseen protocol and service
    unseen_df = pd.DataFrame({
        "protocol_type": ["unseen_proto", "tcp"],
        "service": ["novel_service", "http"],
        "flag": ["SF", "SF"],
        "duration": [0, 1],
        "src_bytes": [100, 200],
        "dst_bytes": [200, 300],
        "logged_in": [0, 1],
    })

    # Should transform without raising an error
    X_proc = preprocessor.transform(unseen_df)
    assert X_proc.shape[0] == 2
    assert X_proc.shape[1] == len(preprocessor.get_feature_names())


def test_missing_features_imputation(dummy_train_df):
    """Test that missing columns during inference are imputed with reference stats."""
    preprocessor = TrafficPreprocessor(
        categorical_features=["protocol_type", "service", "flag"],
    )
    preprocessor.fit(dummy_train_df)

    # Subset missing 'src_bytes'
    incomplete_df = pd.DataFrame({
        "protocol_type": ["tcp"],
        "service": ["http"],
        "flag": ["SF"],
        "duration": [0],
        "dst_bytes": [500],
        "logged_in": [1],
    })

    X_proc = preprocessor.transform(incomplete_df)
    assert X_proc.shape[0] == 1
    assert not np.isnan(X_proc).any()


def test_transform_before_fit_raises_error():
    """Test that calling transform before fit raises ValueError."""
    preprocessor = TrafficPreprocessor()
    df = pd.DataFrame({"protocol_type": ["tcp"], "duration": [0]})
    with pytest.raises(ValueError):
        preprocessor.transform(df)
