"""
Unit Tests for Isolation Forest Detector and Anomaly Scoring.
"""

import joblib
import numpy as np
import pytest

from src.detector import IsolationForestDetector


@pytest.fixture
def synthetic_features():
    """Create a synthetic dataset with normal cluster and distinct outliers."""
    np.random.seed(42)
    # 200 normal samples centered at 0
    normal = np.random.normal(loc=0.0, scale=1.0, size=(200, 10))
    # 10 anomalous outliers centered at 8
    anomalies = np.random.normal(loc=8.0, scale=1.0, size=(10, 10))
    return np.vstack([normal, anomalies])


def test_detector_fit_and_predict(synthetic_features):
    """Test fitting detector and generating binary predictions."""
    detector = IsolationForestDetector(n_estimators=50, contamination=0.05, random_state=42)
    detector.fit(synthetic_features)

    preds = detector.predict(synthetic_features)
    assert len(preds) == len(synthetic_features)
    # Verify predictions are strictly binary integers {0, 1}
    assert set(np.unique(preds)).issubset({0, 1})


def test_normalized_anomaly_scores(synthetic_features):
    """Test continuous anomaly score bounds [0.0, 1.0]."""
    detector = IsolationForestDetector(n_estimators=50, contamination=0.05, random_state=42)
    detector.fit(synthetic_features)

    scores = detector.predict_anomaly_scores(synthetic_features)
    assert len(scores) == len(synthetic_features)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)

    # Anomaly points (last 10) should have higher anomaly scores than normal points (first 10)
    mean_normal_score = np.mean(scores[:10])
    mean_anomaly_score = np.mean(scores[-10:])
    assert mean_anomaly_score > mean_normal_score


def test_model_serialization(synthetic_features, tmp_path):
    """Test saving and reloading detector via joblib."""
    detector = IsolationForestDetector(n_estimators=50, contamination=0.05, random_state=42)
    detector.fit(synthetic_features)

    scores_orig = detector.predict_anomaly_scores(synthetic_features[:10])
    save_path = tmp_path / "detector_test.joblib"
    joblib.dump(detector, save_path)

    loaded_detector = joblib.load(save_path)
    scores_loaded = loaded_detector.predict_anomaly_scores(synthetic_features[:10])

    np.testing.assert_allclose(scores_orig, scores_loaded, rtol=1e-5)


def test_custom_threshold_tuning(synthetic_features):
    """Test adjusting decision threshold modifies prediction sensitivity."""
    detector = IsolationForestDetector(n_estimators=50, contamination=0.10, random_state=42)
    detector.fit(synthetic_features)

    # Default threshold
    preds_default = detector.predict(synthetic_features)

    # Stricter threshold (higher threshold flags more anomalies)
    detector.set_threshold(0.05)
    preds_strict = detector.predict(synthetic_features)

    assert preds_strict.sum() >= preds_default.sum()
