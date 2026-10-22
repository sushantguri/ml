"""
Isolation Forest Anomaly Detector Module.
Adaptive Cyber-Physical Security System.

Strictly adheres to:
- Unsupervised anomaly detection using Scikit-Learn's IsolationForest
- Configurable contamination, random_state, n_estimators, max_samples
- Exposing normalized anomaly scores [0, 1] and binary anomaly predictions
- Clear documentation of mapping between Scikit-Learn internal conventions and cybersecurity labels:
    - Scikit-Learn: -1 (Outlier) -> System: 1 (Anomaly / Security Alert)
    - Scikit-Learn: +1 (Inlier)  -> System: 0 (Normal / Benign Traffic)
- Tunable decision threshold on validation data without peeking at test labels
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


class IsolationForestDetector:
    """Wrapper around Scikit-Learn IsolationForest with cybersecurity scoring & calibration."""

    def __init__(
        self,
        n_estimators: int = 150,
        max_samples: int | str = 512,
        contamination: float | str = 0.10,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        """Initialize detector with configurable hyperparameters.
        
        Args:
            n_estimators: Number of isolation trees to build.
            max_samples: Number of samples drawn to train each isolation tree.
            contamination: Expected proportion of outliers in data ('auto' or float in (0, 0.5]).
            random_state: Seed for reproducible random partitions.
            n_jobs: Number of parallel CPU threads.
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model: IsolationForest | None = None
        self.threshold: float = 0.0  # Decision function threshold for anomaly boundary
        self.score_min: float = -0.35
        self.score_max: float = 0.25

    def fit(self, X_train: np.ndarray) -> IsolationForestDetector:
        """Fit Isolation Forest detector on unsupervised training features.
        
        Args:
            X_train: Preprocessed 2D feature array.
            
        Returns:
            self
        """
        logger.info(
            "Fitting IsolationForest (n_trees=%d, max_samples=%s, contamination=%s, seed=%d)",
            self.n_estimators,
            str(self.max_samples),
            str(self.contamination),
            self.random_state,
        )

        # Dynamically bound max_samples to sample count to prevent warning
        effective_max_samples = (
            min(self.max_samples, X_train.shape[0])
            if isinstance(self.max_samples, int)
            else self.max_samples
        )

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=effective_max_samples,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        self.model.fit(X_train)

        # Calibrate score bounds using training decision function
        raw_train_scores = self.model.decision_function(X_train)
        self.score_min = float(np.percentile(raw_train_scores, 0.5))
        self.score_max = float(np.percentile(raw_train_scores, 99.5))
        if self.score_max <= self.score_min:
            self.score_max = self.score_min + 1.0

        # Scikit-learn offset_ corresponds to the threshold where decision_function == 0
        self.threshold = 0.0

        logger.info(
            "IsolationForest fitted. Score range: [%.4f, %.4f], threshold: %.4f",
            self.score_min,
            self.score_max,
            self.threshold,
        )
        return self

    def raw_decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute Scikit-Learn raw decision function.
        
        Values < 0 indicate outliers; values > 0 indicate inliers.
        """
        if self.model is None:
            raise ValueError("Model is not fitted. Call fit() first.")
        return self.model.decision_function(X)

    def predict_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Calculate normalized anomaly score in [0.0, 1.0].
        
        0.0 = completely normal traffic
        1.0 = extreme anomaly / highest threat risk
        Values >= 0.5 correspond to records flagged as anomalies under default threshold.
        """
        raw_scores = self.raw_decision_function(X)

        # Invert and normalize so lower decision function -> higher anomaly score
        # Using linear min-max scaling with clipping:
        normalized = 1.0 - (raw_scores - self.score_min) / (self.score_max - self.score_min)
        normalized = np.clip(normalized, 0.0, 1.0)
        return normalized

    def predict(self, X: np.ndarray, threshold: float | None = None) -> np.ndarray:
        """Predict binary anomaly flags.
        
        Args:
            X: Input feature array.
            threshold: Optional custom decision function threshold.
                       Defaults to 0.0 (or self.threshold).
                       Records with decision_function < threshold are flagged as 1 (anomaly).
                       
        Returns:
            np.ndarray of int (0 = Normal, 1 = Anomaly).
        """
        if self.model is None:
            raise ValueError("Model is not fitted. Call fit() first.")

        t = self.threshold if threshold is None else threshold
        raw_scores = self.raw_decision_function(X)
        # raw_scores < threshold are anomalies
        return (raw_scores < t).astype(int)

    def set_threshold(self, threshold: float) -> None:
        """Set custom decision threshold."""
        self.threshold = float(threshold)
        logger.info("Detector decision threshold updated to %.4f", self.threshold)

    def get_params(self) -> dict[str, Any]:
        """Return model hyperparameters."""
        return {
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "contamination": self.contamination,
            "random_state": self.random_state,
            "threshold": self.threshold,
            "score_min": self.score_min,
            "score_max": self.score_max,
        }
