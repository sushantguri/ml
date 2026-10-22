"""
Inference and Real-Time Scoring Pipeline.
Adaptive Cyber-Physical Security System.

Provides:
- Model loading from serialized artifacts
- Input schema validation and graceful fallback for missing/extra columns
- Normalized anomaly scoring and classification
- SOC review priority assignment (CRITICAL, HIGH, MEDIUM, LOW)
- Rule-free explainability based on empirical feature deviations from training baseline
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.detector import IsolationForestDetector
from src.feature_engineering import PCAManager
from src.preprocessing import TrafficPreprocessor

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class InferencePipeline:
    """End-to-end inference engine for network traffic anomaly detection."""

    def __init__(self, models_dir: str | Path | None = None):
        self.models_dir = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
        self.preprocessor: TrafficPreprocessor | None = None
        self.detector: IsolationForestDetector | None = None
        self.pca_manager: PCAManager | None = None
        self.metadata: dict[str, Any] = {}
        self.is_loaded: bool = False

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load serialized pipeline components from models directory."""
        preprocessor_path = self.models_dir / "preprocessor.joblib"
        detector_path = self.models_dir / "isolation_forest.joblib"
        pca_path = self.models_dir / "pca_transformer.joblib"
        metadata_path = self.models_dir / "model_metadata.json"

        if not preprocessor_path.exists() or not detector_path.exists():
            logger.warning("Model artifacts not found in %s. Inference unavailable until trained.", self.models_dir)
            self.is_loaded = False
            return

        try:
            self.preprocessor = joblib.load(preprocessor_path)
            self.detector = joblib.load(detector_path)
            if pca_path.exists():
                self.pca_manager = joblib.load(pca_path)
            if metadata_path.exists():
                with open(metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)

            self.is_loaded = True
            logger.info("Successfully loaded model artifacts from %s", self.models_dir)
        except (OSError, EOFError, json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error("Error loading model artifacts: %s", e)
            self.is_loaded = False

    def validate_input_schema(self, df: pd.DataFrame) -> tuple[bool, list[str], list[str]]:
        """Validate input DataFrame against required schema.
        
        Returns:
            Tuple of (is_valid, missing_columns, extra_columns)
        """
        if self.preprocessor is None:
            return False, ["Preprocessor not loaded"], []

        expected_features = set(self.preprocessor.categorical_features + self.preprocessor.numerical_features)
        input_columns = set(df.columns)

        missing = list(expected_features - input_columns)
        extra = list(input_columns - expected_features)

        # File is considered valid if at least 50% of expected features match
        is_valid = len(missing) < (len(expected_features) * 0.5)
        return is_valid, missing, extra

    def _compute_anomaly_reasons(self, record_row: pd.Series) -> str:
        """Identify features that deviate the most from baseline normal distribution."""
        if self.preprocessor is None or not self.preprocessor.reference_stats:
            return "Multivariate isolation anomaly"

        deviations = []
        for col, stats in self.preprocessor.reference_stats.items():
            if col in record_row:
                try:
                    val = float(record_row[col])
                    mean = stats["mean"]
                    std = stats["std"]
                    z_score = abs(val - mean) / std if std > 0 else 0.0
                    if z_score >= 3.0:  # 3 standard deviations
                        direction = "+" if val > mean else "-"
                        deviations.append((col, z_score, f"{col} ({direction}{z_score:.1f}σ, val={val:.1f})"))
                except (ValueError, TypeError):
                    continue

        if not deviations:
            return "Unusual combination of normal individual feature values"

        # Sort by largest deviation z-score
        deviations.sort(key=lambda x: x[1], reverse=True)
        top_reasons = [d[2] for d in deviations[:3]]
        return "; ".join(top_reasons)

    def assign_review_priority(self, score: float, is_anomaly: int) -> str:
        """Assign SOC review priority based on calibrated anomaly score."""
        if not is_anomaly:
            return "LOW"
        if score >= 0.85:
            return "CRITICAL"
        elif score >= 0.70:
            return "HIGH"
        else:
            return "MEDIUM"

    def predict_records(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run anomaly detection inference on an input DataFrame of network records.
        
        Args:
            df: DataFrame containing network traffic features.
            
        Returns:
            Enriched DataFrame with prediction columns:
              - anomaly_prediction ('Normal' or 'Anomaly')
              - is_anomaly (0 or 1)
              - anomaly_score (0.0 to 1.0)
              - review_priority ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
              - anomaly_reasons (top deviating features)
        """
        if not self.is_loaded or self.preprocessor is None or self.detector is None:
            raise RuntimeError("Inference pipeline is not ready. Train or provide model artifacts first.")

        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        # Ignore label or difficulty if present
        features_df = df.copy()
        for drop_col in ["label", "difficulty_level", "is_anomaly", "anomaly_score", "anomaly_prediction"]:
            if drop_col in features_df.columns:
                features_df = features_df.drop(columns=[drop_col])

        # Preprocess features
        X_processed = self.preprocessor.transform(features_df)

        # Generate scores and predictions
        is_anomaly = self.detector.predict(X_processed)
        anomaly_scores = self.detector.predict_anomaly_scores(X_processed)

        # Build output DataFrame
        result_df = df.copy()
        result_df["is_anomaly"] = is_anomaly
        result_df["anomaly_score"] = np.round(anomaly_scores, 4)
        result_df["anomaly_prediction"] = np.where(is_anomaly == 1, "Anomaly", "Normal")

        # Assign priority and reasons
        priorities = [
            self.assign_review_priority(score, flag)
            for score, flag in zip(anomaly_scores, is_anomaly)
        ]
        result_df["review_priority"] = priorities

        # Explain deviations for flagged anomalies
        reasons = []
        for idx in range(len(result_df)):
            if is_anomaly[idx] == 1:
                reasons.append(self._compute_anomaly_reasons(df.iloc[idx]))
            else:
                reasons.append("Normal traffic baseline pattern")
        result_df["anomaly_reasons"] = reasons

        # If PCA is available, attach 2D coordinates for instant plotting
        if self.pca_manager is not None:
            pca_coords = self.pca_manager.transform_2d(X_processed)
            result_df["pca_component_1"] = np.round(pca_coords[:, 0], 4)
            result_df["pca_component_2"] = np.round(pca_coords[:, 1], 4)

        return result_df

    def predict_csv(self, input_csv_path: str | Path, output_csv_path: str | Path | None = None) -> pd.DataFrame:
        """Run batch inference on a CSV file and optionally save enriched results."""
        input_path = Path(input_csv_path)
        if not input_path.exists():
            raise FileNotFoundError(f"CSV file not found: {input_path}")

        df = pd.read_csv(input_path)
        results = self.predict_records(df)

        if output_csv_path:
            out_p = Path(output_csv_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            results.to_csv(out_p, index=False)
            logger.info("Saved batch predictions to %s", out_p)

        return results
