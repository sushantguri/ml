"""
Data Preprocessing Pipeline Module.
Adaptive Cyber-Physical Security System.

Strictly adheres to:
- Scikit-Learn Pipeline and ColumnTransformer
- StandardScaler for numerical continuous features
- OneHotEncoder(handle_unknown='ignore') for categorical features
- Preprocessor fitted strictly on training data to prevent data leakage
- Feature reference baseline calculation for downstream anomaly explainability
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


class TrafficPreprocessor:
    """Encapsulates Scikit-Learn ColumnTransformer preprocessing for network traffic."""

    def __init__(
        self,
        categorical_features: list[str] | None = None,
        binary_features: list[str] | None = None,
        numerical_features: list[str] | None = None,
    ):
        self.categorical_features = categorical_features or ["protocol_type", "service", "flag"]
        self.binary_features = binary_features or [
            "land", "logged_in", "root_shell", "su_attempted", "is_host_login", "is_guest_login"
        ]
        self.numerical_features = numerical_features or []
        self.transformer: ColumnTransformer | None = None
        self.feature_names_out: list[str] = []
        self.reference_stats: dict[str, dict[str, float]] = {}

    def _determine_columns(self, X: pd.DataFrame) -> tuple[list[str], list[str]]:
        """Identify available categorical vs numerical columns from input DataFrame."""
        cat_cols = [c for c in self.categorical_features if c in X.columns]
        # Treat binary and continuous numerical columns together for standard scaling
        num_cols = [c for c in X.columns if c not in cat_cols]
        return cat_cols, num_cols

    def fit(self, X_train: pd.DataFrame) -> TrafficPreprocessor:
        """Fit ColumnTransformer on training features only.
        
        Args:
            X_train: Training DataFrame with traffic features.
            
        Returns:
            self
        """
        cat_cols, num_cols = self._determine_columns(X_train)
        self.categorical_features = cat_cols
        self.numerical_features = num_cols

        logger.info(
            "Fitting preprocessor: %d categorical features, %d numerical/binary features.",
            len(cat_cols),
            len(num_cols),
        )

        transformers = []
        if len(num_cols) > 0:
            transformers.append(
                ("num", StandardScaler(), num_cols)
            )
        if len(cat_cols) > 0:
            transformers.append(
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
            )

        self.transformer = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )

        self.transformer.fit(X_train)

        # Retrieve feature names out
        try:
            self.feature_names_out = list(self.transformer.get_feature_names_out())
        except (AttributeError, ValueError):
            # Fallback if names cannot be retrieved
            self.feature_names_out = [f"feat_{i}" for i in range(self.transformer.transform(X_train.iloc[:2]).shape[1])]

        # Compute baseline statistics for explainability (mean and std of numericals)
        self.reference_stats = {}
        for col in num_cols:
            self.reference_stats[col] = {
                "mean": float(X_train[col].mean()),
                "std": float(X_train[col].std()) if X_train[col].std() > 0 else 1.0,
                "median": float(X_train[col].median()),
            }

        logger.info("Preprocessor successfully fitted. Total transformed dimensions: %d", len(self.feature_names_out))
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform input features using fitted ColumnTransformer.
        
        Args:
            X: Input DataFrame.
            
        Returns:
            Dense 2D NumPy array of scaled and encoded features.
        """
        if self.transformer is None:
            raise ValueError("TrafficPreprocessor has not been fitted yet. Call fit() first.")

        # Ensure all required columns exist; fill missing columns if needed
        X_copy = X.copy()
        for col in self.categorical_features:
            if col not in X_copy.columns:
                X_copy[col] = "unknown"
            else:
                X_copy[col] = X_copy[col].astype(str)

        for col in self.numerical_features:
            if col not in X_copy.columns:
                # Fill with reference mean or 0.0
                default_val = self.reference_stats.get(col, {}).get("mean", 0.0)
                X_copy[col] = default_val
            else:
                X_copy[col] = pd.to_numeric(X_copy[col], errors="coerce").fillna(0.0)

        transformed = self.transformer.transform(X_copy)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        return np.asarray(transformed, dtype=np.float32)

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        """Fit on training data and transform it."""
        return self.fit(X_train).transform(X_train)

    def get_feature_names(self) -> list[str]:
        """Return names of output features after one-hot encoding."""
        return self.feature_names_out
