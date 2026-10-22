"""
Dimensionality Reduction and Feature Engineering Module.
Adaptive Cyber-Physical Security System.

Strictly adheres to:
- Principal Component Analysis (PCA) fitted strictly on training data
- Computation of cumulative explained variance
- Identification of number of components needed to explain 90% of variance
- 2D PCA projection for visualization
- Ground-truth labels used exclusively for post-hoc evaluation coloring, never for fitting PCA
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class PCAManager:
    """Manages PCA decomposition, variance diagnostics, and 2D projections."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.pca_full: PCA | None = None
        self.pca_2d: PCA | None = None
        self.explained_variance_ratio: np.ndarray = np.array([])
        self.cumulative_variance: np.ndarray = np.array([])
        self.n_components_90: int = 2
        self.total_variance_explained_2d: float = 0.0

    def fit(self, X_train_processed: np.ndarray, variance_threshold: float = 0.90) -> PCAManager:
        """Fit PCA on preprocessed training feature matrix.
        
        Args:
            X_train_processed: 2D array of scaled/encoded training features.
            variance_threshold: Target cumulative explained variance ratio (e.g., 0.90).
            
        Returns:
            self
        """
        logger.info("Fitting PCA on training features with shape: %s", X_train_processed.shape)
        max_components = min(X_train_processed.shape[0], X_train_processed.shape[1])

        # Fit full PCA to compute complete spectrum
        self.pca_full = PCA(n_components=min(max_components, 100), random_state=self.random_state)
        self.pca_full.fit(X_train_processed)

        self.explained_variance_ratio = self.pca_full.explained_variance_ratio_
        self.cumulative_variance = np.cumsum(self.explained_variance_ratio)

        # Find components for variance_threshold
        indices_above_threshold = np.where(self.cumulative_variance >= variance_threshold)[0]
        if len(indices_above_threshold) > 0:
            self.n_components_90 = int(indices_above_threshold[0] + 1)
        else:
            self.n_components_90 = len(self.cumulative_variance)

        # Fit dedicated 2D PCA for fast visual projections
        self.pca_2d = PCA(n_components=2, random_state=self.random_state)
        self.pca_2d.fit(X_train_processed)
        self.total_variance_explained_2d = float(np.sum(self.pca_2d.explained_variance_ratio_))

        logger.info(
            "PCA fitted. Components for %.0f%% variance: %d. 2D explained variance: %.2f%%",
            variance_threshold * 100,
            self.n_components_90,
            self.total_variance_explained_2d * 100,
        )
        return self

    def transform_2d(self, X_processed: np.ndarray) -> np.ndarray:
        """Project processed features onto the first 2 principal components.
        
        Args:
            X_processed: 2D feature array.
            
        Returns:
            2D array of shape (N, 2) representing (PC1, PC2).
        """
        if self.pca_2d is None:
            raise ValueError("PCA has not been fitted yet. Call fit() first.")
        return self.pca_2d.transform(X_processed)

    def get_variance_summary(self) -> dict[str, Any]:
        """Return variance diagnostics dictionary for reporting and visualization."""
        return {
            "n_components_90_variance": int(self.n_components_90),
            "pc1_explained_variance": float(self.explained_variance_ratio[0]) if len(self.explained_variance_ratio) > 0 else 0.0,
            "pc2_explained_variance": float(self.explained_variance_ratio[1]) if len(self.explained_variance_ratio) > 1 else 0.0,
            "total_2d_explained_variance": float(self.total_variance_explained_2d),
            "explained_variance_ratios": [float(v) for v in self.explained_variance_ratio[:30]],
            "cumulative_variance_ratios": [float(v) for v in self.cumulative_variance[:30]],
        }
