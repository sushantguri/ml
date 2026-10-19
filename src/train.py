"""
Training Pipeline Orchestrator.
Adaptive Cyber-Physical Security System.

Strictly adheres to:
- Unsupervised anomaly detection: Ground-truth labels are NEVER passed into model
- Preprocessor and PCA fitted strictly on training data
- Separation of Training, Validation, and Held-out Test sets
- Comparison of multiple Isolation Forest configurations on validation data
- Unbiased final evaluation on benchmark test set
- Complete model persistence (joblib + metadata JSON)
"""

from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import load_config, load_dataset
from src.preprocessing import TrafficPreprocessor
from src.feature_engineering import PCAManager
from src.detector import IsolationForestDetector
from src.evaluation import evaluate_predictions, compare_configurations

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent


def run_training_pipeline(
    config_path: Optional[str | Path] = None,
    train_path: Optional[str | Path] = None,
    test_path: Optional[str | Path] = None,
    training_strategy: Optional[str] = None,
    contamination: Optional[float] = None,
    n_estimators: Optional[int] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Execute end-to-end unsupervised training and benchmark evaluation.
    
    Args:
        config_path: Path to YAML configuration.
        train_path: Path to training data file.
        test_path: Path to held-out test data file.
        training_strategy: 'normal_only' or 'unsupervised_mixture'.
        contamination: Contamination rate for Isolation Forest.
        n_estimators: Number of trees in Isolation Forest.
        random_state: Random seed for reproducibility.
        
    Returns:
        Dictionary of complete training results and benchmark metrics.
    """
    config = load_config(config_path)

    # Resolve paths
    train_file = Path(train_path) if train_path else ROOT_DIR / config["dataset"]["paths"]["train_file"]
    test_file = Path(test_path) if test_path else ROOT_DIR / config["dataset"]["paths"]["test_file"]
    models_dir = ROOT_DIR / config["persistence"]["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)

    strategy = training_strategy or config["model"].get("training_strategy", "normal_only")
    contam = contamination if contamination is not None else config["model"]["parameters"]["contamination"]
    n_trees = n_estimators if n_estimators is not None else config["model"]["parameters"]["n_estimators"]
    max_samples = config["model"]["parameters"].get("max_samples", 512)

    logger.info("=" * 70)
    logger.info("CYBER ANOMALY DETECTION: TRAINING PIPELINE")
    logger.info("Dataset: %s", config["dataset"]["name"])
    logger.info("Training Strategy: %s", strategy)
    logger.info("Contamination: %s, Trees: %d, Seed: %d", str(contam), n_trees, random_state)
    logger.info("=" * 70)

    # 1. Ingest Data
    logger.info("Loading training data from %s...", train_file)
    X_raw, y_raw, cat_raw = load_dataset(train_file, config=config)

    logger.info("Loading held-out test data from %s...", test_file)
    X_test, y_test, cat_test = load_dataset(test_file, config=config)

    # 2. Train/Validation Split (80/20) for hyperparameter validation without test peeking
    X_train_df, X_val_df, y_train_s, y_val_s, cat_train_s, cat_val_s = train_test_split(
        X_raw, y_raw, cat_raw, test_size=0.20, random_state=random_state, stratify=y_raw
    )
    logger.info("Data Split: Train=%d, Validation=%d, Test=%d", len(X_train_df), len(X_val_df), len(X_test))

    # 3. Apply Training Strategy for Unsupervised Modeling
    if strategy == "normal_only":
        # Train strictly on normal records to learn the baseline normal distribution
        normal_mask = (y_train_s == 0)
        X_train_fit = X_train_df[normal_mask].copy()
        logger.info(
            "Strategy 'normal_only': Training on %d benign normal records (excluded %d attack records from fit).",
            len(X_train_fit),
            int((~normal_mask).sum()),
        )
    else:
        # Train on unsupervised mixture
        X_train_fit = X_train_df.copy()
        logger.info("Strategy 'unsupervised_mixture': Training on entire unlabelled mixture (%d records).", len(X_train_fit))

    # 4. Preprocessing: Fit ColumnTransformer on Training Data ONLY
    logger.info("Fitting ColumnTransformer on training features only...")
    preprocessor = TrafficPreprocessor(
        categorical_features=config["schema"]["categorical_features"],
        binary_features=config["schema"]["binary_features"],
    )
    preprocessor.fit(X_train_fit)

    X_train_proc = preprocessor.transform(X_train_fit)
    X_val_proc = preprocessor.transform(X_val_df)
    X_test_proc = preprocessor.transform(X_test)

    # 5. PCA Dimensionality Reduction
    logger.info("Fitting PCA on preprocessed training data...")
    pca_manager = PCAManager(random_state=random_state)
    pca_manager.fit(X_train_proc, variance_threshold=config["pca"].get("variance_threshold", 0.90))
    pca_summary = pca_manager.get_variance_summary()

    # 6. Fit Primary Isolation Forest Detector
    logger.info("Fitting primary Isolation Forest detector...")
    primary_detector = IsolationForestDetector(
        n_estimators=n_trees,
        max_samples=max_samples,
        contamination=contam,
        random_state=random_state,
        n_jobs=-1,
    )
    primary_detector.fit(X_train_proc)

    # 7. Compare Multiple Configurations on Validation Data
    logger.info("Comparing candidate configurations on validation split...")
    candidate_configs = {
        f"Contamination=0.05": IsolationForestDetector(n_estimators=n_trees, max_samples=max_samples, contamination=0.05, random_state=random_state).fit(X_train_proc),
        f"Contamination=0.10 (Default)": primary_detector if contam == 0.10 else IsolationForestDetector(n_estimators=n_trees, max_samples=max_samples, contamination=0.10, random_state=random_state).fit(X_train_proc),
        f"Contamination=0.15": IsolationForestDetector(n_estimators=n_trees, max_samples=max_samples, contamination=0.15, random_state=random_state).fit(X_train_proc),
    }

    comparison_df = compare_configurations(
        candidate_configs,
        X_val_proc,
        y_val_s.to_numpy(),
        attack_cat_val=cat_val_s,
    )
    logger.info("Validation Comparison Results:\n%s", comparison_df.to_string(index=False))

    # 8. Unbiased Final Evaluation on Held-Out Test Set (KDDTest+)
    logger.info("Evaluating primary detector on held-out benchmark test set...")
    y_test_pred = primary_detector.predict(X_test_proc)
    y_test_scores = primary_detector.predict_anomaly_scores(X_test_proc)

    test_metrics = evaluate_predictions(
        y_true=y_test.to_numpy(),
        y_pred=y_test_pred,
        anomaly_scores=y_test_scores,
        attack_categories=cat_test,
    )

    logger.info("=" * 60)
    logger.info("FINAL BENCHMARK TEST METRICS (KDDTest+)")
    logger.info("Precision:  %.4f", test_metrics["precision"])
    logger.info("Recall:     %.4f", test_metrics["recall"])
    logger.info("F1-Score:   %.4f", test_metrics["f1_score"])
    logger.info("Accuracy:   %.4f", test_metrics["accuracy"])
    logger.info("False Alarm Rate (FPR): %.2f%% (%d normal flagged)", test_metrics["false_positive_rate"] * 100, test_metrics["false_positives_count"])
    logger.info("Missed Attack Rate (FNR): %.2f%% (%d attacks missed)", test_metrics["false_negative_rate"] * 100, test_metrics["false_negatives_count"])
    if "pr_curve" in test_metrics:
        logger.info("PR-AUC (Average Precision): %.4f", test_metrics["pr_curve"]["average_precision"])
    logger.info("=" * 60)

    # 9. Persist Models and Metadata
    detector_path = ROOT_DIR / config["persistence"]["detector_file"]
    preprocessor_path = ROOT_DIR / config["persistence"]["preprocessor_file"]
    pca_path = ROOT_DIR / config["persistence"]["pca_file"]
    metadata_path = ROOT_DIR / config["persistence"]["metadata_file"]

    logger.info("Saving artifacts to %s...", models_dir)
    joblib.dump(primary_detector, detector_path)
    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(pca_manager, pca_path)

    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "project": config["project"]["name"],
        "version": config["project"]["version"],
        "dataset": {
            "name": config["dataset"]["name"],
            "source": config["dataset"]["source"],
            "train_file": str(train_file.name),
            "test_file": str(test_file.name),
            "train_samples": len(X_train_df),
            "fit_samples": len(X_train_fit),
            "val_samples": len(X_val_df),
            "test_samples": len(X_test),
        },
        "strategy": strategy,
        "hyperparameters": primary_detector.get_params(),
        "pca_diagnostics": pca_summary,
        "validation_comparison": comparison_df.to_dict(orient="records"),
        "test_evaluation": test_metrics,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Training pipeline completed and all artifacts serialized successfully!")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Train unsupervised network anomaly detection pipeline.")
    parser.add_argument("--train-path", type=str, default=None, help="Path to training dataset file")
    parser.add_argument("--test-path", type=str, default=None, help="Path to test dataset file")
    parser.add_argument("--strategy", type=str, choices=["normal_only", "unsupervised_mixture"], default=None, help="Training data subset strategy")
    parser.add_argument("--contamination", type=float, default=None, help="Contamination rate (e.g. 0.10)")
    parser.add_argument("--n-estimators", type=int, default=None, help="Number of trees")
    parser.add_argument("--seed", type=int, default=42, help="Random state seed")
    args = parser.parse_args()

    run_training_pipeline(
        train_path=args.train_path,
        test_path=args.test_path,
        training_strategy=args.strategy,
        contamination=args.contamination,
        n_estimators=args.n_estimators,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()
