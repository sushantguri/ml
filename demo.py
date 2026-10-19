#!/usr/bin/env python3
"""
Demonstration Script: Adaptive Cyber-Physical Security Anomaly Detection
Walks through the complete workflow:
1. Verifies dataset and configuration
2. Executes unsupervised pipeline training
3. Runs benchmark evaluation on held-out test data
4. Performs batch inference on sample CSV files
5. Displays triage alerts with explainability deviations
"""

import logging
from pathlib import Path
import sys

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import load_config
from src.train import run_training_pipeline
from src.predict import InferencePipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    print("=" * 75)
    print("ADAPTIVE CYBER-PHYSICAL SECURITY: ANOMALY DETECTION DEMO")
    print("=" * 75)

    config = load_config()
    print(f"Project:     {config['project']['name']}")
    print(f"Dataset:     {config['dataset']['name']} ({config['dataset']['source']})")
    print(f"Algorithm:   {config['model']['algorithm']} (Unsupervised)")
    print("-" * 75)

    # 1. Check for Model Artifacts or Train
    models_dir = ROOT_DIR / "models"
    detector_file = models_dir / "isolation_forest.joblib"

    if not detector_file.exists():
        print("\n[STEP 1] No existing model found. Running training pipeline...")
        metadata = run_training_pipeline()
    else:
        print("\n[STEP 1] Found existing trained model. Initializing inference pipeline...")

    pipeline = InferencePipeline()
    print("Pipeline Loaded Successfully:", pipeline.is_loaded)

    # 2. Run Batch Inference on Demonstration Sample
    sample_csv = ROOT_DIR / "data" / "samples" / "sample_mixed_traffic.csv"
    if sample_csv.exists():
        print(f"\n[STEP 2] Running batch inference on: {sample_csv.name}")
        results = pipeline.predict_csv(sample_csv)

        total = len(results)
        anomalies = (results["is_anomaly"] == 1).sum()
        normal = (results["is_anomaly"] == 0).sum()

        print(f"Total Sessions Processed: {total}")
        print(f"Normal Sessions:          {normal} ({(normal/total)*100:.1f}%)")
        print(f"Flagged Anomalies:        {anomalies} ({(anomalies/total)*100:.1f}%)")

        print("\n[STEP 3] Review Priority Breakdown:")
        print(results["review_priority"].value_counts().to_string())

        print("\n[STEP 4] Top 5 Flagged Security Alerts:")
        alert_cols = [
            "review_priority",
            "anomaly_score",
            "protocol_type",
            "service",
            "flag",
            "src_bytes",
            "anomaly_reasons",
        ]
        alerts = results[results["is_anomaly"] == 1]
        for idx, row in alerts.head(5).iterrows():
            print(f"\nAlert #{idx} [{row['review_priority']}] - Anomaly Score: {row['anomaly_score']:.4f}")
            print(f"  Protocol: {row['protocol_type']} | Service: {row['service']} | Flag: {row['flag']} | Src Bytes: {row['src_bytes']}")
            print(f"  Explanation: {row['anomaly_reasons']}")

    print("\n" + "=" * 75)
    print("Demo completed successfully! To launch the interactive SOC dashboard, run:")
    print("  streamlit run app.py")
    print("=" * 75)


if __name__ == "__main__":
    main()
