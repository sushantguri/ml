#!/usr/bin/env python3
"""
NSL-KDD Dataset Downloader and Sample Generator
Adaptive Cyber-Physical Security System

Source: Canadian Institute for Cybersecurity (CIC), University of New Brunswick (UNB)
Dataset: NSL-KDD (Improvement of KDD Cup 1999 benchmark)
License/Usage: Public benchmark for academic and scientific research in intrusion detection.
Reference: Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009).
           "A detailed analysis of the KDD CUP 99 data set." IEEE CISDA.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import requests
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"


def load_config() -> dict:
    """Load configuration from config.yaml."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_file(url: str, dest_path: Path, force: bool = False) -> bool:
    """Download a file with streaming and error handling."""
    if dest_path.exists() and not force:
        logger.info("File already exists: %s (%d bytes). Skipping download.", dest_path, dest_path.stat().st_size)
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", url, dest_path)

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_bytes = 0
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    total_bytes += len(chunk)

        logger.info("Download completed successfully (%d bytes saved).", total_bytes)
        return True
    except (requests.RequestException, OSError) as e:
        logger.error("Failed to download from %s: %s", url, e)
        if dest_path.exists():
            dest_path.unlink()
        return False


def generate_samples(train_path: Path, test_path: Path, samples_dir: Path, columns: list) -> None:
    """Generate small standalone CSV sample files with header for testing and UI demo."""
    samples_dir.mkdir(parents=True, exist_ok=True)
    feature_columns = columns[:-2]  # exclude 'label' and 'difficulty_level'

    if not train_path.exists() or not test_path.exists():
        logger.warning("Train or test file missing; skipping sample generation.")
        return

    logger.info("Generating evaluation and test sample CSVs in %s...", samples_dir)

    # Read slice from test dataset
    df_test = pd.read_csv(test_path, header=None, names=columns, nrows=1000)

    # 1. Normal traffic sample (features only, no target column)
    df_normal = df_test[df_test["label"] == "normal"].head(100)[feature_columns]
    df_normal.to_csv(samples_dir / "sample_normal_traffic.csv", index=False)

    # 2. Attack traffic sample (features only, no target column)
    df_attack = df_test[df_test["label"] != "normal"].head(100)[feature_columns]
    df_attack.to_csv(samples_dir / "sample_attack_traffic.csv", index=False)

    # 3. Mixed traffic sample (50 normal, 50 attack, features only)
    df_mixed = pd.concat([df_normal.head(50), df_attack.head(50)], ignore_index=True)
    df_mixed.to_csv(samples_dir / "sample_mixed_traffic.csv", index=False)

    # 4. Labeled evaluation test sample (first 300 records of test set with labels)
    df_eval = df_test.head(300)[columns]
    df_eval.to_csv(samples_dir / "sample_labeled_eval.csv", index=False)

    logger.info("Sample datasets created successfully:")
    for sample_file in samples_dir.glob("*.csv"):
        logger.info(" - %s (%d rows)", sample_file.name, len(pd.read_csv(sample_file)))


def main():
    parser = argparse.ArgumentParser(description="Download NSL-KDD dataset benchmark files.")
    parser.add_argument("--full", action="store_true", help="Download full 125k train set instead of 20%% subset")
    parser.add_argument("--force", action="store_true", help="Force re-download if file already exists")
    args = parser.parse_args()

    config = load_config()
    raw_dir = ROOT_DIR / config["dataset"]["paths"]["raw_dir"]
    samples_dir = ROOT_DIR / config["dataset"]["paths"]["samples_dir"]
    columns = config["schema"]["columns"]

    # Choose train URL
    if args.full:
        train_url = config["dataset"]["urls"]["train_full"]
        train_dest = raw_dir / "KDDTrain+.txt"
    else:
        train_url = config["dataset"]["urls"]["train_20percent"]
        train_dest = raw_dir / "KDDTrain+_20Percent.txt"

    test_url = config["dataset"]["urls"]["test"]
    test_dest = raw_dir / "KDDTest+.txt"

    logger.info("=" * 60)
    logger.info("NSL-KDD Benchmark Dataset Downloader")
    logger.info("Source: %s", config["dataset"]["source"])
    logger.info("License: %s", config["dataset"]["license"])
    logger.info("=" * 60)

    train_ok = download_file(train_url, train_dest, force=args.force)
    test_ok = download_file(test_url, test_dest, force=args.force)

    if not train_ok or not test_ok:
        logger.error("Dataset download failed. Please check internet connection or manually place files in %s", raw_dir)
        sys.exit(1)

    # Verify counts
    train_df = pd.read_csv(train_dest, header=None)
    test_df = pd.read_csv(test_dest, header=None)
    logger.info("Verification passed:")
    logger.info("Training records: %d rows, %d columns", train_df.shape[0], train_df.shape[1])
    logger.info("Test records:     %d rows, %d columns", test_df.shape[0], test_df.shape[1])

    # Generate sample demonstration files
    generate_samples(train_dest, test_dest, samples_dir, columns)
    logger.info("Dataset setup is fully complete!")


if __name__ == "__main__":
    main()
