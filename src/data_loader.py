"""
Data Loader and Ingestion Module for Network Intrusion Data.
Adaptive Cyber-Physical Security System.

Dataset: NSL-KDD Network Benchmark
Source: Canadian Institute for Cybersecurity (CIC), University of New Brunswick (UNB)
Usage Terms: Research and educational benchmark for evaluating intrusion detection systems.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load system configuration from YAML file.
    
    Args:
        config_path: Optional path to config file. Defaults to config/config.yaml.
        
    Returns:
        Dict containing project configuration.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_attack_taxonomy(config: dict[str, Any] | None = None) -> dict[str, str]:
    """Create a flat mapping of specific attack types to high-level attack categories.
    
    Categories:
        - dos: Denial of Service
        - probe: Surveillance, port scanning, network mapping
        - r2l: Remote to Local unauthorized access
        - u2r: User to Root privilege escalation
        - normal: Benign normal traffic
    """
    if config is None:
        config = load_config()

    taxonomy_map: dict[str, str] = {"normal": "normal"}
    attack_categories = config.get("schema", {}).get("attack_categories", {})

    for cat_name, attack_list in attack_categories.items():
        for attack in attack_list:
            taxonomy_map[attack.lower()] = cat_name.lower()

    return taxonomy_map


def load_dataset(
    filepath: str | Path,
    config: dict[str, Any] | None = None,
    drop_metadata: bool = True,
    nrows: int | None = None,
) -> tuple[pd.DataFrame, pd.Series | None, pd.Series | None]:
    """Robustly load a network traffic dataset file (CSV or TXT).
    
    Automatically detects:
      1. Presence of header row vs. raw delimiter-separated values without header.
      2. 43-column full NSL-KDD format (41 features + label + difficulty).
      3. 42-column format (41 features + label).
      4. 41-column inference format (41 features only).
      
    Args:
        filepath: Path to dataset file.
        config: Optional configuration dictionary.
        drop_metadata: Whether to exclude non-feature columns (difficulty_level).
        nrows: Optional row limit for quick inspection.
        
    Returns:
        Tuple of:
          - X (pd.DataFrame): Input traffic features only (no target/labels).
          - y (Optional[pd.Series]): Binary attack label (0=normal, 1=attack) or None.
          - attack_cat (Optional[pd.Series]): High-level attack category string or None.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    if config is None:
        config = load_config()

    full_columns: list[str] = config["schema"]["columns"]
    feature_columns: list[str] = [c for c in full_columns if c not in ("label", "difficulty_level")]
    target_col: str = config["schema"].get("target_column", "label")
    metadata_cols: list[str] = config["schema"].get("metadata_columns", ["difficulty_level"])

    # Inspect first line to detect if header exists
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip()
    first_tokens = [t.strip().strip('"\'') for t in first_line.split(",")]

    has_header = False
    # Check if first token matches known column names
    if first_tokens[0] in full_columns or "duration" in first_tokens or "protocol_type" in first_tokens:
        has_header = True

    if has_header:
        df = pd.read_csv(path, nrows=nrows)
    else:
        # Determine number of columns
        num_cols = len(first_tokens)
        if num_cols == len(full_columns):  # 43 columns
            df = pd.read_csv(path, header=None, names=full_columns, nrows=nrows)
        elif num_cols == len(full_columns) - 1:  # 42 columns (without difficulty)
            cols_42 = [c for c in full_columns if c != "difficulty_level"]
            df = pd.read_csv(path, header=None, names=cols_42, nrows=nrows)
        elif num_cols == len(feature_columns):  # 41 columns (features only)
            df = pd.read_csv(path, header=None, names=feature_columns, nrows=nrows)
        else:
            logger.warning("File has %d columns, expected 41, 42, or 43. Reading with auto-header.", num_cols)
            df = pd.read_csv(path, header=None, nrows=nrows)

    logger.info("Loaded dataset from %s. Initial shape: %s", path.name, df.shape)

    # Check for missing values and report
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        logger.warning("Dataset contains %d missing values. Filling with defaults.", missing_count)
        # Numerical columns with median, categorical with mode
        for col in df.columns:
            if df[col].dtype in (np.float64, np.int64, float, int):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "unknown")

    # Extract target label if present
    y_binary: pd.Series | None = None
    attack_categories: pd.Series | None = None

    if target_col in df.columns:
        raw_labels = df[target_col].astype(str).str.strip().str.lower().str.rstrip(".")
        taxonomy = get_attack_taxonomy(config)

        # Binary label: 0 for normal, 1 for any attack
        y_binary = (raw_labels != "normal").astype(int)
        attack_categories = raw_labels.map(lambda lbl: taxonomy.get(lbl, "unknown_attack" if lbl != "normal" else "normal"))

        # Drop label from features
        df = df.drop(columns=[target_col])

    # Drop metadata columns from model features
    if drop_metadata:
        for m_col in metadata_cols:
            if m_col in df.columns:
                df = df.drop(columns=[m_col])

    # Retain only recognized feature columns if available
    available_features = [col for col in feature_columns if col in df.columns]
    if len(available_features) > 0:
        X = df[available_features].copy()
    else:
        # Fallback if custom schema
        X = df.copy()

    return X, y_binary, attack_categories


def get_dataset_summary(
    X: pd.DataFrame,
    y: pd.Series | None = None,
    attack_cat: pd.Series | None = None,
) -> dict[str, Any]:
    """Generate exploration summary statistics for network traffic records."""
    summary: dict[str, Any] = {
        "total_records": len(X),
        "total_features": X.shape[1],
        "columns": list(X.columns),
        "dtypes": {col: str(dtype) for col, dtype in X.dtypes.items()},
        "missing_values": int(X.isnull().sum().sum()),
        "has_labels": y is not None,
    }

    if y is not None:
        normal_count = int((y == 0).sum())
        attack_count = int((y == 1).sum())
        summary["label_distribution"] = {
            "normal_count": normal_count,
            "attack_count": attack_count,
            "normal_pct": round(normal_count / len(y) * 100, 2),
            "attack_pct": round(attack_count / len(y) * 100, 2),
        }

    if attack_cat is not None:
        summary["attack_category_counts"] = attack_cat.value_counts().to_dict()

    return summary
