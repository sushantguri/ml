"""
Model Evaluation and Diagnostics Module.
Adaptive Cyber-Physical Security System.

Calculates:
- Precision, Recall, F1-score, Specificity, Accuracy
- Confusion Matrix (TN, FP, FN, TP)
- False Positive Rate (FPR) and False Negative Rate (FNR)
- Precision-Recall curve coordinates and Average Precision (PR-AUC)
- Per-attack-category recall (DoS, Probe, R2L, U2R)
- Multi-model hyperparameter comparison
- Actionable operational cybersecurity explanations of false positives/negatives
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    average_precision_score,
    roc_auc_score,
    accuracy_score,
)

logger = logging.getLogger(__name__)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    anomaly_scores: Optional[np.ndarray] = None,
    attack_categories: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics against ground-truth labels.
    
    Conventions:
        0 = Normal (Negative class)
        1 = Attack / Anomaly (Positive class)
        
    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Predicted binary anomaly flags (0 or 1).
        anomaly_scores: Continuous anomaly scores in [0, 1].
        attack_categories: Optional series of attack category strings for per-category recall.
        
    Returns:
        Dictionary of computed metrics and operational security diagnostics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Core rates
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))

    # Cybersecurity specific metrics
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    metrics: Dict[str, Any] = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "specificity": round(specificity, 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "false_positives_count": int(fp),
        "false_negatives_count": int(fn),
        "total_evaluated": int(len(y_true)),
        "normal_count": int(tn + fp),
        "attack_count": int(tp + fn),
    }

    # Precision-Recall curve if continuous scores are supplied
    if anomaly_scores is not None:
        try:
            precisions, recalls, thresholds = precision_recall_curve(y_true, anomaly_scores)
            avg_prec = average_precision_score(y_true, anomaly_scores)
            roc_auc = roc_auc_score(y_true, anomaly_scores)

            # Subsample curve coordinates for lean serialization/JSON display
            step = max(1, len(precisions) // 50)
            metrics["pr_curve"] = {
                "precisions": [float(p) for p in precisions[::step]],
                "recalls": [float(r) for r in recalls[::step]],
                "thresholds": [float(t) for t in thresholds[::step]] if len(thresholds) > 0 else [],
                "average_precision": round(float(avg_prec), 4),
                "roc_auc": round(float(roc_auc), 4),
            }
        except Exception as e:
            logger.warning("Could not calculate PR curve: %s", e)

    # Per attack category recall
    if attack_categories is not None:
        cat_series = attack_categories.reset_index(drop=True)
        pred_series = pd.Series(y_pred)
        cat_metrics: Dict[str, Dict[str, Any]] = {}

        for cat in cat_series.unique():
            if str(cat).lower() == "normal":
                continue
            cat_mask = cat_series == cat
            total_cat = int(cat_mask.sum())
            if total_cat > 0:
                detected = int((pred_series[cat_mask] == 1).sum())
                cat_recall = round(detected / total_cat, 4)
                cat_metrics[str(cat)] = {
                    "total": total_cat,
                    "detected": detected,
                    "missed": total_cat - detected,
                    "recall": cat_recall,
                }
        metrics["per_category_metrics"] = cat_metrics

    # Explanations of operational trade-offs
    metrics["operational_tradeoff_notes"] = {
        "why_accuracy_is_insufficient": (
            "Accuracy measures overall correct predictions across normal and attack traffic. "
            "In network security with heavy class imbalance or high-stakes intrusions, a model "
            "that flags nothing might score high accuracy while missing 100% of stealth intrusions."
        ),
        "false_positives_impact": (
            f"Model flagged {fp} normal network sessions as anomalous (FPR = {fpr:.2%}). "
            "In a Security Operations Center (SOC), excessive false positives cause analyst fatigue "
            "and can lead to real threats being ignored."
        ),
        "false_negatives_impact": (
            f"Model failed to flag {fn} active attacks as anomalous (FNR = {fnr:.2%}). "
            "False negatives represent unmitigated intrusions, lateral movement, or data exfiltration."
        ),
    }

    return metrics


def compare_configurations(
    models_dict: Dict[str, Any],
    X_val_processed: np.ndarray,
    y_val: np.ndarray,
    attack_cat_val: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Compare multiple detector configurations on validation data.
    
    Args:
        models_dict: Dict mapping configuration names to fitted detector instances.
        X_val_processed: Preprocessed validation feature matrix.
        y_val: Validation ground truth binary labels.
        attack_cat_val: Validation attack categories.
        
    Returns:
        pd.DataFrame containing side-by-side comparison metrics.
    """
    rows = []
    for name, detector in models_dict.items():
        y_pred = detector.predict(X_val_processed)
        scores = detector.predict_anomaly_scores(X_val_processed)
        eval_res = evaluate_predictions(y_val, y_pred, anomaly_scores=scores, attack_categories=attack_cat_val)

        row = {
            "Configuration": name,
            "Contamination": getattr(detector, "contamination", "N/A"),
            "Precision": eval_res["precision"],
            "Recall": eval_res["recall"],
            "F1-Score": eval_res["f1_score"],
            "FPR (False Alarm %)": f"{eval_res['false_positive_rate']:.2%}",
            "FNR (Missed Attack %)": f"{eval_res['false_negative_rate']:.2%}",
            "False Positives": eval_res["false_positives_count"],
            "Missed Attacks": eval_res["false_negatives_count"],
            "PR-AUC": eval_res.get("pr_curve", {}).get("average_precision", "N/A"),
        }
        rows.append(row)

    return pd.DataFrame(rows)
