"""
Evaluation metrics for DegradoMap.

Includes standard classification metrics and domain-specific evaluations:
  - Per-E3 performance breakdown
  - Lysine prediction precision@k
  - E3 recommendation MRR
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                    y_score: np.ndarray) -> Dict:
    """
    Compute standard binary classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted binary labels
        y_score: Predicted scores/probabilities

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, average_precision_score,
        f1_score, precision_score, recall_score, confusion_matrix,
        matthews_corrcoef,
    )

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

    if len(np.unique(y_true)) > 1:
        metrics["auroc"] = roc_auc_score(y_true, y_score)
        metrics["auprc"] = average_precision_score(y_true, y_score)
    else:
        metrics["auroc"] = 0.5
        metrics["auprc"] = 0.0

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        metrics["tn"] = int(cm[0, 0])
        metrics["fp"] = int(cm[0, 1])
        metrics["fn"] = int(cm[1, 0])
        metrics["tp"] = int(cm[1, 1])

    return metrics


def compute_per_e3_metrics(y_true: np.ndarray, y_score: np.ndarray,
                           e3_names: List[str]) -> Dict[str, Dict]:
    """
    Compute metrics broken down by E3 ligase family.

    Args:
        y_true: Ground truth labels
        y_score: Predicted scores
        e3_names: E3 ligase name for each sample

    Returns:
        Dictionary mapping E3 name to metrics
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    unique_e3s = sorted(set(e3_names))
    per_e3 = {}

    for e3 in unique_e3s:
        mask = np.array([n == e3 for n in e3_names])
        if mask.sum() < 2:
            continue

        e3_true = y_true[mask]
        e3_score = y_score[mask]
        e3_pred = (e3_score > 0.5).astype(float)

        metrics = {
            "n_samples": int(mask.sum()),
            "n_positive": int(e3_true.sum()),
            "n_negative": int((1 - e3_true).sum()),
            "accuracy": float((e3_pred == e3_true).mean()),
        }

        if len(np.unique(e3_true)) > 1:
            metrics["auroc"] = float(roc_auc_score(e3_true, e3_score))
            metrics["auprc"] = float(average_precision_score(e3_true, e3_score))
        else:
            metrics["auroc"] = 0.5
            metrics["auprc"] = 0.0

        per_e3[e3] = metrics

    return per_e3


def lysine_precision_at_k(predicted_scores: np.ndarray,
                           predicted_indices: np.ndarray,
                           true_ub_sites: List[int],
                           k_values: List[int] = None) -> Dict:
    """
    Compute precision@k for lysine ubiquitination site prediction.

    Args:
        predicted_scores: UGS scores for each predicted lysine
        predicted_indices: Residue indices of predicted lysines
        true_ub_sites: Known ubiquitination site positions
        k_values: Values of k to evaluate

    Returns:
        Dictionary with precision@k for each k
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    if len(predicted_scores) == 0 or len(true_ub_sites) == 0:
        return {f"p@{k}": 0.0 for k in k_values}

    # Sort by score (descending)
    sorted_idx = np.argsort(-predicted_scores)
    sorted_positions = predicted_indices[sorted_idx]

    true_set = set(true_ub_sites)
    results = {}

    for k in k_values:
        top_k = sorted_positions[:k]
        hits = sum(1 for pos in top_k if pos in true_set)
        results[f"p@{k}"] = hits / min(k, len(sorted_positions))

    return results


def e3_recommendation_mrr(target_scores: Dict[str, float],
                          true_e3: str) -> float:
    """
    Compute Mean Reciprocal Rank for E3 ligase recommendation.

    Args:
        target_scores: Dict mapping E3 name to DegradoScore
        true_e3: Ground truth optimal E3 ligase

    Returns:
        Reciprocal rank (1/rank of true E3 in sorted predictions)
    """
    if true_e3 not in target_scores:
        return 0.0

    sorted_e3s = sorted(target_scores.keys(), key=lambda x: -target_scores[x])
    for rank, e3 in enumerate(sorted_e3s, 1):
        if e3 == true_e3:
            return 1.0 / rank

    return 0.0


def full_evaluation_report(results: Dict, split_name: str = "test") -> str:
    """Generate a formatted evaluation report."""
    report = []
    report.append(f"\n{'='*60}")
    report.append(f"DegradoMap Evaluation Report - {split_name}")
    report.append(f"{'='*60}")

    # Overall metrics
    if "overall" in results:
        report.append("\nOverall Metrics:")
        for metric, value in sorted(results["overall"].items()):
            if isinstance(value, float):
                report.append(f"  {metric:20s}: {value:.4f}")
            else:
                report.append(f"  {metric:20s}: {value}")

    # Per-E3 breakdown
    if "per_e3" in results:
        report.append("\nPer-E3 Ligase Breakdown:")
        for e3, metrics in sorted(results["per_e3"].items()):
            report.append(f"\n  {e3}:")
            for metric, value in sorted(metrics.items()):
                if isinstance(value, float):
                    report.append(f"    {metric:18s}: {value:.4f}")
                else:
                    report.append(f"    {metric:18s}: {value}")

    # Lysine prediction
    if "lysine" in results:
        report.append("\nLysine Ubiquitination Prediction:")
        for metric, value in sorted(results["lysine"].items()):
            report.append(f"  {metric:20s}: {value:.4f}")

    report.append(f"\n{'='*60}\n")

    return "\n".join(report)
