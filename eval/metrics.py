"""Evaluation metrics: AUROC, conformal p-values, TPR, FPR, latency."""
from typing import Dict, List, Sequence, Tuple, Union
import numpy as np


def compute_auroc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC).
    
    Uses Wilcoxon-Mann-Whitney U statistic with proper tie handling.
    Exact match to sklearn.metrics.roc_auc_score.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(y_score)
    ranks = np.empty(len(y_score), dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)

    # Tie handling: assign mean rank to ties
    unique_scores, inverse, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    if len(unique_scores) < len(y_score):
        tie_ranks = np.zeros(len(unique_scores), dtype=np.float64)
        np.add.at(tie_ranks, inverse, ranks)
        tie_ranks /= counts
        ranks = tie_ranks[inverse]

    rank_sum_pos = np.sum(ranks[y_true == 1])
    u = rank_sum_pos - (n_pos * (n_pos + 1.0)) / 2.0
    return float(u / (n_pos * n_neg))


def compute_conformal_p_values(calib_scores: Sequence[float], test_scores: Sequence[float]) -> np.ndarray:
    """Compute conformal p-values for test scores given reference calibration scores.
    
    Formula: p = (1 + sum(calib_scores >= score)) / (n_calib + 1)
    """
    calib = np.asarray(calib_scores, dtype=np.float64)
    test = np.asarray(test_scores, dtype=np.float64)
    n = len(calib)
    if n == 0:
        return np.ones(len(test), dtype=np.float64)

    # For each test score, count how many calib scores are >= test score
    # Shape: (len(test), len(calib))
    counts = np.sum(calib[None, :] >= test[:, None], axis=1)
    return (1.0 + counts) / (n + 1.0)


def evaluate_binary_predictions(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    """Calculate TPR (recall), FPR, precision, accuracy."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))

    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    acc = float((tp + tn) / len(y_true)) if len(y_true) > 0 else 0.0

    return {
        "tpr": tpr,
        "fpr": fpr,
        "precision": prec,
        "accuracy": acc,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }
