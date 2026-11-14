"""Binary probability metrics and validation-only threshold selection."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _checked_arrays(y, probability) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y)
    probability = np.asarray(probability, dtype=float)
    if y.ndim != 1 or probability.ndim != 1 or len(y) != len(probability) or not len(y):
        raise ValueError("Expected equally sized, nonempty one-dimensional labels and probabilities.")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Evaluation requires both binary classes 0 and 1.")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Probabilities must be finite and in [0, 1].")
    return y, probability


def select_f1_threshold(y, probability) -> float:
    """Maximize F1 on validation; break exact ties with the highest threshold."""
    y, probability = _checked_arrays(y, probability)
    precision, recall, thresholds = precision_recall_curve(y, probability)
    # The last precision/recall point has no corresponding threshold.
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    best = np.flatnonzero(f1 == f1.max())[-1]
    return float(thresholds[best])


def binary_metrics(y, probability, *, threshold: float) -> dict:
    """Compute ranking, probability and threshold-dependent metrics."""
    y, probability = _checked_arrays(y, probability)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0, 1].")
    predicted = (probability >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "brier_score": float(brier_score_loss(y, probability)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }
