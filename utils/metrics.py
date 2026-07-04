from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BinaryMetrics:
    auroc: float
    auprc: float
    f1: float
    sensitivity: float
    specificity: float
    accuracy: float
    threshold: float


def _import_sklearn_metrics():
    try:
        from sklearn.metrics import (  # type: ignore
            accuracy_score,
            average_precision_score,
            confusion_matrix,
            f1_score,
            precision_recall_curve,
            roc_auc_score,
            roc_curve,
        )
    except Exception as exc:
        raise RuntimeError("scikit-learn is required for metric computation.") from exc
    return {
        "accuracy_score": accuracy_score,
        "average_precision_score": average_precision_score,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "precision_recall_curve": precision_recall_curve,
        "roc_auc_score": roc_auc_score,
        "roc_curve": roc_curve,
    }


def validate_binary_targets(y_true: np.ndarray) -> None:
    labels = np.unique(y_true)
    if labels.shape[0] < 2:
        raise ValueError("Binary metric computation requires both classes to be present.")


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    strategy: str = "youden",
) -> float:
    metrics = _import_sklearn_metrics()
    validate_binary_targets(y_true)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if strategy == "youden":
        fpr, tpr, thresholds = metrics["roc_curve"](y_true, y_prob)
        thresholds = np.asarray(thresholds, dtype=float)
        scores = tpr - fpr
        finite = np.isfinite(thresholds)
        if finite.any():
            thresholds = thresholds[finite]
            scores = scores[finite]
        best_idx = int(np.nanargmax(scores))
        return float(np.clip(thresholds[best_idx], 0.0, 1.0))
    if strategy == "f1":
        precision, recall, thresholds = metrics["precision_recall_curve"](y_true, y_prob)
        if thresholds.shape[0] == 0:
            return 0.5
        f1 = (2 * precision[:-1] * recall[:-1]) / np.clip(precision[:-1] + recall[:-1], 1e-8, None)
        best_idx = int(np.nanargmax(f1))
        return float(np.clip(thresholds[best_idx], 0.0, 1.0))
    raise ValueError(f"Unsupported threshold strategy: {strategy}")


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> BinaryMetrics:
    metrics = _import_sklearn_metrics()
    validate_binary_targets(y_true)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    auroc = float(metrics["roc_auc_score"](y_true, y_prob))
    auprc = float(metrics["average_precision_score"](y_true, y_prob))
    f1 = float(metrics["f1_score"](y_true, y_pred))
    accuracy = float(metrics["accuracy_score"](y_true, y_pred))
    tn, fp, fn, tp = metrics["confusion_matrix"](y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = float(tp / max(tp + fn, 1))
    specificity = float(tn / max(tn + fp, 1))
    return BinaryMetrics(
        auroc=auroc,
        auprc=auprc,
        f1=f1,
        sensitivity=sensitivity,
        specificity=specificity,
        accuracy=accuracy,
        threshold=float(threshold),
    )
