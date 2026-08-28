from __future__ import annotations

import numpy as np


def binary_segmentation_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction, target = prediction.astype(bool), target.astype(bool)
    tp = np.logical_and(prediction, target).sum()
    fp = np.logical_and(prediction, ~target).sum()
    fn = np.logical_and(~prediction, target).sum()
    tn = np.logical_and(~prediction, ~target).sum()
    return {
        "iou": float(tp / max(tp + fp + fn, 1)),
        "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "overall_accuracy": float((tp + tn) / max(tp + fp + fn + tn, 1)),
    }


def exact_match(predictions: list[str], targets: list[str]) -> float:
    if not targets:
        return 0.0
    normalize = lambda value: " ".join(value.lower().strip().split())
    return sum(normalize(a) == normalize(b) for a, b in zip(predictions, targets)) / len(targets)
