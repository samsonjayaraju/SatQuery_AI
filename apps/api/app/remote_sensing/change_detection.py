from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def appearance_change_probability(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Estimate broad material change while suppressing global brightness shifts.

    This path is intentionally sensor-agnostic and complements the LEVIR-trained
    structural model for environmental events such as water recession, flooding,
    vegetation loss, fire scars, and exposed soil.
    """
    if before.shape != after.shape:
        raise ValueError("Appearance change inputs must be aligned and have identical shapes")
    radius = max(1, round(min(before.shape[:2]) / 256))
    before_blurred = np.asarray(
        Image.fromarray(before.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius)),
        dtype=np.float32,
    ) / 255.0
    after_blurred = np.asarray(
        Image.fromarray(after.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius)),
        dtype=np.float32,
    ) / 255.0
    difference = after_blurred - before_blurred
    difference -= np.median(difference.reshape(-1, difference.shape[2]), axis=0)
    magnitude = np.sqrt(np.mean(np.square(difference), axis=2))
    return np.clip((magnitude - 0.04) / 0.24, 0.0, 1.0).astype(np.float32)


def semantic_change_probability(
    before: dict[str, np.ndarray],
    after: dict[str, np.ndarray],
    target: str | None = None,
) -> np.ndarray:
    labels = [label for label in before if label in after]
    if not labels:
        raise ValueError("Semantic change requires at least one shared class")
    if target in labels:
        delta = np.abs(before[target] - after[target])
    else:
        before_stack = np.stack([before[label] for label in labels])
        after_stack = np.stack([after[label] for label in labels])
        delta = 0.5 * np.abs(before_stack - after_stack).sum(axis=0)
    return np.clip(delta / 0.5, 0.0, 1.0).astype(np.float32)


def hybrid_change_probability(
    structural: np.ndarray,
    appearance: np.ndarray,
    semantic: np.ndarray,
    target: str | None = None,
) -> tuple[np.ndarray, str]:
    environmental = 0.65 * appearance + 0.35 * semantic
    if target == "built_up":
        probability = np.maximum(structural, 0.35 * appearance + 0.65 * semantic)
        method = "structural_semantic_hybrid"
    else:
        # Generic and environmental questions must not be suppressed by a
        # building-specific checkpoint. Structural evidence remains a valid
        # independent signal when it is present.
        probability = np.maximum(structural, environmental)
        method = "environmental_semantic_hybrid"
    return np.clip(probability, 0.0, 1.0).astype(np.float32), method
