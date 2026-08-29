from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SatFusionResult:
    probability: np.ndarray
    agreement: float
    optical_weight: float
    sar_weight: float


class SatFusionService:
    """Lightweight, replaceable optical/SAR weighted-feature fusion baseline."""

    model_name = "SatFusion weighted feature fusion v1"

    def predict(
        self,
        optical_probability: np.ndarray,
        sar_probability: np.ndarray,
        sar_texture: np.ndarray,
    ) -> SatFusionResult:
        if optical_probability.shape != sar_probability.shape or sar_probability.shape != sar_texture.shape:
            raise ValueError("SatFusion inputs must share one co-registered pixel grid")
        texture_quality = float(np.clip(1.0 - sar_texture.mean(), 0.0, 1.0))
        sar_weight = float(np.clip(0.46 + 0.14 * texture_quality, 0.46, 0.60))
        optical_weight = 1.0 - sar_weight
        probability = np.clip(
            optical_probability * optical_weight + sar_probability * sar_weight,
            0.0,
            1.0,
        )
        agreement = float(np.clip(1.0 - np.mean(np.abs(optical_probability - sar_probability)), 0.0, 1.0))
        return SatFusionResult(probability, agreement, optical_weight, sar_weight)
