from __future__ import annotations

import numpy as np

from app.api.schemas.analysis import ConfidenceResult


class ConfidenceEngine:
    def from_probability(
        self,
        probability: np.ndarray,
        threshold: float,
        *,
        agreement: float | None = None,
        spatial_quality: float = 0.82,
    ) -> ConfidenceResult:
        selected = probability[probability >= threshold]
        strength = float(selected.mean()) if selected.size else float(probability.mean())
        components = {"evidence_strength": round(strength, 3), "spatial_quality": round(spatial_quality, 3)}
        if agreement is not None:
            components["cross_sensor_agreement"] = round(float(agreement), 3)
        overall = float(np.mean(list(components.values())))
        return ConfidenceResult(
            overall=round(max(0.0, min(1.0, overall)), 3),
            type="heuristic",
            components=components,
            note="Heuristic confidence derived from pixel evidence and spatial consistency; it is not a calibrated model probability.",
        )
