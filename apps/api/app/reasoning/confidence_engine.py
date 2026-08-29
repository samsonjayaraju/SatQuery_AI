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
        learned: bool = False,
        model_score: float | None = None,
        semantic_consistency: float | None = None,
        input_quality: float = 1.0,
        registration_quality: float | None = None,
    ) -> ConfidenceResult:
        selected = probability[probability >= threshold]
        strength = float(selected.mean()) if selected.size else float(probability.mean())
        components = {
            "evidence_strength": round(strength, 3),
            "spatial_quality": round(spatial_quality, 3),
            "input_quality": round(float(input_quality), 3),
        }
        if agreement is not None:
            components["cross_sensor_agreement"] = round(float(agreement), 3)
        if model_score is not None:
            components["model_score"] = round(float(model_score), 3)
        if semantic_consistency is not None:
            components["semantic_consistency"] = round(float(semantic_consistency), 3)
        if registration_quality is not None:
            components["registration_quality"] = round(float(registration_quality), 3)
        overall = float(np.mean(list(components.values())))
        if registration_quality is not None:
            # Registration is a hard constraint for quantitative pair analysis:
            # other strong signals cannot compensate for a poorly aligned pair.
            overall = min(overall, float(registration_quality) + 0.15)
        if learned:
            # No task-specific calibration set is installed yet. Keep an honest
            # ceiling so contrastive similarity is never presented as calibrated
            # probability, while still exposing all contributing evidence.
            overall = min(overall, 0.74, max(0.35, float(input_quality) + 0.1))
        return ConfidenceResult(
            overall=round(max(0.0, min(1.0, overall)), 3),
            type="mixed" if learned else "heuristic",
            components=components,
            note=(
                "Uncalibrated learned evidence score; capped until task-specific confidence calibration is measured."
                if learned
                else "Heuristic confidence derived from pixel evidence and spatial consistency; it is not a calibrated model probability."
            ),
        )
