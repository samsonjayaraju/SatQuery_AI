from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.core.config import Settings
from app.core.exceptions import SatQueryError
from app.models.satfusion import SatFusionService
from app.remote_sensing.visualization import largest_polygon, retain_largest_component
from app.registry.model_registry import ModelRegistry
from app.registry.tool_registry import ToolRegistry
from app.services.analysis_service import AnalysisService
from app.services.history_service import HistoryService


class FakeChangeFormer:
    available = True
    loaded = False

    @staticmethod
    def predict(before: np.ndarray, after: np.ndarray) -> np.ndarray:
        probability = np.zeros(before.shape[:2], dtype=np.float32)
        probability[8:24, 10:30] = 0.92
        return probability


def analysis_service(tmp_path: Path, changeformer=None) -> AnalysisService:
    settings = Settings(
        data_dir=tmp_path / "data",
        model_dir=tmp_path / "models",
        temp_dir=tmp_path / "temp",
        mock_mode=False,
        model_unload_after_request=False,
    )
    return AnalysisService(
        settings,
        ModelRegistry(settings.model_dir, "cpu"),
        ToolRegistry(),
        HistoryService(settings.data_dir),
        changeformer=changeformer,
    )


def test_satfusion_blends_co_registered_optical_and_sar_features():
    optical = np.full((8, 8), 0.8, dtype=np.float32)
    sar = np.full((8, 8), 0.6, dtype=np.float32)
    texture = np.full((8, 8), 0.2, dtype=np.float32)
    result = SatFusionService().predict(optical, sar, texture)
    assert result.probability.shape == optical.shape
    assert result.optical_weight + result.sar_weight == pytest.approx(1.0)
    assert result.agreement == pytest.approx(0.8)
    assert np.all(result.probability < optical)
    assert np.all(result.probability > sar)


def test_largest_polygon_returns_normalized_closed_outline():
    probability = np.zeros((20, 30), dtype=np.float32)
    probability[4:15, 7:24] = 0.9
    polygon = largest_polygon(probability, 0.5)
    assert polygon is not None
    assert polygon[0] == polygon[-1]
    assert all(0 <= coordinate <= 1 for point in polygon for coordinate in point)


def test_grounding_keeps_only_the_largest_connected_region():
    probability = np.zeros((12, 12), dtype=np.float32)
    probability[1:3, 1:3] = 0.9
    probability[5:10, 5:11] = 0.8
    filtered = retain_largest_component(probability, 0.5)
    assert np.count_nonzero(filtered) == 30
    assert filtered[1, 1] == 0


def test_learned_change_path_reports_real_changeformer(tmp_path: Path):
    service = analysis_service(tmp_path, FakeChangeFormer())
    before = np.zeros((32, 40, 3), dtype=np.uint8)
    after = np.full_like(before, 80)
    _, stats, evidence, confidence, models, _, learned = service._change(before, after, "analysis", None)
    assert learned is True
    assert stats["change_threshold"] == 0.5
    assert stats["changed_area_percent"] == 25.0
    assert confidence.type == "mixed"
    assert models[0].startswith("ChangeFormer V6")
    assert "Official ChangeFormer" in evidence[0].description
    assert any(item.type == "polygon" for item in evidence)


def test_real_mode_rejects_change_when_checkpoint_is_missing(tmp_path: Path):
    service = analysis_service(tmp_path)
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    with pytest.raises(SatQueryError) as error:
        service._change(image, image, "analysis", None)
    assert error.value.code == "MODEL_UNAVAILABLE"
