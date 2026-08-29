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


class FakeZeroChangeFormer:
    available = True
    loaded = False

    @staticmethod
    def predict(before: np.ndarray, after: np.ndarray) -> np.ndarray:
        return np.zeros(before.shape[:2], dtype=np.float32)


class FakeLandcover:
    available = True
    adapter_available = False

    @staticmethod
    def landcover_probabilities(image: np.ndarray):
        pixels = image.astype(np.float32) / 255.0
        water = (pixels[..., 2] > pixels[..., 0] + 0.2).astype(np.float32)
        bare = 1.0 - water
        zeros = np.zeros_like(water)
        return {
            "water": water,
            "vegetation": zeros,
            "built_up": zeros,
            "bare_land": bare,
            "agriculture": zeros,
        }, "Synthetic land-cover model"


def analysis_service(tmp_path: Path, changeformer=None, remoteclip=None) -> AnalysisService:
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
        remoteclip=remoteclip,
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


def test_normalized_landcover_percentages_sum_to_one_hundred(tmp_path: Path):
    service = analysis_service(tmp_path)
    probabilities = {
        "water": np.full((6, 8), 0.8, dtype=np.float32),
        "vegetation": np.full((6, 8), 0.6, dtype=np.float32),
        "built_up": np.full((6, 8), 0.5, dtype=np.float32),
        "bare_land": np.full((6, 8), 0.1, dtype=np.float32),
        "agriculture": np.full((6, 8), 0.2, dtype=np.float32),
    }
    normalized = service._normalize_landcover(probabilities)
    percentages = [service._soft_percentage(value) for value in normalized.values()]
    assert sum(percentages) == pytest.approx(100.0, abs=0.05)


def test_learned_change_path_reports_real_changeformer(tmp_path: Path):
    service = analysis_service(tmp_path, FakeChangeFormer())
    before = np.zeros((32, 40, 3), dtype=np.uint8)
    after = np.full_like(before, 80)
    _, stats, evidence, confidence, models, _, learned, transitions = service._change(before, after, "analysis", None)
    assert learned is True
    assert stats["change_threshold"] == 0.5
    assert stats["changed_area_percent"] == 25.0
    assert confidence.type == "mixed"
    assert models[0].startswith("ChangeFormer V6")
    assert "ChangeFormer V6" in evidence[0].description
    assert any(item.type == "polygon" for item in evidence)
    assert any(item.type == "transition" for item in evidence)
    assert transitions


def test_hybrid_change_detects_water_recession_when_changeformer_is_silent(tmp_path: Path):
    service = analysis_service(tmp_path, FakeZeroChangeFormer(), FakeLandcover())
    before = np.full((100, 100, 3), (180, 160, 100), dtype=np.uint8)
    after = before.copy()
    before[20:80, 20:80] = (10, 40, 130)
    after[20:50, 20:50] = (10, 40, 130)

    answer, stats, _, confidence, models, _, learned, transitions = service._change(
        before,
        after,
        "water-loss",
        None,
        input_quality=0.54,
        alignment_method="pixel_space_resize",
    )

    assert learned is True
    assert stats["structural_change_percent"] == 0.0
    assert 20.0 < stats["changed_area_percent"] < 35.0
    assert stats["water_change_pp"] == pytest.approx(-27.0, abs=0.1)
    assert stats["change_method"] == "environmental_semantic_hybrid"
    assert "water decreased" in answer
    assert "approximate" in answer
    assert confidence.overall < 0.74
    assert "Environmental Semantic Change" in models[0]
    assert any(item.from_class == "water" and item.to_class == "bare_land" for item in transitions)


def test_real_mode_rejects_change_when_checkpoint_is_missing(tmp_path: Path):
    service = analysis_service(tmp_path)
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    with pytest.raises(SatQueryError) as error:
        service._change(image, image, "analysis", None)
    assert error.value.code == "MODEL_UNAVAILABLE"
