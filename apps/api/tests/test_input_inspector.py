from pathlib import Path

from PIL import Image
import numpy as np
import rasterio
from rasterio.transform import from_origin

from app.remote_sensing.input_inspector import inspect_inputs
from app.remote_sensing.alignment import align_visual_pair


def image(path: Path, size=(80, 60), color=(40, 100, 160)) -> Path:
    Image.new("RGB", size, color).save(path)
    return path


def test_inspects_benchmark_png(tmp_path):
    response = inspect_inputs([image(tmp_path / "scene.png")], "single")
    assert response.valid
    assert response.images[0].width == 80
    assert response.images[0].band_count == 3
    assert response.images[0].georeferenced is False


def test_rejects_multi_panel_document_figure(tmp_path):
    generator = np.random.default_rng(7)
    canvas = np.full((420, 420, 3), 255, dtype=np.uint8)
    for row in range(3):
        for column in range(3):
            top = 16 + row * 136
            left = 16 + column * 136
            canvas[top : top + 116, left : left + 116] = generator.integers(
                20, 220, size=(116, 116, 3), dtype=np.uint8
            )
    path = tmp_path / "paper-figure.png"
    Image.fromarray(canvas).save(path)

    response = inspect_inputs([path], "single")

    assert response.valid is False
    assert response.visual_quality.status == "unsupported"
    assert "composite_figure" in response.visual_quality.flags
    assert "original satellite panel" in response.warnings[0]


def test_pixel_space_pair_compatibility(tmp_path):
    first = image(tmp_path / "t1.png")
    second = image(tmp_path / "t2.png")
    response = inspect_inputs([first, second], "bi_temporal")
    assert response.compatibility.dimensions_match is True
    assert response.compatibility.co_registered is True
    assert response.warnings


def test_geotiff_metadata_and_georeferencing(tmp_path):
    path = tmp_path / "sentinel-2.tif"
    with rasterio.open(path, "w", driver="GTiff", width=16, height=12, count=4, dtype="uint16", crs="EPSG:4326", transform=from_origin(77.0, 13.0, 0.0001, 0.0001)) as dataset:
        dataset.write(np.ones((4, 12, 16), dtype=np.uint16) * 1200)
    response = inspect_inputs([path], "single")
    metadata = response.images[0]
    assert metadata.georeferenced is True
    assert metadata.crs == "EPSG:4326"
    assert metadata.band_count == 4
    assert metadata.pixel_resolution == [0.0001, 0.0001]


def test_feature_registration_returns_measured_transform(tmp_path):
    generator = np.random.default_rng(42)
    first_array = generator.integers(0, 256, size=(240, 260, 3), dtype=np.uint8)
    second_array = np.zeros_like(first_array)
    second_array[12:, 18:] = first_array[:-12, :-18]
    first = tmp_path / "before.png"
    second = tmp_path / "after.png"
    Image.fromarray(first_array).save(first)
    Image.fromarray(second_array).save(second)

    result = align_visual_pair(first, second)

    assert result.method == "feature_matching"
    assert result.confidence >= 0.55
    assert result.transform is not None
    assert result.valid_mask.mean() > 0.8
