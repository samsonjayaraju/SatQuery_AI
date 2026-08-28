from pathlib import Path

from PIL import Image
import numpy as np
import rasterio
from rasterio.transform import from_origin

from app.remote_sensing.input_inspector import inspect_inputs


def image(path: Path, size=(80, 60), color=(40, 100, 160)) -> Path:
    Image.new("RGB", size, color).save(path)
    return path


def test_inspects_benchmark_png(tmp_path):
    response = inspect_inputs([image(tmp_path / "scene.png")], "single")
    assert response.valid
    assert response.images[0].width == 80
    assert response.images[0].band_count == 3
    assert response.images[0].georeferenced is False


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
