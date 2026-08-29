from __future__ import annotations

import numpy as np
from PIL import Image

from app.remote_sensing.alignment import align_visual_pair
from app.remote_sensing.input_inspector import inspect_inputs
from app.remote_sensing.tiling import tile_windows, tiled_dict_predict, tiled_predict


def test_tile_windows_cover_full_extent_and_stitch_exactly():
    image = np.arange(777 * 911 * 3, dtype=np.float32).reshape(777, 911, 3)
    windows = tile_windows(777, 911, tile_size=256, overlap=48)
    assert windows[0].top == 0 and windows[0].left == 0
    assert max(window.top + window.height for window in windows) == 777
    assert max(window.left + window.width for window in windows) == 911

    expected = image.mean(axis=2)
    stitched = tiled_predict(image, lambda tile: tile.mean(axis=2), tile_size=256, overlap=48)
    assert np.allclose(stitched, expected)


def test_dictionary_predictions_are_stitched_in_one_pass():
    image = np.full((620, 730, 3), 12, dtype=np.uint8)
    result = tiled_dict_predict(
        image,
        lambda tile: {"mean": tile.mean(axis=2), "maximum": tile.max(axis=2)},
        tile_size=256,
        overlap=32,
    )
    assert set(result) == {"mean", "maximum"}
    assert np.allclose(result["mean"], 12)
    assert np.allclose(result["maximum"], 12)


def test_pixel_alignment_resizes_second_image(tmp_path):
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    Image.new("RGB", (120, 80), (10, 20, 30)).save(first_path)
    Image.new("RGB", (60, 40), (40, 50, 60)).save(second_path)
    aligned = align_visual_pair(first_path, second_path)
    assert aligned.first.shape == aligned.second.shape == (80, 120, 3)
    assert aligned.method == "pixel_space_resize"
    assert aligned.reprojected is False


def test_georeferenced_pair_is_reprojected_to_reference_grid(tmp_path):
    import rasterio
    from rasterio.transform import from_bounds

    first_path = tmp_path / "first.tif"
    second_path = tmp_path / "second.tif"
    with rasterio.open(
        first_path,
        "w",
        driver="GTiff",
        width=32,
        height=24,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_bounds(0, 0, 1, 1, 32, 24),
    ) as dataset:
        dataset.write(np.full((3, 24, 32), 70, dtype=np.uint8))
    with rasterio.open(
        second_path,
        "w",
        driver="GTiff",
        width=20,
        height=18,
        count=3,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_bounds(0, 0, 111319, 111325, 20, 18),
    ) as dataset:
        dataset.write(np.full((3, 18, 20), 110, dtype=np.uint8))

    inspection = inspect_inputs([first_path, second_path], "bi_temporal")
    aligned = align_visual_pair(first_path, second_path)
    assert inspection.compatibility.crs_match is False
    assert inspection.compatibility.overlap is not None and inspection.compatibility.overlap > 0.95
    assert aligned.method == "geospatial_reprojection"
    assert aligned.reprojected is True
    assert aligned.first.shape == aligned.second.shape == (24, 32, 3)
