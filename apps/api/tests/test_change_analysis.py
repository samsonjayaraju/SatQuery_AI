from __future__ import annotations

import numpy as np

from app.api.schemas.analysis import RasterMetadata
from app.remote_sensing.change_analysis import land_cover_transitions, spatial_change_statistics


def probabilities(labels: np.ndarray) -> dict[str, np.ndarray]:
    names = ("water", "vegetation", "built_up", "bare_land", "agriculture")
    return {name: (labels == index).astype(np.float32) for index, name in enumerate(names)}


def projected_metadata() -> RasterMetadata:
    return RasterMetadata(
        filename="scene.tif", file_size_bytes=100, width=4, height=3, band_count=3,
        data_type="uint8", crs="EPSG:32643", transform=[10, 0, 500000, 0, -10, 1400000],
        bounds=[500000, 1399970, 500040, 1400000], pixel_resolution=[10, 10], nodata=None,
        georeferenced=True, modality="optical", format="GeoTIFF",
    )


def test_transition_matrix_and_projected_area_are_pixel_derived():
    before = np.array([[0, 0, 1, 1], [0, 2, 2, 1], [3, 3, 4, 4]])
    after = before.copy()
    after[0, 0] = 3
    after[1, 1] = 1

    transitions, index_map, valid = land_cover_transitions(
        probabilities(before), probabilities(after), np.ones_like(before, dtype=bool), projected_metadata()
    )

    lookup = {(item.from_class, item.to_class): item for item in transitions}
    assert lookup[("unchanged", "unchanged")].pixel_count == 10
    assert lookup[("water", "bare_land")].pixel_count == 1
    assert lookup[("water", "bare_land")].area_square_metres == 100
    assert lookup[("built_up", "vegetation")].area_hectares == 0.01
    assert np.count_nonzero(index_map >= 0) == 2
    assert valid.all()


def test_pixel_space_statistics_never_invent_real_world_area():
    probability = np.zeros((10, 10), dtype=np.float32)
    probability[:4, :4] = 0.9
    stats = spatial_change_statistics(probability, 0.5, None, None)
    assert stats["changed_pixel_count"] == 16
    assert stats["area_measurement_basis"] == "pixel_space"
    assert "changed_area_hectares" not in stats
    assert "image region" in str(stats["largest_change_location"])
