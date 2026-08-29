from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.remote_sensing.preprocessing import load_visual, resize_like, stretch_bands


@dataclass(frozen=True)
class AlignmentResult:
    first: np.ndarray
    second: np.ndarray
    method: str
    reprojected: bool


def align_visual_pair(first_path: Path, second_path: Path) -> AlignmentResult:
    """Align the second raster to the first grid, using geospatial reprojection when possible."""
    first = load_visual(first_path)
    if first_path.suffix.lower() in {".tif", ".tiff"} and second_path.suffix.lower() in {".tif", ".tiff"}:
        try:
            import rasterio
            from rasterio.warp import Resampling, reproject

            with rasterio.open(first_path) as reference, rasterio.open(second_path) as source:
                if reference.crs and source.crs:
                    indexes = list(range(1, min(source.count, 3) + 1))
                    destination = np.zeros((len(indexes), reference.height, reference.width), dtype=np.float32)
                    for destination_index, source_index in enumerate(indexes):
                        reproject(
                            source=rasterio.band(source, source_index),
                            destination=destination[destination_index],
                            src_transform=source.transform,
                            src_crs=source.crs,
                            dst_transform=reference.transform,
                            dst_crs=reference.crs,
                            resampling=Resampling.bilinear,
                            dst_nodata=np.nan,
                        )
                    second = stretch_bands(destination)
                    return AlignmentResult(first, second, "geospatial_reprojection", True)
        except (ImportError, OSError, ValueError):
            pass
    unaligned = load_visual(second_path)
    native_match = unaligned.shape[:2] == first.shape[:2]
    second = resize_like(unaligned, first)
    method = "native_pixel_grid" if native_match else "pixel_space_resize"
    return AlignmentResult(first, second, method, False)
