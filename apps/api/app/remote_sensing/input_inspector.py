from __future__ import annotations

import math
from pathlib import Path

from PIL import Image

from app.api.schemas.analysis import Compatibility, InspectionResponse, RasterMetadata
from app.core.exceptions import SatQueryError

SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def _infer_modality(path: Path, band_count: int) -> str:
    value = path.name.lower()
    if any(token in value for token in ("sar", "sentinel-1", "s1_", "_vv", "_vh")):
        return "sar"
    if band_count == 1:
        return "sar"
    if band_count > 4:
        return "multispectral"
    return "optical"


def inspect_raster(path: Path, public_url: str | None = None) -> RasterMetadata:
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise SatQueryError(
            "UNSUPPORTED_FILE_TYPE",
            f"{path.name} is unsupported. Use GeoTIFF, TIFF, PNG, JPEG or JPG.",
            415,
        )

    try:
        if path.suffix.lower() in {".tif", ".tiff"}:
            try:
                import rasterio

                with rasterio.open(path) as dataset:
                    transform = list(dataset.transform)[:6]
                    bounds = list(dataset.bounds)
                    res = list(dataset.res)
                    crs = str(dataset.crs) if dataset.crs else None
                    return RasterMetadata(
                        filename=path.name,
                        file_size_bytes=path.stat().st_size,
                        width=dataset.width,
                        height=dataset.height,
                        band_count=dataset.count,
                        data_type=", ".join(dataset.dtypes),
                        crs=crs,
                        transform=transform,
                        bounds=bounds,
                        pixel_resolution=res,
                        nodata=dataset.nodata,
                        georeferenced=crs is not None,
                        modality=_infer_modality(path, dataset.count),
                        format="GeoTIFF" if crs else "TIFF",
                        thumbnail_url=public_url,
                    )
            except ImportError:
                pass

        with Image.open(path) as image:
            bands = len(image.getbands())
            return RasterMetadata(
                filename=path.name,
                file_size_bytes=path.stat().st_size,
                width=image.width,
                height=image.height,
                band_count=bands,
                data_type=image.mode,
                georeferenced=False,
                modality=_infer_modality(path, bands),
                format=image.format or path.suffix[1:].upper(),
                thumbnail_url=public_url,
            )
    except SatQueryError:
        raise
    except Exception as exc:
        raise SatQueryError("INVALID_RASTER", f"{path.name} could not be read as a raster image.") from exc


def _intersection_over_min_area(a: list[float], b: list[float]) -> float:
    left, bottom = max(a[0], b[0]), max(a[1], b[1])
    right, top = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, top - bottom)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    minimum = min(area_a, area_b)
    return intersection / minimum if minimum else 0.0


def inspect_inputs(paths: list[Path], mode: str, urls: list[str] | None = None) -> InspectionResponse:
    expected = 1 if mode == "single" else 2
    if mode not in {"single", "bi_temporal", "cross_modal"}:
        raise SatQueryError("INVALID_INPUT_MODE", "Input mode must be single, bi_temporal or cross_modal.")
    if len(paths) != expected:
        raise SatQueryError("WRONG_FILE_COUNT", f"{mode} mode requires exactly {expected} image(s).")

    urls = urls or [None] * len(paths)
    images = [inspect_raster(path, urls[index]) for index, path in enumerate(paths)]
    compatibility = Compatibility()
    warnings: list[str] = []
    if len(images) == 2:
        first, second = images
        compatibility.dimensions_match = first.width == second.width and first.height == second.height
        compatibility.crs_match = bool(first.crs and second.crs and first.crs == second.crs)
        compatibility.resolution_compatible = bool(
            first.pixel_resolution
            and second.pixel_resolution
            and all(
                math.isclose(a, b, rel_tol=0.1)
                for a, b in zip(first.pixel_resolution, second.pixel_resolution)
            )
        )
        comparison_bounds = second.bounds
        if first.bounds and second.bounds and first.crs and second.crs and not compatibility.crs_match:
            try:
                from rasterio.warp import transform_bounds

                comparison_bounds = list(transform_bounds(second.crs, first.crs, *second.bounds))
                warnings.append("Coordinate systems differ; the second raster will be reprojected onto the first grid.")
            except (ImportError, ValueError):
                comparison_bounds = None
                warnings.append("Coordinate systems differ and overlap could not be precomputed; reprojection will be attempted during analysis.")
        if first.bounds and comparison_bounds and first.crs and second.crs:
            compatibility.overlap = round(_intersection_over_min_area(first.bounds, comparison_bounds), 4)
            compatibility.co_registered = bool(
                compatibility.overlap >= 0.9
                and (compatibility.resolution_compatible or not compatibility.crs_match)
            )
        else:
            compatibility.overlap = None
            compatibility.co_registered = compatibility.dimensions_match
            warnings.append("Pair has no shared georeferencing; pixel-space alignment will be used.")
        if not compatibility.dimensions_match:
            warnings.append("Dimensions differ; the second image will be aligned to the first in pixel space.")
        if compatibility.overlap is not None and compatibility.overlap < 0.1:
            raise SatQueryError("NO_SPATIAL_OVERLAP", "Paired GeoTIFF files do not overlap sufficiently.")
        if mode == "cross_modal":
            if first.modality == "sar":
                raise SatQueryError("UNSUPPORTED_MODALITY", "The first cross-modal input must be optical or multispectral; place SAR second.")
            if second.modality != "sar":
                warnings.append("The second file was not confidently identified as SAR; slot assignment will be used as the modality declaration.")
        if mode == "bi_temporal" and first.modality != second.modality:
            warnings.append("Temporal inputs appear to use different modalities; change evidence may be unreliable.")
        compatibility.warnings = warnings.copy()

    return InspectionResponse(
        valid=True,
        input_mode=mode,
        images=images,
        compatibility=compatibility,
        warnings=warnings,
    )
