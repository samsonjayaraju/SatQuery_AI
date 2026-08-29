from __future__ import annotations

from collections import Counter

import numpy as np

from app.api.schemas.analysis import RasterMetadata, TransitionItem


LAND_COVER_CLASSES = ("water", "vegetation", "built_up", "bare_land", "agriculture")


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    return np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (shape[1], shape[0]), Image.Resampling.NEAREST
        )
    ) >= 128


def _pixel_area_square_metres(metadata: RasterMetadata | None) -> float | None:
    """Return projected pixel area in square metres, never degrees squared."""
    if metadata is None or not metadata.georeferenced or not metadata.crs or not metadata.transform:
        return None
    try:
        from rasterio.crs import CRS

        crs = CRS.from_user_input(metadata.crs)
        if not crs.is_projected:
            return None
        a, b, _, d, e, _ = metadata.transform[:6]
        native_area = abs(float(a) * float(e) - float(b) * float(d))
        units = crs.linear_units_factor
        factor = float(units[1] if isinstance(units, tuple) else units)
        return native_area * factor * factor
    except (ImportError, TypeError, ValueError):
        return None


def land_cover_transitions(
    before: dict[str, np.ndarray],
    after: dict[str, np.ndarray],
    valid_mask: np.ndarray | None,
    metadata: RasterMetadata | None,
) -> tuple[list[TransitionItem], np.ndarray, np.ndarray]:
    """Build an auditable, pixel-derived transition matrix over the valid overlap."""
    labels = [label for label in LAND_COVER_CLASSES if label in before and label in after]
    before_stack = np.stack([before[label] for label in labels])
    after_stack = np.stack([after[label] for label in labels])
    before_index = np.argmax(before_stack, axis=0)
    after_index = np.argmax(after_stack, axis=0)
    valid = np.ones(before_index.shape, dtype=bool) if valid_mask is None else valid_mask.astype(bool)
    if valid.shape != before_index.shape:
        valid = _resize_mask(valid, before_index.shape)

    valid_count = int(valid.sum())
    transition_index = np.full(before_index.shape, -1, dtype=np.int16)
    semantic_change = valid & (before_index != after_index)
    transition_index[semantic_change] = after_index[semantic_change]
    pixel_area = _pixel_area_square_metres(metadata)

    counts: Counter[tuple[str, str]] = Counter()
    if valid_count:
        same = int(np.count_nonzero(valid & ~semantic_change))
        counts[("unchanged", "unchanged")] = same
        pairs = zip(before_index[semantic_change].tolist(), after_index[semantic_change].tolist())
        counts.update((labels[source], labels[target]) for source, target in pairs)

    transitions: list[TransitionItem] = []
    for (source, target), count in sorted(
        counts.items(), key=lambda item: (item[0] != ("unchanged", "unchanged"), -item[1])
    ):
        if count <= 0:
            continue
        percent = round(count / max(valid_count, 1) * 100, 2)
        area = count * pixel_area if pixel_area is not None else None
        transitions.append(
            TransitionItem(
                from_class=source,
                to_class=target,
                percent=percent,
                pixel_count=count,
                area_square_metres=round(area, 2) if area is not None else None,
                area_hectares=round(area / 10_000, 4) if area is not None else None,
                area_square_kilometres=round(area / 1_000_000, 6) if area is not None else None,
            )
        )
    return transitions, transition_index, valid


def spatial_change_statistics(
    probability: np.ndarray,
    threshold: float,
    valid_mask: np.ndarray | None,
    metadata: RasterMetadata | None,
) -> dict[str, float | int | str]:
    valid = np.ones(probability.shape, dtype=bool) if valid_mask is None else valid_mask.astype(bool)
    if valid.shape != probability.shape:
        valid = _resize_mask(valid, probability.shape)
    changed = (probability >= threshold) & valid
    valid_count = int(valid.sum())
    changed_count = int(changed.sum())
    result: dict[str, float | int | str] = {
        "valid_pixel_count": valid_count,
        "changed_pixel_count": changed_count,
        "area_measurement_basis": "projected_geospatial_grid" if _pixel_area_square_metres(metadata) else "pixel_space",
    }
    if not changed_count:
        result["largest_change_location"] = "No contiguous region passed the change threshold."
        return result

    from skimage.measure import label, regionprops

    components = label(changed, connectivity=2)
    region = max(regionprops(components), key=lambda item: item.area)
    row, column = region.centroid
    if metadata and metadata.georeferenced and metadata.transform:
        try:
            from rasterio.transform import Affine

            x, y = Affine(*metadata.transform[:6]) * (column + 0.5, row + 0.5)
            result["largest_change_location"] = f"projected coordinate ({x:.3f}, {y:.3f})"
            min_row, min_column, max_row, max_column = region.bbox
            x0, y0 = Affine(*metadata.transform[:6]) * (min_column, min_row)
            x1, y1 = Affine(*metadata.transform[:6]) * (max_column, max_row)
            result["largest_change_bounds"] = f"({x0:.3f}, {y0:.3f}) to ({x1:.3f}, {y1:.3f})"
        except (ImportError, TypeError, ValueError):
            result["largest_change_location"] = _image_region(row, column, probability.shape)
    else:
        result["largest_change_location"] = _image_region(row, column, probability.shape)

    pixel_area = _pixel_area_square_metres(metadata)
    if pixel_area is not None:
        area = changed_count * pixel_area
        result.update(
            changed_area_square_metres=round(area, 2),
            changed_area_hectares=round(area / 10_000, 4),
            changed_area_square_kilometres=round(area / 1_000_000, 6),
        )
    return result


def _image_region(row: float, column: float, shape: tuple[int, int]) -> str:
    vertical = ("upper", "central", "lower")[min(2, int(3 * row / max(shape[0], 1)))]
    horizontal = ("left", "centre", "right")[min(2, int(3 * column / max(shape[1], 1)))]
    if vertical == "central" and horizontal == "centre":
        return "centre of the image"
    return f"{vertical}-{horizontal} image region"
