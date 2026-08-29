from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np

from app.remote_sensing.preprocessing import load_visual, resize_like, stretch_bands


@dataclass(frozen=True)
class AlignmentResult:
    first: np.ndarray
    second: np.ndarray
    method: str
    reprojected: bool
    confidence: float
    transform: list[float] | None
    valid_mask: np.ndarray
    warnings: tuple[str, ...] = ()


def _feature_register(first: np.ndarray, second: np.ndarray, min_confidence: float) -> AlignmentResult | None:
    try:
        from skimage.color import rgb2gray
        from skimage.feature import ORB, match_descriptors
        from skimage.measure import ransac
        from skimage.transform import AffineTransform, resize, warp

        max_side = max(first.shape[:2])
        scale = min(1.0, 768.0 / max_side)
        shape = (max(32, round(first.shape[0] * scale)), max(32, round(first.shape[1] * scale)))
        first_gray = resize(rgb2gray(first.astype(np.float32) / 255.0), shape, anti_aliasing=True)
        second_gray = resize(rgb2gray(second.astype(np.float32) / 255.0), shape, anti_aliasing=True)
        first_orb = ORB(n_keypoints=1500, fast_threshold=0.04)
        second_orb = ORB(n_keypoints=1500, fast_threshold=0.04)
        first_orb.detect_and_extract(first_gray)
        second_orb.detect_and_extract(second_gray)
        matches = match_descriptors(
            first_orb.descriptors,
            second_orb.descriptors,
            cross_check=True,
            max_ratio=0.8,
        )
        if len(matches) < 8:
            return None

        destination = first_orb.keypoints[matches[:, 0]][:, ::-1]
        source = second_orb.keypoints[matches[:, 1]][:, ::-1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            model, inliers = ransac(
                (source, destination),
                AffineTransform,
                min_samples=3,
                residual_threshold=4.0,
                max_trials=1000,
                rng=26167,
            )
        if model is None or inliers is None or int(inliers.sum()) < 6:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            residuals = model.residuals(source[inliers], destination[inliers])
        inlier_ratio = float(inliers.mean())
        residual_score = float(np.exp(-float(np.median(residuals)) / 3.0))
        points = destination[inliers]
        coverage = float(
            max(0.0, np.ptp(points[:, 0])) * max(0.0, np.ptp(points[:, 1])) / max(shape[0] * shape[1], 1)
        )
        coverage_score = min(1.0, coverage / 0.25)
        confidence = float(np.clip(0.55 * inlier_ratio + 0.25 * residual_score + 0.20 * coverage_score, 0, 1))
        if confidence < min_confidence:
            return None

        scaling = np.diag([scale, scale, 1.0])
        full_matrix = np.linalg.inv(scaling) @ model.params @ scaling
        full_model = AffineTransform(matrix=full_matrix)
        with warnings.catch_warnings(), np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            registered = warp(
                second,
                inverse_map=full_model.inverse,
                output_shape=first.shape[:2],
                order=1,
                preserve_range=True,
                mode="constant",
                cval=0,
            )
            valid = warp(
                np.ones(second.shape[:2], dtype=np.uint8),
                inverse_map=full_model.inverse,
                output_shape=first.shape[:2],
                order=0,
                preserve_range=True,
                mode="constant",
                cval=0,
            ) >= 0.5
        return AlignmentResult(
            first=first,
            second=np.clip(registered, 0, 255).astype(np.uint8),
            method="feature_matching",
            reprojected=False,
            confidence=round(confidence, 3),
            transform=[round(float(value), 8) for value in full_matrix.reshape(-1)],
            valid_mask=valid,
        )
    except (ImportError, RuntimeError, ValueError):
        return None


def align_visual_pair(first_path: Path, second_path: Path, min_confidence: float = 0.55) -> AlignmentResult:
    """Align the second raster to the first grid, using geospatial reprojection when possible."""
    first = load_visual(first_path)
    if first_path.suffix.lower() in {".tif", ".tiff"} and second_path.suffix.lower() in {".tif", ".tiff"}:
        try:
            import rasterio
            from rasterio.warp import Resampling, reproject

            with rasterio.open(first_path) as reference, rasterio.open(second_path) as source:
                if reference.crs and source.crs:
                    indexes = list(range(1, min(source.count, 3) + 1))
                    destination = np.full((len(indexes), reference.height, reference.width), np.nan, dtype=np.float32)
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
                    valid = np.all(np.isfinite(destination), axis=0)
                    return AlignmentResult(
                        first,
                        second,
                        "geospatial_reprojection",
                        True,
                        0.98,
                        None,
                        valid,
                    )
        except (ImportError, OSError, ValueError):
            pass
    unaligned = load_visual(second_path)
    native_match = unaligned.shape[:2] == first.shape[:2]
    second = resize_like(unaligned, first)
    registered = _feature_register(first, second, min_confidence)
    if registered is not None:
        return registered
    method = "native_pixel_grid" if native_match else "pixel_space_resize"
    confidence = 0.45 if native_match else 0.35
    warning = (
        "Feature registration was unavailable or below threshold; controlled pixel-space alignment was used."
    )
    return AlignmentResult(
        first,
        second,
        method,
        False,
        confidence,
        None,
        np.ones(first.shape[:2], dtype=bool),
        (warning,),
    )
