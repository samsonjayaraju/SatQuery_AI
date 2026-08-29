from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TileWindow:
    top: int
    left: int
    height: int
    width: int

    @property
    def rows(self) -> slice:
        return slice(self.top, self.top + self.height)

    @property
    def columns(self) -> slice:
        return slice(self.left, self.left + self.width)


def _origins(length: int, tile_size: int, overlap: int) -> list[int]:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be non-negative and smaller than tile_size")
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    origins = list(range(0, max(1, length - tile_size + 1), stride))
    final = length - tile_size
    if origins[-1] != final:
        origins.append(final)
    return origins


def tile_windows(height: int, width: int, tile_size: int = 512, overlap: int = 64) -> list[TileWindow]:
    return [
        TileWindow(top, left, min(tile_size, height - top), min(tile_size, width - left))
        for top in _origins(height, tile_size, overlap)
        for left in _origins(width, tile_size, overlap)
    ]


def tiled_predict(
    image: np.ndarray,
    predictor: Callable[[np.ndarray], np.ndarray],
    tile_size: int = 512,
    overlap: int = 64,
) -> np.ndarray:
    """Run a spatial predictor per tile and blend overlaps back to image coordinates."""
    height, width = image.shape[:2]
    total: np.ndarray | None = None
    weights = np.zeros((height, width), dtype=np.float32)
    for window in tile_windows(height, width, tile_size, overlap):
        prediction = np.asarray(predictor(image[window.rows, window.columns]), dtype=np.float32)
        expected = (window.height, window.width)
        if prediction.shape[:2] != expected:
            raise ValueError(f"predictor returned {prediction.shape[:2]}, expected {expected}")
        if total is None:
            total = np.zeros((height, width, *prediction.shape[2:]), dtype=np.float32)
        total[window.rows, window.columns] += prediction
        weights[window.rows, window.columns] += 1.0
    if total is None:
        raise ValueError("cannot tile an empty image")
    divisor = weights if total.ndim == 2 else weights[(...,) + (None,) * (total.ndim - 2)]
    return total / np.maximum(divisor, 1e-6)


def tiled_dict_predict(
    image: np.ndarray,
    predictor: Callable[[np.ndarray], dict[str, np.ndarray]],
    tile_size: int = 512,
    overlap: int = 64,
) -> dict[str, np.ndarray]:
    height, width = image.shape[:2]
    totals: dict[str, np.ndarray] = {}
    weights = np.zeros((height, width), dtype=np.float32)
    for window in tile_windows(height, width, tile_size, overlap):
        outputs = predictor(image[window.rows, window.columns])
        for key, value in outputs.items():
            prediction = np.asarray(value, dtype=np.float32)
            if prediction.shape[:2] != (window.height, window.width):
                raise ValueError(f"predictor returned {prediction.shape[:2]}, expected {(window.height, window.width)}")
            if key not in totals:
                totals[key] = np.zeros((height, width, *prediction.shape[2:]), dtype=np.float32)
            totals[key][window.rows, window.columns] += prediction
        weights[window.rows, window.columns] += 1.0
    if not totals:
        raise ValueError("predictor returned no outputs")
    return {
        key: value / np.maximum(
            weights if value.ndim == 2 else weights[(...,) + (None,) * (value.ndim - 2)], 1e-6
        )
        for key, value in totals.items()
    }
