from __future__ import annotations

import numpy as np


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denominator = a.astype(np.float32) + b.astype(np.float32)
    return np.divide(
        a.astype(np.float32) - b.astype(np.float32),
        denominator,
        out=np.zeros_like(denominator, dtype=np.float32),
        where=np.abs(denominator) > 1e-6,
    )


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    return normalized_difference(nir, red)


def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return normalized_difference(green, nir)


def ndbi(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return normalized_difference(swir, nir)
