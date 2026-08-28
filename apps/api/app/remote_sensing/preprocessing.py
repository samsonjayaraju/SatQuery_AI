from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_visual(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image: Image.Image
    if path.suffix.lower() in {".tif", ".tiff"}:
        try:
            import rasterio

            with rasterio.open(path) as dataset:
                indexes = list(range(1, min(dataset.count, 3) + 1))
                data = dataset.read(indexes).astype(np.float32)
                channels = []
                for band in data:
                    valid = band[np.isfinite(band)]
                    if valid.size:
                        low, high = np.percentile(valid, [2, 98])
                        scaled = np.clip((band - low) / max(high - low, 1e-6), 0, 1)
                    else:
                        scaled = np.zeros_like(band)
                    channels.append((scaled * 255).astype(np.uint8))
                while len(channels) < 3:
                    channels.append(channels[-1])
                image = Image.fromarray(np.stack(channels[:3], axis=-1), "RGB")
        except (ImportError, Exception):
            image = Image.open(path).convert("RGB")
    else:
        image = Image.open(path).convert("RGB")
    if size:
        image = image.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def resize_like(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if image.shape[:2] == reference.shape[:2]:
        return image
    return np.asarray(
        Image.fromarray(image).resize((reference.shape[1], reference.shape[0]), Image.Resampling.BILINEAR)
    )


def optical_probabilities(image: np.ndarray) -> dict[str, np.ndarray]:
    rgb = image.astype(np.float32) / 255.0
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    brightness = rgb.mean(axis=2)
    saturation = rgb.max(axis=2) - rgb.min(axis=2)
    water = np.clip((blue - (red + green) / 2) * 2.2 + (0.48 - brightness) * 0.35 + 0.35, 0, 1)
    vegetation = np.clip((green - (red + blue) / 2) * 2.5 + 0.35, 0, 1)
    built = np.clip(brightness * 0.75 + (0.28 - saturation) * 1.15 - vegetation * 0.35, 0, 1)
    bare = np.clip((red - blue) * 1.2 + brightness * 0.45 - vegetation * 0.25, 0, 1)
    agriculture = np.clip(vegetation * (0.8 - saturation * 0.25) + 0.08, 0, 1)
    return {
        "water": water,
        "vegetation": vegetation,
        "built_up": built,
        "bare_land": bare,
        "agriculture": agriculture,
    }


def sar_probabilities(image: np.ndarray) -> dict[str, np.ndarray]:
    intensity = image.astype(np.float32).mean(axis=2) / 255.0
    water = np.clip((0.48 - intensity) * 1.8 + 0.35, 0, 1)
    built = np.clip((intensity - 0.42) * 1.8 + 0.35, 0, 1)
    texture = np.abs(intensity - np.roll(intensity, 1, axis=0)) + np.abs(
        intensity - np.roll(intensity, 1, axis=1)
    )
    return {"water": water, "built_up": built, "texture": np.clip(texture * 2, 0, 1)}
