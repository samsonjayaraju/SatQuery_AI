from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


COLORS = {
    "water": (55, 184, 255),
    "vegetation": (91, 220, 145),
    "built_up": (251, 177, 80),
    "change": (245, 95, 114),
    "fused": (82, 224, 196),
}


def save_image(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(path)


def overlay_mask(image: np.ndarray, probability: np.ndarray, path: Path, label: str, threshold: float = 0.58) -> None:
    base = image.astype(np.float32)
    mask = probability >= threshold
    color = np.array(COLORS.get(label, COLORS["fused"]), dtype=np.float32)
    base[mask] = base[mask] * 0.45 + color * 0.55
    save_image(np.clip(base, 0, 255), path)


def heatmap(probability: np.ndarray, path: Path, label: str = "change") -> None:
    color = np.array(COLORS.get(label, COLORS["change"]), dtype=np.float32)
    output = np.zeros((*probability.shape, 4), dtype=np.uint8)
    output[..., :3] = color.astype(np.uint8)
    output[..., 3] = np.clip(probability * 205, 0, 205).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output).save(path)


def bounding_box(probability: np.ndarray, threshold: float = 0.58) -> list[list[float]] | None:
    points = np.argwhere(probability >= threshold)
    if not len(points):
        return None
    y0, x0 = points.min(axis=0)
    y1, x1 = points.max(axis=0)
    height, width = probability.shape
    return [
        [round(x0 / width, 4), round(y0 / height, 4)],
        [round(x1 / width, 4), round(y1 / height, 4)],
    ]


def retain_largest_component(probability: np.ndarray, threshold: float) -> np.ndarray:
    from skimage.measure import label

    components = label(probability >= threshold, connectivity=2)
    identifiers, counts = np.unique(components[components > 0], return_counts=True)
    if not len(identifiers):
        return probability
    largest = identifiers[int(counts.argmax())]
    return np.where(components == largest, probability, 0.0).astype(np.float32)


def largest_polygon(
    probability: np.ndarray,
    threshold: float = 0.58,
    max_points: int = 48,
) -> list[list[float]] | None:
    from skimage.measure import find_contours

    mask = probability >= threshold
    if not mask.any():
        return None
    contours = find_contours(np.pad(mask.astype(np.uint8), 1), 0.5)
    if not contours:
        return None
    contour = max(contours, key=len) - 1
    if len(contour) < 3:
        return None
    step = max(1, int(np.ceil(len(contour) / max_points)))
    sampled = contour[::step]
    height, width = probability.shape
    coordinates = [
        [
            round(float(np.clip(x, 0, width - 1)) / max(width - 1, 1), 4),
            round(float(np.clip(y, 0, height - 1)) / max(height - 1, 1), 4),
        ]
        for y, x in sampled
    ]
    if coordinates[0] != coordinates[-1]:
        coordinates.append(coordinates[0])
    return coordinates


def draw_box(image: np.ndarray, coordinates: list[list[float]], path: Path, label: str) -> None:
    canvas = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    (x0, y0), (x1, y1) = coordinates
    box = (int(x0 * width), int(y0 * height), int(x1 * width), int(y1 * height))
    draw.rectangle(box, outline=COLORS.get(label, COLORS["fused"]), width=max(2, width // 180))
    draw.rectangle((box[0], box[1], min(box[0] + 130, width), box[1] + 24), fill=(5, 15, 22))
    draw.text((box[0] + 6, box[1] + 5), label.replace("_", " ").upper(), fill=(235, 247, 247))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
