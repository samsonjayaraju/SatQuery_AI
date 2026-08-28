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
