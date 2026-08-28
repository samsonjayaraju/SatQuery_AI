from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1] / "demo"
WIDTH, HEIGHT = 960, 620


def optical_scene(seed: int, developed: bool = False) -> Image.Image:
    rng = np.random.default_rng(seed)
    array = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
    array[:] = (58, 78, 54)
    tile_w, tile_h = 120, 95
    palette = [(76, 111, 62), (108, 125, 69), (119, 103, 63), (68, 101, 58), (136, 123, 79)]
    for y in range(0, HEIGHT, tile_h):
        for x in range(0, WIDTH, tile_w):
            color = np.array(palette[(x // tile_w + y // tile_h + seed) % len(palette)], dtype=np.float32)
            jitter = rng.normal(0, 4, (min(tile_h - 4, HEIGHT - y), min(tile_w - 4, WIDTH - x), 1))
            array[y : min(y + tile_h - 4, HEIGHT), x : min(x + tile_w - 4, WIDTH)] = color + jitter
    image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(.45))
    draw = ImageDraw.Draw(image)
    river = [(0, 475), (160, 430), (310, 455), (455, 400), (610, 420), (760, 350), (960, 365)]
    draw.line(river, fill=(35, 92, 127), width=64, joint="curve")
    draw.line(river, fill=(45, 111, 151), width=43, joint="curve")
    draw.line([(90, 0), (230, 190), (430, 270), (710, 300), (960, 275)], fill=(72, 72, 68), width=17)
    draw.line([(90, 0), (230, 190), (430, 270), (710, 300), (960, 275)], fill=(165, 157, 137), width=5)
    blocks = [(610, 65, 885, 230), (410, 305, 600, 420)]
    if developed:
        blocks.extend([(160, 110, 355, 270), (670, 410, 910, 555)])
    for x0, y0, x1, y1 in blocks:
        draw.rectangle((x0, y0, x1, y1), fill=(118, 119, 112))
        for y in range(y0 + 8, y1 - 6, 18):
            for x in range(x0 + 7, x1 - 8, 23):
                draw.rectangle((x, y, x + 13, y + 9), fill=(174, 170, 154))
    return image


def sar_from_optical(optical: Image.Image, seed: int) -> Image.Image:
    source = np.asarray(optical, dtype=np.float32) / 255
    intensity = source.mean(axis=2)
    blue_dominant = source[..., 2] > source[..., 1] * 1.12
    bright = np.clip(intensity * 1.25, 0, 1)
    bright[blue_dominant] *= .18
    rng = np.random.default_rng(seed)
    speckle = rng.gamma(shape=2.5, scale=.4, size=bright.shape)
    sar = np.clip(bright * speckle, 0, 1)
    sar = np.stack([sar * .78, sar * .9, sar], axis=-1)
    return Image.fromarray((sar * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(.35))


def save(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, optimize=True)


def main() -> None:
    before = optical_scene(8, developed=False)
    after = optical_scene(8, developed=True)
    optical = optical_scene(17, developed=True)
    save(optical_scene(3, developed=False), ROOT / "single" / "agriculture-river.png")
    save(before, ROOT / "change" / "t1.png")
    save(after, ROOT / "change" / "t2.png")
    save(optical, ROOT / "cross_modal" / "optical.png")
    save(sar_from_optical(optical, 19), ROOT / "cross_modal" / "sar.png")
    print(f"Generated deterministic demo imagery under {ROOT}")


if __name__ == "__main__":
    main()
