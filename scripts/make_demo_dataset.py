from __future__ import annotations

import math
import os
import shutil
import stat
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "demo"
IMAGE_SIZE = 48


def main() -> None:
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR, onexc=clear_readonly)
    for split, count in [("train", 14), ("test", 6)]:
        for class_name in ["cats", "dogs"]:
            folder = DATA_DIR / split / class_name
            folder.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                rng = np.random.default_rng(hash((split, class_name, index)) & 0xFFFF_FFFF)
                image = draw_cat(rng) if class_name == "cats" else draw_dog(rng)
                image.save(folder / f"{class_name[:-1]}_{index:02d}.png")

    print(f"created demo dataset: {DATA_DIR}")


def background(rng: np.random.Generator) -> Image.Image:
    base = int(rng.integers(215, 246))
    array = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), base, dtype=np.uint8)
    noise = rng.normal(0, 8, array.shape)
    array = np.clip(array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(array, "RGB")


def draw_cat(rng: np.random.Generator) -> Image.Image:
    image = background(rng)
    draw = ImageDraw.Draw(image)
    fur = tuple(int(value) for value in rng.integers(55, 130, size=3))
    cx = int(rng.integers(28, 37))
    cy = int(rng.integers(28, 36))
    radius = int(rng.integers(14, 18))

    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=fur)
    draw.polygon([(cx - 13, cy - 11), (cx - 5, cy - 29), (cx + 1, cy - 10)], fill=fur)
    draw.polygon([(cx + 13, cy - 11), (cx + 5, cy - 29), (cx - 1, cy - 10)], fill=fur)
    draw.ellipse([cx - 8, cy - 3, cx - 4, cy + 1], fill=(20, 20, 20))
    draw.ellipse([cx + 4, cy - 3, cx + 8, cy + 1], fill=(20, 20, 20))
    draw.polygon([(cx, cy + 3), (cx - 3, cy + 7), (cx + 3, cy + 7)], fill=(220, 130, 140))

    for side in [-1, 1]:
        for offset in [-3, 2, 7]:
            draw.line([(cx + side * 3, cy + 8), (cx + side * 22, cy + offset)], fill=(35, 35, 35), width=1)

    return image.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0, 0.55))))


def clear_readonly(function, path, exc_info) -> None:
    os.chmod(path, stat.S_IWRITE)
    function(path)


def draw_dog(rng: np.random.Generator) -> Image.Image:
    image = background(rng)
    draw = ImageDraw.Draw(image)
    fur = tuple(int(value) for value in rng.integers(90, 170, size=3))
    cx = int(rng.integers(28, 37))
    cy = int(rng.integers(30, 39))
    head_w = int(rng.integers(28, 34))
    head_h = int(rng.integers(24, 30))

    draw.ellipse([cx - head_w // 2, cy - head_h // 2, cx + head_w // 2, cy + head_h // 2], fill=fur)
    draw.ellipse([cx - 24, cy - 11, cx - 12, cy + 17], fill=fur)
    draw.ellipse([cx + 12, cy - 11, cx + 24, cy + 17], fill=fur)
    draw.ellipse([cx - 9, cy + 1, cx + 9, cy + 14], fill=tuple(min(255, v + 35) for v in fur))
    draw.ellipse([cx - 8, cy - 5, cx - 4, cy - 1], fill=(20, 20, 20))
    draw.ellipse([cx + 4, cy - 5, cx + 8, cy - 1], fill=(20, 20, 20))
    draw.ellipse([cx - 3, cy + 5, cx + 3, cy + 10], fill=(25, 25, 25))
    draw.arc([cx - 8, cy + 5, cx, cy + 17], 10, 85, fill=(40, 40, 40), width=1)
    draw.arc([cx, cy + 5, cx + 8, cy + 17], 95, 170, fill=(40, 40, 40), width=1)

    angle = float(rng.uniform(-0.08, 0.08))
    if abs(angle) > 0.02:
        image = image.rotate(angle * 180 / math.pi, resample=Image.Resampling.BILINEAR)
    return image.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0, 0.45))))


if __name__ == "__main__":
    main()
