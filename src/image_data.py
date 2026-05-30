from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


CLASSES = ["cats", "dogs"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_image(path: str | Path, size: int = 24) -> np.ndarray:
    image = Image.open(path).convert("L")
    image = ImageOps.fit(image, (size, size), method=Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array[None, :, :]


def augment_image(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    augmented = image.copy()
    if rng.random() < 0.5:
        augmented = augmented[:, :, ::-1]
    augmented = augmented * rng.uniform(0.85, 1.15) + rng.uniform(-0.08, 0.08)
    if rng.random() < 0.35:
        shift_y = int(rng.integers(-2, 3))
        shift_x = int(rng.integers(-2, 3))
        augmented = np.roll(augmented, shift=(shift_y, shift_x), axis=(1, 2))
    return np.clip(augmented, 0.0, 1.0).astype(np.float32)


def load_split(data_dir: str | Path, split: str, augment: bool = False, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    data_dir = Path(data_dir)
    images: list[np.ndarray] = []
    labels: list[int] = []
    rng = np.random.default_rng(seed)

    for label, class_name in enumerate(CLASSES):
        class_dir = data_dir / split / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing dataset folder: {class_dir}")
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            image = load_image(path)
            images.append(image)
            labels.append(label)
            if augment:
                for _ in range(1):
                    images.append(augment_image(image, rng))
                    labels.append(label)

    if not images:
        raise ValueError(f"No images found in {data_dir / split}")

    return np.stack(images).astype(np.float32), np.array(labels, dtype=np.int64)


def one_hot(labels: np.ndarray, class_count: int) -> np.ndarray:
    encoded = np.zeros((labels.size, class_count), dtype=np.float32)
    encoded[np.arange(labels.size), labels] = 1.0
    return encoded
