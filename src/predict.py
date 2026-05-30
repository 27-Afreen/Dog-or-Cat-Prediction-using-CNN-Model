from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from cnn_model import SimpleCNN
from image_data import load_image


ASCII_CHARS = " .:-=+*#%@"


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict whether an image is a cat or dog.")
    parser.add_argument("--image", required=True, help="Image path.")
    parser.add_argument("--model", default="models/cats_dogs_cnn.npz")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--preview", action="store_true", help="Print a small terminal preview of the image.")
    parser.add_argument("--open-image", action="store_true", help="Open the image in the default Windows image viewer.")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Image file not found: {image_path}")

    if args.open_image:
        open_image(image_path)
    if args.preview:
        print_terminal_preview(image_path)

    model, classes = SimpleCNN.load(args.model)
    image = load_image(image_path, size=model.image_size)
    probabilities = model.predict_proba(image)
    predicted_index = int(probabilities.argmax())

    if args.verbose:
        print(f"Image: {image_path}")
        for index, class_name in enumerate(classes):
            print(f"{class_name}: {probabilities[index] * 100:.2f}%")
        print("Prediction:")
    print(classes[predicted_index])


def print_terminal_preview(image_path: Path, width: int = 48) -> None:
    preview = Image.open(image_path).convert("L")
    aspect_ratio = preview.height / max(1, preview.width)
    height = max(8, int(width * aspect_ratio * 0.45))
    preview = preview.resize((width, height), Image.Resampling.BILINEAR)
    pixels = np.asarray(preview)

    print(f"Preview: {image_path}")
    for row in range(height):
        line = ""
        for col in range(width):
            value = int(pixels[row, col])
            line += ASCII_CHARS[value * (len(ASCII_CHARS) - 1) // 255]
        print(line)
    print()


def open_image(image_path: Path) -> None:
    try:
        import os

        os.startfile(image_path.resolve())
    except OSError as exc:
        raise SystemExit(f"Could not open image preview: {exc}") from exc


if __name__ == "__main__":
    main()
