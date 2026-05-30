from __future__ import annotations

import argparse
from pathlib import Path

from cnn_model import SimpleCNN
from image_data import CLASSES, load_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CNN cat/dog image classifier.")
    parser.add_argument("--data-dir", default="data/demo")
    parser.add_argument("--model", default="models/cats_dogs_cnn.npz")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--filters", type=int, default=4)
    args = parser.parse_args()

    x_train, y_train = load_split(args.data_dir, "train", augment=True)
    print(f"loaded training images: {x_train.shape[0]}")

    model = SimpleCNN(filters=args.filters, learning_rate=args.learning_rate)
    model.fit(x_train, y_train, epochs=args.epochs)

    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path), CLASSES)
    print(f"saved model: {model_path}")


if __name__ == "__main__":
    main()
