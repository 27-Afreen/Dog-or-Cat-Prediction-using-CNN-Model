from __future__ import annotations

import argparse

from cnn_model import SimpleCNN
from image_data import load_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained CNN cat/dog classifier.")
    parser.add_argument("--data-dir", default="data/demo")
    parser.add_argument("--model", default="models/cats_dogs_cnn.npz")
    args = parser.parse_args()

    model, classes = SimpleCNN.load(args.model)
    x_test, y_test = load_split(args.data_dir, "test", augment=False)

    correct = 0
    for image, expected in zip(x_test, y_test):
        predicted = model.predict(image)
        correct += int(predicted == int(expected))

    total = len(y_test)
    print(f"Accuracy: {correct / total:.2%} ({correct}/{total})")
    print("Classes:", ", ".join(classes))


if __name__ == "__main__":
    main()
