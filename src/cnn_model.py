from __future__ import annotations

import numpy as np


class SimpleCNN:
    """Small CNN: conv -> ReLU -> max pool -> dense -> softmax."""

    def __init__(
        self,
        image_size: int = 24,
        filters: int = 4,
        kernel_size: int = 3,
        class_count: int = 2,
        learning_rate: float = 0.01,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.image_size = image_size
        self.filters = filters
        self.kernel_size = kernel_size
        self.class_count = class_count
        self.learning_rate = learning_rate
        self.conv_w = rng.normal(0, 0.08, (filters, 1, kernel_size, kernel_size)).astype(np.float32)
        self.conv_b = np.zeros(filters, dtype=np.float32)
        conv_size = image_size - kernel_size + 1
        pooled_size = conv_size // 2
        self.flatten_size = filters * pooled_size * pooled_size
        self.dense_w = rng.normal(0, 0.08, (self.flatten_size, class_count)).astype(np.float32)
        self.dense_b = np.zeros(class_count, dtype=np.float32)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int = 25) -> None:
        indices = np.arange(x_train.shape[0])
        for epoch in range(1, epochs + 1):
            np.random.shuffle(indices)
            total_loss = 0.0
            correct = 0
            for idx in indices:
                loss, prediction = self._train_one(x_train[idx], int(y_train[idx]))
                total_loss += loss
                correct += int(prediction == int(y_train[idx]))
            accuracy = correct / len(indices)
            if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
                print(f"epoch={epoch:03d} loss={total_loss / len(indices):.4f} accuracy={accuracy:.2%}")

    def predict_proba(self, image: np.ndarray) -> np.ndarray:
        flat, _ = self._forward_features(image)
        return _softmax(flat @ self.dense_w + self.dense_b)

    def predict(self, image: np.ndarray) -> int:
        return int(np.argmax(self.predict_proba(image)))

    def save(self, path: str, classes: list[str]) -> None:
        np.savez(
            path,
            conv_w=self.conv_w,
            conv_b=self.conv_b,
            dense_w=self.dense_w,
            dense_b=self.dense_b,
            image_size=self.image_size,
            filters=self.filters,
            kernel_size=self.kernel_size,
            class_count=self.class_count,
            learning_rate=self.learning_rate,
            classes=np.array(classes),
        )

    @classmethod
    def load(cls, path: str) -> tuple["SimpleCNN", list[str]]:
        data = np.load(path, allow_pickle=True)
        model = cls(
            image_size=int(data["image_size"]),
            filters=int(data["filters"]),
            kernel_size=int(data["kernel_size"]),
            class_count=int(data["class_count"]),
            learning_rate=float(data["learning_rate"]),
        )
        model.conv_w = data["conv_w"]
        model.conv_b = data["conv_b"]
        model.dense_w = data["dense_w"]
        model.dense_b = data["dense_b"]
        return model, [str(item) for item in data["classes"].tolist()]

    def _train_one(self, image: np.ndarray, target: int) -> tuple[float, int]:
        flat, cache = self._forward_features(image)
        logits = flat @ self.dense_w + self.dense_b
        probabilities = _softmax(logits)
        loss = -np.log(probabilities[target] + 1e-9)
        prediction = int(np.argmax(probabilities))

        dlogits = probabilities
        dlogits[target] -= 1.0

        ddense_w = np.outer(flat, dlogits)
        ddense_b = dlogits
        dflat = dlogits @ self.dense_w.T
        dpool = dflat.reshape(cache["pool"].shape)
        drelu = _max_pool_backward(dpool, cache["relu"], cache["pool_mask"])
        dconv = drelu * (cache["conv"] > 0)

        dconv_w = np.zeros_like(self.conv_w)
        dconv_b = dconv.sum(axis=(1, 2))
        for f in range(self.filters):
            for y in range(dconv.shape[1]):
                for x in range(dconv.shape[2]):
                    region = image[:, y : y + self.kernel_size, x : x + self.kernel_size]
                    dconv_w[f] += dconv[f, y, x] * region

        for gradient in (dconv_w, dconv_b, ddense_w, ddense_b):
            np.clip(gradient, -2.5, 2.5, out=gradient)

        self.conv_w -= self.learning_rate * dconv_w
        self.conv_b -= self.learning_rate * dconv_b
        self.dense_w -= self.learning_rate * ddense_w
        self.dense_b -= self.learning_rate * ddense_b

        return float(loss), prediction

    def _forward_features(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        conv = _conv2d_valid(image, self.conv_w, self.conv_b)
        relu = np.maximum(conv, 0.0)
        pool, pool_mask = _max_pool2d(relu)
        flat = pool.reshape(-1)
        return flat, {"conv": conv, "relu": relu, "pool": pool, "pool_mask": pool_mask}


def _conv2d_valid(image: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    filters, _, kernel_size, _ = weights.shape
    output_size = image.shape[1] - kernel_size + 1
    output = np.zeros((filters, output_size, output_size), dtype=np.float32)
    for f in range(filters):
        for y in range(output_size):
            for x in range(output_size):
                region = image[:, y : y + kernel_size, x : x + kernel_size]
                output[f, y, x] = float(np.sum(region * weights[f]) + bias[f])
    return output


def _max_pool2d(features: np.ndarray, size: int = 2) -> tuple[np.ndarray, np.ndarray]:
    channels, height, width = features.shape
    pooled = np.zeros((channels, height // size, width // size), dtype=np.float32)
    mask = np.zeros_like(features, dtype=np.float32)
    for c in range(channels):
        for y in range(0, height - 1, size):
            for x in range(0, width - 1, size):
                region = features[c, y : y + size, x : x + size]
                max_index = np.unravel_index(np.argmax(region), region.shape)
                pooled[c, y // size, x // size] = region[max_index]
                mask[c, y + max_index[0], x + max_index[1]] = 1.0
    return pooled, mask


def _max_pool_backward(dpool: np.ndarray, relu: np.ndarray, mask: np.ndarray, size: int = 2) -> np.ndarray:
    drelu = np.zeros_like(relu)
    for c in range(dpool.shape[0]):
        for y in range(dpool.shape[1]):
            for x in range(dpool.shape[2]):
                region = mask[c, y * size : y * size + size, x * size : x * size + size]
                drelu[c, y * size : y * size + size, x * size : x * size + size] += dpool[c, y, x] * region
    return drelu


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)
