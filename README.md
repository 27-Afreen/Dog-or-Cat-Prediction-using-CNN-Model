# CNN Cats vs Dogs Image Classification

This project demonstrates a complete computer vision workflow for classifying cat and dog images using a Convolutional Neural Network (CNN).

The starter version uses a lightweight NumPy CNN so it can run locally without installing TensorFlow or PyTorch. It includes image preprocessing, data augmentation, convolutional feature extraction, model training, evaluation, and prediction.

## Project Structure

```text
CNN-Cats-vs-Dogs-Image-Classification/
  .vscode/              VS Code run/debug setup
  data/
    demo/               Generated demo cat/dog images
  models/               Saved CNN model files
  scripts/              Dataset helper scripts
  src/                  Python source code
```

## Quick Start

Run these commands from the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts\make_demo_dataset.py
python src\train.py --data-dir data\demo --model models\cats_dogs_cnn.npz --epochs 12
python src\evaluate.py --data-dir data\demo --model models\cats_dogs_cnn.npz
python src\predict.py --image data\demo\test\cats\cat_00.png --model models\cats_dogs_cnn.npz
```

If `python` is not recognized, use:

```powershell
.\.venv\Scripts\python.exe src\predict.py --image data\demo\test\cats\cat_00.png --model models\cats_dogs_cnn.npz
```

## Using Your Own Cat/Dog Dataset

Place images in this folder structure:

```text
data/my_dataset/
  train/
    cats/
    dogs/
  test/
    cats/
    dogs/
```

Then run:

```powershell
python src\train.py --data-dir data\my_dataset --model models\cats_dogs_cnn.npz --epochs 12
python src\evaluate.py --data-dir data\my_dataset --model models\cats_dogs_cnn.npz
python src\predict.py --image "C:\path\to\cat_or_dog.jpg" --model models\cats_dogs_cnn.npz
```

## Notes

- The demo dataset is synthetic and is meant to prove the full CNN pipeline works.
- For high accuracy on real photos, use a larger real dataset and consider upgrading to TensorFlow/Keras or PyTorch transfer learning.
