"""Train a small PyTorch CNN to classify cats and dogs.

Run this module directly to download the dataset (when necessary) and train the
model.  Importing it is side-effect free, so ``CatDogCNN`` and
``predict_image`` can also be reused from another script.
"""

from __future__ import annotations

import argparse
import os
import random
import zipfile
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Use a non-interactive backend so figure saving never waits for a desktop GUI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATASET_URL = "https://storage.googleapis.com/tensorflow-1-public/course2/cats_and_dogs_filtered.zip"
IMAGE_SIZE = 150
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)
DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}


def set_seed(seed: int) -> None:
    """Make the data order and model initialization reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    """Extract an archive only when every member stays below destination."""
    destination = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if os.path.commonpath((str(destination), str(target))) != str(destination):
            raise ValueError(f"Unsafe archive path: {member.filename}")
    archive.extractall(destination)


def download_dataset(work_dir: Path, timeout: int = 30) -> Path:
    """Download and safely unpack the Cats vs Dogs dataset if it is absent."""
    data_dir = work_dir / "cats_and_dogs_filtered"
    train_dir = data_dir / "train"
    validation_dir = data_dir / "validation"
    if train_dir.is_dir() and validation_dir.is_dir():
        print(f"Dataset already exists: {data_dir}")
        return data_dir

    archive_path = work_dir / "cats_and_dogs_filtered.zip"
    temporary_path = archive_path.with_suffix(".zip.part")
    if not archive_path.is_file():
        print("Downloading dataset (about 65 MB)...")
        try:
            with requests.get(DATASET_URL, headers=DOWNLOAD_HEADERS, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                with temporary_path.open("wb") as file:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            file.write(chunk)
            if not zipfile.is_zipfile(temporary_path):
                raise zipfile.BadZipFile("The downloaded file is not a valid ZIP archive.")
            temporary_path.replace(archive_path)
        except requests.RequestException as exc:
            temporary_path.unlink(missing_ok=True)
            raise RuntimeError(f"Dataset download failed: {exc}") from exc
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    print("Extracting dataset...")
    try:
        with zipfile.ZipFile(archive_path) as archive:
            _safe_extract(archive, work_dir)
    except (OSError, zipfile.BadZipFile, ValueError) as exc:
        raise RuntimeError(f"Dataset extraction failed: {exc}") from exc

    if not (train_dir.is_dir() and validation_dir.is_dir()):
        raise RuntimeError("Archive extraction completed but dataset directories are missing.")
    return data_dir


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])
    validation_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])
    return train_transform, validation_transform


class CatDogCNN(nn.Module):
    """CNN whose classifier works with different image dimensions."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5), nn.Linear(128, 512), nn.ReLU(inplace=True), nn.Linear(512, num_classes)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Run one training or validation pass and return loss and accuracy."""
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        raise RuntimeError("Data loader is empty.")
    return total_loss / total, correct / total


def plot_history(history: dict[str, list[float]], output_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, history["train_loss"], label="Train loss")
    axes[0].plot(epochs, history["val_loss"], label="Validation loss")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Training and validation loss")
    axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], label="Train accuracy")
    axes[1].plot(epochs, history["val_acc"], label="Validation accuracy")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Training and validation accuracy")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def predict_image(
    img_path: str | Path | Image.Image,
    model: nn.Module,
    device: torch.device,
    class_names: Iterable[str],
) -> tuple[str, float]:
    """Return the most likely class and its softmax probability for one image."""
    _, validation_transform = build_transforms()
    image = img_path.convert("RGB") if isinstance(img_path, Image.Image) else Image.open(img_path).convert("RGB")
    image_tensor = validation_transform(image).unsqueeze(0).to(device)
    names = list(class_names)
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(image_tensor), dim=1)
        confidence, prediction = probabilities.max(dim=1)
    predicted_class = names[prediction.item()]
    return predicted_class, confidence.item()


def train(
    work_dir: Path,
    epochs: int,
    batch_size: int,
    seed: int,
    data_dir: Path | None = None,
    num_workers: int = 2,
) -> None:
    set_seed(seed)
    data_dir = data_dir if data_dir is not None else download_dataset(work_dir)
    if not ((data_dir / "train").is_dir() and (data_dir / "validation").is_dir()):
        raise RuntimeError("Dataset directory must contain train/ and validation/ subdirectories.")
    train_transform, validation_transform = build_transforms()
    train_dataset = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    validation_dataset = datasets.ImageFolder(data_dir / "validation", transform=validation_transform)
    if train_dataset.class_to_idx != validation_dataset.class_to_idx:
        raise RuntimeError("Training and validation class mappings differ.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    loader_options = {"num_workers": num_workers, "pin_memory": pin_memory, "persistent_workers": num_workers > 0}
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, **loader_options)
    model = CatDogCNN(num_classes=len(train_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"Device: {device}; classes: {train_dataset.class_to_idx}")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, validation_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(f"Epoch [{epoch:02d}/{epochs}] Train loss: {train_loss:.4f}, acc: {train_acc:.4f}; "
              f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_to_idx": train_dataset.class_to_idx,
        "image_size": IMAGE_SIZE,
        "normalize_mean": NORMALIZE_MEAN,
        "normalize_std": NORMALIZE_STD,
    }
    checkpoint_path = work_dir / "cat_dog_cnn.pth"
    temporary_checkpoint_path = checkpoint_path.with_suffix(".pth.tmp")
    torch.save(checkpoint, temporary_checkpoint_path)
    temporary_checkpoint_path.replace(checkpoint_path)
    plot_history(history, work_dir / "training_history_pytorch.png")
    print("Saved model checkpoint and training history.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a cats-vs-dogs CNN.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2, help="Number of data-loading worker processes.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Existing dataset directory containing train/ and validation/. Skips downloading.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("--epochs and --batch-size must be positive; --num-workers cannot be negative.")
    train(Path(__file__).resolve().parent, args.epochs, args.batch_size, args.seed, args.data_dir, args.num_workers)
