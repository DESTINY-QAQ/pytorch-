"""Flask backend for the offline machine-learning teaching laboratory."""

from __future__ import annotations

import base64
import io
from functools import lru_cache
from pathlib import Path

import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from werkzeug.datastructures import FileStorage

from dogVScat import CatDogCNN, predict_image


APP_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = APP_DIR / "cat_dog_cnn.pth"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_IMAGE_PIXELS = 20_000_000
CLASS_LABELS = {"cats": "猫", "cat": "猫", "dogs": "狗", "dog": "狗"}

app = Flask(__name__)
app.config.update(MAX_CONTENT_LENGTH=10 * 1024 * 1024)


@lru_cache(maxsize=2)
def load_model(checkpoint_mtime_ns: int) -> tuple[CatDogCNN, list[str]]:
    """Load the checkpoint and refresh it when the file modification time changes."""
    del checkpoint_mtime_ns
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    class_to_idx = checkpoint["class_to_idx"]
    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    model = CatDogCNN(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names


def get_model() -> tuple[CatDogCNN, list[str]]:
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError("未找到训练模型 cat_dog_cnn.pth，请先完成模型训练。")
    return load_model(CHECKPOINT_PATH.stat().st_mtime_ns)


def validate_image(upload: FileStorage) -> tuple[Image.Image, str]:
    """Validate an upload and return a safe RGB image plus a compact data-URL preview."""
    extension = Path(upload.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError("文件格式不支持，请上传 JPG、PNG、BMP 或 WEBP 图片。")

    data = upload.read()
    if not data:
        raise ValueError("上传的图片为空，请重新选择。")

    probe = Image.open(io.BytesIO(data))
    probe.verify()
    image = Image.open(io.BytesIO(data)).convert("RGB")
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ValueError("图片分辨率过高，请上传不超过 2000 万像素的图片。")

    preview_buffer = io.BytesIO()
    preview = image.copy()
    preview.thumbnail((1200, 900))
    preview.save(preview_buffer, format="JPEG", quality=88, optimize=True)
    preview_url = "data:image/jpeg;base64," + base64.b64encode(preview_buffer.getvalue()).decode("ascii")
    return image, preview_url


@app.get("/")
def index():
    return render_template("index.html", model_ready=CHECKPOINT_PATH.is_file())


@app.post("/api/cnn/predict")
def cnn_predict():
    upload = request.files.get("image")
    if upload is None or not upload.filename:
        return jsonify(ok=False, error="请选择一张需要识别的图片。"), 400

    try:
        image, preview_url = validate_image(upload)
        model, class_names = get_model()
        predicted_class, confidence = predict_image(image, model, torch.device("cpu"), class_names)
        return jsonify(
            ok=True,
            prediction={
                "id": predicted_class,
                "label": CLASS_LABELS.get(predicted_class.lower(), predicted_class),
                "confidence": round(float(confidence), 6),
            },
            preview=preview_url,
            model={
                "name": "CatDogCNN",
                "runtime": "PyTorch CPU",
                "input_shape": [1, 3, 150, 150],
                "checkpoint": CHECKPOINT_PATH.name,
            },
        )
    except (ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        message = str(exc) or "图片无法解析，请换一张图片后重试。"
        return jsonify(ok=False, error=message), 400
    except (FileNotFoundError, KeyError, RuntimeError, OSError) as exc:
        return jsonify(ok=False, error=f"识别失败：{exc}"), 500


@app.get("/health")
def health():
    model_ready = CHECKPOINT_PATH.is_file()
    return jsonify(
        status="ok",
        version="2.0.0",
        checkpoint={"ready": model_ready, "file": CHECKPOINT_PATH.name},
        modules={
            "catdog_cnn": "real" if model_ready else "unavailable",
            "resnet50": "teaching-simulation",
            "vit_b_16": "teaching-simulation",
            "transformer_translation": "teaching-simulation",
            "yolo11": "teaching-simulation",
        },
        offline_frontend=True,
    )


@app.errorhandler(413)
def too_large(_error):
    return jsonify(ok=False, error="文件过大，请上传不超过 10 MB 的图片。"), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
