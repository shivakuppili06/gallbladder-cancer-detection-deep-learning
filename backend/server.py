import io
import os
import json
import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path

from gradcam import GradCAM, overlay_heatmap, get_target_layer

app = Flask(__name__)

# Configurable CORS origin for production (Vercel domain or wildcard)
FRONTEND_URL = os.environ.get("FRONTEND_URL", "*")

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = FRONTEND_URL if FRONTEND_URL != "*" else "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CLASS_NAMES_FILE = CHECKPOINT_DIR / "class_names.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CHECKPOINTS = {
    "EfficientNetB0": "efficientnetb0_gbc.pth",
    "EfficientNetB1": "efficientnetb1_gbc.pth",
    "EfficientNetB2": "efficientnetb2_gbc.pth",
    "EfficientNetB3": "efficientnetb3_gbc.pth",
    "EfficientNetB4": "efficientnetb4_gbc.pth",
    "DenseNet121": "densenet121_gbc.pth",
    "MobileNetV2": "mobilenetv2_gbc.pth",
    "ResNet18": "resnet18_gbc.pth",
    "ResNet50": "resnet50_gbc.pth",
}

# Default class names if class_names.json is unreadable
DEFAULT_CLASSES = ["abn", "bmt", "malg", "nml", "stn"]

if CLASS_NAMES_FILE.exists():
    try:
        with open(CLASS_NAMES_FILE) as f:
            CLASS_NAMES = json.load(f)
    except Exception:
        CLASS_NAMES = DEFAULT_CLASSES
else:
    CLASS_NAMES = DEFAULT_CLASSES

# LRU Model Cache to prevent RAM exhaustion on serverless/cloud instances
_model_cache = {}
MAX_CACHED_MODELS = int(os.environ.get("MAX_CACHED_MODELS", 3))


def build_architecture(model_name, num_classes):
    if model_name == "EfficientNetB0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif model_name == "EfficientNetB1":
        m = models.efficientnet_b1(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif model_name == "EfficientNetB2":
        m = models.efficientnet_b2(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif model_name == "EfficientNetB3":
        m = models.efficientnet_b3(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif model_name == "EfficientNetB4":
        m = models.efficientnet_b4(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif model_name == "DenseNet121":
        m = models.densenet121(weights=None)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    elif model_name == "MobileNetV2":
        m = models.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif model_name == "ResNet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif model_name == "ResNet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        m = None
    return m


def get_model(model_name):
    if model_name in _model_cache:
        return _model_cache[model_name]

    num_classes = len(CLASS_NAMES)
    model = build_architecture(model_name, num_classes)
    if model is None:
        return None, f"Unknown architecture selected: '{model_name}'"

    checkpoint_file = MODEL_CHECKPOINTS.get(model_name)
    checkpoint_path = CHECKPOINT_DIR / checkpoint_file if checkpoint_file else None

    warning = None
    if checkpoint_path and checkpoint_path.exists():
        try:
            state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state_dict)
        except Exception as e:
            warning = f"Error loading checkpoint '{checkpoint_file}': {str(e)}"
    else:
        warning = f"No trained checkpoint found at '{checkpoint_file}'."

    model = model.to(DEVICE)
    model.eval()

    # Evict oldest model if cache size exceeds limit to save memory
    if len(_model_cache) >= MAX_CACHED_MODELS:
        oldest_key = next(iter(_model_cache))
        del _model_cache[oldest_key]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _model_cache[model_name] = (model, warning)
    return model, warning


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


@app.route("/")
def index():
    return jsonify({
        "status": "online",
        "service": "Gallbladder Cancer Detection AI API",
        "device": str(DEVICE),
        "endpoints": ["/status", "/predict", "/metrics"]
    })


@app.route("/status", methods=["GET", "OPTIONS"])
def status():
    if request.method == "OPTIONS":
        return "", 204
    result = {}
    for name, filename in MODEL_CHECKPOINTS.items():
        has_checkpoint = (CHECKPOINT_DIR / filename).exists()
        result[name] = has_checkpoint
    return jsonify(result)


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 204

    model_name = request.form.get("model", "EfficientNetB0")
    if model_name not in MODEL_CHECKPOINTS:
        return jsonify({
            "error": f"Invalid model architecture '{model_name}'. Valid options: {list(MODEL_CHECKPOINTS.keys())}"
        }), 400

    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify({"error": "No image uploaded. Please attach an ultrasound image file."}), 400

    try:
        image_bytes = file.read()
        if not image_bytes:
            return jsonify({"error": "Empty image file uploaded."}), 400
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    except Exception as e:
        return jsonify({"error": f"Invalid or corrupt image format: {str(e)}"}), 400

    try:
        model, warning = get_model(model_name)
        if model is None:
            return jsonify({"error": warning or "Model failed to load."}), 400

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1).squeeze().cpu().tolist()
    except Exception as e:
        _model_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return jsonify({"error": f"Model inference error: {str(e)}"}), 500

    max_idx = probs.index(max(probs))

    # Grad-CAM heatmap visualization
    gradcam_image_b64 = None
    try:
        cam_input = transform(image).unsqueeze(0).to(DEVICE)
        cam_input.requires_grad_()
        target_layer = get_target_layer(model, model_name)
        cam_engine = GradCAM(model, target_layer)
        cam, _ = cam_engine.generate(cam_input, class_idx=max_idx)
        cam_engine.remove_hooks()

        img_224 = image.resize((224, 224))
        original_bgr = cv2.cvtColor(np.array(img_224), cv2.COLOR_RGB2BGR)
        overlay_b64 = overlay_heatmap(cam, original_bgr)
        gradcam_image_b64 = f"data:image/jpeg;base64,{overlay_b64}"
    except Exception as cam_err:
        print(f"GradCAM warning for {model_name}: {cam_err}")

    result = {
        "prediction": CLASS_NAMES[max_idx],
        "confidence": float(probs[max_idx] * 100),
        "distribution": dict(zip(CLASS_NAMES, [float(p * 100) for p in probs])),
        "gradcam_image": gradcam_image_b64,
        "warning": warning
    }
    return jsonify(result)


@app.route("/metrics", methods=["GET", "OPTIONS"])
def metrics():
    import csv
    summary_path = CHECKPOINT_DIR / "summary_metrics.csv"
    summary_data = []
    if summary_path.exists():
        with open(summary_path) as f:
            reader = csv.DictReader(f)
            summary_data = list(reader)
    
    per_class_data = {}
    for name in MODEL_CHECKPOINTS:
        p_path = CHECKPOINT_DIR / f"per_class_metrics_{name}.csv"
        if p_path.exists():
            with open(p_path) as f:
                reader = csv.DictReader(f)
                per_class_data[name] = list(reader)

    return jsonify({
        "summary": summary_data,
        "per_class": per_class_data
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
