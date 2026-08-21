import io
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


@app.after_request
def add_cors_headers(response):
    # index.html is served from a different port (app.py) than this API
    # (server.py, port 5000), so the browser needs explicit CORS permission
    # for the fetch() calls in index.html to succeed. Fine for local dev;
    # tighten Access-Control-Allow-Origin to a specific origin if deploying.
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
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

IMPLEMENTED_ARCHITECTURES = {
    "EfficientNetB0", "EfficientNetB1", "EfficientNetB2", "EfficientNetB3", "EfficientNetB4",
    "DenseNet121", "MobileNetV2", "ResNet18", "ResNet50",
}

# Maps raw dataset folder codes -> human-readable labels for DISPLAY ONLY.
DISPLAY_LABELS = {
    "abn": "Abnormal",
    "bmt": "Benign",
    "malg": "Malignant",
    "nml": "Normal",
    "stn": "Gallstone",
}

if CLASS_NAMES_FILE.exists():
    with open(CLASS_NAMES_FILE) as f:
        CLASS_NAMES = json.load(f)
else:
    CLASS_NAMES = None

_model_cache = {}


def display_label(raw_class_name):
    """Human-readable label for a raw folder-code class name, for API output only."""
    return DISPLAY_LABELS.get(raw_class_name, raw_class_name)


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
    elif model_name == "ConvNeXtTiny":
        m = models.convnext_tiny(weights=None)
        in_features = m.classifier[2].in_features
        m.classifier[2] = nn.Linear(in_features, num_classes)
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

    if CLASS_NAMES is None:
        return None, ("class_names.json not found in checkpoints/. Run train.py first — "
                       "it writes this file automatically alongside the checkpoints.")

    num_classes = len(CLASS_NAMES)
    model = build_architecture(model_name, num_classes)
    if model is None:
        return None, f"Unknown architecture selected: {model_name}"

    checkpoint_file = MODEL_CHECKPOINTS.get(model_name)
    checkpoint_path = CHECKPOINT_DIR / checkpoint_file if checkpoint_file else None

    warning = None
    if checkpoint_path and checkpoint_path.exists():
        state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
    else:
        warning = (f"No trained checkpoint found at '{checkpoint_path}'. "
                   f"Predictions below are from an UNTRAINED model and are statistically meaningless.")

    model = model.to(DEVICE)
    model.eval()
    _model_cache[model_name] = (model, warning)
    return model, warning


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


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
    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "No image uploaded"}), 400

    image_bytes = file.read()
    if not image_bytes:
        return jsonify({"error": "Empty image file"}), 400

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    except Exception as e:
        return jsonify({"error": f"Invalid image format: {str(e)}"}), 400

    try:
        model, warning = get_model(model_name)
        if model is None:
            return jsonify({"error": warning}), 400

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1).squeeze().cpu().tolist()
    except RuntimeError as e:
        # Clear model cache if memory allocation fails and retry once
        _model_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            model, warning = get_model(model_name)
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1).squeeze().cpu().tolist()
        except Exception as retry_err:
            return jsonify({"error": f"Model inference error: {str(retry_err)}"}), 500

    max_idx = probs.index(max(probs))

    # Generate Grad-CAM visualization
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
        print(f"GradCAM generation warning: {cam_err}")

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
    app.run(host="0.0.0.0", port=5001, debug=False)