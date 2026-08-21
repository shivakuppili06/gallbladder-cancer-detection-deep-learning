import io
import os
import gc
import json
import time
import logging
import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path

from gradcam import GradCAM, overlay_heatmap, get_target_layer

# Configure server-side logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

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

DEFAULT_CLASSES = ["abn", "bmt", "malg", "nml", "stn"]

if CLASS_NAMES_FILE.exists():
    try:
        with open(CLASS_NAMES_FILE) as f:
            CLASS_NAMES = json.load(f)
    except Exception:
        CLASS_NAMES = DEFAULT_CLASSES
else:
    CLASS_NAMES = DEFAULT_CLASSES

# LRU Model Cache (Strict MAX_CACHED_MODELS=1 for Render CPU memory safety)
_model_cache = {}
MAX_CACHED_MODELS = int(os.environ.get("MAX_CACHED_MODELS", 1))


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
        return _model_cache[model_name], None

    num_classes = len(CLASS_NAMES)

    # PRIMARY FIX 1: Evict existing cached model BEFORE constructing/loading the new model
    if len(_model_cache) >= MAX_CACHED_MODELS:
        for old_name in list(_model_cache.keys()):
            logging.info(f"Evicting cached model '{old_name}' before loading '{model_name}'")
            del _model_cache[old_name]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logging.info(f"Checkpoint load start for {model_name}")
    checkpoint_file = MODEL_CHECKPOINTS.get(model_name)
    checkpoint_path = CHECKPOINT_DIR / checkpoint_file if checkpoint_file else None

    # PRIMARY FIX 2 & 10: Fail safely if checkpoint is missing or load fails
    if not checkpoint_path or not checkpoint_path.exists():
        err_msg = f"Trained checkpoint file '{checkpoint_file}' not found for model '{model_name}'."
        logging.error(err_msg)
        return None, {
            "error": f"Failed to load trained checkpoint for {model_name}",
            "details": err_msg,
            "trained_model_available": False
        }

    try:
        model = build_architecture(model_name, num_classes)
        if model is None:
            return None, {
                "error": f"Failed to load trained checkpoint for {model_name}",
                "details": f"Unknown architecture '{model_name}'.",
                "trained_model_available": False
            }

        try:
            try:
                state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True, mmap=True)
            except Exception:
                try:
                    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
                except Exception:
                    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        except Exception as load_err:
            logging.error(f"Error loading checkpoint file '{checkpoint_file}': {load_err}")
            return None, {
                "error": f"Failed to load trained checkpoint for {model_name}",
                "details": str(load_err),
                "trained_model_available": False
            }

        model.load_state_dict(state_dict)
        del state_dict
        gc.collect()

        model = model.to(DEVICE)
        model.eval()
        logging.info(f"Checkpoint load success for {model_name}")
    except Exception as e:
        err_msg = f"Exception loading checkpoint '{checkpoint_file}': {str(e)}"
        logging.error(err_msg)
        return None, {
            "error": f"Failed to load trained checkpoint for {model_name}",
            "details": err_msg,
            "trained_model_available": False
        }

    _model_cache[model_name] = model
    return model, None


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


@app.route("/")
def index():
    if (BASE_DIR / "frontend" / "index.html").exists():
        return send_from_directory(BASE_DIR / "frontend", "index.html")
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


@app.route("/predict", methods=["GET", "POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 204

    if request.method == "GET":
        return jsonify({
            "status": "online",
            "message": "The /predict endpoint accepts POST requests with multipart/form-data containing an ultrasound image.",
            "models": list(MODEL_CHECKPOINTS.keys())
        })

    start_time = time.time()
    model_name = request.form.get("model", "EfficientNetB0")
    gradcam_param = request.form.get("gradcam", "true").lower()
    gradcam_enabled = gradcam_param in ("true", "1")

    logging.info(f"Model requested: {model_name} (gradcam={gradcam_enabled})")

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

    model, err_dict = get_model(model_name)
    if model is None:
        return jsonify(err_dict), 500

    try:
        logging.info(f"Inference start for {model_name}")
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1).squeeze().cpu().tolist()

        del input_tensor
        del outputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        inference_time = time.time() - start_time
        logging.info(f"Inference complete for {model_name} in {inference_time:.3f}s")
    except Exception as e:
        _model_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.error(f"Inference error for {model_name}: {str(e)}")
        return jsonify({"error": f"Model inference error for {model_name}: {str(e)}"}), 500

    max_idx = probs.index(max(probs))

    # Grad-CAM heatmap visualization
    gradcam_image_b64 = None
    if gradcam_enabled:
        cam_engine = None
        try:
            logging.info(f"Grad-CAM start for {model_name}")
            cam_input = transform(image).unsqueeze(0).to(DEVICE)
            cam_input.requires_grad_()
            target_layer = get_target_layer(model, model_name)
            cam_engine = GradCAM(model, target_layer)
            cam, _ = cam_engine.generate(cam_input, class_idx=max_idx)

            img_224 = image.resize((224, 224))
            original_bgr = cv2.cvtColor(np.array(img_224), cv2.COLOR_RGB2BGR)
            overlay_b64 = overlay_heatmap(cam, original_bgr)
            gradcam_image_b64 = f"data:image/jpeg;base64,{overlay_b64}"

            del cam_input
            del cam
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logging.info(f"Grad-CAM complete for {model_name}")
        except Exception as cam_err:
            logging.warning(f"GradCAM warning for {model_name}: {cam_err}")
            gradcam_image_b64 = None
        finally:
            if cam_engine is not None:
                try:
                    cam_engine.remove_hooks()
                except Exception:
                    pass

    logging.info(f"Memory cleanup complete for {model_name}")

    result = {
        "prediction": CLASS_NAMES[max_idx],
        "confidence": float(probs[max_idx] * 100),
        "distribution": dict(zip(CLASS_NAMES, [float(p * 100) for p in probs])),
        "gradcam_image": gradcam_image_b64,
        "inference_time_seconds": round(inference_time, 3),
        "trained_model_available": True,
        "warning": None
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
