import py_compile
import glob
import os
import urllib.request
import json
import io
import requests
from PIL import Image

print("=== 1. PYTHON SYNTAX AUDIT ===")
for f in sorted(glob.glob("*.py")):
    py_compile.compile(f, doraise=True)
    print(f"  [PASS] {f}")

print("\n=== 2. MODEL CHECKPOINTS AUDIT ===")
ckpt_dir = "checkpoints"
models = [
    "efficientnetb0_gbc.pth", "efficientnetb1_gbc.pth", "efficientnetb2_gbc.pth",
    "efficientnetb3_gbc.pth", "efficientnetb4_gbc.pth", "resnet18_gbc.pth",
    "resnet50_gbc.pth", "mobilenetv2_gbc.pth", "densenet121_gbc.pth"
]
for m in models:
    path = os.path.join(ckpt_dir, m)
    exists = os.path.exists(path)
    size_mb = os.path.getsize(path) / (1024*1024) if exists else 0
    status_str = "PASS" if exists else "FAIL"
    print(f"  [{status_str}] {m}: {size_mb:.2f} MB")

print("\n=== 3. SERVER & API ENDPOINTS AUDIT ===")
try:
    res = json.loads(urllib.request.urlopen("http://127.0.0.1:5001/status").read())
    print("  [PASS] GET /status:", res)
except Exception as e:
    print("  [FAIL] GET /status:", e)

try:
    img = Image.new("RGB", (224, 224), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    r = requests.post("http://127.0.0.1:5001/predict", data={"model": "ResNet18"}, files={"image": ("test.jpg", buf, "image/jpeg")})
    res_j = r.json()
    conf = res_j.get('confidence', 0.0)
    print(f"  [PASS] POST /predict (ResNet18): Status {r.status_code} | Diagnosis: {res_j.get('prediction')} | Confidence: {conf:.1f}% | GradCAM Length: {len(res_j.get('gradcam_image', ''))}")
except Exception as e:
    print("  [FAIL] POST /predict:", e)

try:
    ui_status = urllib.request.urlopen("http://127.0.0.1:8501/").status
    print(f"  [PASS] GET Frontend UI (Port 8501): HTTP {ui_status}")
except Exception as e:
    print("  [FAIL] GET Frontend UI:", e)
