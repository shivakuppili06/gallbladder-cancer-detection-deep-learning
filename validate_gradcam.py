import sys
import json
import torch
import base64
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms

from train import build_model, MODEL_CHECKPOINTS, CHECKPOINT_DIR
from gradcam import GradCAM, overlay_heatmap, get_target_layer

IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else None
OUT_DIR = Path("gradcam_validation")
OUT_DIR.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if IMAGE_PATH is None or not Path(IMAGE_PATH).exists():
    # Find a sample image automatically from data/test if not provided
    sample_images = list(Path("data/test").rglob("*.png")) + list(Path("data/test").rglob("*.jpg"))
    if sample_images:
        IMAGE_PATH = str(sample_images[0])
        print(f"No valid image path provided. Auto-selected sample image: {IMAGE_PATH}")
    else:
        print("Error: No test image provided and no sample images found in data/test.")
        sys.exit(1)

with open(CHECKPOINT_DIR / "class_names.json") as f:
    class_names = json.load(f)
    if isinstance(class_names, dict):
        class_names = [class_names[str(i)] for i in range(len(class_names))]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img = Image.open(IMAGE_PATH).convert("RGB")
img_224 = img.resize((224, 224))
original_bgr = cv2.cvtColor(np.array(img_224), cv2.COLOR_RGB2BGR)

print(f"Evaluating Grad-CAM across available checkpoints on: {IMAGE_PATH}\n" + "="*60)

for model_name, ckpt_file in MODEL_CHECKPOINTS.items():
    ckpt_path = CHECKPOINT_DIR / ckpt_file
    if not ckpt_path.exists():
        print(f"[SKIP] {model_name}: no checkpoint found at {ckpt_path}")
        continue

    try:
        model = build_model(model_name, num_classes=len(class_names)).to(device)
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        input_tensor = transform(img_224).unsqueeze(0).to(device)
        input_tensor.requires_grad_()

        target_layer = get_target_layer(model, model_name)
        cam_engine = GradCAM(model, target_layer)

        output = model(input_tensor)
        pred_idx = output.argmax(dim=1).item()
        cam, _ = cam_engine.generate(input_tensor, class_idx=pred_idx)
        cam_engine.remove_hooks()

        heatmap_b64 = overlay_heatmap(cam, original_bgr)
        out_path = OUT_DIR / f"{model_name}_gradcam.jpg"
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(heatmap_b64))

        print(f"[RUN ] {model_name:<15} -> Predicted: {class_names[pred_idx]:<8} Saved: {out_path}")
    except Exception as e:
        print(f"[FAIL] {model_name}: {e}")

print("="*60)
print(f"Grad-CAM validation complete. Visualizations saved in '{OUT_DIR}/'")
