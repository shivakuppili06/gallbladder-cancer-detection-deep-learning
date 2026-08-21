"""
validate_gradcam_batch.py
Runs Grad-CAM validation across all 10 model checkpoints against
multiple sample images per class (5 classes). Outputs per-model/per-class
heatmaps plus an aggregated CSV summary for report/viva use.
"""
import sys
import json
import csv
import random
from pathlib import Path

import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

from train import build_model, MODEL_CHECKPOINTS, CHECKPOINT_DIR
from gradcam import GradCAM, overlay_heatmap, get_target_layer

DATA_TEST_DIR = Path("data/test")
OUT_DIR = Path("gradcam_validation_batch")
OUT_DIR.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(CHECKPOINT_DIR / "class_names.json") as f:
    class_names_raw = json.load(f)
    class_names = (
        [class_names_raw[str(i)] for i in range(len(class_names_raw))]
        if isinstance(class_names_raw, dict) else class_names_raw
    )

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def pick_sample_images(n_per_class=1, seed=42):
    """One (or n) random image per class from data/test/."""
    rng = random.Random(seed)
    samples = {}
    for class_dir in sorted(DATA_TEST_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        images = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
        if not images:
            continue
        samples[class_dir.name] = rng.sample(images, min(n_per_class, len(images)))
    return samples


def analyze_heatmap(overlay_bgr):
    red_channel = overlay_bgr[:, :, 2]
    peak_idx = np.unravel_index(red_channel.argmax(), red_channel.shape)
    hot_area_pct = float((red_channel > 200).sum() / red_channel.size * 100)
    return {"peak_y": int(peak_idx[0]), "peak_x": int(peak_idx[1]), "hot_area_pct": round(hot_area_pct, 2)}


def load_model(model_name):
    model = build_model(model_name, num_classes=len(class_names)).to(device)
    ckpt_path = CHECKPOINT_DIR / MODEL_CHECKPOINTS[model_name]
    if not ckpt_path.exists():
        return None
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_gradcam(model, model_name, img_224, original_bgr):
    input_tensor = transform(img_224).unsqueeze(0).to(device)
    input_tensor.requires_grad_()

    target_layer = get_target_layer(model, model_name)
    cam_engine = GradCAM(model, target_layer)

    output = model(input_tensor)
    pred_idx = output.argmax(dim=1).item()
    cam, _ = cam_engine.generate(input_tensor, class_idx=pred_idx)
    cam_engine.remove_hooks()

    overlay_b64 = overlay_heatmap(cam, original_bgr)
    import base64
    overlay_bytes = base64.b64decode(overlay_b64)
    overlay_bgr = cv2.imdecode(np.frombuffer(overlay_bytes, np.uint8), cv2.IMREAD_COLOR)

    return class_names[pred_idx], overlay_bgr


def main():
    n_per_class = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    samples = pick_sample_images(n_per_class=n_per_class)

    print(f"Selected {sum(len(v) for v in samples.values())} images across {len(samples)} classes")

    rows = []
    for model_name in MODEL_CHECKPOINTS:
        print(f"\n=== {model_name} ===")
        model = load_model(model_name)
        if model is None:
            print(f"[SKIP] no checkpoint for {model_name}")
            continue

        for true_class, img_paths in samples.items():
            for img_path in img_paths:
                img = Image.open(img_path).convert("RGB")
                img_224 = img.resize((224, 224))
                original_bgr = cv2.cvtColor(np.array(img_224), cv2.COLOR_RGB2BGR)

                try:
                    pred_label, overlay_bgr = run_gradcam(model, model_name, img_224, original_bgr)
                except Exception as e:
                    print(f"[FAIL] {model_name} on {img_path.name}: {e}")
                    continue

                metrics = analyze_heatmap(overlay_bgr)
                correct = (pred_label == true_class)

                out_name = f"{model_name}_{true_class}_{img_path.stem}.jpg"
                cv2.imwrite(str(OUT_DIR / out_name), overlay_bgr)

                rows.append({
                    "model": model_name,
                    "true_class": true_class,
                    "image": img_path.name,
                    "predicted": pred_label,
                    "correct": correct,
                    **metrics,
                })
                status = "[OK]" if correct else "[ERR]"
                print(f"  {status:5s} true={true_class:5s} pred={pred_label:5s} "
                      f"peak=({metrics['peak_x']},{metrics['peak_y']}) hot_area={metrics['hot_area_pct']}%")

    # Write aggregated CSV
    csv_path = OUT_DIR / "gradcam_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Per-model average hot-area summary (the key stat for your report)
    print("\n" + "=" * 60)
    print("PER-MODEL AVERAGE HOT-ZONE AREA (lower = more focused)")
    print("=" * 60)
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r["hot_area_pct"])
    for model_name, areas in by_model.items():
        avg_area = sum(areas) / len(areas)
        acc = sum(1 for r in rows if r["model"] == model_name and r["correct"]) / len(areas) * 100
        print(f"{model_name:16s}  avg_hot_area={avg_area:6.2f}%   accuracy_on_sample={acc:5.1f}%")

    print(f"\nFull results: {csv_path}")
    print(f"Heatmap images: {OUT_DIR}/")


if __name__ == "__main__":
    main()
