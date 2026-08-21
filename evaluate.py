"""
evaluate.py — Generates metrics and plots for every trained checkpoint.

Run AFTER training (all checkpoints should already exist in checkpoints/).
Outputs go to evaluation_results/:
    confusion_matrix_<model>.png   — per model
    roc_curve_<model>.png          — per model, one-vs-rest per class
    per_class_metrics_<model>.csv  — precision/recall/f1/support per class
    summary_metrics.csv            — accuracy + macro P/R/F1 for every model
    comparison_accuracy_f1.png     — bar chart comparing all models

Usage:
    python evaluate.py --data-dir data
    python evaluate.py --data-dir data --models EfficientNetB0 ResNet18
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

# Reuse the exact architecture-building logic already in server.py, so the
# evaluated model definitions are guaranteed identical to what the server loads.
from server import (
    build_architecture,
    MODEL_CHECKPOINTS,
    IMPLEMENTED_ARCHITECTURES,
    CHECKPOINT_DIR,
    CLASS_NAMES,
    DISPLAY_LABELS,
)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "evaluation_results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def readable(label):
    return DISPLAY_LABELS.get(label, label)


def get_test_loader(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return dataset, loader


def evaluate_model(model_name, test_loader, num_classes):
    model = build_architecture(model_name, num_classes)
    checkpoint_path = CHECKPOINT_DIR / MODEL_CHECKPOINTS[model_name]
    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()

    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, class_names, model_name, out_dir):
    labels_readable = [readable(c) for c in class_names]
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(labels_readable, rotation=45, ha="right")
    ax.set_yticklabels(labels_readable)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name}")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / f"confusion_matrix_{model_name}.png", dpi=150)
    plt.close(fig)


def plot_roc_curves(y_true, y_probs, class_names, model_name, out_dir):
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for i, cname in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{readable(cname)} (AUC={roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves (one-vs-rest) — {model_name}")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"roc_curve_{model_name}.png", dpi=150)
    plt.close(fig)


def write_per_class_csv(y_true, y_pred, class_names, model_name, out_dir):
    labels_readable = [readable(c) for c in class_names]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )
    path = out_dir / f"per_class_metrics_{model_name}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Class", "Precision", "Recall", "F1", "Support"])
        for name, p, r, f1_, s in zip(labels_readable, precision, recall, f1, support):
            writer.writerow([name, f"{p:.4f}", f"{r:.4f}", f"{f1_:.4f}", int(s)])


def plot_comparison(summary_rows, out_dir):
    models = [r["Model"] for r in summary_rows]
    acc = [r["Accuracy"] for r in summary_rows]
    f1 = [r["Macro_F1"] for r in summary_rows]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 0.9), 5))
    ax.bar(x - width / 2, acc, width, label="Accuracy")
    ax.bar(x + width / 2, f1, width, label="Macro F1")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=40, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Test Set Accuracy vs Macro F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_accuracy_f1.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained GBC models on the test set")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Subset of models to evaluate (default: all with a ready checkpoint)")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if CLASS_NAMES is None:
        raise SystemExit("checkpoints/class_names.json not found — train at least one model first.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {DEVICE}")
    print(f"Classes (index order): {CLASS_NAMES}")

    test_dataset, test_loader = get_test_loader(args.data_dir, args.batch_size)
    print(f"Test set size: {len(test_dataset)} images")

    if test_dataset.classes != CLASS_NAMES:
        print("WARNING: data/test class folder order does not match checkpoints/class_names.json.")
        print(f"  test folders: {test_dataset.classes}")
        print(f"  trained on:   {CLASS_NAMES}")
        print("  Metrics below may be mislabeled — resolve this mismatch before trusting the results.")

    candidate_models = args.models if args.models else list(MODEL_CHECKPOINTS.keys())
    summary_rows = []

    for model_name in candidate_models:
        if model_name not in MODEL_CHECKPOINTS:
            print(f"Skipping unknown model: {model_name}")
            continue
        if model_name not in IMPLEMENTED_ARCHITECTURES:
            print(f"Skipping {model_name}: architecture not implemented yet")
            continue
        checkpoint_path = CHECKPOINT_DIR / MODEL_CHECKPOINTS[model_name]
        if not checkpoint_path.exists():
            print(f"Skipping {model_name}: no checkpoint at {checkpoint_path}")
            continue

        print(f"\nEvaluating {model_name}...")
        y_true, y_pred, y_probs = evaluate_model(model_name, test_loader, len(CLASS_NAMES))

        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        print(f"  Accuracy: {acc:.4f}  Macro P: {precision:.4f}  Macro R: {recall:.4f}  Macro F1: {f1:.4f}")

        plot_confusion_matrix(y_true, y_pred, CLASS_NAMES, model_name, RESULTS_DIR)
        plot_roc_curves(y_true, y_probs, CLASS_NAMES, model_name, RESULTS_DIR)
        write_per_class_csv(y_true, y_pred, CLASS_NAMES, model_name, RESULTS_DIR)

        summary_rows.append({
            "Model": model_name,
            "Accuracy": acc,
            "Macro_Precision": precision,
            "Macro_Recall": recall,
            "Macro_F1": f1,
            "Test_Size": len(y_true),
        })

    if not summary_rows:
        print("\nNo models were evaluated — check that checkpoints exist and match IMPLEMENTED_ARCHITECTURES.")
        return

    summary_path = RESULTS_DIR / "summary_metrics.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    plot_comparison(summary_rows, RESULTS_DIR)

    print(f"\nDone. All metrics and plots saved to: {RESULTS_DIR}")
    print(f"  - confusion_matrix_<model>.png  ({len(summary_rows)} files)")
    print(f"  - roc_curve_<model>.png         ({len(summary_rows)} files)")
    print(f"  - per_class_metrics_<model>.csv ({len(summary_rows)} files)")
    print("  - summary_metrics.csv")
    print("  - comparison_accuracy_f1.png")


if __name__ == "__main__":
    main()