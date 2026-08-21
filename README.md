# 🦠 Gallbladder Cancer Detection System from Ultrasound Images

A deep learning system and web application for classifying gallbladder ultrasound images across 5 clinical diagnostic categories using 9 convolutional neural network (CNN) architectures and multi-model ensemble analysis.

---

## 📌 Features & System Overview

- **Single-Page Diagnostic Dashboard (`index.html`):**
  - **Minimalist Dual Theme System:** Slate Navy (Dark) and Cool Porcelain (Light) theme switcher with preference persistence (`localStorage`).
  - **Clickable Model Status Chip:** Displays real-time server readiness (`9 / 9 models ready`) with interactive retry capability.
  - **Drag-and-Drop Dropzone:** Clickable and keyboard-accessible (`Tab`, `Enter`, `Space`) upload box with file chip badges and removable (`×`) action.
  - **Image Resolution Tag:** Floating metadata badge displaying actual uploaded image dimensions (e.g. `224 × 224 px`).
  - **Shimmer Loading Bar:** Pulsing glow dot and gentle gradient shimmer sweep during PyTorch inference.
- **Dual Analysis Modes:**
  - **🔍 Single Model Triage:** Select any individual model architecture to view diagnostic alerts, confidence metric cards, and multi-class probability distribution.
  - **📊 Compare All Models Ensemble:** Run parallel inference across all 9 models to calculate consensus agreement, animated horizontal confidence bar charts, and staggered breakdown tables.
  - **Segmented View Switcher:** Pill switcher control allowing users to toggle between `📊 Bar Chart View` and `📋 Breakdown Table View` one at a time.
- **Flask REST API Backend (`server.py`):**
  - Serves static dashboard at `http://localhost:5000/`.
  - CORS-enabled REST API with memory error recovery and automatic cache purging.
- **Streamlit Web UI (`app.py`):**
  - Alternative Streamlit dashboard interface for quick model evaluation.

---

## 📁 Dataset & Clinical Diagnostic Categories

Dataset source: [Gallbladder Cancer Ultrasound Dataset](https://www.kaggle.com/datasets/aneerbansaha/gallbladder-cancer/data) (2,294 ultrasound images).

The system classifies ultrasound images into 5 clinical diagnostic categories:

| Dataset Code | Diagnostic Class | Clinical Alert Tag | Badge Color | Color HEX |
| :--- | :--- | :--- | :--- | :--- |
| `nml` | **Normal** | `LOW CONCERN` | Emerald Green | `#10B981` |
| `bmt` | **Benign** | `MONITOR` | Amber Gold | `#F59E0B` |
| `stn` | **Gallstone** | `MONITOR` | Coral Orange | `#F97316` |
| `abn` | **Abnormal** | `REVIEW` | Violet Purple | `#8B5CF6` |
| `malg` | **Malignant** | `HIGH PRIORITY` | Crimson Red | `#EF4444` |

---

## 🧠 Supported Deep Learning Architectures

The system integrates 9 active model architectures trained using PyTorch transfer learning:

1. **DenseNet121** (`densenet121_gbc.pth`)
2. **EfficientNetB0** (`efficientnetb0_gbc.pth`)
3. **EfficientNetB1** (`efficientnetb1_gbc.pth`)
4. **EfficientNetB2** (`efficientnetb2_gbc.pth`)
5. **EfficientNetB3** (`efficientnetb3_gbc.pth`)
6. **EfficientNetB4** (`efficientnetb4_gbc.pth`)
7. **MobileNetV2** (`mobilenetv2_gbc.pth`)
8. **ResNet18** (`resnet18_gbc.pth`)
9. **ResNet50** (`resnet50_gbc.pth`)

---

## 🔌 API Endpoints (`server.py`)

The Flask server runs on `http://localhost:5000` and provides the following routes:

| Route | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | Serves the main single-page web dashboard ([index.html](file:///d:/Gallblader-Cancer-Detection-DL-main/index.html)). |
| `/status` | `GET` | Returns JSON dictionary indicating checkpoint readiness for all 9 model architectures. |
| `/predict` | `POST` | Accepts `multipart/form-data` with parameters `model` (string) and `image` (file). Returns JSON prediction, top confidence score, and 5-class distribution. |

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Slightsmile/gallbladder-cancer-detection-dl.git
cd gallbladder-cancer-detection-dl
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train / Prepare Model Checkpoints
```bash
python train.py --data-dir data --epochs 10
```
*Trained model weights (`.pth`) and class mapping (`class_names.json`) are automatically written to the `checkpoints/` directory.*

### 4. Launch the Web Application

Start the Flask server:
```bash
python server.py
```
Open your browser and navigate to:
```
http://localhost:5000/
```

*(Optional)* In a separate terminal, launch the Streamlit interface:
```bash
python -m streamlit run app.py
```

---

## 📈 Comprehensive Visual Plots Inventory

The evaluation pipeline automatically generates high-resolution diagnostic graphs categorized into organized subdirectories within `plots/`:

### 📊 Master Plot Catalog

| Category | Plot Name & File Path | Description |
| :--- | :--- | :--- |
| **01. Dataset Distribution** | [`plots/01_dataset_distribution/dataset_class_distribution.png`](file:///d:/Gallblader-Cancer-Detection-DL-main/plots/01_dataset_distribution/dataset_class_distribution.png) | Bar chart displaying ultrasound image counts across 5 categories (`abn`, `bmt`, `malg`, `nml`, `stn`) for Train, Validation, and Test splits. |
| **02. Confusion Matrices (Raw)** | [`plots/02_confusion_matrices/raw/confusion_matrices_grid_raw.png`](file:///d:/Gallblader-Cancer-Detection-DL-main/plots/02_confusion_matrices/raw/confusion_matrices_grid_raw.png) | 3x3 grid showing raw sample classification counts for all 9 model architectures. |
| | [`plots/02_confusion_matrices/raw/confusion_matrix_raw_<model>.png`](file:///d:/Gallblader-Cancer-Detection-DL-main/plots/02_confusion_matrices/raw) | Individual per-model raw count confusion matrix heatmaps (e.g. `confusion_matrix_raw_ResNet18.png`). |
| **02. Confusion Matrices (Norm)** | [`plots/02_confusion_matrices/normalized/confusion_matrices_grid_norm.png`](file:///d:/Gallblader-Cancer-Detection-DL-main/plots/02_confusion_matrices/normalized/confusion_matrices_grid_norm.png) | 3x3 grid displaying normalized classification percentage heatmaps across all 9 models. |
| | [`plots/02_confusion_matrices/normalized/confusion_matrix_norm_<model>.png`](file:///d:/Gallblader-Cancer-Detection-DL-main/plots/02_confusion_matrices/normalized) | Individual per-model normalized percentage confusion matrix heatmaps (e.g. `confusion_matrix_norm_ResNet18.png`). |
| **03. ROC Curves (One-vs-Rest)** | [`plots/03_roc_and_pr_curves/roc/roc_curve_<model>.png`](file:///d:/Gallblader-Cancer-Detection-DL-main/plots/03_roc_and_pr_curves/roc) | Multi-class Receiver Operating Characteristic (ROC) curves with AUC scores for each model architecture. |
| **03. Precision-Recall Curves** | [`plots/03_roc_and_pr_curves/pr/pr_curve_<model>.png`](file:///d:/Gallblader-Cancer-Detection-DL-main/plots/03_roc_and_pr_curves/pr) | One-vs-Rest Precision-Recall (PR) curves with Average Precision (AP) scores per class per model. |
| **04. Model Comparisons** | [`plots/04_model_comparisons/accuracy_vs_macro_f1_comparison.png`](file:///d:/Gallblader-Cancer-Detection-DL-main/plots/04_model_comparisons/accuracy_vs_macro_f1_comparison.png) | Grouped bar chart comparing Overall Accuracy (%) vs Macro F1-Score (%) across all 9 models. |
| | [`plots/04_model_comparisons/malignant_recall_comparison.png`](file:///d:/Gallblader-Cancer-Detection-DL-main/plots/04_model_comparisons/malignant_recall_comparison.png) | Bar chart callout highlighting Malignant-specific Diagnostic Sensitivity / Recall across all 9 models. |
| | [`plots/04_model_comparisons/accuracy_vs_model_size_latency.png`](file:///d:/Gallblader-Cancer-Detection-DL-main/plots/04_model_comparisons/accuracy_vs_model_size_latency.png) | Scatter plot auditing the trade-off between Model Size (Parameters in M), Inference Latency (ms), and Accuracy. |
| | [`plots/04_model_comparisons/per_class_metrics_resnet18.png`](file:///d:/Gallblader-Cancer-Detection-DL-main/plots/04_model_comparisons/per_class_metrics_resnet18.png) | Precision, Recall, and F1-Score breakdown across Abnormal, Benign, Malignant, Normal, and Gallstone classes. |
| **05. Training Curves** | [`plots/05_training_curves/training_loss_curves.png`](file:///d:/Gallblader-Cancer-Detection-DL-main/plots/05_training_curves/training_loss_curves.png) | Epoch-by-epoch Cross-Entropy Loss convergence curve for Training vs Validation phases. |
| | [`plots/05_training_curves/training_validation_accuracy_curves.png`](file:///d:/Gallblader-Cancer-Detection-DL-main/plots/05_training_curves/training_validation_accuracy_curves.png) | Epoch-by-epoch Diagnostic Accuracy (%) progression curve for Training vs Validation phases. |

---

## 📦 Project Directory Structure

```
├── checkpoints/              # Trained PyTorch model weights (.pth) & class_names.json
│   ├── class_names.json
│   ├── densenet121_gbc.pth
│   ├── efficientnetb0_gbc.pth
│   ├── efficientnetb1_gbc.pth
│   ├── efficientnetb2_gbc.pth
│   ├── efficientnetb3_gbc.pth
│   ├── efficientnetb4_gbc.pth
│   ├── mobilenetv2_gbc.pth
│   ├── resnet18_gbc.pth
│   └── resnet50_gbc.pth
├── plots/                    # Categorized diagnostic plot subdirectories
│   ├── 01_dataset_distribution/
│   ├── 02_confusion_matrices/
│   │   ├── raw/
│   │   └── normalized/
│   ├── 03_roc_and_pr_curves/
│   │   ├── roc/
│   │   └── pr/
│   ├── 04_model_comparisons/
│   └── 05_training_curves/
├── docs/                     # Documentation & audit reports
│   ├── DATASET_DOCUMENTATION.md # Comprehensive dataset breakdown & class statistics
│   ├── EVALUATION_REPORT.md      # Real PyTorch evaluation audit report
│   └── METRICS_DOCUMENTATION.md  # Detailed statistical breakdown across all models
├── index.html                # Single-page diagnostic web dashboard
├── server.py                 # Flask REST API server & static web host
├── app.py                    # Streamlit web application interface
├── evaluate.py               # Comprehensive model evaluation & metrics audit script
├── train.py                  # PyTorch model training & history logger script
├── requirements.txt          # Python package dependencies
└── README.md                 # Project documentation
```

---

## 📄 License & Acknowledgments

Developed for deep learning research in gallbladder ultrasound image classification and computer-aided diagnostics.
