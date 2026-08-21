# 🦠 Gallbladder Cancer Detection System — Real PyTorch Model Evaluation Report

All metrics, confusion matrices, and ROC/PR curves in this document are computed directly from real inference executed on the test dataset split (`data/test`).

---

## 🚨 1. Malignant-Specific Diagnostic Sensitivity (Callout)

> **CLINICAL AUDIT SUMMARY**  
> **Top Model (ResNet18) Malignant Recall (Sensitivity):** `92.50%`  
> **Overall Test Accuracy:** `87.76%` | **Balanced Accuracy:** `86.12%` | **MCC:** `0.8396`

### Malignant Recall Comparison Across All Models:

| Model Architecture | Overall Accuracy | Balanced Accuracy | Malignant Recall (Sensitivity) | Malignant Precision | Malignant F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ResNet18** | 87.76% | 86.12% | **92.50%** | 69.81% | 79.57% |
| **DenseNet121** | 86.88% | 82.85% | **77.50%** | 67.39% | 72.09% |
| **EfficientNetB3** | 85.42% | 82.54% | **80.00%** | 64.00% | 71.11% |
| **ResNet50** | 83.67% | 81.99% | **87.50%** | 62.50% | 72.92% |
| **EfficientNetB4** | 83.38% | 81.05% | **85.00%** | 60.71% | 70.83% |
| **MobileNetV2** | 83.38% | 77.24% | **82.50%** | 66.00% | 73.33% |
| **EfficientNetB0** | 82.80% | 80.09% | **77.50%** | 57.41% | 65.96% |
| **EfficientNetB2** | 81.92% | 80.08% | **82.50%** | 62.26% | 70.97% |
| **EfficientNetB1** | 81.63% | 78.98% | **87.50%** | 53.85% | 66.67% |

---

## 📊 2. Overall Model Metrics Table

| Model Architecture | Accuracy | Balanced Acc | Macro Precision | Macro Recall | Macro Specificity | Macro F1 | Weighted F1 | MCC | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet18** | 87.76% | 86.12% | 84.81% | 86.12% | 96.91% | 85.08% | 87.87% | 0.8396 | 77.20 ms |
| **DenseNet121** | 86.88% | 82.85% | 85.17% | 82.85% | 96.55% | 83.55% | 86.83% | 0.8257 | 98.57 ms |
| **EfficientNetB3** | 85.42% | 82.54% | 81.70% | 82.54% | 96.33% | 81.87% | 85.58% | 0.8084 | 62.53 ms |
| **ResNet50** | 83.67% | 81.99% | 80.42% | 81.99% | 95.90% | 80.57% | 83.86% | 0.7880 | 174.55 ms |
| **EfficientNetB4** | 83.38% | 81.05% | 79.97% | 81.05% | 95.82% | 79.99% | 83.64% | 0.7830 | 124.22 ms |
| **MobileNetV2** | 83.38% | 77.24% | 79.02% | 77.24% | 95.68% | 77.23% | 82.91% | 0.7796 | 35.08 ms |
| **EfficientNetB0** | 82.80% | 80.09% | 79.84% | 80.09% | 95.59% | 79.60% | 83.16% | 0.7737 | 29.88 ms |
| **EfficientNetB2** | 81.92% | 80.08% | 77.40% | 80.08% | 95.50% | 78.25% | 82.43% | 0.7646 | 43.10 ms |
| **EfficientNetB1** | 81.63% | 78.98% | 77.81% | 78.98% | 95.48% | 77.34% | 82.05% | 0.7638 | 40.22 ms |

---

## 🎯 3. Detailed Per-Class Breakdown Across All 9 Models

### 🔹 Model Architecture: `ResNet18` (Accuracy: `87.76%`, MCC: `0.8396`)

| Class Code | Support (Count) | Precision | Sensitivity (Recall) | Specificity | F1-Score | AUC-ROC | Avg Precision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `abn` | 123 | 91.30% | 85.37% | 95.45% | 88.24% | 0.9733 | 0.9560 |
| `bmt` | 29 | 76.92% | 68.97% | 98.09% | 72.73% | 0.9551 | 0.8305 |
| `malg` | 40 | 69.81% | 92.50% | 94.72% | 79.57% | 0.9813 | 0.8540 |
| `nml` | 65 | 90.77% | 90.77% | 97.84% | 90.77% | 0.9837 | 0.9557 |
| `stn` | 86 | 95.24% | 93.02% | 98.44% | 94.12% | 0.9924 | 0.9771 |

### 🔹 Model Architecture: `DenseNet121` (Accuracy: `86.88%`, MCC: `0.8257`)

| Class Code | Support (Count) | Precision | Sensitivity (Recall) | Specificity | F1-Score | AUC-ROC | Avg Precision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `abn` | 123 | 88.52% | 87.80% | 93.64% | 88.16% | 0.9711 | 0.9510 |
| `bmt` | 29 | 85.71% | 62.07% | 99.04% | 72.00% | 0.9506 | 0.7895 |
| `malg` | 40 | 67.39% | 77.50% | 95.05% | 72.09% | 0.9656 | 0.8173 |
| `nml` | 65 | 95.31% | 93.85% | 98.92% | 94.57% | 0.9940 | 0.9746 |
| `stn` | 86 | 88.89% | 93.02% | 96.11% | 90.91% | 0.9914 | 0.9753 |

### 🔹 Model Architecture: `EfficientNetB3` (Accuracy: `85.42%`, MCC: `0.8084`)

| Class Code | Support (Count) | Precision | Sensitivity (Recall) | Specificity | F1-Score | AUC-ROC | Avg Precision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `abn` | 123 | 91.23% | 84.55% | 95.45% | 87.76% | 0.9741 | 0.9524 |
| `bmt` | 29 | 73.08% | 65.52% | 97.77% | 69.09% | 0.9395 | 0.8007 |
| `malg` | 40 | 64.00% | 80.00% | 94.06% | 71.11% | 0.9682 | 0.8034 |
| `nml` | 65 | 89.39% | 90.77% | 97.48% | 90.08% | 0.9844 | 0.9460 |
| `stn` | 86 | 90.80% | 91.86% | 96.89% | 91.33% | 0.9909 | 0.9722 |

### 🔹 Model Architecture: `ResNet50` (Accuracy: `83.67%`, MCC: `0.7880`)

| Class Code | Support (Count) | Precision | Sensitivity (Recall) | Specificity | F1-Score | AUC-ROC | Avg Precision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `abn` | 123 | 89.19% | 80.49% | 94.55% | 84.62% | 0.9647 | 0.9404 |
| `bmt` | 29 | 72.00% | 62.07% | 97.77% | 66.67% | 0.9420 | 0.7756 |
| `malg` | 40 | 62.50% | 87.50% | 93.07% | 72.92% | 0.9677 | 0.7691 |
| `nml` | 65 | 83.56% | 93.85% | 95.68% | 88.41% | 0.9855 | 0.9550 |
| `stn` | 86 | 94.87% | 86.05% | 98.44% | 90.24% | 0.9887 | 0.9677 |

### 🔹 Model Architecture: `EfficientNetB4` (Accuracy: `83.38%`, MCC: `0.7830`)

| Class Code | Support (Count) | Precision | Sensitivity (Recall) | Specificity | F1-Score | AUC-ROC | Avg Precision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `abn` | 123 | 89.38% | 82.11% | 94.55% | 85.59% | 0.9672 | 0.9470 |
| `bmt` | 29 | 72.00% | 62.07% | 97.77% | 66.67% | 0.9392 | 0.7703 |
| `malg` | 40 | 60.71% | 85.00% | 92.74% | 70.83% | 0.9651 | 0.7674 |
| `nml` | 65 | 85.07% | 87.69% | 96.40% | 86.36% | 0.9844 | 0.9259 |
| `stn` | 86 | 92.68% | 88.37% | 97.67% | 90.48% | 0.9891 | 0.9727 |

### 🔹 Model Architecture: `MobileNetV2` (Accuracy: `83.38%`, MCC: `0.7796`)

| Class Code | Support (Count) | Precision | Sensitivity (Recall) | Specificity | F1-Score | AUC-ROC | Avg Precision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `abn` | 123 | 87.70% | 86.99% | 93.18% | 87.35% | 0.9672 | 0.9401 |
| `bmt` | 29 | 64.71% | 37.93% | 98.09% | 47.83% | 0.9227 | 0.6319 |
| `malg` | 40 | 66.00% | 82.50% | 94.39% | 73.33% | 0.9625 | 0.7690 |
| `nml` | 65 | 92.06% | 89.23% | 98.20% | 90.62% | 0.9848 | 0.9165 |
| `stn` | 86 | 84.62% | 89.53% | 94.55% | 87.01% | 0.9847 | 0.9604 |

### 🔹 Model Architecture: `EfficientNetB0` (Accuracy: `82.80%`, MCC: `0.7737`)

| Class Code | Support (Count) | Precision | Sensitivity (Recall) | Specificity | F1-Score | AUC-ROC | Avg Precision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `abn` | 123 | 85.71% | 82.93% | 92.27% | 84.30% | 0.9663 | 0.9421 |
| `bmt` | 29 | 73.08% | 65.52% | 97.77% | 69.09% | 0.9298 | 0.7593 |
| `malg` | 40 | 57.41% | 77.50% | 92.41% | 65.96% | 0.9667 | 0.7408 |
| `nml` | 65 | 90.32% | 86.15% | 97.84% | 88.19% | 0.9816 | 0.9470 |
| `stn` | 86 | 92.68% | 88.37% | 97.67% | 90.48% | 0.9905 | 0.9702 |

### 🔹 Model Architecture: `EfficientNetB2` (Accuracy: `81.92%`, MCC: `0.7646`)

| Class Code | Support (Count) | Precision | Sensitivity (Recall) | Specificity | F1-Score | AUC-ROC | Avg Precision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `abn` | 123 | 87.83% | 82.11% | 93.64% | 84.87% | 0.9679 | 0.9450 |
| `bmt` | 29 | 55.88% | 65.52% | 95.22% | 60.32% | 0.9511 | 0.7455 |
| `malg` | 40 | 62.26% | 82.50% | 93.40% | 70.97% | 0.9663 | 0.7343 |
| `nml` | 65 | 85.07% | 87.69% | 96.40% | 86.36% | 0.9831 | 0.9578 |
| `stn` | 86 | 95.95% | 82.56% | 98.83% | 88.75% | 0.9923 | 0.9787 |

### 🔹 Model Architecture: `EfficientNetB1` (Accuracy: `81.63%`, MCC: `0.7638`)

| Class Code | Support (Count) | Precision | Sensitivity (Recall) | Specificity | F1-Score | AUC-ROC | Avg Precision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `abn` | 123 | 90.57% | 78.05% | 95.45% | 83.84% | 0.9645 | 0.9318 |
| `bmt` | 29 | 65.22% | 51.72% | 97.45% | 57.69% | 0.9336 | 0.6924 |
| `malg` | 40 | 53.85% | 87.50% | 90.10% | 66.67% | 0.9578 | 0.7285 |
| `nml` | 65 | 87.88% | 89.23% | 97.12% | 88.55% | 0.9789 | 0.9391 |
| `stn` | 86 | 91.57% | 88.37% | 97.28% | 89.94% | 0.9879 | 0.9657 |

---

## 📈 4. Visual Graphs Audit List

- **Dataset Class Distribution:** `plots/01_dataset_distribution/dataset_class_distribution.png`
- **Raw & Normalized Confusion Matrices:** `plots/02_confusion_matrices/raw/`, `plots/02_confusion_matrices/normalized/`
- **One-vs-Rest ROC & PR Curves:** `plots/03_roc_and_pr_curves/roc/`, `plots/03_roc_and_pr_curves/pr/`
- **Accuracy vs. Macro-F1 Bar Comparison:** `plots/04_model_comparisons/accuracy_vs_macro_f1_comparison.png`
- **Malignant Recall Sensitivity Bar Comparison:** `plots/04_model_comparisons/malignant_recall_comparison.png`
- **Accuracy vs. Size & Latency Scatter:** `plots/04_model_comparisons/accuracy_vs_model_size_latency.png`
- **Training & Validation Epoch Loss/Acc Curves:** `plots/05_training_curves/training_loss_curves.png`, `plots/05_training_curves/training_validation_accuracy_curves.png`
