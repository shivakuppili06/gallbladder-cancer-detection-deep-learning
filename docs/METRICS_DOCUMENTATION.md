# 📊 Gallbladder Cancer Detection System — Complete Model Metrics Documentation

This document provides a comprehensive statistical breakdown of all evaluated performance metrics across all 9 deep learning model architectures.

---

## 🚨 1. Malignant-Class Diagnostic Sensitivity (Recall Audit)

| Model Architecture | Overall Accuracy | Balanced Accuracy | Malignant Recall (Sensitivity) | Malignant Precision | Malignant F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet18** | 87.76% | 86.12% | **92.50%** | 69.81% | 79.57% | 40 |
| **DenseNet121** | 86.88% | 82.85% | **77.50%** | 67.39% | 72.09% | 40 |
| **EfficientNetB3** | 85.42% | 82.54% | **80.00%** | 64.00% | 71.11% | 40 |
| **ResNet50** | 83.67% | 81.99% | **87.50%** | 62.50% | 72.92% | 40 |
| **EfficientNetB4** | 83.38% | 81.05% | **85.00%** | 60.71% | 70.83% | 40 |
| **MobileNetV2** | 83.38% | 77.24% | **82.50%** | 66.00% | 73.33% | 40 |
| **EfficientNetB0** | 82.80% | 80.09% | **77.50%** | 57.41% | 65.96% | 40 |
| **EfficientNetB2** | 81.92% | 80.08% | **82.50%** | 62.26% | 70.97% | 40 |
| **EfficientNetB1** | 81.63% | 78.98% | **87.50%** | 53.85% | 66.67% | 40 |

---

## 📊 2. Overall Model Metrics Table (Macro & Weighted Averages)

| Model Architecture | Accuracy | Balanced Acc | Macro Precision | Macro Recall | Macro Specificity | Macro F1 | Weighted F1 | MCC | Latency (ms) | Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet18** | 87.76% | 86.12% | 84.81% | 86.12% | 96.91% | 85.08% | 87.87% | 0.8396 | 77.20 ms | 11.18M |
| **DenseNet121** | 86.88% | 82.85% | 85.17% | 82.85% | 96.55% | 83.55% | 86.83% | 0.8257 | 98.57 ms | 6.96M |
| **EfficientNetB3** | 85.42% | 82.54% | 81.70% | 82.54% | 96.33% | 81.87% | 85.58% | 0.8084 | 62.53 ms | 10.70M |
| **ResNet50** | 83.67% | 81.99% | 80.42% | 81.99% | 95.90% | 80.57% | 83.86% | 0.7880 | 174.55 ms | 23.52M |
| **EfficientNetB4** | 83.38% | 81.05% | 79.97% | 81.05% | 95.82% | 79.99% | 83.64% | 0.7830 | 124.22 ms | 17.56M |
| **MobileNetV2** | 83.38% | 77.24% | 79.02% | 77.24% | 95.68% | 77.23% | 82.91% | 0.7796 | 35.08 ms | 2.23M |
| **EfficientNetB0** | 82.80% | 80.09% | 79.84% | 80.09% | 95.59% | 79.60% | 83.16% | 0.7737 | 29.88 ms | 4.01M |
| **EfficientNetB2** | 81.92% | 80.08% | 77.40% | 80.08% | 95.50% | 78.25% | 82.43% | 0.7646 | 43.10 ms | 7.71M |
| **EfficientNetB1** | 81.63% | 78.98% | 77.81% | 78.98% | 95.48% | 77.34% | 82.05% | 0.7638 | 40.22 ms | 6.52M |

---

## 🎯 3. Complete Per-Class Metrics Breakdown for All 9 Models

### 🔹 ResNet18

| Class Code | Class Name | Support | Precision | Sensitivity (Recall) | Specificity | F1-Score | AUC-ROC | Avg Precision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `abn` | **ABN** | 123 | 91.30% | 85.37% | 95.45% | 88.24% | 0.9733 | 0.9560 |
| `bmt` | **BMT** | 29 | 76.92% | 68.97% | 98.09% | 72.73% | 0.9551 | 0.8305 |
| `malg` | **MALG** | 40 | 69.81% | 92.50% | 94.72% | 79.57% | 0.9813 | 0.8540 |
| `nml` | **NML** | 65 | 90.77% | 90.77% | 97.84% | 90.77% | 0.9837 | 0.9557 |
| `stn` | **STN** | 86 | 95.24% | 93.02% | 98.44% | 94.12% | 0.9924 | 0.9771 |

### 🔹 DenseNet121

| Class Code | Class Name | Support | Precision | Sensitivity (Recall) | Specificity | F1-Score | AUC-ROC | Avg Precision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `abn` | **ABN** | 123 | 88.52% | 87.80% | 93.64% | 88.16% | 0.9711 | 0.9510 |
| `bmt` | **BMT** | 29 | 85.71% | 62.07% | 99.04% | 72.00% | 0.9506 | 0.7895 |
| `malg` | **MALG** | 40 | 67.39% | 77.50% | 95.05% | 72.09% | 0.9656 | 0.8173 |
| `nml` | **NML** | 65 | 95.31% | 93.85% | 98.92% | 94.57% | 0.9940 | 0.9746 |
| `stn` | **STN** | 86 | 88.89% | 93.02% | 96.11% | 90.91% | 0.9914 | 0.9753 |

### 🔹 EfficientNetB3

| Class Code | Class Name | Support | Precision | Sensitivity (Recall) | Specificity | F1-Score | AUC-ROC | Avg Precision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `abn` | **ABN** | 123 | 91.23% | 84.55% | 95.45% | 87.76% | 0.9741 | 0.9524 |
| `bmt` | **BMT** | 29 | 73.08% | 65.52% | 97.77% | 69.09% | 0.9395 | 0.8007 |
| `malg` | **MALG** | 40 | 64.00% | 80.00% | 94.06% | 71.11% | 0.9682 | 0.8034 |
| `nml` | **NML** | 65 | 89.39% | 90.77% | 97.48% | 90.08% | 0.9844 | 0.9460 |
| `stn` | **STN** | 86 | 90.80% | 91.86% | 96.89% | 91.33% | 0.9909 | 0.9722 |

### 🔹 ResNet50

| Class Code | Class Name | Support | Precision | Sensitivity (Recall) | Specificity | F1-Score | AUC-ROC | Avg Precision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `abn` | **ABN** | 123 | 89.19% | 80.49% | 94.55% | 84.62% | 0.9647 | 0.9404 |
| `bmt` | **BMT** | 29 | 72.00% | 62.07% | 97.77% | 66.67% | 0.9420 | 0.7756 |
| `malg` | **MALG** | 40 | 62.50% | 87.50% | 93.07% | 72.92% | 0.9677 | 0.7691 |
| `nml` | **NML** | 65 | 83.56% | 93.85% | 95.68% | 88.41% | 0.9855 | 0.9550 |
| `stn` | **STN** | 86 | 94.87% | 86.05% | 98.44% | 90.24% | 0.9887 | 0.9677 |

### 🔹 EfficientNetB4

| Class Code | Class Name | Support | Precision | Sensitivity (Recall) | Specificity | F1-Score | AUC-ROC | Avg Precision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `abn` | **ABN** | 123 | 89.38% | 82.11% | 94.55% | 85.59% | 0.9672 | 0.9470 |
| `bmt` | **BMT** | 29 | 72.00% | 62.07% | 97.77% | 66.67% | 0.9392 | 0.7703 |
| `malg` | **MALG** | 40 | 60.71% | 85.00% | 92.74% | 70.83% | 0.9651 | 0.7674 |
| `nml` | **NML** | 65 | 85.07% | 87.69% | 96.40% | 86.36% | 0.9844 | 0.9259 |
| `stn` | **STN** | 86 | 92.68% | 88.37% | 97.67% | 90.48% | 0.9891 | 0.9727 |

### 🔹 MobileNetV2

| Class Code | Class Name | Support | Precision | Sensitivity (Recall) | Specificity | F1-Score | AUC-ROC | Avg Precision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `abn` | **ABN** | 123 | 87.70% | 86.99% | 93.18% | 87.35% | 0.9672 | 0.9401 |
| `bmt` | **BMT** | 29 | 64.71% | 37.93% | 98.09% | 47.83% | 0.9227 | 0.6319 |
| `malg` | **MALG** | 40 | 66.00% | 82.50% | 94.39% | 73.33% | 0.9625 | 0.7690 |
| `nml` | **NML** | 65 | 92.06% | 89.23% | 98.20% | 90.62% | 0.9848 | 0.9165 |
| `stn` | **STN** | 86 | 84.62% | 89.53% | 94.55% | 87.01% | 0.9847 | 0.9604 |

### 🔹 EfficientNetB0

| Class Code | Class Name | Support | Precision | Sensitivity (Recall) | Specificity | F1-Score | AUC-ROC | Avg Precision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `abn` | **ABN** | 123 | 85.71% | 82.93% | 92.27% | 84.30% | 0.9663 | 0.9421 |
| `bmt` | **BMT** | 29 | 73.08% | 65.52% | 97.77% | 69.09% | 0.9298 | 0.7593 |
| `malg` | **MALG** | 40 | 57.41% | 77.50% | 92.41% | 65.96% | 0.9667 | 0.7408 |
| `nml` | **NML** | 65 | 90.32% | 86.15% | 97.84% | 88.19% | 0.9816 | 0.9470 |
| `stn` | **STN** | 86 | 92.68% | 88.37% | 97.67% | 90.48% | 0.9905 | 0.9702 |

### 🔹 EfficientNetB2

| Class Code | Class Name | Support | Precision | Sensitivity (Recall) | Specificity | F1-Score | AUC-ROC | Avg Precision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `abn` | **ABN** | 123 | 87.83% | 82.11% | 93.64% | 84.87% | 0.9679 | 0.9450 |
| `bmt` | **BMT** | 29 | 55.88% | 65.52% | 95.22% | 60.32% | 0.9511 | 0.7455 |
| `malg` | **MALG** | 40 | 62.26% | 82.50% | 93.40% | 70.97% | 0.9663 | 0.7343 |
| `nml` | **NML** | 65 | 85.07% | 87.69% | 96.40% | 86.36% | 0.9831 | 0.9578 |
| `stn` | **STN** | 86 | 95.95% | 82.56% | 98.83% | 88.75% | 0.9923 | 0.9787 |

### 🔹 EfficientNetB1

| Class Code | Class Name | Support | Precision | Sensitivity (Recall) | Specificity | F1-Score | AUC-ROC | Avg Precision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `abn` | **ABN** | 123 | 90.57% | 78.05% | 95.45% | 83.84% | 0.9645 | 0.9318 |
| `bmt` | **BMT** | 29 | 65.22% | 51.72% | 97.45% | 57.69% | 0.9336 | 0.6924 |
| `malg` | **MALG** | 40 | 53.85% | 87.50% | 90.10% | 66.67% | 0.9578 | 0.7285 |
| `nml` | **NML** | 65 | 87.88% | 89.23% | 97.12% | 88.55% | 0.9789 | 0.9391 |
| `stn` | **STN** | 86 | 91.57% | 88.37% | 97.28% | 89.94% | 0.9879 | 0.9657 |

