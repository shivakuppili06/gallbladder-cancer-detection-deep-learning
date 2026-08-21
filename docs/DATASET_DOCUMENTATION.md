# 🩺 Gallbladder Cancer Ultrasound Dataset — Comprehensive Documentation

This document provides an exhaustive technical and clinical breakdown of the ultrasound image dataset used for training, validating, and evaluating 9 deep learning architectures for **Gallbladder Cancer (GBC) Detection and Diagnostic Classification**.

---

## 📌 1. Overview & Clinical Context

Gallbladder Cancer (GBC) is an aggressive gastrointestinal malignancy with high mortality if detected at advanced stages. Transabdominal Ultrasound (US) is the primary non-invasive modality used for gallbladder screening. 

This dataset comprises **2,294 high-resolution ultrasound images** categorized into **5 distinct diagnostic classes** representing normal anatomy, benign pathologies, inflammatory conditions, gallstones, and malignant carcinoma.

---

## 🏷️ 2. Diagnostic Category Definitions

| Class Code | Class Name | Clinical & Radiologic Description | Risk & Significance |
| :--- | :--- | :--- | :--- |
| `nml` | **Normal (Normal Gallbladder)** | Thin, smooth gallbladder wall (< 3mm), clear anechoic lumen without sludge, acoustic shadowing, or focal lesions. | Normal / Healthy baseline |
| `abn` | **Abnormal (Wall Thickening / Inflammation)** | Diffuse or focal wall thickening (> 3mm), mural edema, or acute/chronic cholecystitis signs without discrete mass. | Inflammatory / High-monitoring required |
| `bmt` | **Benign Mass / Tumor** | Well-circumscribed mucosal polyps, adenomyomatosis, or benign tumors (< 10mm) lacking invasion. | Low-risk benign neoplasm / Surveillance |
| `stn` | **Gallstones (Cholelithiasis)** | Echogenic foci within the gallbladder lumen producing distinct posterior acoustic shadowing. | Non-malignant stone disease / Very common |
| `malg` | **Malignant (Gallbladder Carcinoma)** | Irregular polypoid/fungating lumen mass, asymmetric wall replacement, direct liver parenchymal invasion, or loss of wall interface. | **High-Risk Malignancy (Urgent Intervention)** |

---

## 📊 3. Dataset Split Statistics

The dataset is partitioned into **Training (70%)**, **Validation (15%)**, and **Test (15%)** sets using stratified sampling to maintain class proportions across splits.

### Numerical Distribution Table:

| Diagnostic Category | Class Code | Training Set | Validation Set | Test Set | Total Images | Class Share (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Abnormal (Inflammation)** | `abn` | 576 | 124 | 123 | **823** | 35.88% |
| **Benign Mass / Tumor** | `bmt` | 138 | 30 | 29 | **197** | 8.59% |
| **Malignant Carcinoma** | `malg` | 190 | 41 | 40 | **271** | 11.81% |
| **Normal Gallbladder** | `nml` | 302 | 65 | 65 | **432** | 18.83% |
| **Gallstones (Cholelithiasis)** | `stn` | 399 | 86 | 86 | **571** | 24.89% |
| **TOTAL** | -- | **1,605** | **346** | **343** | **2,294** | **100.00%** |

---

## ⚙️ 4. Image Preprocessing & Input Pipelines

All images undergo standardized preprocessing prior to model feature extraction and classification:

1. **Resolution & Resizing:** Resized to $224 \times 224$ pixels using bilinear interpolation.
2. **Channel Format:** RGB 3-channel input format.
3. **Normalization (Standard ImageNet):**
   $$\text{Mean: } [0.485, 0.456, 0.406], \quad \text{Std: } [0.229, 0.224, 0.225]$$
4. **Data Augmentation (Training Only):**
   - Random Horizontal & Vertical Flips ($p = 0.5$)
   - Random Affine Rotations ($\pm 15^\circ$)
   - Color Jitter (Brightness/Contrast Adjustments)

---

## 📂 5. Directory Structure Mapping

```
data/
├── training/
│   ├── abn/ (576 images)
│   ├── bmt/ (138 images)
│   ├── malg/ (190 images)
│   ├── nml/ (302 images)
│   └── stn/ (399 images)
├── validation/
│   ├── abn/ (124 images)
│   ├── bmt/ (30 images)
│   ├── malg/ (41 images)
│   ├── nml/ (65 images)
│   └── stn/ (86 images)
└── test/
    ├── abn/ (123 images)
    ├── bmt/ (29 images)
    ├── malg/ (40 images)
    ├── nml/ (65 images)
    └── stn/ (86 images)
```

---

## 📈 6. Visual Class Distribution Plot

The distribution chart is saved at:
`plots/01_dataset_distribution/dataset_class_distribution.png`
