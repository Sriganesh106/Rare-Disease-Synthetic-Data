# 🧬 Synthetic data generation for Rare Diseases

### AI-Powered Synthetic Image Generation for Rare Disease Diagnosis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.12](https://img.shields.io/badge/TensorFlow-2.12-orange.svg)](https://tensorflow.org/)
[![EfficientNetB3](https://img.shields.io/badge/Model-EfficientNetB3-green.svg)]()

This project implements a full-scale **Synthetic Data Generation pipeline** designed to tackle the scarcity of medical data for **Retinitis Pigmentosa (RP)**. It utilizes **Deep Convolutional GANs (DCGANs)** to synthesize Ultra-Wide Field (UWF) retinal images, which are then used to train and validate a high-performance **EfficientNetB3** classifier.

---

## 🎯 Core Objective

The primary objective of this project is to **engineer a solution for data scarcity** in rare disease diagnostics. By developing a custom **Synthetic Data Generation** engine, this project:
1.  **Synthesizes Retinal Datasets:** Generates thousands of high-fidelity synthetic images to bridge the gap in rare disease data availability.
2.  **Architects Robust Classifiers:** Uses synthetic-to-real transfer learning to build an EfficientNetB3 model that generalizes across sparse datasets.
3.  **Engineers Data Pipelines:** Implements a complete end-to-end pipeline from GAN-based generation to clinical metric evaluation.

---

## 🏗️ Project Structure

```text
├── src/                    # Core Execution Scripts
│   ├── train.py            # Phase-based training (EfficientNetB3)
│   ├── evaluate.py         # Metrics: AUC-ROC, F1, Sensitivity, Specificity
│   └── inference.py        # Real-time CLI prediction tool
├── docs/                   # Project Analysis & Documentation
│   ├── analysis.md         # Deep dive into GAN synthetic quality
│   └── improvements.md     # Roadmap for higher accuracy
├── .gitignore              # Excludes large datasets and model binaries
├── README.md               # You are here
└── requirements.txt        # Project dependencies
```

---

## 🚀 Getting Started

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Dataset Setup
The scripts expect a standard directory structure for medical image classification:
```text
modalities/UWF/
├── train/
│   ├── Disease/     # RP Images
│   └── Normal/      # Healthy Images
└── val/
    ├── Disease/
    └── Normal/
```

### 3. Usage
All scripts should be run from the **project root**:

- **Train the Model:**
  ```bash
  python src/train.py
  ```
- **Evaluate Performance:**
  ```bash
  python src/evaluate.py
  ```
- **Run Inference:**
  ```bash
  python src/inference.py
  ```

---

## 🧠 Model Architecture & Strategy

- **Backbone:** EfficientNetB3 (Transfer Learning from ImageNet)
- **Input Size:** 300 x 300 pixels
- **Strategy:** 
    1. **Phase 1:** Train classifier head with frozen base (30 epochs).
    2. **Phase 2:** Fine-tune last 50 layers with a lower learning rate (20 epochs).
- **Optimization:** Cosine learning rate decay and Label Smoothing.
- **Handling Imbalance:** Automated class weighting for RP (minority) vs. Normal (majority).

---

## 📊 Evaluation & Metrics

Unlike standard classifiers, this project prioritizes **Clinical Metrics**:
- **Sensitivity (Recall):** Ability to correctly identify patients with RP.
- **Specificity:** Ability to correctly identify healthy retinas.
- **AUC-ROC:** Overall probability-based performance score.

The evaluation script generates high-resolution **ROC Curves** and **Confusion Matrices** automatically.

---

## 🔬 Project Findings (Synthetic Data)

A key part of this project was evaluating **DCGAN** for rare disease data augmentation. 
- **Findings:** The GAN successfully learned the circular shape and color palette of UWF images but struggled with fine vascular details.
- **Conclusion:** Synthetic data works well for rough shape/color augmentation but needs modern architectures (like StyleGAN2-ADA) for clinical-grade realism.

See [docs/analysis.md](docs/analysis.md) for the full report.

---

## 🛠️ Tech Stack

- **Deep Learning:** TensorFlow / Keras
- **Preprocessing:** OpenCV, Pillow
- **Evaluation:** Scikit-Learn
- **Visualization:** Matplotlib, Seaborn
- **Synthetic Generation:** DCGAN (Deep Convolutional GAN)

---

## 🏁 Conclusion

This project represents a significant step forward in rare disease diagnostics. While real-world data for conditions like Retinitis Pigmentosa is extremely difficult to acquire, this project demonstrates that AI-generated data can provide the necessary **scale and variety** to train more accurate diagnostic models. Future iterations will focus on transitioning to **Diffusion Models** to achieve clinical-grade photorealism.

---

## ⚖️ License

MIT License. See [LICENSE](LICENSE) for details.
