# Rare Disease Detection — Project Analysis

## What This Project Is About

This is a **deep learning project for detecting Retinitis Pigmentosa (RP)** — a rare, inherited eye disease that causes progressive vision loss — using **Ultra-Wide Field (UWF) retinal fundus images** captured by **Optos** cameras.

The project has **two main components**:

### 1. GAN-Based Synthetic Data Generation
A **DCGAN (Deep Convolutional Generative Adversarial Network)** was trained to generate synthetic retinal images for **data augmentation**, addressing the limited availability of rare disease datasets.

| Class | Real Images | Synthetic Images Generated |
|-------|------------|---------------------------|
| Normal | 223 (OPTOS_NORMAL) | 2,500 epochs of 4×4 grids |
| RP (Disease) | 150 (UWPC_RP) | 1,066 epochs of 4×4 grids |

---

## Synthetic Image Quality Assessment

### Training Progression ✅
The GAN training shows the **expected learning curve**:

| Stage | Epoch | Observation |
|-------|-------|-------------|
| Early | ~1 | Pure grey/noise — generator has not learned anything yet |
| Mid | ~300-500 | Circular blob shapes emerge, green/orange color palette appears |
| Late | ~1000-2500 | Recognizable retinal-like circular shapes with color gradients |

### Comparison: Real vs. Synthetic

**Real images** (Optos UWF) are:
- High resolution (~700-1000KB each)
- Show clear optic disc, blood vessels, retinal vasculature, and the characteristic circular UWF field of view
- RP images show distinct bone spicule pigmentation, vessel attenuation, and macular changes

**Synthetic images** (GAN output at final epochs) are:
- Low resolution (~100-130KB, displayed as 4×4 grids of small thumbnails)
- Show the general **circular shape** and **green/orange color palette** of UWF images
- Capture the overall **color distribution** and **dark-border-to-bright-center** gradient

### Verdict: Partially Correct, But Low Quality

> [!WARNING]
> The synthetic images are **NOT clinically realistic**. They are low-resolution blobs that only approximate the gross color/shape characteristics of UWF fundus images.

**What the GAN learned ✅:**
- Overall circular fundus shape
- Green/red-orange color palette typical of Optos UWF imaging
- Dark peripheral borders
- Basic color difference between Normal (more orange/red tones) and RP (more green/muted tones)

**What the GAN did NOT learn ❌:**
- Blood vessel structure — completely absent
- Optic disc — not visible
- Fine retinal details or texture
- RP-specific features (bone spicule pigmentation, vessel narrowing)
- Realistic resolution — images are very small and blurry

> [!IMPORTANT]
> **For a technical project**, this level of synthetic data is common when using a basic DCGAN on a very small dataset (~150-223 images). The synthetic images might still help as rough augmentation to prevent overfitting, but they would **not pass clinical review** and should not be used as standalone training data.
