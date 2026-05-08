# 🚀 How to Get the Best Results — Rare Disease Detection

## 🏆 Complete Improvement Plan (Priority Order)

### 1. Fix Preprocessing Consistency (CRITICAL)
Your **training** used `preprocess_input()` (scales to [-1, 1]), but **evaluation** used `/255.0`. This mismatch has been fixed in the current `src/` scripts.

### 2. Use a Stronger Backbone Model
MobileNetV2 is for speed. We have switched the project to **EfficientNetB3** for better accuracy.

| Model | Top-1 Accuracy (ImageNet) | Params | Best For |
|-------|--------------------------|--------|----------|
| MobileNetV2 | 71.3% | 3.4M | Speed |
| **EfficientNetB3** | 81.6% | 12M | ⭐ Balance |
| **EfficientNetV2-S** | 83.9% | 21M | ⭐⭐ Max Accuracy |

### 3. Use Larger Image Size
Retinal details (vessels, pigment) are fine. We moved from **224px** to **300px**.

### 4. Add Class Weights to Handle Imbalance
With 223 Normal vs 150 RP, the model needs to "pay more attention" to the RP cases. The new training script handles this automatically.

### 5. Improve the GAN (Synthetic Data)
Your current DCGAN produces low-quality blobs. 
- **Option A**: Use **StyleGAN2-ADA** (designed for small datasets).
- **Option B**: Use **Diffusion Models** for photorealistic retinal generation.

### 6. Medical Evaluation Metrics
Accuracy is not enough. We have added **AUC-ROC, Sensitivity, and Specificity** to the evaluation script.

### 7. Explainability with Grad-CAM
In the future, adding Grad-CAM will allow you to see exactly *which* part of the retina the AI is looking at to make its decision.
