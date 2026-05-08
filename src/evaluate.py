import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    roc_curve
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

DATA_DIR = "modalities/UWF_fixed"
MODEL_PATH = "models/UWF_EfficientNetB3_BEST.h5"

IMG_SIZE = 300  # ↑ was 224 — matches new EfficientNetB3 training size

print("📂 Loading test dataset...")

# FIXED: Use EfficientNet preprocessing instead of rescale=1/255
test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)

test_gen = test_datagen.flow_from_directory(
    DATA_DIR + "/test",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=1,
    class_mode="categorical",
    shuffle=False
)

print("🔍 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Predict all images
pred_probs = model.predict(test_gen)
y_pred = np.argmax(pred_probs, axis=1)
y_true = test_gen.classes

labels = list(test_gen.class_indices.keys())

# ═══════════════════════════════════════════════════════════
# COMPREHENSIVE METRICS (not just accuracy)
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 50)
print("📊 CLASSIFICATION REPORT")
print("=" * 50)
print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))

print("\n📌 Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
print(cm)

# AUC-ROC Score
try:
    auc_score = roc_auc_score(y_true, pred_probs[:, 1])
    print(f"\n🎯 AUC-ROC Score: {auc_score:.4f}")
except Exception as e:
    print(f"\n⚠️ Could not compute AUC-ROC: {e}")

# F1 Score
f1 = f1_score(y_true, y_pred, average="weighted")
print(f"📊 F1-Score (weighted): {f1:.4f}")

# Sensitivity (Recall for Disease class) and Specificity
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"\n🏥 Medical Metrics:")
print(f"   Sensitivity (RP detection rate):  {sensitivity:.4f} ({sensitivity*100:.1f}%)")
print(f"   Specificity (Normal correct rate): {specificity:.4f} ({specificity*100:.1f}%)")
print(f"   True Positives:  {tp}")
print(f"   True Negatives:  {tn}")
print(f"   False Positives: {fp}")
print(f"   False Negatives: {fn}")

# ═══════════════════════════════════════════════════════════
# ROC CURVE PLOT
# ═══════════════════════════════════════════════════════════
try:
    fpr, tpr, thresholds = roc_curve(y_true, pred_probs[:, 1])

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#2196F3', lw=2, label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve — RP Detection', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=150)
    plt.show()
    print("\n📈 ROC curve saved as roc_curve.png")
except Exception as e:
    print(f"\n⚠️ Could not plot ROC curve: {e}")

# ═══════════════════════════════════════════════════════════
# CONFUSION MATRIX PLOT
# ═══════════════════════════════════════════════════════════
try:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=labels,
        yticklabels=labels,
        ylabel='True Label',
        xlabel='Predicted Label',
        title='Confusion Matrix'
    )

    # Add text annotations
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=16, fontweight="bold")

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()
    print("📊 Confusion matrix saved as confusion_matrix.png")
except Exception as e:
    print(f"\n⚠️ Could not plot confusion matrix: {e}")
