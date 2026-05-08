import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers.schedules import CosineDecay

import os
import numpy as np

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════
IMG_SIZE = 300           # ↑ was 224 — EfficientNetB3 optimal input
BATCH = 16
EPOCHS = 30              # Phase 1: frozen base training
FINETUNE_EPOCHS = 20     # Phase 2: fine-tuning (was 15)
DROPOUT_RATE = 0.5       # ↑ was 0.3 — better regularization for small datasets
LABEL_SMOOTHING = 0.1    # NEW: prevents overconfident predictions

DATA_DIR = "modalities/UWF"

train_dir = os.path.join(DATA_DIR, "train")
val_dir = os.path.join(DATA_DIR, "val")

print("📂 Loading dataset...")

train_ds = image_dataset_from_directory(
    train_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    label_mode="categorical",
    seed=42
)

val_ds = image_dataset_from_directory(
    val_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    label_mode="categorical",
    seed=42
)

class_names = train_ds.class_names
print("Classes:", class_names)

# ═══════════════════════════════════════════════════════════
# CLASS WEIGHTS (handles imbalanced dataset: 223 Normal vs 150 RP)
# ═══════════════════════════════════════════════════════════
# Adjust these numbers based on your actual train split counts
# Formula: total / (num_classes * class_count)
NUM_NORMAL = 223
NUM_RP = 150
TOTAL = NUM_NORMAL + NUM_RP

class_weight = {
    0: TOTAL / (2 * NUM_RP),      # Disease (RP) — upweight minority class
    1: TOTAL / (2 * NUM_NORMAL),   # Normal — downweight majority class
}
print(f"📊 Class Weights: {class_weight}")

# AUTOTUNE
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

# ═══════════════════════════════════════════════════════════
# 🟢 ENHANCED DATA AUGMENTATION
# ═══════════════════════════════════════════════════════════
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomFlip("vertical"),            # NEW: retinal images can be flipped vertically
    tf.keras.layers.RandomRotation(0.15),              # ↑ was 0.10
    tf.keras.layers.RandomZoom((-0.15, 0.15)),         # ↑ wider zoom range
    tf.keras.layers.RandomBrightness(factor=0.2),
    tf.keras.layers.RandomContrast(0.3),               # ↑ was 0.2
    tf.keras.layers.RandomTranslation(0.1, 0.1),
], name="augmentation")

# ═══════════════════════════════════════════════════════════
# 🧠 BUILD MODEL — EfficientNetB3 (replaces MobileNetV2)
# ═══════════════════════════════════════════════════════════
print("🚀 Building EfficientNetB3 model...")

base_model = EfficientNetB3(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # Phase 1: freeze base

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = data_augmentation(inputs)
x = tf.keras.applications.efficientnet.preprocess_input(x)  # FIXED: matching preprocessing
x = base_model(x, training=False)

x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)            # NEW: stabilizes training
x = Dropout(DROPOUT_RATE)(x)           # ↑ was 0.3 → 0.5
x = Dense(128, activation="relu")(x)   # NEW: additional dense layer
x = Dropout(0.3)(x)                    # NEW: second dropout
outputs = Dense(2, activation="softmax")(x)

model = Model(inputs, outputs)

# ═══════════════════════════════════════════════════════════
# PHASE 1: TRAIN WITH FROZEN BASE
# ═══════════════════════════════════════════════════════════
# Cosine decay LR schedule (better than constant LR for small datasets)
steps_per_epoch = TOTAL // BATCH
phase1_lr = CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=EPOCHS * steps_per_epoch,
    alpha=1e-6
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(phase1_lr),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),                    # NEW: AUC-ROC
        tf.keras.metrics.Precision(name="precision"),          # NEW
        tf.keras.metrics.Recall(name="recall"),                # NEW
    ]
)

model.summary()

# 🟡 CALLBACKS
callbacks = [
    EarlyStopping(
        monitor="val_auc",        # CHANGED: monitor AUC instead of val_loss
        patience=7,
        restore_best_weights=True,
        mode="max"                 # AUC should be maximized
    ),
    ReduceLROnPlateau(
        monitor="val_auc",
        patience=3,
        factor=0.3,
        mode="max"
    ),
    ModelCheckpoint(
        "UWF_EfficientNetB3_BEST.h5",
        monitor="val_auc",
        save_best_only=True,
        mode="max"
    )
]

print("🚀 Phase 1: Training with frozen base...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight       # NEW: handle class imbalance
)

# ═══════════════════════════════════════════════════════════
# PHASE 2: FINE-TUNE LAST LAYERS
# ═══════════════════════════════════════════════════════════
print("🔓 Phase 2: Fine-tuning last layers...")

base_model.trainable = True
# Freeze all layers except the last 50 (was 40)
for layer in base_model.layers[:-50]:
    layer.trainable = False

# Count trainable layers
trainable_count = sum(1 for layer in model.layers if layer.trainable)
print(f"📊 Trainable layers: {trainable_count}")

# Lower LR for fine-tuning with cosine decay
phase2_lr = CosineDecay(
    initial_learning_rate=1e-5,
    decay_steps=FINETUNE_EPOCHS * steps_per_epoch,
    alpha=1e-7
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(phase2_lr),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]
)

history_finetune = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINETUNE_EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight
)

model.save("UWF_EfficientNetB3_BEST.h5")
print("🎉 Training complete! Best model saved as UWF_EfficientNetB3_BEST.h5")
