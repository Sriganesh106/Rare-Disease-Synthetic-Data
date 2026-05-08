import tensorflow as tf
import numpy as np
import os

MODEL_PATH = "models/UWF_EfficientNetB3_BEST.h5"
CLASS_NAMES = ["Disease", "Normal"]
IMG_SIZE = 300  # ↑ was 224 — matches new EfficientNetB3 training size

def load_and_prepare(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.utils.img_to_array(img)

    # FIXED: Use EfficientNet preprocessing (was img_array / 255.0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    return np.expand_dims(img_array, axis=0)

def main():

    print("🔍 Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!\n")

    while True:
        # Ask user for path
        img_path = input("Enter image path (or type exit): ").strip()

        # Exit condition
        if img_path.lower() == "exit":
            print("👋 Exiting prediction tool.")
            break

        # Normalize slashes
        img_path = img_path.replace("\\", "/")

        # Check file
        if not os.path.exists(img_path):
            print(f"❌ File not found:\n{img_path}\n")
            continue

        print(f"📸 Image: {img_path}")

        # Preprocess
        img = load_and_prepare(img_path)

        # Predict
        preds = model.predict(img)
        confidence = float(np.max(preds))
        label = CLASS_NAMES[int(np.argmax(preds))]

        # Show all class probabilities
        print("\n" + "=" * 40)
        print(f"🧠 PREDICTION: {label}")
        print(f"📊 Confidence: {confidence*100:.2f}%")
        print("-" * 40)
        for i, name in enumerate(CLASS_NAMES):
            bar = "█" * int(preds[0][i] * 30)
            print(f"   {name:>8}: {preds[0][i]*100:6.2f}% {bar}")
        print("=" * 40 + "\n")


if __name__ == "__main__":
    main()
