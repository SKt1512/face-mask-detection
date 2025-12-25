import os
import random
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
MODEL_PATH = ".."   # parent folder = face_mask_detector
IMAGE_DIR = "../test_images"
OUTPUT_DIR = "test_predictions"
NUM_IMAGES = 10
IMG_SIZE = 224

CLASSES = ["with_mask", "without_mask", "mask_weared_incorrect"]
# ----------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load SavedModel
model = tf.keras.models.load_model(MODEL_PATH)
infer = model.signatures["serve"]

# Collect images
all_images = [
    os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
]

sample_images = random.sample(all_images, NUM_IMAGES)

for idx, img_path in enumerate(sample_images):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    # Preprocess
    img_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img_norm = img_resized / 255.0
    img_input = np.expand_dims(img_norm, axis=0).astype("float32")

    # Inference
    outputs = infer(tf.constant(img_input))

    for v in outputs.values():
        if v.shape[-1] == 4:
            bbox = v.numpy()
        elif v.shape[-1] == 3:
            cls = v.numpy()

    box = bbox[0]
    class_id = int(np.argmax(cls[0]))
    label = CLASSES[class_id]
    confidence = float(np.max(cls[0]))

    xmin = int(box[0] * w)
    ymin = int(box[1] * h)
    xmax = int(box[2] * w)
    ymax = int(box[3] * h)

    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    cv2.putText(
        image,
        f"{label} ({confidence:.2f})",
        (xmin, ymin - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    out_path = os.path.join(OUTPUT_DIR, f"prediction_{idx+1}.jpg")
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")

print("\n✅ Saved predictions for 10 random images.")
