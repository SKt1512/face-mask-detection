import tensorflow as tf
import numpy as np

# Load exported SavedModel
model = tf.keras.models.load_model(".")
print("✅ Model loaded successfully")

# Prepare dummy input
dummy = np.random.rand(1, 224, 224, 3).astype("float32")

# Run inference using SavedModel signature
infer = model.signatures["serve"]
outputs = infer(tf.constant(dummy))

bbox, cls = outputs.values()

print("BBox shape:", bbox.shape)
print("Class shape:", cls.shape)
