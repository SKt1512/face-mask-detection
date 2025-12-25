import cv2
import numpy as np
import tensorflow as tf

# Class labels
CLASSES = ["with_mask", "without_mask", "mask_weared_incorrect"]

# Load SavedModel
model = tf.keras.models.load_model("..")
infer = model.signatures["serve"]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # Preprocess
    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype("float32")

    #Inference
    outputs = infer(tf.constant(img))
    outputs_dict = {k: v.numpy() for k, v in outputs.items()}

    for v in outputs_dict.values():
        if v.shape[-1] == 4:
            bbox = v
        elif v.shape[-1] == 3:
            cls = v

    box = bbox[0]
    class_id = int(np.argmax(cls[0]))
    label = CLASSES[class_id]

    # Convert normalized bbox to pixel coords
    xmin = int(box[0] * w)
    ymin = int(box[1] * h)
    xmax = int(box[2] * w)
    ymax = int(box[3] * h)

    # Draw
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    cv2.putText(frame, label, (xmin, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
