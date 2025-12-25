import streamlit as st
import cv2
import numpy as np
from PIL import Image
from utils import load_model, preprocess_image, predict, CLASSES

import os
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..")

st.set_page_config(page_title="Face Mask Detector", layout="centered")

st.title("😷 Face Mask Detection App")
st.write("Upload an image to detect face mask usage")

@st.cache_resource
def load():
    return load_model(MODEL_PATH)

model = load()

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_input = preprocess_image(image_np)
    bbox, class_id = predict(model, img_input)

    h, w, _ = image_np.shape
    xmin = int(bbox[0] * w)
    ymin = int(bbox[1] * h)
    xmax = int(bbox[2] * w)
    ymax = int(bbox[3] * h)

    cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    label = CLASSES[class_id]
    cv2.putText(image_np, label, (xmin, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    st.image(image_np, caption=f"Prediction: {label}", use_container_width=True)
