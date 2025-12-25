import cv2
import numpy as np
import tensorflow as tf

CLASSES = ["Mask", "No Mask", "Incorrect Mask"]  # adjust if needed

def load_model(model_path):
    model = tf.saved_model.load(model_path)
    infer = model.signatures["serving_default"]
    return infer

def preprocess_image(image):
    img = cv2.resize(image, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict(model, image):
    outputs = model(tf.constant(image))
    
    # robustly find bbox and keys based on shape
    # Expected shapes: bbox: (1, 4) or (1, N, 4), class: (1, num_classes)
    # But usually model output for single image might be (1, 4) if it's a regression model or similar.
    # However, for 2 classes (mask/no mask) + regression:
    # Let's inspect shapes
    res = {}
    for k, v in outputs.items():
        res[k] = v.numpy()
        
    # Heuristic: 
    # bbox usually has last dim 4
    # class usually has last dim num_classes (3 here: Mask, No Mask, Incorrect)
    
    bbox = None
    classes = None
    
    for k, v in res.items():
        if v.shape[-1] == 4:
            bbox = v[0]
        elif v.shape[-1] == 3: # 3 classes
            classes = v[0]
            
    # Fallback if shapes don't match expectation (e.g. if it's wrong)
    if bbox is None:
        # separate logic or just take first
        bbox = list(res.values())[0][0]
    if classes is None and len(res) > 1:
        classes = list(res.values())[1][0]
        
    # Ensure shapes
    bbox = np.clip(bbox, 0, 1)
    class_id = np.argmax(classes)
    
    return bbox, class_id
