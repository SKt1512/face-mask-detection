import tensorflow as tf

model_path = "."
try:
    model = tf.saved_model.load(model_path)
    print("Available signatures:", list(model.signatures.keys()))
    
    if "serving_default" in model.signatures:
        infer = model.signatures["serving_default"]
        print("Using 'serving_default'")
    else:
        key = list(model.signatures.keys())[0]
        infer = model.signatures[key]
        print(f"Using '{key}'")

    print("Inputs:", infer.structured_input_signature)
    print("Outputs:", infer.structured_outputs)
except Exception as e:
    print("Error loading model:", e)
