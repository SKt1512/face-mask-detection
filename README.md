# Face Mask Detection App

A deep learning-based application to detect whether a person is wearing a face mask, not wearing one, or wearing it incorrectly. This project uses a TensorFlow/Keras model deployed via a Streamlit web interface.

## Features
- **Real-time Detection**: Upload images to detect face masks.
- **3 Class Classification**: 
  - `Mask`
  - `No Mask`
  - `Incorrect Mask`
- **User-Friendly Interface**: Built with [Streamlit](https://streamlit.io/).

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SKt1512/face-mask-detection.git
   cd face-mask-detection
   ```

2. **Install Dependencies:**
   Ensure you have Python installed (recommended Python 3.9 - 3.12).
   
   ```bash
   pip install tensorflow Streamlit opencv-python-headless pillow numpy
   ```
   > **Note:** If you encounter `numpy` incompatibilities with TensorFlow, downgrade numpy:
   > ```bash
   > pip install "numpy<2.0"
   > ```

## Usage

1. **Navigate to the deployment directory:**
   ```bash
   cd deployment
   ```

2. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```

3. **Open in Browser:**
   The app will automatically open in your default browser at `http://localhost:8501`.

## Project Structure
- `deployment/`: Contains the Streamlit app (`app.py`), inference logic (`utils.py`), and visualization scripts.
- `saved_model.pb` & `variables/`: Two standard TensorFlow SavedModel files.
- `assets/`: Project assets.

## Model Details
- **Framework**: TensorFlow / Keras
- **Input**: Images resized to 224x224
- **Output**: Bounding box coordinates and Class probabilities

