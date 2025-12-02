import os
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import gradio as gr

# -----------------------
# Environment configuration
# -----------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU (no CUDA)

# -----------------------
# Load the trained model safely
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "uniform_model.keras")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    raise FileNotFoundError(
        f"Model not found or incompatible. Make sure {MODEL_PATH} exists. Original error: {e}"
    )

CLASS_NAMES = ["non_uniform", "uniform"]  # 0 = non_uniform, 1 = uniform

# -----------------------
# Prediction function for Gradio
# -----------------------
def predict_gradio(img: Image.Image):
    """
    img: PIL image from Gradio
    returns:
      - label scores dict for Gradio Label
      - processed image with text overlay
    """
    try:
        # Preprocess for model
        resized = img.resize((128, 128))
        img_array = np.array(resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array)[0][0]  # scalar between 0 and 1
        prob_uniform = float(prediction)
        prob_non_uniform = float(1 - prediction)

        # Decide label & confidence
        label_idx = 1 if prediction > 0.5 else 0
        label = CLASS_NAMES[label_idx]
        confidence = prob_uniform if label_idx == 1 else prob_non_uniform

        # Convert PIL → OpenCV (BGR) for text drawing
        img_np = np.array(img)  # RGB
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        text = f"{label} ({confidence:.2%})"
        cv2.putText(
            img_bgr,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Back to RGB + PIL for Gradio
        processed_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        processed_pil = Image.fromarray(processed_rgb)

        # For gr.Label, return dict: class -> probability
        scores = {
            "uniform": prob_uniform,
            "non_uniform": prob_non_uniform,
        }

        return scores, processed_pil

    except Exception as e:
        # In case of error, just return zero scores and original image
        print("Prediction error:", e)
        scores = {
            "uniform": 0.0,
            "non_uniform": 0.0,
        }
        return scores, img

# -----------------------
# Gradio Interface
# -----------------------
demo = gr.Interface(
    fn=predict_gradio,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=[
        gr.Label(num_top_classes=2, label="Prediction"),
        gr.Image(type="pil", label="Processed Image"),
    ],
    title="Uniform Detection System",
    description="Upload an image to check if the person is wearing a uniform or not.",
)

if __name__ == "__main__":
    demo.launch()
