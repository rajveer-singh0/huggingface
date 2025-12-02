
import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

# -----------------------
# Environment configuration
# -----------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU (no CUDA)

# -----------------------
# Flask App Initialization
# -----------------------
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit

# -----------------------
# Load the trained model safely
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'uniform_model.keras')

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    raise FileNotFoundError(
        f"Model not found or incompatible. Make sure {MODEL_PATH} exists. Original error: {e}"
    )

CLASS_NAMES = ['non_uniform', 'uniform']  # 0 = non_uniform, 1 = uniform

# -----------------------
# Helper Functions
# -----------------------
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_image(img_path):
    try:
        # Preprocess the image
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array)[0][0]
        label_idx = 1 if prediction > 0.5 else 0
        label = CLASS_NAMES[label_idx]
        confidence = prediction if label_idx == 1 else 1 - prediction

        # Visualization
        img = cv2.imread(img_path)
        if img is not None:
            text = f"{label} ({confidence:.2%})"
            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
            processed_path = os.path.join(
                app.config['UPLOAD_FOLDER'],
                'processed_' + os.path.basename(img_path)
            )
            cv2.imwrite(processed_path, img)
        else:
            processed_path = img_path

        return {"label": label, "confidence": round(confidence * 100, 2), "processed_image": processed_path}

    except Exception as e:
        print(f"Prediction error: {e}")
        return {"label": "Error", "confidence": 0, "error": str(e)}

# -----------------------
# Routes
# -----------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image_route():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result = predict_image(filepath)
        if result.get('error'):
            return jsonify({"success": False, "error": result['error']})

        response = {
            "success": True,
            "detections": [{
                "class": result['label'],
                "confidence": result['confidence'] / 100
            }],
            "processed_image": f"/{result['processed_image']}"
        }
        return jsonify(response)

    return jsonify({"success": False, "error": "Invalid file type"})

# -----------------------
# Run App
# -----------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
