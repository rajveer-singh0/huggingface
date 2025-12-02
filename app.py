import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from gradio_client import Client, handle_file

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
# Gradio Client (Hugging Face Space)
# -----------------------
# Use your actual Space ID here:
client = Client("rajveer0singh/uniform_detection")
API_NAME = "/predict_gradio"  # same as you used in your test script

# -----------------------
# Helper Functions
# -----------------------
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def predict_image(img_path):
    """
    Calls your Hugging Face Gradio Space with the uploaded image.
    Expects the Space to return [scores_dict, processed_image].
    """
    try:
        # Call your Gradio Space
        result = client.predict(
            img=handle_file(img_path),
            api_name=API_NAME
        )

        # For your Space, result should be:
        # result[0] = dict like {"uniform": 0.93, "non_uniform": 0.07}
        # result[1] = processed image file path (downloaded by gradio_client)
        scores = result[0]
        processed_image_remote = result[1]

        # Pick best label
        label = max(scores, key=scores.get)
        confidence = round(float(scores[label]) * 100, 2)

        # If you want, you can ignore processed_image_remote
        # and just show the original upload on your UI.
        # Or you can copy that file into static/uploads.
        processed_path = img_path  # simplest: use original image

        return {"label": label, "confidence": confidence, "processed_image": processed_path}

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
                "confidence": result['confidence'] / 100.0
            }],
            # serve the original uploaded file from /static/uploads/...
            "processed_image": f"/{result['processed_image']}"
        }
        return jsonify(response)

    return jsonify({"success": False, "error": "Invalid file type"})


# -----------------------
# Run App (local dev only)
# -----------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
