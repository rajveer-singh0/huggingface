import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from gradio_client import Client, handle_file   # ✅ make sure this line exists

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB

# ✅ Use your Space ID
client = Client("rajveer0singh/uniform_detection")
API_NAME = "/predict_gradio"

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_image(img_path):
    try:
        result = client.predict(
            img=handle_file(img_path),
            api_name=API_NAME
        )

        # Expect: [scores_dict, processed_image]
        scores = result[0]
        label = max(scores, key=scores.get)
        confidence = round(float(scores[label]) * 100, 2)

        return {"label": label, "confidence": confidence, "processed_image": img_path}

    except Exception as e:
        print("Prediction error:", e)
        return {"label": "Error", "confidence": 0, "error": str(e)}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image_route():
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "No selected file"}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            result = predict_image(filepath)
            if result.get('error'):
                return jsonify({"success": False, "error": result['error']}), 500

            response = {
                "success": True,
                "detections": [{
                    "class": result['label'],
                    "confidence": result['confidence'] / 100.0
                }],
                "processed_image": f"/{result['processed_image']}"
            }
            return jsonify(response), 200

        return jsonify({"success": False, "error": "Invalid file type"}), 400

    except Exception as e:
        # This prevents Flask from crashing and gives you JSON instead
        print("Route error:", e)
        return jsonify({"success": False, "error": str(e)}), 500
