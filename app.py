import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from gradio_client import Client, handle_file

# -----------------------
# Flask setup
# -----------------------
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB

# -----------------------
# Hugging Face Space client
# -----------------------
# Space ID from your URL: https://huggingface.co/spaces/rajveer0singh/uniform_detection
client = Client("rajveer0singh/uniform_detection")

# API name from "Use via API" on the Space
API_NAME = "/predict_gradio"   # change only if your Space shows a different one


def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


def predict_image(img_path: str) -> dict:
    """
    Send image to Hugging Face Space via gradio_client and return
    {label, confidence, processed_image} or {error,...} on failure.
    """
    try:
        result = client.predict(
            img=handle_file(img_path),
            api_name=API_NAME,
        )

        # Expect: result[0] = scores dict, result[1] = processed image (from Space)
        scores = result[0]
        if not isinstance(scores, dict) or len(scores) == 0:
            raise ValueError(f"Unexpected scores format: {scores}")

        label = max(scores, key=scores.get)
        confidence = round(float(scores[label]) * 100.0, 2)

        # We just show the original upload on our UI
        return {
            "label": label,
            "confidence": confidence,
            "processed_image": img_path,
        }

    except Exception as e:
        print("Prediction error:", e)
        return {"label": "Error", "confidence": 0, "error": str(e)}


# -----------------------
# Routes
# -----------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/process_image", methods=["POST"])
def process_image_route():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"success": False, "error": "No selected file"}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            result = predict_image(filepath)
            if result.get("error"):
                return jsonify({"success": False, "error": result["error"]}), 500

            response = {
                "success": True,
                "detections": [
                    {
                        "class": result["label"],
                        "confidence": result["confidence"] / 100.0,
                    }
                ],
                # Flask will serve /static/... by default
                "processed_image": f"/{result['processed_image']}",
            }
            return jsonify(response), 200

        return jsonify({"success": False, "error": "Invalid file type"}), 400

    except Exception as e:
        print("Route error:", e)
        return jsonify({"success": False, "error": str(e)}), 500


# -----------------------
# Local dev entrypoint
# (Render will ignore this and use gunicorn)
# -----------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
