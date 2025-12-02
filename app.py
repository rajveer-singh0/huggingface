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
    Call the Hugging Face Space and return a simple dict:
    {label, confidence, processed_image}.
    Assumes the Space returns a standard classification output:
      - list of {"label": ..., "score": ...}
      - or a single {"label": ..., "score": ...}
      - or just a label string.
    """
    try:
        hf_result = client.predict(
            img=handle_file(img_path),
            api_name=API_NAME,
        )

        # Debug: see exactly what HF returned
        print("HF raw result:", repr(hf_result))

        label = "unknown"
        confidence = 100.0  # default, if we don't find a score
        processed_image = img_path  # default to original image

        # Handle different possible return formats from Hugging Face API
        if isinstance(hf_result, tuple) and len(hf_result) >= 2:
            # If it's a tuple, first element is usually the label/prediction, 
            # second might be the processed image
            if isinstance(hf_result[0], str):
                label = hf_result[0]
            elif isinstance(hf_result[0], dict) and "label" in hf_result[0]:
                label = hf_result[0]["label"]
                if "score" in hf_result[0] and hf_result[0]["score"] is not None:
                    confidence = round(float(hf_result[0]["score"]) * 100.0, 2)
            
            # Check if second element is an image path
            if isinstance(hf_result[1], str) and (hf_result[1].endswith('.jpg') or hf_result[1].endswith('.png') or hf_result[1].endswith('.jpeg')):
                processed_image = hf_result[1]
        elif isinstance(hf_result, list) and len(hf_result) > 0:
            first = hf_result[0]

            # list of {"label": ..., "score": ...}
            if isinstance(first, dict) and "label" in first:
                # choose the dict with max "score" (or 0 if missing)
                best = max(
                    hf_result,
                    key=lambda x: float(x.get("score", 0) or 0)
                    if isinstance(x, dict)
                    else 0,
                )
                label = best.get("label", "unknown")
                if "score" in best and best["score"] is not None:
                    confidence = round(float(best["score"]) * 100.0, 2)
            else:
                # list of plain labels: ["uniform", "non_uniform", ...]
                label = str(first)

        # Case 2: single dict {"label": ..., "score": ...}
        elif isinstance(hf_result, dict) and "label" in hf_result:
            label = hf_result.get("label", "unknown")
            if "score" in hf_result and hf_result["score"] is not None:
                confidence = round(float(hf_result["score"]) * 100.0, 2)
            # Check if dict contains an image path
            for key in hf_result:
                if isinstance(hf_result[key], str) and (hf_result[key].endswith('.jpg') or hf_result[key].endswith('.png') or hf_result[key].endswith('.jpeg')):
                    processed_image = hf_result[key]
                    break

        # Case 3: just a plain string: "uniform"
        elif isinstance(hf_result, str):
            label = hf_result

        return {
            "label": label,
            "confidence": confidence,
            "processed_image": processed_image,
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
                "processed_image": result['processed_image'] if result['processed_image'].startswith('/') else f"/{result['processed_image']}",
                "debug_raw": repr(result)  # 👈 add this line
            }
            return jsonify(response), 200

        return jsonify({"success": False, "error": "Invalid file type"}), 400

    except Exception as e:
        print("Route error:", e)
        return jsonify({"success": False, "error": str(e)}), 500


# -----------------------
# Local dev entrypoint
# -----------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
