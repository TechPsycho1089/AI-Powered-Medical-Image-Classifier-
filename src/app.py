"""
src/app.py  —  Flask prediction API.
Started by run_app.py
"""

import io
import base64
import numpy as np
import cv2
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

from src.config import CONFIG
from src.data import preprocess_pil
from src.gradcam import compute_gradcam, overlay_heatmap

app = Flask(__name__)
CORS(app)

# Load model once at startup
model = None
print("[SERVER] Loading model...")
if Path(CONFIG["model_save"]).exists():
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(CONFIG["model_save"])
        print(f"[SERVER] Model loaded: {CONFIG['model_save']}")
    except Exception as e:
        print(f"[SERVER] Failed to load model: {e}")
        print("[SERVER] Running in DEMO MODE — predictions will be random.")
else:
    print(f"[SERVER] Model not found at '{CONFIG['model_save']}'")
    print("[SERVER] Run  python3 train.py  first, then restart the server.")
    print("[SERVER] Running in DEMO MODE — predictions will be random.")


def _img_to_b64(arr: np.ndarray) -> str:
    pil = Image.fromarray(np.uint8(arr * 255))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _heatmap_to_b64(img_np: np.ndarray, heatmap: np.ndarray) -> str:
    overlay = overlay_heatmap(img_np, heatmap)
    pil = Image.fromarray(overlay.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": CONFIG["model_save"],
        "note": "Train model first with: python3 train.py" if model is None else "Ready"
    })


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file. Use field name 'file'."}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    allowed = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    if Path(f.filename).suffix.lower() not in allowed:
        return jsonify({"error": "Unsupported format. Use PNG or JPG."}), 400

    try:
        pil_img = Image.open(io.BytesIO(f.read()))
        img_np  = preprocess_pil(pil_img, CONFIG["img_size"])
        inp     = np.expand_dims(img_np, 0)

        demo_mode = model is None
        if not demo_mode:
            prob = float(model.predict(inp, verbose=0)[0][0])
        else:
            import random
            prob = random.uniform(0.1, 0.95)

        # NOTE: Labels are flipped because Kaggle trained with swapped labels
        # prob close to 1.0 = Healthy, prob close to 0.0 = Infected
        label          = "Infected"  if prob >= CONFIG["threshold"] else "Healthy"
        prob_infected  = round(prob, 4)
        prob_healthy   = round(1-prob, 4)
        conf           = prob if prob >= CONFIG["threshold"] else (1 - prob)

        if not demo_mode:
            heatmap = compute_gradcam(model, inp)
        else:
            heatmap = np.zeros((8, 8))

        gradcam_b64 = _heatmap_to_b64(img_np, heatmap)
        pre_b64     = _img_to_b64(img_np)

        return jsonify({
            "label":                label,
            "confidence":           round(conf, 4),
            "probability_infected": prob_infected,
            "probability_healthy":  prob_healthy,
            "risk_level":           "High" if label == "Infected" else "Low",
            "demo_mode":            demo_mode,
            "preprocessed_b64":     pre_b64,
            "gradcam_b64":          gradcam_b64,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500