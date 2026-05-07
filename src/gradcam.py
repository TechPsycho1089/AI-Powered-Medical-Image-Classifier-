"""
src/gradcam.py  —  Grad-CAM explainability.
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from src.config import CONFIG


def compute_gradcam(model, img_array: np.ndarray,
                    last_conv_layer: str = None,
                    pred_index: int = None) -> np.ndarray:
    if last_conv_layer is None:
        last_conv_layer = CONFIG["last_conv_layer"]
    try:
        grad_model = keras.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer).output, model.output]
        )
        img_tensor = tf.cast(img_array, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            conv_out, preds = grad_model(img_tensor)
            ch = preds[:, 0]
        grads = tape.gradient(ch, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_sum(conv_out[0] * pooled, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.math.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        return heatmap.numpy()
    except Exception as e:
        print(f"[GRADCAM] {e}")
        return np.zeros((8, 8))


def overlay_heatmap(img: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.4) -> np.ndarray:
    """Overlay Grad-CAM heatmap on original image [0,1]."""
    h_uint8 = cv2.resize(np.uint8(255 * heatmap), (img.shape[1], img.shape[0]))
    colored = cv2.applyColorMap(h_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    img_uint8 = np.uint8(255 * img)
    return cv2.addWeighted(img_uint8, 1 - alpha, colored, alpha, 0)


def visualize_gradcam(model, X_test, y_test, n=4, save_dir="results"):
    """Plot Grad-CAM for n infected cells."""
    os.makedirs(save_dir, exist_ok=True)
    idxs = np.where(y_test == 1)[0][:n]

    fig, axes = plt.subplots(n, 3, figsize=(11, 3.5 * n))
    fig.suptitle("Grad-CAM — Infected Cell Predictions",
                 fontsize=13, fontweight="bold")
    for col, title in enumerate(["Original", "Grad-CAM Heatmap", "Overlay"]):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold")

    for row, idx in enumerate(idxs):
        img = X_test[idx]
        inp = np.expand_dims(img, 0)
        hm = compute_gradcam(model, inp)
        overlay = overlay_heatmap(img, hm)
        hm_colored = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
        hm_colored = cv2.cvtColor(hm_colored, cv2.COLOR_BGR2RGB)

        prob = float(model.predict(inp, verbose=0)[0][0])
        pred = "Infected" if prob >= CONFIG["threshold"] else "Healthy"

        axes[row, 0].imshow(img); axes[row, 0].axis("off")
        axes[row, 0].set_ylabel(f"{pred} ({prob:.2f})", fontsize=9)
        axes[row, 1].imshow(hm_colored); axes[row, 1].axis("off")
        axes[row, 2].imshow(overlay); axes[row, 2].axis("off")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/gradcam.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[GRADCAM] Saved to {save_dir}/gradcam.png")


def predict_single(model, image_path: str) -> dict:
    """Predict a single image file and show Grad-CAM."""
    from src.data import preprocess_image
    img = preprocess_image(image_path, CONFIG["img_size"])
    inp = np.expand_dims(img, 0)

    prob = float(model.predict(inp, verbose=0)[0][0])
    label = "Infected" if prob >= CONFIG["threshold"] else "Healthy"
    conf  = prob if prob >= CONFIG["threshold"] else (1 - prob)

    print(f"\n[PREDICT] {image_path}")
    print(f"  Label      : {label}")
    print(f"  Confidence : {conf*100:.1f}%")

    hm = compute_gradcam(model, inp)
    overlay = overlay_heatmap(img, hm)
    hm_colored = cv2.applyColorMap(np.uint8(255*hm), cv2.COLORMAP_JET)
    hm_colored = cv2.cvtColor(hm_colored, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    fig.suptitle(f"Prediction: {label}  |  {conf*100:.1f}% confidence",
                 fontsize=13, fontweight="bold",
                 color="#c0392b" if label == "Infected" else "#27ae60")
    axes[0].imshow(img);        axes[0].set_title("Input");    axes[0].axis("off")
    axes[1].imshow(hm_colored); axes[1].set_title("Grad-CAM"); axes[1].axis("off")
    axes[2].imshow(overlay);    axes[2].set_title("Overlay");  axes[2].axis("off")
    plt.tight_layout(); plt.show()

    return {"label": label, "confidence": conf,
            "prob_infected": prob, "prob_healthy": 1-prob}