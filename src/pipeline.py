"""
src/pipeline.py  —  Full training pipeline orchestrator.
Called by train.py
"""

import os
import numpy as np
import tensorflow as tf

from src.config import CONFIG
from src.data import (
    load_image_paths, build_dataset, split_data, get_class_weights,
    plot_class_distribution, plot_sample_images, plot_augmentation
)
from src.model import build_custom_cnn, compile_model
from src.train import train, plot_history
from src.evaluate import evaluate
from src.gradcam import visualize_gradcam


def print_flow_diagram():
    print("""
╔══════════════════════════════════════════════════════════════╗
║         MALARIA CELL CLASSIFICATION — PIPELINE FLOW         ║
╚══════════════════════════════════════════════════════════════╝
  1. DATA LOADING        →  Parasitized + Uninfected images
  2. EDA                 →  Class distribution, sample visualization
  3. PREPROCESSING       →  BGR→RGB, Resize, CLAHE, Normalize
  4. IMBALANCE HANDLING  →  Computed class weights
  5. AUGMENTATION        →  Rotation, flip, zoom, brightness
  6. MODEL BUILD         →  Custom CNN (4 blocks) + head
  7. TRAINING            →  Adam + EarlyStopping + ReduceLR
  8. EVALUATION          →  Accuracy, AUC, F1, CM, ROC
  9. GRAD-CAM            →  Heatmap explainability
 10. SAVE                →  models/malaria_cnn_best.h5
""")


def main():
    np.random.seed(42)
    tf.random.set_seed(42)

    print("\n" + "="*60)
    print("  BIOCON LTD. — MALARIA CLASSIFIER  (UCS321 #4)")
    print("="*60)
    print_flow_diagram()

    os.makedirs(CONFIG["results_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG["model_save"]), exist_ok=True)

    # 1. Load
    all_data = load_image_paths(CONFIG["infected_dir"], CONFIG["healthy_dir"])

    # 2. EDA
    plot_sample_images(CONFIG["infected_dir"], CONFIG["healthy_dir"],
                       save_dir=CONFIG["results_dir"])
    plot_class_distribution(all_data, save_dir=CONFIG["results_dir"])

    # 3. Preprocess & build arrays
    X, y = build_dataset(all_data, CONFIG["img_size"])

    # 4. Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # 5. Class weights
    class_weights = get_class_weights(y_train)

    # 6. Augmentation preview
    plot_augmentation(X_train, y_train, save_dir=CONFIG["results_dir"])

    # 7. Build model
    print("\n[MODEL] Building Custom CNN...")
    model = build_custom_cnn(input_shape=(*CONFIG["img_size"], 3))
    model.summary()
    model = compile_model(model, lr=CONFIG["learning_rate"])

    # 8. Train
    history = train(model, X_train, y_train, X_val, y_val, class_weights)
    plot_history(history, save_dir=CONFIG["results_dir"])

    # 9. Evaluate
    metrics = evaluate(model, X_test, y_test, save_dir=CONFIG["results_dir"])

    # 10. Grad-CAM
    visualize_gradcam(model, X_test, y_test, save_dir=CONFIG["results_dir"])

    # 11. Save final model
    model.save(CONFIG["model_save"])
    print(f"\n[DONE] Model saved: {CONFIG['model_save']}")

    print("\n" + "="*55)
    print("  FINAL RESULTS")
    print("="*55)
    for k, v in metrics.items():
        print(f"  {k.upper():<12}: {v:.4f}")
    print(f"\n  Plots saved in: {CONFIG['results_dir']}/")
    print("="*55)
