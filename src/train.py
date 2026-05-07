"""
src/train.py  —  Model training logic and callbacks.
"""

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)

from src.config import CONFIG
from src.data import get_augmentation_generator


def get_callbacks():
    os.makedirs(os.path.dirname(CONFIG["model_save"]), exist_ok=True)
    return [
        EarlyStopping(
            monitor="val_auc", patience=7,
            restore_best_weights=True, mode="max", verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=4, min_lr=1e-7, verbose=1
        ),
        ModelCheckpoint(
            filepath=CONFIG["model_save"],
            monitor="val_auc", save_best_only=True,
            mode="max", verbose=1
        ),
    ]


def train(model, X_train, y_train, X_val, y_val, class_weights, augment=True):
    """
    Train the model.
    augment=True  → on-the-fly ImageDataGenerator augmentation
    augment=False → plain batch training
    """
    callbacks = get_callbacks()

    if augment:
        datagen = get_augmentation_generator()
        gen = datagen.flow(X_train, y_train, batch_size=CONFIG["batch_size"])
        steps = len(X_train) // CONFIG["batch_size"]
        history = model.fit(
            gen,
            steps_per_epoch=steps,
            epochs=CONFIG["epochs"],
            validation_data=(X_val, y_val),
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
        )
    else:
        history = model.fit(
            X_train, y_train,
            batch_size=CONFIG["batch_size"],
            epochs=CONFIG["epochs"],
            validation_data=(X_val, y_val),
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
        )

    print(f"\n[TRAINING] Best model saved to: {CONFIG['model_save']}")
    return history


def plot_history(history, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    metrics = [m for m in ["loss", "accuracy", "auc", "precision", "recall"]
               if m in history.history]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    fig.suptitle("Training History", fontsize=13, fontweight="bold")

    for ax, m in zip(axes, metrics):
        ax.plot(history.history[m], label="Train", linewidth=2)
        if f"val_{m}" in history.history:
            ax.plot(history.history[f"val_{m}"], label="Val",
                    linewidth=2, linestyle="--")
        ax.set_title(m.upper())
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_history.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[VIZ] Training history saved.")
