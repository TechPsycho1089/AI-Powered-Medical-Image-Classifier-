"""
src/data.py  —  Data loading, preprocessing, augmentation, splitting.
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.config import CONFIG


# ─────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────

def load_image_paths(infected_dir: str, healthy_dir: str):
    """Return list of (filepath, label) tuples. Label 0=Healthy, 1=Infected."""
    exts = ["*.png", "*.jpg", "*.jpeg"]
    infected, healthy = [], []
    for ext in exts:
        infected += [(str(p), 1) for p in Path(infected_dir).glob(ext)]
        healthy  += [(str(p), 0) for p in Path(healthy_dir).glob(ext)]

    all_data = infected + healthy
    print(f"[DATA] Infected : {len(infected)}")
    print(f"[DATA] Healthy  : {len(healthy)}")
    print(f"[DATA] Total    : {len(all_data)}")
    return all_data


# ─────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────

def preprocess_image(filepath: str, img_size=(128, 128)) -> np.ndarray:
    """
    Load → RGB → Resize → CLAHE → Normalize [0,1].
    CLAHE handles staining variability in microscopy images.
    """
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return img.astype(np.float32) / 255.0


def preprocess_pil(pil_img, img_size=(128, 128)) -> np.ndarray:
    """Preprocess a PIL Image (used by Flask API)."""
    img = np.array(pil_img.convert("RGB"), dtype=np.uint8)
    img = cv2.resize(img, img_size)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return img.astype(np.float32) / 255.0


def build_dataset(all_data, img_size=(128, 128)):
    """Load and preprocess all images into numpy arrays."""
    print("[PREPROCESSING] Loading images...")
    images, labels = [], []
    for i, (fp, label) in enumerate(all_data):
        try:
            images.append(preprocess_image(fp, img_size))
            labels.append(label)
        except Exception as e:
            print(f"  [WARN] Skipped {fp}: {e}")
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(all_data)}")

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    print(f"[PREPROCESSING] Done. Shape: X={X.shape}, y={y.shape}")
    return X, y


# ─────────────────────────────────────────────────────────────
# SPLITTING
# ─────────────────────────────────────────────────────────────

def split_data(X, y):
    """Stratified 70 / 15 / 15 train / val / test split."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=CONFIG["test_split"], random_state=42, stratify=y
    )
    val_ratio = CONFIG["val_split"] / (1 - CONFIG["test_split"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
    )
    print(f"[SPLIT] Train={len(X_train)}  Val={len(X_val)}  Test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_class_weights(y_train):
    """Compute class weights to handle imbalanced dataset."""
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    cw = dict(zip(classes, weights))
    print(f"[CLASS WEIGHTS] {cw}")
    return cw


# ─────────────────────────────────────────────────────────────
# AUGMENTATION
# ─────────────────────────────────────────────────────────────

def get_augmentation_generator():
    """ImageDataGenerator for training augmentation."""
    return ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )


# ─────────────────────────────────────────────────────────────
# VISUALIZATIONS
# ─────────────────────────────────────────────────────────────

def plot_class_distribution(all_data, save_dir="results"):
    import pandas as pd
    os.makedirs(save_dir, exist_ok=True)
    labels = [item[1] for item in all_data]
    counts = pd.Series(labels).value_counts()
    colors = ["#2196F3", "#F44336"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Class Distribution", fontsize=13, fontweight="bold")
    axes[0].bar(CONFIG["classes"], [counts.get(0, 0), counts.get(1, 0)],
                color=colors, edgecolor="black", linewidth=0.8)
    axes[0].set_ylabel("Count")
    for bar, c in zip(axes[0].patches, [counts.get(0,0), counts.get(1,0)]):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
                     str(c), ha="center", va="bottom", fontweight="bold")
    axes[1].pie([counts.get(0, 0), counts.get(1, 0)],
                labels=CONFIG["classes"], colors=colors, autopct="%1.1f%%")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/class_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_sample_images(infected_dir, healthy_dir, n=5, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, n, figsize=(3*n, 7))
    fig.suptitle("Sample Images — Healthy vs Infected", fontsize=13, fontweight="bold")
    for i, (d, label) in enumerate([(healthy_dir, "Healthy"), (infected_dir, "Infected")]):
        imgs = list(Path(d).glob("*.png"))[:n]
        for j, imgp in enumerate(imgs):
            img = cv2.imread(str(imgp))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i][j].imshow(img)
            axes[i][j].set_title(label, fontsize=10)
            axes[i][j].axis("off")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/sample_images.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_augmentation(X_train, y_train, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    datagen = get_augmentation_generator()
    idx = np.where(y_train == 1)[0][0]
    sample = X_train[idx:idx+1]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle("Augmentation Examples", fontsize=13, fontweight="bold")
    axes.flat[0].imshow(X_train[idx]); axes.flat[0].set_title("Original"); axes.flat[0].axis("off")
    gen = datagen.flow(sample, batch_size=1)
    for i in range(1, 8):
        axes.flat[i].imshow(next(gen)[0]); axes.flat[i].set_title(f"Aug {i}"); axes.flat[i].axis("off")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/augmentation_examples.png", dpi=150, bbox_inches="tight")
    plt.show()
