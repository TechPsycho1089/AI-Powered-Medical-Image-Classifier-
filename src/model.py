"""
src/model.py  —  CNN model architectures.
Two options:
  1. build_custom_cnn()       — scratch-built 4-block CNN
  2. build_transfer_model()   — VGG16 or ResNet50 with fine-tuning head
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.optimizers import Adam

from src.config import CONFIG


def build_custom_cnn(input_shape=(128, 128, 3)) -> Model:
    """
    Custom CNN for malaria cell classification.
    Architecture: Input → 4×[Conv→BN→ReLU→Pool→Dropout] → GAP → Dense → Sigmoid

    The final conv layer is named 'last_conv' for Grad-CAM compatibility.
    """
    inp = keras.Input(shape=input_shape, name="input_image")

    # Block 1
    x = layers.Conv2D(32, 3, padding="same", name="conv1a")(inp)
    x = layers.BatchNormalization(name="bn1a")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, 3, padding="same", name="conv1b")(x)
    x = layers.BatchNormalization(name="bn1b")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, name="pool1")(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding="same", name="conv2a")(x)
    x = layers.BatchNormalization(name="bn2a")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, 3, padding="same", name="conv2b")(x)
    x = layers.BatchNormalization(name="bn2b")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, name="pool2")(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding="same", name="conv3a")(x)
    x = layers.BatchNormalization(name="bn3a")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, 3, padding="same", name="conv3b")(x)
    x = layers.BatchNormalization(name="bn3b")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, name="pool3")(x)
    x = layers.Dropout(0.25)(x)

    # Block 4 — 'last_conv' is used by Grad-CAM
    x = layers.Conv2D(256, 3, padding="same", name="conv4a")(x)
    x = layers.BatchNormalization(name="bn4a")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(256, 3, padding="same", name="last_conv")(x)  # ← Grad-CAM target
    x = layers.BatchNormalization(name="bn4b")(x)
    x = layers.Activation("relu")(x)

    # Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu", name="dense2")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    return Model(inp, out, name="CustomCNN_Malaria")


def build_transfer_model(backbone="VGG16", input_shape=(128, 128, 3)):
    """
    Transfer learning model.
    Phase 1: base frozen, only head trained.
    Phase 2: call fine_tune() to unfreeze top layers.

    Returns: (model, base_model)
    """
    if backbone == "VGG16":
        base = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    else:
        base = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)

    base.trainable = False

    inp = keras.Input(shape=input_shape, name="input_image")
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu", name="dense2")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inp, out, name=f"TransferLearning_{backbone}")
    return model, base


def compile_model(model, lr=1e-4):
    """Compile with Adam + binary crossentropy + key metrics."""
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def fine_tune(model, base_model, unfreeze_from=15):
    """Unfreeze top layers of base model for Phase 2 fine-tuning."""
    base_model.trainable = True
    for layer in base_model.layers[:unfreeze_from]:
        layer.trainable = False
    return compile_model(model, lr=1e-5)
