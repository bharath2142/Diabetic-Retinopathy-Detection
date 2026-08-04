from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import InceptionV3, ResNet50
from tensorflow.keras.applications.inception_v3 import preprocess_input as inc_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as res_preprocess
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from preprocessing import compute_class_weights


# ================= CONFIG =================
@dataclass
class HybridTrainingConfig:
    input_shape: Tuple[int, int, int] = (224, 224, 3)  # keep 224 for compatibility
    num_classes: int = 5
    learning_rate: float = 1e-4
    dense_units: int = 256
    dropout_rate: float = 0.4
    epochs: int = 15
    fine_tune_at_inception: int = 249
    fine_tune_at_resnet: int = 143


# ================= MODEL =================
def build_hybrid_model(config: HybridTrainingConfig):

    inputs = layers.Input(shape=config.input_shape)

    # Apply preprocessing separately (IMPORTANT)
    inc_input = layers.Lambda(inc_preprocess)(inputs)
    res_input = layers.Lambda(res_preprocess)(inputs)

    # Load pretrained models
    inception = InceptionV3(include_top=False, weights="imagenet", input_tensor=inc_input)
    resnet = ResNet50(include_top=False, weights="imagenet", input_tensor=res_input)

    # Freeze layers (partial fine-tuning)
    for layer in inception.layers[:config.fine_tune_at_inception]:
        layer.trainable = False

    for layer in resnet.layers[:config.fine_tune_at_resnet]:
        layer.trainable = False

    # Feature extraction
    inc_feat = layers.GlobalAveragePooling2D()(inception.output)
    res_feat = layers.GlobalAveragePooling2D()(resnet.output)

    # Fusion
    x = layers.Concatenate()([inc_feat, res_feat])
    x = layers.BatchNormalization()(x)
    x = layers.Dense(config.dense_units, activation="relu")(x)
    x = layers.Dropout(config.dropout_rate)(x)

    outputs = layers.Dense(config.num_classes, activation="softmax")(x)

    # Models
    model = Model(inputs, outputs)
    feature_model = Model(inputs, x)  # used for feature extraction / Grad-CAM support

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model, feature_model, "mixed10"  # keep for Grad-CAM compatibility


# ================= TRAIN =================
def train_hybrid_model(model, train_ds, val_ds, y_train, epochs=15):

    Path("artifacts").mkdir(exist_ok=True)

    class_weights = compute_class_weights(y_train)

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(patience=2),
        ModelCheckpoint("artifacts/hybrid_model.keras", save_best_only=True),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    return history


# ================= PLOT =================
def plot_training_curves(history):

    h = history.history
    epochs = range(len(h["loss"]))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, h["accuracy"], label="train")
    plt.plot(epochs, h["val_accuracy"], label="val")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, h["loss"], label="train")
    plt.plot(epochs, h["val_loss"], label="val")
    plt.title("Loss")
    plt.legend()

    plt.savefig("artifacts/training_curves.png")
    plt.close()