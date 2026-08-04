from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def make_gradcam_heatmap(
    image_array: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: str,
    pred_index: int | None = None,
) -> np.ndarray:
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output],
    )

    image_tensor = tf.convert_to_tensor(image_array[None, ...], dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_tensor)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize safely
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        return np.zeros_like(heatmap.numpy())
    heatmap /= max_val

    return heatmap.numpy()


def overlay_heatmap(
    heatmap: np.ndarray,
    original_image: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    # Convert heatmap to color
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(
        heatmap_uint8,
        (original_image.shape[1], original_image.shape[0])
    )
    colored_heatmap = cv2.applyColorMap(heatmap_resized, colormap)

    # Convert original image
    original_uint8 = np.uint8(np.clip(original_image * 255.0, 0, 255))

    # CREATE CIRCULAR MASK (KEY FIX)
    h, w, _ = original_uint8.shape
    center = (w // 2, h // 2)
    radius = int(min(center[0], center[1]) * 0.95)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)

    # Apply mask → remove background influence
    colored_heatmap[mask == 0] = 0
    original_uint8[mask == 0] = 0

    # Overlay
    overlay = cv2.addWeighted(original_uint8, 1 - alpha, colored_heatmap, alpha, 0)

    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def save_gradcam_visualization(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    output_path: str | Path,
    title: str = "Grad-CAM",
) -> None:
    overlay = overlay_heatmap(heatmap, original_image)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title("Preprocessed Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap="jet")
    plt.title("Heatmap")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title(title)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()