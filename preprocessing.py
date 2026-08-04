from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


CLASS_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}


@dataclass
class DatasetConfig:
    csv_path: str
    image_dir: str
    image_col: str = "id_code"
    label_col: str = "diagnosis"
    image_ext: str = ".png"
    target_size: Tuple[int, int] = (224, 224)
    test_size: float = 0.2
    random_state: int = 42


# ================= FIXED PREPROCESS FUNCTION =================
def preprocess_fundus_image(image_path, target_size=(224, 224)):

    # 🔥 CRITICAL FIX: convert bytes → string
    if isinstance(image_path, bytes):
        image_path = image_path.decode("utf-8")

    image_bgr = cv2.imread(str(image_path))

    # 🔥 Instead of crashing → skip safely
    if image_bgr is None:
        print(f"⚠️ Skipping corrupted image: {image_path}")
        return np.zeros((224, 224, 3), dtype=np.float32)

    image_bgr = cv2.resize(image_bgr, target_size)

    green_channel = image_bgr[:, :, 1]

    # Morphological enhancement
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    top_hat = cv2.morphologyEx(green_channel, cv2.MORPH_TOPHAT, kernel)
    bottom_hat = cv2.morphologyEx(green_channel, cv2.MORPH_BLACKHAT, kernel)

    enhanced = cv2.add(green_channel, top_hat)
    enhanced = cv2.subtract(enhanced, bottom_hat)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(enhanced)

    # Convert to RGB
    image_rgb = cv2.merge([clahe_image, clahe_image, clahe_image])
    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

    # Normalize
    image_rgb = image_rgb.astype(np.float32) / 255.0

    return image_rgb


# ================= AUGMENT =================
def augment_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    return image


# ================= DATASET =================
def dataframe_to_dataset(
    df,
    image_dir,
    image_col,
    label_col,
    image_ext=".png",
    target_size=(224, 224),
    batch_size=16,
    training=False,
):

    image_paths = [
        str(Path(image_dir) / f"{img_id}{image_ext}")
        for img_id in df[image_col]
    ]

    labels = df[label_col].astype(int).values

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    def process(path, label):
        img = tf.numpy_function(
            preprocess_fundus_image, [path], tf.float32
        )
        img.set_shape((224, 224, 3))

        if training:
            img = augment_image(img)

        return img, label

    ds = ds.map(process, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.shuffle(1000)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds


# ================= CLASS WEIGHTS =================
def compute_class_weights(labels):
    classes, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    num_classes = len(classes)

    return {
        int(cls): float(total / (num_classes * count))
        for cls, count in zip(classes, counts)
    }