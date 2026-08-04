from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from feature_extraction import (
    HybridTrainingConfig,
    build_hybrid_model,
    train_hybrid_model,
    plot_training_curves,
)
from preprocessing import preprocess_fundus_image


# ================= DATA LOADER =================
def load_data(csv_path, image_dir):
    df = pd.read_csv(csv_path)

    image_paths = []
    labels = []

    for _, row in df.iterrows():
        img_path = Path(image_dir) / f"{row['id_code']}.png"

        if img_path.exists():
            image_paths.append(str(img_path))
            labels.append(int(row["diagnosis"]))

    print(f"Loaded {len(image_paths)} images")
    return image_paths, labels


# ================= DATASET =================
def build_dataset(image_paths, labels, batch_size=16, training=True):

    def process(path, label):
        img = tf.numpy_function(preprocess_fundus_image, [path], tf.float32)
        img.set_shape((224, 224, 3))
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.map(process, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.shuffle(1000)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds


# ================= MAIN =================
def main(args):

    print("📂 Loading data...")
    image_paths, labels = load_data(args.csv_path, args.image_dir)

    print("🔀 Splitting dataset...")
    X_train, X_val, y_train, y_val = train_test_split(
        image_paths,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=42,
    )

    print("📦 Building datasets...")
    train_ds = build_dataset(X_train, y_train, training=True)
    val_ds = build_dataset(X_val, y_val, training=False)

    print("🧠 Building model...")
    config = HybridTrainingConfig()
    model, feature_model, last_conv_layer = build_hybrid_model(config)

    print("🚀 Training model...")
    history = train_hybrid_model(
        model,
        train_ds,
        val_ds,
        y_train,
        epochs=config.epochs,
    )

    print("📊 Saving training curves...")
    plot_training_curves(history)

    print("✅ Training completed successfully!")


# ================= ENTRY =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--image-dir", type=str, required=True)

    args = parser.parse_args()
    main(args)