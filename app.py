from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from classifiers import get_classifiers
from feature_extraction import HybridTrainingConfig, build_hybrid_model
from gradcam import make_gradcam_heatmap, overlay_heatmap
from preprocessing import CLASS_NAMES, preprocess_fundus_image


ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "hybrid_model.keras"
BEST_CLASSIFIER_PATH = ARTIFACTS_DIR / "classifiers" / "random_forest.joblib"


@st.cache_resource
def load_models():
    config = HybridTrainingConfig()
    hybrid_model, feature_model, last_conv_layer_name = build_hybrid_model(config)

    if MODEL_PATH.exists():
        hybrid_model = tf.keras.models.load_model(MODEL_PATH)

        try:
            feature_layer = hybrid_model.get_layer("fused_features")
        except:
            feature_layer = hybrid_model.get_layer("concatenate_2")

        feature_model = tf.keras.Model(
            inputs=hybrid_model.input,
            outputs=feature_layer.output,
        )

    classifier = None
    if BEST_CLASSIFIER_PATH.exists():
        classifier = joblib.load(BEST_CLASSIFIER_PATH)

    return hybrid_model, feature_model, classifier, last_conv_layer_name


def main():
    st.set_page_config(page_title="Diabetic Retinopathy Detection", layout="wide")

    st.title("Diabetic Retinopathy Detection and Grading System")
    st.write(
        "Upload a retinal fundus image to predict the diabetic retinopathy grade and visualize Grad-CAM attention."
    )

    uploaded_file = st.file_uploader("Upload fundus image", type=["png", "jpg", "jpeg"])
    if uploaded_file is None:
        return

    # Save uploaded image
    temp_path = Path("artifacts") / uploaded_file.name
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_bytes(uploaded_file.read())

    # Preprocess image
    image = preprocess_fundus_image(temp_path)

    # Load models
    hybrid_model, feature_model, classifier, last_conv_layer_name = load_models()

    # Prediction
    probabilities = hybrid_model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
    probabilities = probabilities.flatten()

    deep_pred = int(np.argmax(probabilities))

    fused_features = feature_model.predict(np.expand_dims(image, axis=0), verbose=0)

    if classifier is not None:
        pred_class = int(classifier.predict(fused_features)[0])
        confidence = float(np.max(classifier.predict_proba(fused_features)))
        model_used = "Classical classifier on fused InceptionV3 + ResNet50 features"
    else:
        pred_class = deep_pred
        confidence = float(np.max(probabilities))
        model_used = "Hybrid deep model"

    # Grad-CAM
    heatmap = make_gradcam_heatmap(
        image, hybrid_model, last_conv_layer_name, pred_index=pred_class
    )
    overlay = overlay_heatmap(heatmap, image)

    # ===== UI =====

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(image, width="stretch")   # FIXED

    with col2:
        st.subheader("Grad-CAM Overlay")
        st.image(overlay, width="stretch")  # FIXED
        st.caption("Red regions show where the model is focusing.")

    # Prediction info
    st.subheader("Prediction Details")
    st.write(f"Model: {model_used}")
    st.write(f"Predicted Class: {pred_class}")
    st.write(f"Label: {CLASS_NAMES[pred_class]}")
    st.write(f"Confidence: {confidence:.4f}")

    # All probabilities
    st.subheader("Class Probabilities")
    for i, p in enumerate(probabilities):
        st.write(f"{CLASS_NAMES[i]}: {p:.2%}")

    # DataFrame for chart
    df = pd.DataFrame(list(zip(CLASS_NAMES, probabilities)), columns=["Class", "Probability"])
    st.bar_chart(df.set_index("Class"))


if __name__ == "__main__":
    main()