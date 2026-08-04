# Diabetic Retinopathy Detection and Grading System

Research-oriented end-to-end diabetic retinopathy (DR) detection pipeline using hybrid deep learning feature fusion from **InceptionV3 + ResNet50**, followed by classical machine learning classifiers and Grad-CAM explainability.

## Features

- 5-class DR grading: `0` No DR, `1` Mild, `2` Moderate, `3` Severe, `4` Proliferative DR
- Fundus preprocessing with:
  - resize to `224x224`
  - green channel extraction
  - top-hat and bottom-hat transforms
  - CLAHE enhancement
  - normalization
- Data augmentation with rotation, flips, zoom, and intensity jitter
- Transfer learning with ImageNet-pretrained `InceptionV3` and `ResNet50`
- Feature fusion through concatenation of global pooled deep features
- Classical classifiers on fused features:
  - Random Forest
  - Linear SVM
  - RBF SVM
  - Decision Tree
  - Naive Bayes
- Evaluation with accuracy, precision, recall, F1-score, and confusion matrices
- Grad-CAM visualization for InceptionV3 attention maps
- Streamlit deployment for single-image inference

## Project Structure

```text
preprocessing.py
feature_extraction.py
classifiers.py
gradcam.py
train.py
app.py
main.ipynb
requirements.txt
README.md
```

## Dataset Format

The code supports APTOS 2019 and EyePACS-style CSV files.

Expected CSV columns by default:

- `id_code`: image identifier or image filename
- `diagnosis`: integer class label in `[0, 1, 2, 3, 4]`

If your CSV uses different column names, pass them through CLI arguments.

Example layout:

```text
dataset/
  train.csv
  train_images/
    000c1434d8d7.png
    001639a390f0.png
```

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py --csv-path dataset/train.csv --image-dir dataset/train_images --epochs 15
```

Optional arguments:

- `--image-col`
- `--label-col`
- `--image-ext`
- `--batch-size`
- `--output-dir`

## Outputs

The training script saves:

- best hybrid deep model: `artifacts/hybrid_model.keras`
- training curves: `artifacts/training_curves.png`
- classical classifier models: `artifacts/classifiers/*.joblib`
- classifier reports and confusion matrices
- summary metrics CSV
- Grad-CAM visualizations

## Streamlit App

```bash
streamlit run app.py
```

The app shows:

- uploaded retinal image
- predicted DR class and label
- confidence score
- Grad-CAM heatmap overlay

## Notes

- The hybrid deep model is trained end to end with class weighting to address imbalance.
- Classical classifiers are trained on the fused feature vectors extracted from the hybrid backbone.
- For best results on EyePACS, use cleaned fundus images and enough compute for fine-tuning.