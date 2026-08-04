from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def get_classifiers(random_state: int = 42) -> Dict[str, object]:
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "linear_svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="linear", probability=True, class_weight="balanced", random_state=random_state)),
            ]
        ),
        "rbf_svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=random_state)),
            ]
        ),
        "decision_tree": DecisionTreeClassifier(class_weight="balanced", random_state=random_state),
        "naive_bayes": Pipeline([("scaler", StandardScaler()), ("clf", GaussianNB())]),
    }


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def train_and_evaluate_classifiers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Dict[int, str],
    output_dir: str = "artifacts/classifiers",
) -> pd.DataFrame:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, clf in get_classifiers().items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        metrics = evaluate_predictions(y_test, y_pred)

        rows.append({"classifier": name, **metrics})
        joblib.dump(clf, output_path / f"{name}.joblib")

        report = classification_report(
            y_test,
            y_pred,
            target_names=[class_names[i] for i in sorted(class_names)],
            zero_division=0,
        )
        (output_path / f"{name}_report.txt").write_text(report, encoding="utf-8")
        save_confusion_matrix(y_test, y_pred, class_names, output_path / f"{name}_confusion_matrix.png", title=name)

    results_df = pd.DataFrame(rows).sort_values(by="f1_macro", ascending=False).reset_index(drop=True)
    results_df.to_csv(output_path / "classifier_results.csv", index=False)
    return results_df


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Dict[int, str],
    output_path: Path,
    title: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(sorted(class_names)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[class_names[i] for i in sorted(class_names)],
        yticklabels=[class_names[i] for i in sorted(class_names)],
    )
    plt.title(f"Confusion Matrix - {title}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()