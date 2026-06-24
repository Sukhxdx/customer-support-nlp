"""Model evaluation and visualization utilities."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from src.config import CATEGORIES, FIGURES_DIR


def print_classification_report(y_true, y_pred, labels=None):
    """Print a detailed sklearn classification report."""
    labels = labels or CATEGORIES
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    print("\nClassification Report")
    print("-" * 60)
    print(report)
    return report


def save_confusion_matrix(y_true, y_pred, labels=None, save_path=None):
    """
    Generate and save a confusion matrix heatmap.

    Default save path: reports/figures/confusion_matrix.png
    """
    labels = labels or CATEGORIES
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = save_path or (FIGURES_DIR / "confusion_matrix.png")

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Confusion Matrix - Support Ticket Classification")
    plt.xlabel("Predicted Category")
    plt.ylabel("Actual Category")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Confusion matrix saved to: {output_path}")
    return output_path


def save_model_comparison_chart(comparison_df, save_path=None):
    """
    Save a bar chart comparing model performance metrics.

    Default save path: reports/figures/model_comparison.png
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = save_path or (FIGURES_DIR / "model_comparison.png")

    plot_df = comparison_df.set_index("model")
    metrics = ["accuracy", "precision", "recall", "f1_score"]

    plt.figure(figsize=(10, 6))
    plot_df[metrics].plot(kind="bar", figsize=(10, 6))
    plt.title("Model Comparison - Classification Metrics")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=15, ha="right")
    plt.legend(loc="lower right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Model comparison chart saved to: {output_path}")
    return output_path


def get_top_features(pipeline, top_n=10):
    """
    Extract top TF-IDF features per class from Logistic Regression or Linear SVM.

    Returns:
        dict mapping category name -> list of top feature words
    """
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = np.array(vectorizer.get_feature_names_out())
    class_labels = classifier.classes_

    # LinearSVC uses coef_ similar to LogisticRegression
    if not hasattr(classifier, "coef_"):
        return {}

    importance_by_class = {}
    for idx, class_name in enumerate(class_labels):
        coefficients = classifier.coef_[idx]
        top_indices = np.argsort(coefficients)[-top_n:][::-1]
        importance_by_class[class_name] = feature_names[top_indices].tolist()

    return importance_by_class


def print_top_features(top_features):
    """Print top important words per category to the console."""
    if not top_features:
        print("\nTop features: Not available for the selected model.")
        return

    print("\nTop Important Words per Category")
    print("-" * 60)
    for category, words in top_features.items():
        print(f"{category}: {', '.join(words)}")


def save_top_features_chart(top_features, save_path=None):
    """Save a chart showing top features per category."""
    if not top_features:
        return None

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = save_path or (FIGURES_DIR / "top_features.png")

    num_classes = len(top_features)
    fig, axes = plt.subplots(num_classes, 1, figsize=(10, 3 * num_classes))

    if num_classes == 1:
        axes = [axes]

    for ax, (category, features) in zip(axes, top_features.items()):
        y_pos = np.arange(len(features))
        ax.barh(y_pos, np.ones(len(features)), color="steelblue")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_title(f"Top Features: {category}")
        ax.set_xlabel("Importance (ranked)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Top features chart saved to: {output_path}")
    return output_path
