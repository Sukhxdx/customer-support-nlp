#!/usr/bin/env python3
"""Train and evaluate the customer support ticket classification model."""

import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on the path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_PATH
from src.data_preprocessing import (
    check_missing_values,
    get_category_distribution,
    load_data,
    preprocess_dataframe,
)
from src.evaluation import (
    get_top_features,
    print_classification_report,
    print_top_features,
)
from src.model_training import split_train_test, train_model


def print_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    print_header("NLP-Based Customer Support Ticket Classification - Training")

    # ------------------------------------------------------------------
    # Step 1: Load and preprocess data
    # ------------------------------------------------------------------
    print("\n[Step 1] Loading and preprocessing dataset...")
    raw_df = load_data()
    print(f"  Raw dataset shape: {raw_df.shape}")

    missing_info = check_missing_values(raw_df)
    if missing_info["total_missing"] > 0:
        print(f"  Missing values found: {missing_info['missing_by_column']}")
    else:
        print("  Missing values: None")

    df = preprocess_dataframe(raw_df)
    print(f"  Cleaned dataset shape: {df.shape}")

    print("\n  Category distribution:")
    distribution = get_category_distribution(df)
    for category, count in distribution.items():
        print(f"    - {category}: {count}")

    # ------------------------------------------------------------------
    # Step 2: Train/test split (preview counts)
    # ------------------------------------------------------------------
    print("\n[Step 2] Splitting data (80% train, 20% test, stratified)...")
    X_train, X_test, y_train, y_test = split_train_test(df)
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples:     {len(X_test)}")

    # ------------------------------------------------------------------
    # Step 3–5: Train, evaluate, and save artifacts
    # ------------------------------------------------------------------
    print("\n[Step 3] Training and comparing models...")
    _, metrics = train_model(save_charts=True)

    comparison_df = pd.DataFrame(metrics["model_comparison"])
    best_model_name = metrics["best_model"]

    print("\n  Model Comparison Results:")
    print("-" * 70)
    print(comparison_df.to_string(index=False))
    print("-" * 70)
    print(f"\n  Best Model (by F1-score): {best_model_name}")

    print("\n[Step 4] Evaluating best model on test set...")
    test_metrics = metrics["test_metrics"]

    print(f"\n  Final Test Accuracy:  {test_metrics['accuracy']:.2%}")
    print(f"  Final Test Precision: {test_metrics['precision']:.2%}")
    print(f"  Final Test Recall:    {test_metrics['recall']:.2%}")
    print(f"  Final Test F1-Score:  {test_metrics['f1_score']:.2%}")

    goal_met = test_metrics["accuracy"] >= 0.85
    print(f"\n  85% Accuracy Goal: {'MET' if goal_met else 'NOT MET'}")

    print("\n[Step 5] Model artifacts and evaluation charts saved.")
    from src.model_training import load_model

    model = load_model()
    y_pred = model.predict(X_test)
    print_classification_report(y_test, y_pred)
    top_features = get_top_features(model)
    print_top_features(top_features)

    print_header("Training Complete")
    print(f"  Best model saved to: {MODEL_PATH}")
    print("  Charts saved to:     reports/figures/")
    print("\n  Next step: streamlit run app.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
