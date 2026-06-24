"""Model training utilities with multi-model comparison."""

import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import (
    CATEGORIES,
    CATEGORIES_PATH,
    CLEANED_TEXT_COLUMN,
    LABEL_COLUMN,
    MAX_FEATURES,
    METRICS_PATH,
    MIN_DF,
    MODEL_PATH,
    MODELS_DIR,
    NGRAM_RANGE,
    RANDOM_STATE,
    TEST_SIZE,
)


def build_tfidf_vectorizer():
    """Create a TF-IDF vectorizer with project settings."""
    return TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        stop_words="english",
        sublinear_tf=True,
    )


def get_model_candidates():
    """
    Return a dictionary of model name -> sklearn classifier.

    All models are paired with the same TF-IDF vectorizer for fair comparison.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            solver="lbfgs",
        ),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(
            random_state=RANDOM_STATE,
            class_weight="balanced",
            max_iter=2000,
        ),
    }


def build_pipeline(classifier):
    """Build a complete TF-IDF + classifier pipeline."""
    return Pipeline(
        [
            ("tfidf", build_tfidf_vectorizer()),
            ("classifier", classifier),
        ]
    )


def split_train_test(df):
    """
    Split cleaned dataframe into train and test sets with stratification.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df[CLEANED_TEXT_COLUMN]
    y = df[LABEL_COLUMN]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def evaluate_predictions(y_true, y_pred):
    """Compute standard classification metrics."""
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(
            precision_score(y_true, y_pred, average="macro", zero_division=0), 4
        ),
        "recall": round(
            recall_score(y_true, y_pred, average="macro", zero_division=0), 4
        ),
        "f1_score": round(
            f1_score(y_true, y_pred, average="macro", zero_division=0), 4
        ),
    }


def compare_models(X_train, X_test, y_train, y_test):
    """
    Train and compare all candidate models.

    Returns:
        comparison_df: DataFrame with metrics for each model
        best_model_name: Name of the model with highest F1-score
        best_pipeline: Trained best pipeline
        all_pipelines: Dict of all trained pipelines
    """
    candidates = get_model_candidates()
    results = []
    all_pipelines = {}

    for name, classifier in candidates.items():
        pipeline = build_pipeline(classifier)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics = evaluate_predictions(y_test, y_pred)
        metrics["model"] = name
        results.append(metrics)
        all_pipelines[name] = pipeline

    comparison_df = pd.DataFrame(results)[
        ["model", "accuracy", "precision", "recall", "f1_score"]
    ]
    comparison_df = comparison_df.sort_values("f1_score", ascending=False).reset_index(
        drop=True
    )

    best_model_name = comparison_df.iloc[0]["model"]
    best_pipeline = all_pipelines[best_model_name]

    return comparison_df, best_model_name, best_pipeline, all_pipelines


def save_best_model(pipeline, comparison_df, best_model_name, y_test, y_pred):
    """Save the best pipeline, category labels, and training metrics."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(list(pipeline.named_steps["classifier"].classes_), CATEGORIES_PATH)

    metrics = {
        "best_model": best_model_name,
        "test_metrics": evaluate_predictions(y_test, y_pred),
        "model_comparison": comparison_df.to_dict(orient="records"),
        "categories": CATEGORIES,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    return metrics


def load_model():
    """Load the saved best model pipeline."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. "
            "Please run 'python train.py' first to train the model."
        )
    return joblib.load(MODEL_PATH)


def train_model(save_charts=True):
    """
    Run the full training pipeline and persist model artifacts.

    Args:
        save_charts: When True, write evaluation charts to reports/figures/.

    Returns:
        tuple: (best_pipeline, metrics dict)
    """
    from src.data_preprocessing import load_data, preprocess_dataframe
    from src.evaluation import (
        get_top_features,
        save_confusion_matrix,
        save_model_comparison_chart,
        save_top_features_chart,
    )

    raw_df = load_data()
    df = preprocess_dataframe(raw_df)
    X_train, X_test, y_train, y_test = split_train_test(df)
    comparison_df, best_model_name, best_pipeline, _ = compare_models(
        X_train, X_test, y_train, y_test
    )
    y_pred = best_pipeline.predict(X_test)
    metrics = save_best_model(
        best_pipeline, comparison_df, best_model_name, y_test, y_pred
    )

    if save_charts:
        save_confusion_matrix(y_test, y_pred)
        save_model_comparison_chart(comparison_df)
        top_features = get_top_features(best_pipeline)
        save_top_features_chart(top_features)

    return best_pipeline, metrics


def ensure_model_trained(save_charts=False):
    """
    Train and save the model if the .pkl artifact is missing.

    Returns:
        bool: True if training ran, False if the model already existed.
    """
    if MODEL_PATH.exists():
        return False

    train_model(save_charts=save_charts)
    return True
