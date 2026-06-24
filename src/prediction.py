"""Inference utilities for ticket classification and auto-reply generation."""

import numpy as np
import joblib

from src.config import AUTO_REPLY_TEMPLATES, CATEGORIES_PATH, MODEL_PATH
from src.data_preprocessing import clean_text
from src.model_training import load_model


def generate_auto_reply(category):
    """
    Return a professional auto-reply for the given support category.

    Args:
        category: Predicted ticket category name.

    Returns:
        str: Template-based support reply message.
    """
    return AUTO_REPLY_TEMPLATES.get(
        category,
        "Thank you for contacting us. Our support team will review your message "
        "and respond shortly.",
    )


def _get_confidence(model, cleaned_message, predicted_category):
    """Extract confidence score from the classifier if supported."""
    classifier = model.named_steps["classifier"]
    tfidf_features = model.named_steps["tfidf"].transform([cleaned_message])

    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(tfidf_features)[0]
        classes = classifier.classes_
        best_idx = probabilities.argmax()
        return str(classes[best_idx]), float(probabilities[best_idx])

    if hasattr(classifier, "decision_function"):
        scores = classifier.decision_function(tfidf_features)
        if scores.ndim == 1:
            scores = scores.reshape(1, -1)
        exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        probabilities = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        best_idx = probabilities[0].argmax()
        return str(classifier.classes_[best_idx]), float(probabilities[0][best_idx])

    return predicted_category, None


def predict_ticket_category(message, model=None):
    """
    Classify a customer support message and return category with confidence.

    Args:
        message: Raw customer complaint text.
        model: Optional pre-loaded pipeline. Loads from disk if not provided.

    Returns:
        dict with keys:
            - predicted_category (str)
            - confidence (float or None)
            - cleaned_message (str)
            - auto_reply (str)

    Raises:
        FileNotFoundError: If the trained model file is missing.
        ValueError: If the message is empty after preprocessing.
    """
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at '{MODEL_PATH}'. "
                "Please run 'python train.py' first to train the model."
            )
        model = load_model()

    cleaned = clean_text(message)
    if not cleaned:
        raise ValueError("Message is empty after preprocessing. Please enter valid text.")

    predicted_category = str(model.predict([cleaned])[0])
    predicted_category, confidence = _get_confidence(model, cleaned, predicted_category)

    return {
        "predicted_category": predicted_category,
        "confidence": confidence,
        "cleaned_message": cleaned,
        "auto_reply": generate_auto_reply(predicted_category),
    }


def load_category_labels():
    """Load saved category label list if available."""
    if CATEGORIES_PATH.exists():
        return joblib.load(CATEGORIES_PATH)
    return None
