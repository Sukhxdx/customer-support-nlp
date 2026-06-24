"""Streamlit app for Customer Support Ticket Classification and Auto Reply."""

import json
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CATEGORIES, METRICS_PATH
from src.prediction import predict_ticket_category

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Support Ticket Classification System",
    page_icon="🎫",
    layout="wide",
)

# Example messages users can try instantly
EXAMPLE_MESSAGES = {
    "Payment Issue": "My payment was deducted but order is not confirmed",
    "Order Issue": "My order is delayed since 5 days and tracking is not updating",
    "Refund Request": "I want a refund for my damaged product received yesterday",
    "Account/Login Issue": "I forgot my password and cannot login to my account",
    "Product/Service Complaint": "The product quality is very poor and not as described",
}


@st.cache_resource(show_spinner="Preparing model (auto-trains on first run if needed)...")
def load_trained_model():
    """Load the trained model, training automatically on first run if needed."""
    from src.model_training import ensure_model_trained, load_model

    ensure_model_trained(save_charts=False)
    return load_model()


def load_training_metrics():
    """Load saved training metrics if available."""
    if METRICS_PATH.exists():
        with open(METRICS_PATH, encoding="utf-8") as file:
            return json.load(file)
    return None


# Ensure model is ready before rendering the UI (no manual train.py step required)
load_trained_model()
metrics = load_training_metrics()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("About Project")
    st.markdown(
        """
        **NLP-Based Customer Support Ticket Classification and Auto Reply System**

        This application classifies customer support messages into predefined
        categories and generates professional automated replies using a local
        machine learning model — no external API required.
        """
    )

    st.divider()
    st.subheader("Model Used")
    if metrics:
        st.info(f"Best Model: **{metrics.get('best_model', 'N/A')}**")
        test_metrics = metrics.get("test_metrics", {})
        if test_metrics:
            st.metric("Test Accuracy", f"{test_metrics.get('accuracy', 0):.1%}")
    else:
        st.warning("Model metrics not available yet.")

    st.divider()
    st.subheader("Categories")
    for category in CATEGORIES:
        st.markdown(f"- {category}")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.title("Customer Support Ticket Classification System")
st.markdown(
    "Enter a customer complaint below to classify it and generate an automated support reply."
)

# Example message buttons
st.markdown("**Try an example:**")
example_cols = st.columns(len(EXAMPLE_MESSAGES))
for col, (category, example_text) in zip(example_cols, EXAMPLE_MESSAGES.items()):
    with col:
        if st.button(category, use_container_width=True, key=f"btn_{category}"):
            st.session_state["customer_message"] = example_text

# Text input (pre-filled when an example is selected)
message = st.text_area(
    "Customer Message",
    value=st.session_state.get("customer_message", ""),
    height=150,
    placeholder="Type the customer complaint or support message here...",
    key="customer_message",
)

# Predict button
if st.button("Predict Category & Generate Reply", type="primary"):
    if not message or not message.strip():
        st.warning("Please enter a customer message before predicting.")
    else:
        try:
            model = load_trained_model()
            result = predict_ticket_category(message, model=model)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Predicted Category")
                st.success(result["predicted_category"])

            with col2:
                st.subheader("Confidence Score")
                if result["confidence"] is not None:
                    st.metric("Confidence", f"{result['confidence']:.1%}")
                else:
                    st.info("Confidence not available for this model type.")

            st.subheader("Auto-Generated Support Reply")
            st.info(result["auto_reply"])

            with st.expander("View cleaned message"):
                st.code(result["cleaned_message"])

        except ValueError as exc:
            st.warning(str(exc))
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

# Footer with model info
st.divider()
if metrics:
    st.caption(
        f"Model: {metrics.get('best_model', 'N/A')} | "
        f"Accuracy: {metrics.get('test_metrics', {}).get('accuracy', 0):.1%} | "
        f"Categories: {len(CATEGORIES)}"
    )
