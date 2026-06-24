"""Central configuration for the Customer Support NLP project."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths (use Path for cross-platform compatibility on Windows/macOS/Linux)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Data and model artifact paths
DATA_PATH = DATA_DIR / "customer_support_tickets.csv"
MODEL_PATH = MODELS_DIR / "support_ticket_classifier.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
CATEGORIES_PATH = MODELS_DIR / "category_labels.pkl"
METRICS_PATH = MODELS_DIR / "training_metrics.json"

# ---------------------------------------------------------------------------
# Dataset column names
# ---------------------------------------------------------------------------
TICKET_ID_COLUMN = "ticket_id"
TEXT_COLUMN = "customer_message"
CLEANED_TEXT_COLUMN = "cleaned_message"
LABEL_COLUMN = "category"
PRIORITY_COLUMN = "priority"

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2

# TF-IDF settings
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)
MIN_DF = 2

# ---------------------------------------------------------------------------
# Support ticket categories and priority levels
# ---------------------------------------------------------------------------
CATEGORIES = [
    "Payment Issue",
    "Order Issue",
    "Refund Request",
    "Account/Login Issue",
    "Product/Service Complaint",
]

PRIORITY_LEVELS = ["Low", "Medium", "High"]

# ---------------------------------------------------------------------------
# Category-wise auto-reply templates (no external API required)
# ---------------------------------------------------------------------------
AUTO_REPLY_TEMPLATES = {
    "Payment Issue": (
        "We are sorry for the inconvenience. Please share your transaction ID "
        "so our support team can verify your payment status."
    ),
    "Order Issue": (
        "We apologize for the delay. Please share your order ID so we can "
        "check the latest delivery status."
    ),
    "Refund Request": (
        "Your refund request has been noted. Our team will verify the issue "
        "and update you regarding the refund process."
    ),
    "Account/Login Issue": (
        "We are sorry you are facing login issues. Please try resetting your "
        "password or share your registered email ID for support."
    ),
    "Product/Service Complaint": (
        "We apologize for the inconvenience caused. Please share more details "
        "about the issue so our team can resolve it quickly."
    ),
}
