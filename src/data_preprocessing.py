"""Data loading and text preprocessing utilities."""

import re

import pandas as pd

from src.config import (
    CLEANED_TEXT_COLUMN,
    DATA_PATH,
    LABEL_COLUMN,
    PRIORITY_COLUMN,
    TEXT_COLUMN,
    TICKET_ID_COLUMN,
)


def load_data(file_path=None):
    """
    Load customer support tickets from a CSV file.

    Args:
        file_path: Optional path to CSV. Defaults to DATA_PATH from config.

    Returns:
        pandas.DataFrame with ticket records.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns are missing.
    """
    path = file_path or DATA_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. "
            "Please ensure data/customer_support_tickets.csv exists."
        )

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Failed to read CSV file '{path}': {exc}") from exc

    required_columns = {TICKET_ID_COLUMN, TEXT_COLUMN, LABEL_COLUMN, PRIORITY_COLUMN}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in dataset: {sorted(missing)}")

    return df


def check_missing_values(df):
    """
    Check and report missing values in the dataset.

    Returns:
        dict with total missing count and per-column missing counts.
    """
    missing_per_column = df.isnull().sum()
    missing_per_column = missing_per_column[missing_per_column > 0]

    return {
        "total_missing": int(df.isnull().sum().sum()),
        "missing_by_column": missing_per_column.to_dict(),
    }


def clean_text(text):
    """
    Clean a single customer message.

    Steps:
        - Convert to lowercase
        - Remove special characters (keep letters, numbers, spaces)
        - Remove extra spaces
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    # Keep only letters, numbers, and spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_dataframe(df):
    """
    Preprocess the raw dataset and return a cleaned dataframe.

    Steps:
        1. Check for missing values and drop rows with missing key fields
        2. Remove duplicate rows
        3. Clean customer_message text
        4. Add cleaned_message column

    Returns:
        Cleaned pandas.DataFrame.
    """
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty. Cannot preprocess.")

    processed = df.copy()

    # Drop rows with missing values in important columns
    key_columns = [TEXT_COLUMN, LABEL_COLUMN]
    processed = processed.dropna(subset=key_columns)

    # Fill optional missing priority with Medium
    if PRIORITY_COLUMN in processed.columns:
        processed[PRIORITY_COLUMN] = processed[PRIORITY_COLUMN].fillna("Medium")

    # Remove exact duplicate rows
    processed = processed.drop_duplicates()

    # Remove duplicate messages within the same category
    processed = processed.drop_duplicates(subset=[TEXT_COLUMN, LABEL_COLUMN])

    # Clean text and create cleaned_message column
    processed[CLEANED_TEXT_COLUMN] = processed[TEXT_COLUMN].astype(str).apply(clean_text)

    # Remove rows where cleaned message is empty
    processed = processed[processed[CLEANED_TEXT_COLUMN].str.len() > 0]
    processed = processed.reset_index(drop=True)

    return processed


def get_category_distribution(df):
    """Return ticket count per category."""
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Column '{LABEL_COLUMN}' not found in dataframe.")
    return df[LABEL_COLUMN].value_counts().to_dict()
